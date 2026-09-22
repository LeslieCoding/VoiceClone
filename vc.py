# -*- coding: utf-8 -*-
# VoiceClone 零样本语音克隆流水线（基于 F5-TTS，2025 新一代架构）：
#   数据准备：MelBand RoFormer 人声分离 -> 去混响 -> 语音切分 -> faster-whisper 识别 -> 标注确认
#   推理合成：F5-TTS 零样本克隆（无需训练，给一段参考音直接合成；vocos 声码器，无电音）
# 完全自包含，不依赖任何外部项目。用法见各子命令 --help，推荐双击 go-vc.bat 启动。
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, ::1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HF 镜像，避免直连超时
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # 屏蔽 Windows 符号链接警告噪音
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for _stream in (sys.stdout, sys.stderr):
    try:
        if not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PYTHON = sys.executable  # 本项目 venv 的 Python（由 go-vc.bat 启动）

# 自带 ffmpeg（tools/ffmpeg-bin），无需系统安装
FFMPEG_BIN = os.path.join(PROJECT_ROOT, "tools", "ffmpeg-bin")
if os.path.isdir(FFMPEG_BIN):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

SLICE_DEFAULTS = dict(threshold=-34, min_length=4000, min_interval=300,
                      hop_size=10, max_sil_kept=500, _max=0.9, alpha=0.25)

SLICER_DIR = os.path.abspath("output/slicer_opt")
ASR_DIR = os.path.abspath("output/asr_opt")
UVR_DIR = os.path.abspath("output/uvr5_opt")
DEREVERB_DIR = os.path.abspath("output/dereverb_opt")
TTS_DIR = os.path.abspath("output/tts")
DEFAULT_LIST = os.path.join(ASR_DIR, "slicer_opt.list")
SUBFIX_PORT = 9871

F5_MODEL = "F5TTS_v1_Base"  # 中英双语零样本克隆模型


def die(msg, code=1):
    print(f"\n[错误] {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd, desc, cwd=None):
    """运行子进程，失败即中止（打印完整命令便于排查）"""
    print(f"执行命令: {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if r.returncode != 0:
        die(f"{desc} 失败（退出码 {r.returncode}），请查看上方输出")


def check_file(path, desc):
    if not os.path.isfile(path):
        die(f"{desc}不存在: {path}")
    return os.path.abspath(path)


class Stager:
    """阶段标题打印器：===== [N/total] 阶段名 ====="""

    def __init__(self, total):
        self.total = total
        self.n = 0

    def next(self, title):
        self.n += 1
        print(f"\n{'='*20} [{self.n}/{self.total}] {title} {'='*20}", flush=True)


# ---------------------------------------------------------------- 数据准备

def cmd_prepare(args):
    import separate
    import sentence_slicer
    import annotate

    st = Stager(5 if not args.skip_uvr else 3)
    src = os.path.abspath(args.source)
    if not os.path.isfile(src):
        die(f"音源文件不存在: {src}")

    # [1] 人声分离（MelBand RoFormer）
    if args.skip_uvr:
        st.next("人声分离（--skip-uvr，跳过）")
        vocal_path = src
        print(f"输入已是纯人声，直接使用: {vocal_path}")
    else:
        st.next("人声分离（MelBand RoFormer）")
        vocal_path = os.path.abspath(separate.separate_vocal(src, UVR_DIR))

        # [2] 去混响：分离残留的混响/伴奏痕迹是合成电音的主要来源（实验证实）
        st.next("去混响（消除分离残留，防止电音）")
        deref = separate.dereverb(vocal_path, DEREVERB_DIR)
        stem = os.path.splitext(os.path.basename(src))[0]
        vocal_path = os.path.abspath(os.path.join(DEREVERB_DIR, f"{stem}_clean.wav"))
        shutil.copyfile(deref, vocal_path)
        print(f"干净人声: {vocal_path}")

    # [3] 句子级智能切分（含语音识别，完整句子不切断，音调相近的句子贴在一起）
    st.next("句子级智能切分 + 语音识别")
    n_old = len(glob.glob(os.path.join(SLICER_DIR, "*.wav")))
    if n_old > 0:
        print(f"提示：输出目录 {SLICER_DIR} 已有 {n_old} 个 wav 文件，新切片将与其并存（如需干净数据请先手动清空）。")
    n_slices, list_path = sentence_slicer.sentence_slice(
        vocal_path, SLICER_DIR, DEFAULT_LIST, model_size=args.whisper_model)
    check_file(list_path, "ASR 标注文件(.list)")

    # [4] 标注确认（唯一需要人工确认的环节）
    st.next("标注确认")
    annotate.run_annotator(list_path, port=SUBFIX_PORT)

    st.next("完成")
    print(f"数据准备完成！共 {n_slices} 个句子片段，标注文件: {list_path}")
    print("接下来可直接运行推理: vc.py infer")


# ---------------------------------------------------------------- 参考音频挑选

def wav_duration(path):
    import soundfile as sf
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0


def pick_reference(list_path, min_sec=4.0, max_sec=15.0):
    """从 .list 标注中自动挑选最优参考音：时长 4~15 秒、文本非空，优先 8~12 秒"""
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) >= 4 and parts[3]:
                wav = parts[0]
                if not os.path.isabs(wav):
                    wav = os.path.abspath(wav)
                if os.path.isfile(wav):
                    entries.append((wav, parts[3]))
    if not entries:
        die(f"标注文件 {list_path} 中没有可用的音频条目")

    scored = []
    for wav, text in entries:
        dur = wav_duration(wav)
        if min_sec <= dur <= max_sec:
            # 8~12 秒最优，其余按偏离度扣分
            ideal = 0.0 if 8.0 <= dur <= 12.0 else min(abs(dur - 8.0), abs(dur - 12.0))
            scored.append((ideal, -dur, wav, text, dur))
    if not scored:
        # 没有合格时放宽到最接近期望区间的
        cand = [(abs(wav_duration(w) - 10.0), w, t, wav_duration(w)) for w, t in entries]
        cand.sort()
        _, wav, text, dur = cand[0]
        print(f"提示：没有 4~15 秒的切片，改用时长最接近 10 秒的: {os.path.basename(wav)} ({dur:.1f}s)")
        return wav, text
    scored.sort()
    _, _, wav, text, dur = scored[0]
    print(f"自动选定参考音频: {wav}\n  时长 {dur:.1f}s | 文本: {text}")
    return wav, text


# ---------------------------------------------------------------- 推理合成

def load_f5(model_name):
    from f5_tts.api import F5TTS
    print(f"正在加载 {model_name}（首次运行会从 hf-mirror 下载模型，约 1.3GB）...")
    return F5TTS(model=model_name)


def synth(f5, ref_file, ref_text, gen_text, out_path, nfe_step=32, cfg_strength=2.0, speed=1.0, seed=-1):
    import soundfile as sf
    # F5-TTS 内部 ODE 采样器偶发 "t must be strictly increasing" 断言（上游已知数值边界），
    # 遇到时换随机种子自动重试，最多 3 次
    last_err = None
    for attempt in range(3):
        try:
            use_seed = seed if (attempt == 0 and seed >= 0) else None  # None = F5 内部随机
            wav, sr, _ = f5.infer(
                ref_file=ref_file, ref_text=ref_text, gen_text=gen_text,
                nfe_step=nfe_step, cfg_strength=cfg_strength, speed=speed, seed=use_seed,
            )
            break
        except AssertionError as e:
            if "strictly increasing" in str(e):
                last_err = e
                print(f"[提示] 采样器偶发断言（第 {attempt+1}/3 次），换随机种子重试...", flush=True)
                continue
            raise
    else:
        raise last_err
    sf.write(out_path, wav, sr)
    return out_path


def cmd_infer(args):
    list_path = args.list or DEFAULT_LIST

    if args.ref:
        ref_file = check_file(args.ref, "参考音频")
        ref_text = args.ref_text
        if ref_text is None and os.path.isfile(list_path):
            ref_text = find_text_in_list(list_path, ref_file) or ""
            if not ref_text:
                print("提示：未在标注文件中找到该音频的文本，ref_text 为空（F5-TTS 可自动识别，但显式文本更准）")
        if ref_text is None:
            ref_text = ""
    else:
        if not os.path.isfile(list_path):
            die(f"标注文件不存在: {list_path}\n请先运行: vc.py prepare <音源文件>（或用 --ref 显式指定参考音频）")
        ref_file, ref_text = pick_reference(list_path)

    os.makedirs(TTS_DIR, exist_ok=True)
    f5 = load_f5(args.model)

    texts = list(args.text or [])
    if not texts:
        print("\n请输入要合成的文本（支持多行，空行结束输入；输入 quit 退出）：")
    idx = len(glob.glob(os.path.join(TTS_DIR, "*.wav"))) + 1
    while True:
        if not texts:
            lines = []
            try:
                while True:
                    line = input()
                    if not line.strip():
                        break
                    if line.strip().lower() == "quit":
                        print("已退出。")
                        return
                    lines.append(line)
            except EOFError:
                return
            if not lines:
                return
            gen_text = "\n".join(lines)
        else:
            gen_text = texts.pop(0)

        out_path = os.path.join(TTS_DIR, f"tts_{idx:04d}.wav")
        print("合成中...", flush=True)
        synth(f5, ref_file, ref_text, gen_text, out_path,
              nfe_step=args.steps, cfg_strength=args.cfg, speed=args.speed, seed=args.seed)
        idx += 1
        print(f"已保存: {out_path}\n")
        if args.text:
            continue
        print("继续输入下一段文本（空行结束，quit 退出）：")


def find_text_in_list(list_path, wav_path):
    target = os.path.normcase(os.path.abspath(wav_path))
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) >= 4:
                p = parts[0]
                if not os.path.isabs(p):
                    p = os.path.abspath(p)
                if os.path.normcase(p) == target:
                    return parts[3]
    return None


# ---------------------------------------------------------------- 网页界面

def cmd_ui(args):
    """启动自研中文推理网页（只保留合成功能）"""
    import infer_ui
    try:
        infer_ui.run_ui(DEFAULT_LIST, TTS_DIR, port=args.port or 9873)
    except KeyboardInterrupt:
        print("\n推理网页已关闭。")


# ---------------------------------------------------------------- 一条龙

def cmd_all(args):
    cmd_prepare(args)
    print("\n数据准备完成，直接进入推理合成。")
    args.list = DEFAULT_LIST
    args.ref = None
    args.ref_text = None
    args.text = None
    args.steps = 32
    args.cfg = 2.0
    args.speed = 1.0
    args.seed = -1
    cmd_infer(args)


# ---------------------------------------------------------------- 向导菜单

def split_dropped_paths(line):
    """把拖入窗口的一行拆成路径列表：支持多个带引号路径、单个未加引号含空格路径"""
    line = line.strip()
    if not line:
        return []
    # 整行去掉首尾引号后就是一个存在的文件（未加引号但路径含空格的情况）
    single = line.strip('"').strip("'")
    if os.path.isfile(single):
        return [single]
    # 否则按引号/空格拆成多个
    tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|([^\s"\']+)', line)
    return [next(g for g in m if g) for m in tokens]


def wizard():
    while True:
        print("\n" + "=" * 50)
        print("  VoiceClone 零样本语音克隆（F5-TTS）")
        print("=" * 50)
        print("  1) 推理合成（终端直接合成，无需训练）")
        print("  2) 打开推理合成网页版（在浏览器里合成，界面更直观）")
        print("  3) 数据准备（新音源用：人声分离 → 去混响 → 切分 → 识别 → 标注）")
        print("  4) 只打开标注校对页（切分和识别已完成、只需校对时用）")
        print("  0) 退出")
        try:
            choice = input("\n请选择 [0-4]: ").strip()
        except EOFError:
            return

        if choice == "0":
            return
        try:
            if choice == "1":
                args = argparse.Namespace(list=None, ref=None, ref_text=None, text=None,
                                          model=F5_MODEL, steps=32, cfg=2.0, speed=1.0, seed=-1)
                cmd_infer(args)
            elif choice == "2":
                cmd_ui(argparse.Namespace(port=None, share=False))
            elif choice == "3":
                srcs = split_dropped_paths(input("请输入音源文件路径（可一次拖入多个文件）: "))
                if not srcs:
                    print("未输入路径。")
                    continue
                missing = [s for s in srcs if not os.path.isfile(s)]
                if missing:
                    print("以下文件不存在，将跳过：")
                    for m in missing:
                        print(f"  {m}")
                    srcs = [s for s in srcs if os.path.isfile(s)]
                if not srcs:
                    continue
                if len(srcs) > 1:
                    print(f"共 {len(srcs)} 个音源，将依次处理。")
                pure = input("音源是否已是纯人声（无背景音乐）？[y/N]: ").strip().lower() == "y"
                for i, s in enumerate(srcs, 1):
                    if len(srcs) > 1:
                        print(f"\n{'~'*15} 第 {i}/{len(srcs)} 个音源: {os.path.basename(s)} {'~'*15}")
                    args = argparse.Namespace(source=s, skip_uvr=pure, whisper_model="large-v3-turbo")
                    try:
                        cmd_prepare(args)
                    except (SystemExit, Exception) as e:
                        print(f"\n[提示] 该音源处理失败（{e}），继续处理下一个。")
            elif choice == "4":
                import annotate
                list_path = check_file(DEFAULT_LIST, "标注文件(.list)")
                annotate.run_annotator(list_path, port=SUBFIX_PORT)
            else:
                print("无效选择。")
        except SystemExit:
            print("\n[提示] 上一步出错已中止，回到主菜单。")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[提示] 上一步出错（{e}），已回到主菜单。")


# ---------------------------------------------------------------- 入口

def main():
    p = argparse.ArgumentParser(prog="vc.py", description="VoiceClone 零样本语音克隆流水线")
    sub = p.add_subparsers(dest="command")

    p_prepare = sub.add_parser("prepare", help="数据准备（分离/去混响/切分/识别/标注）")
    p_prepare.add_argument("source", help="音源文件路径")
    p_prepare.add_argument("--skip-uvr", action="store_true", help="输入已是纯人声，跳过分离和去混响")
    p_prepare.add_argument("--whisper-model", default="large-v3-turbo",
                           help="Whisper 识别模型（默认 large-v3-turbo，可换 large-v3）")

    p_infer = sub.add_parser("infer", help="推理合成（零样本克隆）")
    p_infer.add_argument("--list", help=f"标注文件路径（默认 {DEFAULT_LIST}）")
    p_infer.add_argument("--ref", help="参考音频路径（默认从标注文件自动挑选最优切片）")
    p_infer.add_argument("--ref-text", help="参考音频文本（默认自动从标注文件查找）")
    p_infer.add_argument("--text", action="append", help="要合成的文本（可多次指定；不指定则进入交互输入）")
    p_infer.add_argument("--model", default=F5_MODEL, help=f"F5-TTS 模型名（默认 {F5_MODEL}）")
    p_infer.add_argument("--steps", type=int, default=32, help="采样步数 NFE（默认 32，越大越精细越慢）")
    p_infer.add_argument("--cfg", type=float, default=2.0, help="CFG 强度（默认 2.0）")
    p_infer.add_argument("--speed", type=float, default=1.0, help="语速（默认 1.0）")
    p_infer.add_argument("--seed", type=int, default=-1, help="随机种子（-1 随机）")

    p_ui = sub.add_parser("ui", help="打开 F5-TTS 网页版界面")
    p_ui.add_argument("--port", type=int, help="网页端口（默认自动分配）")
    p_ui.add_argument("--share", action="store_true", help="生成公网分享链接")

    p_anno = sub.add_parser("annotate", help="只打开标注校对页（切片和识别已完成时使用，避免重跑）")
    p_anno.add_argument("--list", help=f"标注文件路径（默认 {DEFAULT_LIST}）")

    p_all = sub.add_parser("all", help="一条龙（数据准备 + 推理合成）")
    p_all.add_argument("source", help="音源文件路径")
    p_all.add_argument("--skip-uvr", action="store_true", help="输入已是纯人声")
    p_all.add_argument("--whisper-model", default="large-v3-turbo", help="Whisper 识别模型")

    args = p.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "ui":
        cmd_ui(args)
    elif args.command == "annotate":
        import annotate
        list_path = check_file(args.list or DEFAULT_LIST, "标注文件(.list)")
        annotate.run_annotator(list_path, port=SUBFIX_PORT)
    elif args.command == "all":
        cmd_all(args)
    else:
        wizard()


if __name__ == "__main__":
    main()
