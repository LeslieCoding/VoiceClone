# -*- coding: utf-8 -*-
# VoiceClone 的 GPT-SoVITS v4 后端：训练数据格式化（1Aa/1Ab/1Ac）、SoVITS/GPT 两步训练、
# 训练权重推理。代码在 gsv/ 目录（GPT-SoVITS 官方源码 + 单卡/编码补丁），
# 本模块负责路径配置、子进程调度和电音防线。
import glob
import json
import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GSV_ROOT = os.path.join(PROJECT_ROOT, "gsv")
GSV_PKG = os.path.join(GSV_ROOT, "GPT_SoVITS")
PRETRAINED = os.path.join(GSV_PKG, "pretrained_models")
CONFIGS_DIR = os.path.join(GSV_PKG, "configs")
TEMP_DIR = os.path.join(GSV_ROOT, "TEMP")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

GSV_VERSION = "v4"
PRETRAINED_S2G = os.path.join(PRETRAINED, "gsv-v4-pretrained", "s2Gv4.pth")
PRETRAINED_S1 = os.path.join(PRETRAINED, "s1v3.ckpt")
BERT_DIR = os.path.join(PRETRAINED, "chinese-roberta-wwm-ext-large")
CNHUBERT_DIR = os.path.join(PRETRAINED, "chinese-hubert-base")
SOVITS_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "SoVITS_weights_v4")
GPT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "GPT_weights_v4")

PYTHON = sys.executable


def gsv_env(extra=None):
    """训练/格式化子进程环境：UTF-8 + 三条 sys.path（gsv 根、包目录、eres2net）"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join([
        GSV_ROOT, GSV_PKG, os.path.join(GSV_PKG, "eres2net"),
        env.get("PYTHONPATH", ""),
    ])
    env.setdefault("no_proxy", "localhost,127.0.0.1,::1")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 减少显存碎片
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_gsv(script, desc, extra_env=None, args=()):
    """跑 gsv 下的脚本，输出直接进终端（训练进度条可见），失败即抛"""
    cmd = [PYTHON, "-s", script, *args]
    print(f"执行命令: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=GSV_ROOT, env=gsv_env(extra_env))
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 失败（退出码 {r.returncode}），请查看上方输出")


def setup_gsv_path():
    """把 gsv 三个路径插进 sys.path（同进程内做推理用）"""
    for p in (GSV_ROOT, GSV_PKG, os.path.join(GSV_PKG, "eres2net")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------- 数据质量门（电音防线·训练前）

def quality_gate(list_path, min_sec=1.0, max_sec=20.0, min_rms=0.01, max_clip_ratio=0.01):
    """过滤不适合训练的切片：过短/过长、近乎静音、削波严重。
    直接改写 list 文件，返回 (保留数, 剔除原因列表)。坏切片是电音和音色漂移的温床。"""
    import numpy as np
    import soundfile as sf

    kept, dropped = [], []
    with open(list_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        wav = parts[0] if parts else ""
        reason = None
        if not wav or not os.path.isfile(wav):
            reason = "文件不存在"
        else:
            try:
                data, sr = sf.read(wav, dtype="float32")
                dur = len(data) / sr
                rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
                clip = float(np.mean(np.abs(data) > 0.99)) if len(data) else 1.0
                if dur < min_sec or dur > max_sec:
                    reason = f"时长 {dur:.1f}s 超出 {min_sec}~{max_sec}s"
                elif rms < min_rms:
                    reason = f"音量过低（RMS {rms:.4f}），接近静音"
                elif clip > max_clip_ratio:
                    reason = f"削波比例 {clip*100:.1f}% 过高"
            except Exception as e:
                reason = f"读取失败: {e}"
        if reason:
            dropped.append((os.path.basename(wav), reason))
        else:
            kept.append(ln)
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    return len(kept), dropped


# ---------------------------------------------------------------- 格式化（1Aa/1Ab/1Ac）

def run_format(exp_name, list_path, wav_dir):
    """一键三连：文本+BERT 特征 → cnhubert 特征 → 语义 token，产物在 logs/{exp}/"""
    import torch
    opt_dir = os.path.join(LOGS_DIR, exp_name)
    base_env = {
        "inp_text": os.path.abspath(list_path),
        "inp_wav_dir": os.path.abspath(wav_dir),
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": str(torch.cuda.is_available()),
    }
    pd_dir = os.path.join(GSV_PKG, "prepare_datasets")

    print("\n----- [1Aa] 文本分词与 BERT 特征提取 -----", flush=True)
    run_gsv(os.path.join(pd_dir, "1-get-text.py"), "文本特征提取",
            dict(base_env, bert_pretrained_dir=BERT_DIR))
    _merge_part(opt_dir, "2-name2text", "txt")

    print("\n----- [1Ab] cnhubert 语音自监督特征提取 -----", flush=True)
    run_gsv(os.path.join(pd_dir, "2-get-hubert-wav32k.py"), "cnhubert 特征提取",
            dict(base_env, cnhubert_base_dir=CNHUBERT_DIR))

    print("\n----- [1Ac] 语义 token 提取 -----", flush=True)
    run_gsv(os.path.join(pd_dir, "3-get-semantic.py"), "语义 token 提取",
            dict(base_env, pretrained_s2G=PRETRAINED_S2G,
                 s2config_path=os.path.join(CONFIGS_DIR, "s2.json")))
    _merge_part(opt_dir, "6-name2semantic", "tsv")
    return opt_dir


def _merge_part(opt_dir, stem, ext):
    """all_parts=1 时产物是 {stem}-0.{ext}，合并（重命名）为 {stem}.{ext}"""
    part = os.path.join(opt_dir, f"{stem}-0.{ext}")
    final = os.path.join(opt_dir, f"{stem}.{ext}")
    if os.path.isfile(part):
        if os.path.isfile(final):
            os.remove(final)
        os.rename(part, final)
    if not os.path.isfile(final) or os.path.getsize(final) == 0:
        raise RuntimeError(f"格式化产物缺失或为空: {final}")
    return final


# ---------------------------------------------------------------- 训练（1Ba SoVITS + 1Bb GPT）

def build_s2_config(exp_name, epochs, batch_size, lora_rank):
    import torch
    with open(os.path.join(CONFIGS_DIR, "s2.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    exp_dir = os.path.join(LOGS_DIR, exp_name)
    cfg["train"].update({
        "fp16_run": torch.cuda.is_available(),
        "batch_size": batch_size,
        "epochs": epochs,
        "text_low_lr_rate": 0.4,
        "pretrained_s2G": PRETRAINED_S2G,
        "pretrained_s2D": PRETRAINED_S2G.replace("s2G", "s2D"),
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": 1,
        "gpu_numbers": "0",
        "grad_ckpt": False,
        "lora_rank": lora_rank,
    })
    cfg["model"]["version"] = GSV_VERSION
    cfg["data"]["exp_dir"] = exp_dir
    cfg["s2_ckpt_dir"] = exp_dir
    cfg["save_weight_dir"] = SOVITS_WEIGHTS_DIR
    os.makedirs(SOVITS_WEIGHTS_DIR, exist_ok=True)
    cfg["name"] = exp_name
    cfg["version"] = GSV_VERSION
    os.makedirs(TEMP_DIR, exist_ok=True)
    out = os.path.join(TEMP_DIR, "tmp_s2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    return out


def build_s1_config(exp_name, epochs, batch_size):
    import torch
    import yaml
    with open(os.path.join(CONFIGS_DIR, "s1longer-v2.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp_dir = os.path.join(LOGS_DIR, exp_name)
    cfg["train"].update({
        "precision": "16-mixed" if torch.cuda.is_available() else "32",
        "batch_size": batch_size,
        "epochs": epochs,
        "save_every_n_epoch": max(1, epochs // 3),
        "if_save_every_weights": True,
        "if_save_latest": False,
        "if_dpo": False,
        "half_weights_save_dir": GPT_WEIGHTS_DIR,
        "exp_name": exp_name,
    })
    os.makedirs(GPT_WEIGHTS_DIR, exist_ok=True)
    cfg["data"]["num_workers"] = 0 if os.name == "nt" else 4
    cfg["pretrained_s1"] = PRETRAINED_S1
    cfg["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text.txt")
    cfg["output_dir"] = os.path.join(exp_dir, f"logs_s1_{GSV_VERSION}")
    os.makedirs(TEMP_DIR, exist_ok=True)
    out = os.path.join(TEMP_DIR, "tmp_s1.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return out


def run_train(exp_name, sovits_epochs=8, sovits_batch=12, gpt_epochs=15,
              gpt_batch=8, lora_rank=32, sovits_only=False, gpt_only=False):
    """1Ba SoVITS 训练 + 1Bb GPT 训练，终端实时显示进度"""
    if not gpt_only:
        print(f"\n----- [1Ba] SoVITS 训练（v4 LoRA，{sovits_epochs} 轮，batch {sovits_batch}，rank {lora_rank}）-----", flush=True)
        cfg = build_s2_config(exp_name, sovits_epochs, sovits_batch, lora_rank)
        run_gsv(os.path.join(GSV_PKG, "s2_train_v3_lora.py"), "SoVITS 训练",
                {"_CUDA_VISIBLE_DEVICES": "0"}, args=["-c", cfg])
    if not sovits_only:
        print(f"\n----- [1Bb] GPT 训练（{gpt_epochs} 轮，batch {gpt_batch}）-----", flush=True)
        cfg = build_s1_config(exp_name, gpt_epochs, gpt_batch)
        run_gsv(os.path.join(GSV_PKG, "s1_train.py"), "GPT 训练",
                {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}, args=["-c", cfg])


# ---------------------------------------------------------------- 权重扫描

def _natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def scan_weights():
    """扫描训练出的权重，返回 (sovits列表, gpt列表)，按文件名自然排序"""
    sovits = sorted(glob.glob(os.path.join(SOVITS_WEIGHTS_DIR, "*.pth")), key=_natural_key)
    gpts = sorted(glob.glob(os.path.join(GPT_WEIGHTS_DIR, "*.ckpt")), key=_natural_key)
    return sovits, gpts


# ---------------------------------------------------------------- 推理（电音防线·推理后）

_DF_STATE = None


def df_denoise(audio, sr):
    """DeepFilterNet3 深度学习降噪：压掉声码器残留的嘶嘶底噪和金属杂音。
    实测对合成音各频段噪声能量都有削减（8-16kHz 降约 30%），SECS 音色几乎无损。"""
    global _DF_STATE
    import numpy as np
    import torch
    from df.enhance import enhance, init_df
    if _DF_STATE is None:
        _DF_STATE = init_df()
    model, df_state, _ = _DF_STATE
    target_sr = df_state.sr()
    audio = np.asarray(audio, dtype="float32")
    if sr != target_sr:
        import librosa
        audio_in = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_in = audio
    with torch.no_grad():
        out = enhance(model, df_state, torch.from_numpy(audio_in).unsqueeze(0))
    out = out.squeeze(0).cpu().numpy().astype("float32")
    if sr != target_sr:
        import librosa
        out = librosa.resample(out, orig_sr=target_sr, target_sr=sr)
    return out


def anti_aliasing_postprocess(audio, sr, target_rms=0.10, max_gain=6.0):
    """推理输出后处理（电音防线最后一道）：
    20Hz 高通去直流 → 软噪声门压底噪 → 响度归一（RMS 目标 + 增益封顶）→ 峰值防爆音。
    注意：不能用峰值归一！v4 声码器输出电平低（RMS~0.03）且有底噪，
    峰值归一会把底噪一起放大 30 倍，电音感就是这么来的。"""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return audio
    nyq = sr / 2.0
    sos = butter(4, 20.0 / nyq, btype="high", output="sos")
    audio = sosfiltfilt(sos, audio).astype(np.float32)
    # 软噪声门：短时 RMS 低于底噪估计 2.5 倍的部分按比例衰减，压制静音段嘶嘶底噪
    frame = max(int(sr * 0.02), 1)
    rms_frames = np.sqrt(np.convolve(audio ** 2, np.ones(frame) / frame, mode="same"))
    noise_floor = float(np.percentile(rms_frames, 20))
    gate_thresh = max(noise_floor * 2.5, 1e-5)
    gate_gain = np.clip(rms_frames / gate_thresh, 0.0, 1.0) ** 0.5
    audio = (audio * np.maximum(gate_gain, 0.1)).astype(np.float32)
    # 响度归一：RMS 对齐目标值，增益封顶，避免把底噪放大成电音
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-6:
        audio = (audio * min(target_rms / rms, max_gain)).astype(np.float32)
    # 峰值防爆音（只在真的爆了时才动）
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = (audio / peak * 0.99).astype(np.float32)
    return audio


class GsvTTS:
    """GPT-SoVITS v4 推理封装：加载训练出的 SoVITS/GPT 权重做合成"""

    def __init__(self, sovits_path, gpt_path, fp16=False):
        setup_gsv_path()
        import torch
        from TTS_infer_pack.TTS import TTS, TTS_Config
        cfg = TTS_Config(os.path.join(CONFIGS_DIR, "tts_infer.yaml"))
        cfg.update_version(GSV_VERSION)
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 默认 fp32 全精度：实测 RTX 5080（Blackwell）上 fp16 推理会让音色相似度
        # 从 0.30 掉到 0.19（SECS），声码器底噪也会变大，16G 显存跑 fp32 绰绰有余
        cfg.is_half = fp16 and cfg.device == "cuda"
        cfg.t2s_weights_path = os.path.abspath(gpt_path)
        cfg.vits_weights_path = os.path.abspath(sovits_path)
        cfg.bert_base_path = BERT_DIR
        cfg.cnhuhbert_base_path = CNHUBERT_DIR
        print(f"加载 GPT 权重: {gpt_path}")
        print(f"加载 SoVITS 权重: {sovits_path}")
        self.tts = TTS(cfg)

    def synth(self, text, ref_audio, prompt_text, prompt_lang="zh", text_lang="zh",
              sample_steps=64, top_k=15, top_p=1.0, temperature=1.0,
              speed_factor=1.0, seed=-1, postprocess=True, denoise=True):
        """合成一段文本，返回 (采样率, float32 波形)。sample_steps 默认 64（官方 32 有电音感）"""
        inputs = {
            "text": text, "text_lang": text_lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text, "prompt_lang": prompt_lang,
            "top_k": top_k, "top_p": top_p, "temperature": temperature,
            "text_split_method": "cut1", "batch_size": 1,
            "speed_factor": speed_factor, "seed": seed,
            "repetition_penalty": 1.35,
            "sample_steps": sample_steps,
            "super_sampling": False,
        }
        sr, audio = next(self.tts.run(inputs))
        if denoise:
            audio = df_denoise(audio, sr)  # DeepFilterNet 压声码器残留杂音（在响度归一之前）
        if postprocess:
            audio = anti_aliasing_postprocess(audio, sr)
        return sr, audio
