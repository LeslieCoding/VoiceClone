# -*- coding: utf-8 -*-
# ASR 语音识别（faster-whisper large-v3-turbo，自动检测中英语言）
# 输出 .list 标注文件，每行: 音频路径|目录名|语种|文本
import glob
import os

WHISPER_MODEL = "large-v3-turbo"


def _add_torch_dll_dir():
    """ctranslate2 需要 cuBLAS/cuDNN 运行时，torch 自带的 DLL 目录加入搜索路径"""
    import sys
    torch_lib = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
    torch_lib = os.path.abspath(torch_lib)
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(torch_lib)
        except OSError:
            pass


def transcribe_dir(input_dir, output_dir, model_size=WHISPER_MODEL, speaker=None):
    _add_torch_dll_dir()
    from faster_whisper import WhisperModel

    wavs = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"目录中没有 wav 文件: {input_dir}")

    print(f"加载 Whisper 模型 {model_size}（首次运行会从 hf-mirror 下载）...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    os.makedirs(output_dir, exist_ok=True)
    speaker = speaker or os.path.basename(os.path.normpath(input_dir))
    list_path = os.path.join(output_dir, f"{os.path.basename(os.path.normpath(input_dir))}.list")

    lines = []
    for i, wav in enumerate(wavs, 1):
        segments, info = model.transcribe(wav, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        lang = (info.language or "zh").upper()
        if not text:
            print(f"  [{i}/{len(wavs)}] {os.path.basename(wav)} | 识别为空，跳过")
            continue
        lines.append(f"{os.path.abspath(wav)}|{speaker}|{lang}|{text}")
        print(f"  [{i}/{len(wavs)}] {os.path.basename(wav)} | {lang} | {text}")

    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"标注文件已生成: {list_path}（{len(lines)} 条）")
    return os.path.abspath(list_path)
