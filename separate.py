# -*- coding: utf-8 -*-
# 人声分离 / 去混响（基于 audio-separator 库 + MelBand RoFormer 现代模型）。
# 模型首次使用自动下载到 models/ 目录，之后离线可用。
import glob
import os
import re

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# MelBand RoFormer 人声分离（Gabox FV4，基准对比中优于 KimberleyJensen 版）
VOCAL_MODEL = "mel_band_roformer_vocals_fv4_gabox.ckpt"
# MelBand RoFormer 去混响（anvuew，SDR 19.17，目前最强档）
DEREVERB_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"

_STEM_RE = re.compile(r"_\(([^)]+)\)_")


def _stem_of(path):
    """输出文件名形如 原名_(主干)_模型名.wav，提取括号里的主干名"""
    m = _STEM_RE.search(os.path.basename(path))
    return m.group(1).lower() if m else ""


def _make_separator(output_dir):
    from audio_separator.separator import Separator
    os.makedirs(output_dir, exist_ok=True)
    return Separator(
        output_dir=output_dir,
        output_format="WAV",
        model_file_dir=MODELS_DIR,
        output_single_stem=None,  # 由 pick 逻辑按关键词挑
    )


def _pick_output(output_dir, src, prefer, avoid):
    """从分离输出里挑出目标主干文件（按文件名中 _(主干)_ 部分精确匹配）"""
    stem = os.path.splitext(os.path.basename(src))[0]
    cands = [f for f in glob.glob(os.path.join(output_dir, "*.wav"))
             if os.path.basename(f).startswith(stem)]
    if not cands:
        cands = glob.glob(os.path.join(output_dir, "*.wav"))
    if not cands:
        raise FileNotFoundError(f"分离输出目录为空: {output_dir}")
    cands.sort(key=os.path.getmtime)
    for kw in prefer:
        hits = [f for f in cands if _stem_of(f) == kw]
        if hits:
            return hits[-1]
    non_avoid = [f for f in cands if _stem_of(f) not in avoid]
    return non_avoid[-1] if non_avoid else cands[-1]


def separate_vocal(src, output_dir):
    """人声分离，返回人声 wav 路径"""
    sep = _make_separator(output_dir)
    sep.load_model(model_filename=VOCAL_MODEL)
    sep.separate(src)
    out = _pick_output(output_dir, src, prefer=["vocals"], avoid=["instrumental"])
    print(f"人声文件: {out}")
    return out


def dereverb(src, output_dir):
    """去混响，返回干净人声 wav 路径"""
    sep = _make_separator(output_dir)
    sep.load_model(model_filename=DEREVERB_MODEL)
    sep.separate(src)
    out = _pick_output(output_dir, src, prefer=["noreverb", "dry"], avoid=["reverb", "no dry"])
    print(f"去混响完成: {out}")
    return out
