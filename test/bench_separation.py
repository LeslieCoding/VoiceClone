# -*- coding: utf-8 -*-
# 人声分离模型基准对比：
# 1) 用 (vocals)+(other) 两个 stem 合成一个 30 秒"带伴奏原曲"
# 2) 各候选模型分别分离
# 3) 计算分离人声对干净人声的 SI-SDR（越高越干净），并生成频谱对比图
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PATH"] = os.path.abspath("tools/ffmpeg-bin") + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import soundfile as sf

VOCAL = "output/uvr5_opt/Tony001_(vocals)_vocals_mel_band_roformer.wav"
OTHER = "output/uvr5_opt/Tony001_(other)_vocals_mel_band_roformer.wav"
DUR = 30

MODELS = [
    "vocals_mel_band_roformer.ckpt",                     # 当前默认（KimberleyJensen）
    "mel_band_roformer_vocals_fv4_gabox.ckpt",           # Gabox FV4
    "bs_roformer_vocals_resurrection_unwa.ckpt",         # Unwa Resurrection（2025 MVSEP 榜首档）
    "mel_band_roformer_karaoke_gabox_v2.ckpt",           # Gabox Karaoke V2（连和声一起去）
]


def si_sdr(ref, est):
    """尺度不变信噪比（dB），越高越好"""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / (np.dot(noise, noise) + 1e-12) + 1e-12)


def main():
    os.makedirs("test/bench", exist_ok=True)
    v, sr_v = sf.read(VOCAL, dtype="float32", always_2d=True)
    o, sr_o = sf.read(OTHER, dtype="float32", always_2d=True)
    n = min(int(DUR * sr_v), len(v), len(o))
    v, o = v[:n].mean(axis=1), o[:n].mean(axis=1)
    # 人声:伴奏 = 1:1 混合（峰值归一后）
    mix = v / np.abs(v).max() * 0.7 + o / np.abs(o).max() * 0.7
    mix_path = "test/bench/mix_30s.wav"
    sf.write(mix_path, mix, sr_v)
    sf.write("test/bench/ref_vocal_30s.wav", v, sr_v)
    print(f"测试混音已生成: {mix_path}（{n/sr_v:.1f}s, {sr_v}Hz）")

    from audio_separator.separator import Separator
    results = {}
    for model in MODELS:
        tag = model.split(".")[0]
        out_dir = f"test/bench/{tag}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n===== {model} =====")
        try:
            sep = Separator(output_dir=out_dir, output_format="WAV", model_file_dir="models")
            sep.load_model(model_filename=model)
            sep.separate(mix_path)
        except Exception as e:
            print(f"  [失败] {e}")
            continue
        # 找 vocals 主干
        import glob
        cands = [f for f in glob.glob(os.path.join(out_dir, "*.wav")) if "vocals" in os.path.basename(f).lower()]
        if not cands:
            cands = glob.glob(os.path.join(out_dir, "*.wav"))
        est, sr_e = sf.read(cands[0], dtype="float32", always_2d=True)
        est = est[:n].mean(axis=1)
        m = min(len(est), len(v))
        score = si_sdr(v[:m], est[:m])
        results[model] = score
        print(f"  SI-SDR = {score:.2f} dB  ({os.path.basename(cands[0])})")

    print("\n===== 结果汇总（SI-SDR 越高越好）=====")
    for m, s in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {s:6.2f} dB  {m}")


if __name__ == "__main__":
    main()
