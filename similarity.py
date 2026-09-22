# -*- coding: utf-8 -*-
"""
音色相似度评测（SECS）：Speaker Embedding Cosine Similarity
用 speechbrain 预训练 ECAPA-TDNN 声纹编码器，提取两段音频的说话人 embedding，
计算余弦相似度。判读参考：>0.8 优秀 / >0.7 合格 / <0.5 基本不是同一人。
另报 F0（基频）均值对比作辅助。

用法（在 venv 中）：
    python similarity.py <参考音.wav> <待评测1.wav> [待评测2.wav ...]
    python similarity.py --baseline <参考音.wav> <同目录其他切片...>   同人上限基线
"""
import os
import sys
import math

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ECAPA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ecapa")

_clf = None


def _load_model():
    global _clf
    if _clf is not None:
        return _clf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clf = EncoderClassifier.from_hparams(
        source=ECAPA_DIR,
        savedir=ECAPA_DIR,
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    return _clf


def _load_audio_16k(path):
    """加载任意 wav → 16kHz 单声道 float32 numpy"""
    import numpy as np
    import librosa
    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype("float32")


def speaker_embedding(path):
    import torch
    clf = _load_model()
    wav = _load_audio_16k(path)
    with torch.no_grad():
        emb = clf.encode_batch(torch.tensor(wav).unsqueeze(0))  # [1, 1, D]
    return emb.squeeze().cpu().numpy()


def secs(path_a, path_b):
    import numpy as np
    ea, eb = speaker_embedding(path_a), speaker_embedding(path_b)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))


def f0_stats(path):
    import numpy as np
    import librosa
    wav = _load_audio_16k(path)
    f0, _, _ = librosa.pyin(wav, fmin=65, fmax=800, sr=16000, frame_length=1024)
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return float("nan"), float("nan")
    return float(np.mean(f0)), float(np.median(f0))


def verdict(score):
    if score >= 0.80:
        return "优秀（音色几乎一致）"
    if score >= 0.70:
        return "合格（同一人音色）"
    if score >= 0.50:
        return "偏弱（有相似感但不够）"
    return "不像（音色差异明显）"


def main():
    args = [a for a in sys.argv[1:] if a.strip()]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    ref = args[0].strip('"')
    cands = [a.strip('"') for a in args[1:]]
    if not os.path.isfile(ref):
        print(f"[错误] 参考音不存在: {ref}")
        sys.exit(1)

    print(f"参考音: {ref}")
    rows = []
    for c in cands:
        if not os.path.isfile(c):
            print(f"[跳过] 不存在: {c}")
            continue
        s = secs(ref, c)
        rows.append((c, s))
        print(f"  SECS={s:.4f}  {verdict(s)}  <-  {os.path.basename(c)}")

    if len(rows) > 1:
        avg = sum(s for _, s in rows) / len(rows)
        print(f"\n平均 SECS = {avg:.4f}  {verdict(avg)}")

    print("\n[F0 基频对比]（辅助指标，均值越接近越好）")
    m, md = f0_stats(ref)
    print(f"  参考音: mean={m:.1f}Hz median={md:.1f}Hz")
    for c, _ in rows:
        m, md = f0_stats(c)
        print(f"  {os.path.basename(c)}: mean={m:.1f}Hz median={md:.1f}Hz")


if __name__ == "__main__":
    main()
