# -*- coding: utf-8 -*-
# 音量门限切分器（自实现，numpy + soundfile）：
# 按 hop_size 计算 RMS 能量曲线，低于 threshold(dB) 的区段视作静音候选切点，
# 切点间隔不小于 min_interval，短于 min_length 的段与后段合并，段首尾最多保留 max_sil_kept 静音。
import os

import numpy as np
import soundfile as sf

SR = 32000  # 统一工作采样率


def load_audio(path, sr=SR):
    audio, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # 转单声道
    if orig_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


def _rms_curve(audio, hop_ms):
    hop = max(1, int(SR * hop_ms / 1000))
    n = len(audio) // hop
    if n == 0:
        return np.array([0.0]), hop
    frames = audio[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12), hop


def slice_audio(audio, threshold=-34, min_length=4000, min_interval=300,
                hop_size=10, max_sil_kept=500):
    """返回 [(chunk, start_frame, end_frame), ...]，帧为采样点"""
    rms, hop = _rms_curve(audio, hop_size)
    thr = 10.0 ** (threshold / 20.0)
    silent = rms < thr

    min_interval_f = int(min_interval / hop_size)
    min_length_f = int(min_length / hop_size)
    max_sil_f = int(max_sil_kept / hop_size)

    # 找候选切点：静音段的中点，且距上一个切点 >= min_interval
    cut_points = []
    i = 0
    n = len(silent)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            mid = (i + j) // 2
            if not cut_points or mid - cut_points[-1] >= min_interval_f:
                cut_points.append(mid)
            i = j
        else:
            i += 1

    # 按切点切段，首尾补边界
    bounds = [0] + cut_points + [n]
    segments = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]

    # 合并过短的段（与后段连起来直到 >= min_length）
    merged = []
    i = 0
    while i < len(segments):
        s, e = segments[i]
        while e - s < min_length_f and i + 1 < len(segments):
            i += 1
            e = segments[i][1]
        merged.append((s, e))
        i += 1

    results = []
    for s, e in merged:
        # 修剪首尾静音，各最多保留 max_sil_kept；全静音段直接丢弃
        seg_lo, seg_hi = s, e
        probe = s
        while probe < e and silent[probe]:
            probe += 1
        if probe >= e:
            continue  # 整段都是静音
        s = max(probe - max_sil_f, seg_lo)
        while e > s and silent[e - 1]:
            e -= 1
        e = min(e + max_sil_f, seg_hi)
        start, end = s * hop, min(e * hop, len(audio))
        if end - start < int(0.3 * SR):  # 丢弃 0.3 秒以下的碎片
            continue
        results.append((audio[start:end], start, end))
    return results


def normalize(chunk, _max=0.9, alpha=0.25):
    """峰值归一化，混入 alpha 比例原始音频保留动态"""
    m = np.abs(chunk).max()
    if m < 1e-8:
        return chunk
    normed = chunk / m * _max
    return normed * alpha + chunk * (1 - alpha)


def slice_file(inp_path, out_dir, threshold=-34, min_length=4000, min_interval=300,
               hop_size=10, max_sil_kept=500, _max=0.9, alpha=0.25):
    """切分单个音频文件，输出 wav 到 out_dir，返回输出文件列表"""
    os.makedirs(out_dir, exist_ok=True)
    audio = load_audio(inp_path)
    stem = os.path.splitext(os.path.basename(inp_path))[0]
    outs = []
    for idx, (chunk, start, end) in enumerate(slice_audio(
            audio, threshold, min_length, min_interval, hop_size, max_sil_kept)):
        chunk = normalize(chunk, _max, alpha)
        out_path = os.path.join(out_dir, f"{stem}_{idx:03d}.wav")
        sf.write(out_path, chunk, SR)
        outs.append(out_path)
    return outs
