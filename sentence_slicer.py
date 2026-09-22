# -*- coding: utf-8 -*-
# 句子级智能切分：先用 faster-whisper 带词级时间戳识别整条音频，
# 然后只在句子边界切分——完整句子打包成 6~15 秒片段，音调差异大的句子不合并，
# 切点在句间微调对准静音点。切片的同时直接生成 .list 标注（文本来自识别结果）。
import os

import numpy as np
import soundfile as sf

from slicer import load_audio, _rms_curve, normalize, SR

END_PUNCT = tuple("。！？!?.…；;")
MIN_CLIP = 6.0      # 片段最短时长（秒）
MAX_CLIP = 15.0     # 片段最长时长（秒）
PAD = 0.15          # 片段首尾留白（秒）
PAUSE_SPLIT = 0.8   # 句间停顿超过此值（秒）且片段够长时，作为天然切点
PITCH_RATIO = 1.5   # 相邻句音调比超过此值则不合并进同一片段
WHISPER_MODEL = "large-v3-turbo"


def _sentences_from_whisper(segments):
    """把 whisper 的词级时间戳按句末标点聚合成句子 [(start, end, text)]"""
    sentences = []
    cur = []
    for seg in segments:
        words = list(getattr(seg, "words", None) or [])
        if not words:
            if seg.text.strip():
                sentences.append((seg.start, seg.end, seg.text.strip()))
            continue
        for w in words:
            cur.append(w)
            if w.word.strip().endswith(END_PUNCT):
                text = "".join(x.word for x in cur).strip()
                sentences.append((cur[0].start, cur[-1].end, text))
                cur = []
    if cur:
        text = "".join(x.word for x in cur).strip()
        if text:
            sentences.append((cur[0].start, cur[-1].end, text))
    return [(s, e, t) for s, e, t in sentences if e - s > 0.3]


def _pitch_median(audio16, start_s, end_s):
    """句子的中位基频（Hz），用于音调相似度判断；过短或无声返回 None"""
    import librosa
    seg = audio16[int(start_s * 16000): int(end_s * 16000)]
    if len(seg) < 3200:
        return None
    try:
        f0 = librosa.yin(seg, fmin=50, fmax=500, sr=16000, frame_length=1024)
    except Exception:
        return None
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else None


def _pack_sentences(sentences, pitches):
    """贪心打包：停顿长、音调突变或超长时封口；连续密集且音调相近的句子贴在一起"""
    clips = []
    cur = []
    cur_pitches = []
    for (s, e, t), p in zip(sentences, pitches):
        if cur:
            dur = cur[-1][1] - cur[0][0]
            gap = s - cur[-1][1]
            pitch_jump = False
            base = [x for x in cur_pitches if x]
            if p and base:
                ratio = p / (sum(base) / len(base))
                pitch_jump = ratio > PITCH_RATIO or ratio < 1.0 / PITCH_RATIO
            projected = e - cur[0][0]
            if (dur >= MIN_CLIP and (gap > PAUSE_SPLIT or projected > MAX_CLIP)) or \
               (dur >= 3.0 and pitch_jump):
                clips.append(cur)
                cur, cur_pitches = [], []
        cur.append((s, e, t))
        cur_pitches.append(p)
    if cur:
        clips.append(cur)
    return clips


def _snap_to_silence(rms, hop, t_sec, search=0.3, thr_ratio=3.0):
    """在 t_sec 附近 ±search 秒内找能量最低的帧作为切点，避免切到发音"""
    center = int(t_sec * SR / hop)
    half = int(search * SR / hop)
    lo, hi = max(0, center - half), min(len(rms), center + half)
    if hi <= lo:
        return t_sec
    thr = np.median(rms) * thr_ratio
    quiet = np.where(rms[lo:hi] < thr)[0]
    if len(quiet):
        frame = lo + quiet[np.argmin(rms[lo:hi][quiet])]
    else:
        frame = lo + int(np.argmin(rms[lo:hi]))
    return frame * hop / SR


def sentence_slice(vocal_path, out_dir, list_path, model_size=WHISPER_MODEL,
                   speaker=None, _max=0.9, alpha=0.25, verbose=True, append=False):
    """句子级切分整条音频，写出切片 wav 和 .list 标注，返回 (切片数, list_path)。
    append=True 时把新条目追加到已有 .list（多音源批量处理用），否则覆盖重写。"""
    from faster_whisper import WhisperModel
    from asr import _add_torch_dll_dir
    import librosa

    _add_torch_dll_dir()
    audio = load_audio(vocal_path)           # 32k 单声道 float32
    audio16 = librosa.resample(audio, orig_sr=SR, target_sr=16000)
    rms, hop = _rms_curve(audio, 10)

    print(f"加载 Whisper 模型 {model_size} 进行句子对齐识别...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(vocal_path, word_timestamps=True, vad_filter=True)
    lang = (info.language or "zh").upper()
    sentences = _sentences_from_whisper(segments)
    if not sentences:
        raise RuntimeError("识别结果为空，无法切分（音频里可能没有人声）")
    print(f"识别到 {len(sentences)} 个句子（语种 {lang}）")

    pitches = [_pitch_median(audio16, s, e) for s, e, _ in sentences]
    clips = _pack_sentences(sentences, pitches)
    print(f"打包为 {len(clips)} 个片段（完整句子不切断，音调相近的句子贴在一起）")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    stem = os.path.splitext(os.path.basename(vocal_path))[0]
    speaker = speaker or os.path.basename(os.path.normpath(out_dir))

    lines = []
    total = len(audio) / SR
    # 相邻片段的切点只算一次（句间中点附近对准静音），首尾共用，避免重叠
    bounds = []
    for idx in range(len(clips) - 1):
        gap_mid = (clips[idx][-1][1] + clips[idx + 1][0][0]) / 2
        bounds.append(_snap_to_silence(rms, hop, gap_mid))

    for idx, clip in enumerate(clips):
        start = bounds[idx - 1] if idx > 0 else max(0.0, clip[0][0] - PAD)
        end = bounds[idx] if idx < len(bounds) else min(total, clip[-1][1] + PAD)
        s_f, e_f = int(start * SR), min(int(end * SR), len(audio))
        if e_f - s_f < int(0.5 * SR):
            continue
        chunk = normalize(audio[s_f:e_f], _max, alpha)
        out_path = os.path.join(out_dir, f"{stem}_s{idx:03d}.wav")
        sf.write(out_path, chunk, SR)
        text = " ".join(t for _, _, t in clip).strip()
        lines.append(f"{os.path.abspath(out_path)}|{speaker}|{lang}|{text}")
        if verbose:
            print(f"  [{idx+1:03d}] {start:7.2f}s ~ {end:7.2f}s ({end-start:4.1f}s) | {text[:50]}")

    _write_list(list_path, lines, append=append)
    print(f"切分完成：{len(lines)} 个片段，标注文件: {list_path}")
    return len(lines), os.path.abspath(list_path)


def _write_list(list_path, lines, append=False):
    """写 .list 标注；append=True 时追加到已有内容之后"""
    if append and os.path.isfile(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            old_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        lines = old_lines + lines
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
