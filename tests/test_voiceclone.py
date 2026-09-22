# -*- coding: utf-8 -*-
# VoiceClone 全模块测试套件（单元 + 边界，不依赖 GPU/模型下载）
# 运行: venv\Scripts\python.exe -m pytest tests/ -v
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import annotate
import sentence_slicer as ss
import slicer
from separate import _pick_output, _stem_of


# ---------------------------------------------------------------- 工具

def make_wav(path, spec, sr=32000):
    """spec: [(秒, 类型)]，类型 'tone'|'silence'，拼合成测试音频"""
    parts = []
    for dur, kind in spec:
        n = int(dur * sr)
        if kind == "tone":
            t = np.arange(n) / sr
            parts.append(0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32))
        else:
            parts.append(np.zeros(n, dtype=np.float32))
    audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    sf.write(str(path), audio, sr)
    return str(path)


# ---------------------------------------------------------------- slicer

class TestSlicer:
    def test_tone_silence_tone_splits_into_two(self, tmp_path):
        src = make_wav(tmp_path / "a.wav", [(5, "tone"), (1.5, "silence"), (5, "tone")])
        outs = slicer.slice_file(src, str(tmp_path / "out"))
        assert len(outs) == 2, f"应切成 2 段，实际 {len(outs)}"
        for p in outs:
            d = sf.info(p).duration
            assert 4.5 < d < 6.5, f"片段时长异常: {d}"

    def test_pure_silence_gives_nothing(self, tmp_path):
        src = make_wav(tmp_path / "b.wav", [(10, "silence")])
        outs = slicer.slice_file(src, str(tmp_path / "out"))
        assert outs == []

    def test_tiny_file_dropped(self, tmp_path):
        src = make_wav(tmp_path / "c.wav", [(0.2, "tone")])
        outs = slicer.slice_file(src, str(tmp_path / "out"))
        assert outs == []

    def test_stereo_input(self, tmp_path):
        sr = 44100
        stereo = np.stack([0.2 * np.sin(np.arange(sr * 5) / 100.0)] * 2, axis=1).astype(np.float32)
        src = tmp_path / "st.wav"
        sf.write(str(src), stereo, sr)
        outs = slicer.slice_file(str(src), str(tmp_path / "out"))
        assert len(outs) == 1

    def test_normalize_peak_within_limit(self, tmp_path):
        src = make_wav(tmp_path / "d.wav", [(5, "tone")])
        outs = slicer.slice_file(src, str(tmp_path / "out"))
        wav, _ = sf.read(outs[0])
        assert np.abs(wav).max() <= 0.95  # _max=0.9 + alpha 混合余量

    def test_continuous_speech_no_cut(self, tmp_path):
        """连续 12 秒无停顿不应被切断"""
        src = make_wav(tmp_path / "e.wav", [(12, "tone")])
        outs = slicer.slice_file(src, str(tmp_path / "out"))
        assert len(outs) == 1


# ---------------------------------------------------------------- sentence_slicer 内部逻辑

class FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class FakeSeg:
    def __init__(self, words=None, start=0.0, end=0.0, text=""):
        self.words = words
        self.start = start
        self.end = end
        self.text = text


class TestSentenceSplit:
    def test_punctuation_splits_sentences(self):
        words = [
            FakeWord("Hello", 0.0, 0.5), FakeWord(" world.", 0.5, 1.0),
            FakeWord(" How", 1.5, 1.8), FakeWord(" are", 1.8, 2.0), FakeWord(" you?", 2.0, 2.5),
        ]
        sents = ss._sentences_from_whisper([FakeSeg(words)])
        assert len(sents) == 2
        assert sents[0][2].endswith(".")
        assert sents[1][0] == 1.5

    def test_chinese_punctuation(self):
        words = [FakeWord("你好。", 0.0, 1.0), FakeWord("再见！", 2.0, 3.0)]
        sents = ss._sentences_from_whisper([FakeSeg(words)])
        assert len(sents) == 2

    def test_no_words_fallback_segment(self):
        sents = ss._sentences_from_whisper([FakeSeg(None, 1.0, 3.0, "  hello  ")])
        assert sents == [(1.0, 3.0, "hello")]

    def test_tiny_segments_dropped(self):
        words = [FakeWord("嗯.", 0.0, 0.1)]
        sents = ss._sentences_from_whisper([FakeSeg(words)])
        assert sents == []


class TestPackSentences:
    def _sent(self, s, e):
        return (s, e, f"text{s}")

    def test_long_pause_splits(self):
        sents = [self._sent(0, 7), self._sent(9.5, 12)]  # 2.5s 停顿
        clips = ss._pack_sentences(sents, [200, 200])
        assert len(clips) == 2

    def test_pitch_jump_splits(self):
        sents = [self._sent(0, 4), self._sent(4.2, 8)]  # 连续但音调突变
        clips = ss._pack_sentences(sents, [200, 400])
        assert len(clips) == 2

    def test_similar_pitch_merges(self):
        sents = [self._sent(0, 4), self._sent(4.2, 8)]
        clips = ss._pack_sentences(sents, [200, 210])
        assert len(clips) == 1

    def test_max_length_caps(self):
        sents = [self._sent(i * 5.2, i * 5.2 + 5) for i in range(6)]  # 连续 31s
        clips = ss._pack_sentences(sents, [200] * 6)
        assert all(c[-1][1] - c[0][0] <= ss.MAX_CLIP + 5.1 for c in clips)
        assert len(clips) >= 2

    def test_short_clip_waits(self):
        """不足 MIN_CLIP 时即使有停顿也继续合并"""
        sents = [self._sent(0, 2), self._sent(3.5, 5.5), self._sent(7, 9)]
        clips = ss._pack_sentences(sents, [200, 200, 200])
        assert len(clips) == 1

    def test_none_pitch_never_splits(self):
        sents = [self._sent(0, 4), self._sent(4.2, 8)]
        clips = ss._pack_sentences(sents, [None, None])
        assert len(clips) == 1


class TestSnapToSilence:
    def test_snaps_to_quiet_frame(self):
        sr = slicer.SR
        audio = np.concatenate([
            0.3 * np.ones(5 * sr, dtype=np.float32),
            np.zeros(1 * sr, dtype=np.float32),
            0.3 * np.ones(5 * sr, dtype=np.float32),
        ])
        rms, hop = slicer._rms_curve(audio, 10)
        snapped = ss._snap_to_silence(rms, hop, 4.8, search=0.5)
        assert 5.0 <= snapped <= 5.8, f"应吸附到静音区(5~6s)，实际 {snapped}"


# ---------------------------------------------------------------- separate 文件挑选（FV4 bug 回归）

class TestPickOutput:
    def test_stem_parsing(self):
        assert _stem_of("mix_(vocals)_mel_band_roformer_vocals_fv4.wav") == "vocals"
        assert _stem_of("mix_(Instrumental)_x.wav") == "instrumental"
        assert _stem_of("plain.wav") == ""

    def test_model_name_containing_vocals_not_confused(self, tmp_path):
        """回归测试：模型名带 vocals 时不得把 Instrumental 当人声"""
        stem = "song"
        inst = tmp_path / f"{stem}_(Instrumental)_mel_band_roformer_vocals_fv4_gabox.wav"
        voc = tmp_path / f"{stem}_(Vocals)_mel_band_roformer_vocals_fv4_gabox.wav"
        inst.write_bytes(b"x")
        voc.write_bytes(b"x")
        out = _pick_output(str(tmp_path), f"{stem}.wav", prefer=["vocals"], avoid=["instrumental"])
        assert "(Vocals)" in os.path.basename(out)

    def test_dereverb_picks_noreverb(self, tmp_path):
        stem = "song"
        (tmp_path / f"{stem}_(reverb)_m.ckpt.wav").write_bytes(b"x")
        (tmp_path / f"{stem}_(noreverb)_m.ckpt.wav").write_bytes(b"x")
        out = _pick_output(str(tmp_path), f"{stem}.wav",
                           prefer=["noreverb", "dry"], avoid=["reverb", "no dry"])
        assert "(noreverb)" in os.path.basename(out)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _pick_output(str(tmp_path), "x.wav", prefer=["vocals"], avoid=[])


# ---------------------------------------------------------------- annotate 数据逻辑

class TestAnnotateData:
    def test_list_roundtrip(self, tmp_path):
        p = tmp_path / "a.list"
        lines = [
            "D:/x/a.wav|slicer_opt|EN|hello world.",
            "D:/x/b.wav|slicer_opt|ZH|你好世界。",
            "坏行",
            "| ||",
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries = annotate.load_list(str(p))
        assert len(entries) == 2
        n = annotate.save_list(str(p), entries[:1])
        assert n == 1
        assert len(annotate.load_list(str(p))) == 1

    def test_find_free_port_skips_occupied(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        used = s.getsockname()[1]
        try:
            got = annotate.find_free_port(used)
            assert got > used
        finally:
            s.close()

    def test_find_free_port_all_occupied_raises(self, monkeypatch):
        class AlwaysBusy:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def bind(self, addr):
                raise OSError("busy")

        monkeypatch.setattr("socket.socket", lambda *a, **k: AlwaysBusy())
        with pytest.raises(RuntimeError):
            annotate.find_free_port(9000, tries=3)


# ---------------------------------------------------------------- vc.py 参考音挑选与重试

class TestPickReference:
    def _mklist(self, tmp_path, durs):
        lines = []
        for i, d in enumerate(durs):
            wav = tmp_path / f"r{i}.wav"
            sf.write(str(wav), np.zeros(int(32000 * d), dtype=np.float32), 32000)
            lines.append(f"{wav}|spk|ZH|文本{i}")
        lp = tmp_path / "t.list"
        lp.write_text("\n".join(lines), encoding="utf-8")
        return str(lp)

    def test_prefers_5_to_9s(self, tmp_path):
        import vc
        lp = self._mklist(tmp_path, [2.0, 7.0, 14.0])
        wav, text, lang = vc.pick_reference(lp)
        assert "r1" in wav
        assert lang == "zh"

    def test_fallback_when_none_qualified(self, tmp_path):
        import vc
        lp = self._mklist(tmp_path, [1.0, 20.0])
        wav, text, lang = vc.pick_reference(lp)
        assert wav  # 不崩溃，挑最接近的

    def test_empty_list_dies(self, tmp_path):
        import vc
        lp = tmp_path / "empty.list"
        lp.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            vc.pick_reference(str(lp))

    def test_missing_files_skipped(self, tmp_path):
        import vc
        lp = tmp_path / "m.list"
        lp.write_text("D:/不存在/x.wav|spk|ZH|文本", encoding="utf-8")
        with pytest.raises(SystemExit):
            vc.pick_reference(str(lp))


class TestAntiAliasing:
    def test_removes_dc_and_limits_peak(self):
        import gsv_backend as gb
        sr = 32000
        t = np.arange(sr) / sr
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3).astype(np.float32)  # 带直流偏移
        out = gb.anti_aliasing_postprocess(audio, sr)
        assert out.dtype == np.float32
        assert abs(float(np.mean(out))) < 0.01, "直流偏移没去掉"
        assert float(np.max(np.abs(out))) <= 0.951, "峰值没限制住"

    def test_empty_input_safe(self):
        import gsv_backend as gb
        out = gb.anti_aliasing_postprocess(np.zeros(0, dtype=np.float32), 32000)
        assert len(out) == 0

    def test_sine_survives(self):
        import gsv_backend as gb
        sr = 32000
        t = np.arange(sr) / sr
        audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = gb.anti_aliasing_postprocess(audio, sr)
        # 440Hz 在通带内，相关度应很高
        corr = float(np.corrcoef(audio[1000:-1000], out[1000:-1000])[0, 1])
        assert corr > 0.95


class TestQualityGate:
    def _mk(self, tmp_path, name, dur=5.0, amp=0.3, clip=False):
        sr = 32000
        t = np.arange(int(sr * dur)) / sr
        data = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        if clip:
            data = np.clip(data * 10, -1, 1)
        p = tmp_path / name
        sf.write(str(p), data, sr)
        return str(p)

    def _list(self, tmp_path, wavs):
        lp = tmp_path / "q.list"
        lp.write_text("\n".join(f"{w}|spk|ZH|文本" for w in wavs), encoding="utf-8")
        return str(lp)

    def test_drops_silent_clipped_short_missing(self, tmp_path):
        import gsv_backend as gb
        good = self._mk(tmp_path, "good.wav")
        silent = self._mk(tmp_path, "silent.wav", amp=0.0001)
        clipped = self._mk(tmp_path, "clip.wav", clip=True)
        short = self._mk(tmp_path, "short.wav", dur=0.3)
        missing = str(tmp_path / "nope.wav")
        lp = self._list(tmp_path, [good, silent, clipped, short, missing])
        kept, dropped = gb.quality_gate(lp)
        assert kept == 1
        assert len(dropped) == 4
        # list 文件只剩好的那条
        content = open(lp, encoding="utf-8").read()
        assert "good.wav" in content and "silent.wav" not in content


class TestScanWeights:
    def test_natural_sort_and_dirs(self, tmp_path, monkeypatch):
        import gsv_backend as gb
        sd, gd = tmp_path / "sov", tmp_path / "gpt"
        sd.mkdir(); gd.mkdir()
        for n in ("exp_e10_s100_l32.pth", "exp_e2_s20_l32.pth", "exp_e1_s9_l32.pth"):
            (sd / n).write_bytes(b"x")
        for n in ("exp-e15.ckpt", "exp-e5.ckpt"):
            (gd / n).write_bytes(b"x")
        monkeypatch.setattr(gb, "SOVITS_WEIGHTS_DIR", str(sd))
        monkeypatch.setattr(gb, "GPT_WEIGHTS_DIR", str(gd))
        sovits, gpts = gb.scan_weights()
        assert [os.path.basename(p) for p in sovits] == [
            "exp_e1_s9_l32.pth", "exp_e2_s20_l32.pth", "exp_e10_s100_l32.pth"]
        assert [os.path.basename(p) for p in gpts] == ["exp-e5.ckpt", "exp-e15.ckpt"]


class TestFindEntryInList:
    def test_finds_by_path(self, tmp_path):
        import vc
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        lp = tmp_path / "x.list"
        lp.write_text(f"{wav}|spk|EN|对应文本\n", encoding="utf-8")
        entry = vc.find_entry_in_list(str(lp), str(wav))
        assert entry[1] == "对应文本"
        assert entry[2] == "en"

    def test_not_found_returns_none(self, tmp_path):
        import vc
        lp = tmp_path / "x.list"
        lp.write_text("D:/y/b.wav|spk|ZH|别的\n", encoding="utf-8")
        assert vc.find_entry_in_list(str(lp), "D:/y/a.wav") is None


class TestSplitDroppedPaths:
    def test_single_quoted(self):
        import vc
        assert vc.split_dropped_paths('"D:/a/b c.wav"') == ["D:/a/b c.wav"]

    def test_multiple_quoted(self):
        import vc
        line = '"D:/x/钢铁侠2.wav" "D:/x/Jarvis001.wav" "D:/x/复联1.wav"'
        assert vc.split_dropped_paths(line) == [
            "D:/x/钢铁侠2.wav", "D:/x/Jarvis001.wav", "D:/x/复联1.wav"]

    def test_unquoted_existing_file_with_space(self, tmp_path):
        import vc
        f = tmp_path / "my voice.wav"
        f.write_bytes(b"x")
        assert vc.split_dropped_paths(str(f)) == [str(f)]

    def test_unquoted_nonexistent_falls_back_to_tokens(self):
        import vc
        assert vc.split_dropped_paths("a.wav b.wav") == ["a.wav", "b.wav"]

    def test_empty(self):
        import vc
        assert vc.split_dropped_paths("   ") == []

    def test_mixed_quotes(self):
        import vc
        line = "'D:/x/one.wav' plain.wav"
        assert vc.split_dropped_paths(line) == ["D:/x/one.wav", "plain.wav"]


class TestWriteList:
    def test_overwrite_default(self, tmp_path):
        lp = str(tmp_path / "a.list")
        ss._write_list(lp, ["x1|s|ZH|一"])
        ss._write_list(lp, ["x2|s|ZH|二"])
        assert open(lp, encoding="utf-8").read() == "x2|s|ZH|二\n"

    def test_append_merges(self, tmp_path):
        lp = str(tmp_path / "a.list")
        ss._write_list(lp, ["x1|s|ZH|一"])
        ss._write_list(lp, ["x2|s|ZH|二"], append=True)
        assert open(lp, encoding="utf-8").read() == "x1|s|ZH|一\nx2|s|ZH|二\n"

    def test_append_to_missing_file_just_writes(self, tmp_path):
        lp = str(tmp_path / "a.list")
        ss._write_list(lp, ["x1|s|ZH|一"], append=True)
        assert open(lp, encoding="utf-8").read() == "x1|s|ZH|一\n"
