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

    def test_prefers_8_to_12s(self, tmp_path):
        import vc
        lp = self._mklist(tmp_path, [2.0, 10.0, 14.0])
        wav, text = vc.pick_reference(lp)
        assert "r1" in wav

    def test_fallback_when_none_qualified(self, tmp_path):
        import vc
        lp = self._mklist(tmp_path, [1.0, 20.0])
        wav, text = vc.pick_reference(lp)
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


class TestSynthRetry:
    def test_retries_on_ode_assert(self, tmp_path):
        import vc

        class FakeF5:
            def __init__(self):
                self.calls = 0

            def infer(self, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise AssertionError("t must be strictly increasing or decreasing")
                return np.zeros(24000, dtype=np.float32), 24000, None

        f5 = FakeF5()
        out = tmp_path / "o.wav"
        vc.synth(f5, "ref.wav", "txt", "合成文本", str(out))
        assert f5.calls == 2 and os.path.isfile(out)

    def test_real_assert_propagates(self, tmp_path):
        import vc

        class FakeF5:
            def infer(self, **kw):
                raise AssertionError("别的断言")

        with pytest.raises(AssertionError):
            vc.synth(FakeF5(), "ref.wav", "t", "g", str(tmp_path / "o.wav"))

    def test_gives_up_after_3(self, tmp_path):
        import vc

        class FakeF5:
            def __init__(self):
                self.calls = 0

            def infer(self, **kw):
                self.calls += 1
                raise AssertionError("t must be strictly increasing or decreasing")

        f5 = FakeF5()
        with pytest.raises(AssertionError):
            vc.synth(f5, "r.wav", "t", "g", str(tmp_path / "o.wav"))
        assert f5.calls == 3


class TestFindTextInList:
    def test_finds_by_path(self, tmp_path):
        import vc
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        lp = tmp_path / "x.list"
        lp.write_text(f"{wav}|spk|ZH|对应文本\n", encoding="utf-8")
        assert vc.find_text_in_list(str(lp), str(wav)) == "对应文本"

    def test_not_found_returns_none(self, tmp_path):
        import vc
        lp = tmp_path / "x.list"
        lp.write_text("D:/y/b.wav|spk|ZH|别的\n", encoding="utf-8")
        assert vc.find_text_in_list(str(lp), "D:/y/a.wav") is None
