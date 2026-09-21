# -*- coding: utf-8 -*-
# 端到端冒烟：vc.py prepare 全链路（跳过人工标注环节）
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import annotate

def fake_run_annotator(list_path, port=None):
    n = len(annotate.load_list(list_path))
    print(f"[smoke] 标注环节已跳过，标注文件共 {n} 条有效条目")
    assert n > 0, "标注文件为空，切分/识别环节有问题"
    return n

annotate.run_annotator = fake_run_annotator

import types
import vc

args = types.SimpleNamespace(
    source=os.path.join("test", "tony_30s.wav"),
    skip_uvr=False,
    whisper_model="large-v3-turbo",
)
vc.cmd_prepare(args)

# 验证产物
import glob
wavs = glob.glob(os.path.join("output", "slicer_opt", "*.wav"))
assert wavs, "没有产出切片"
for w in wavs:
    import soundfile as sf
    info = sf.info(w)
    assert info.duration >= 1.0, f"切片过短: {w} {info.duration}s"
    print(f"  切片 {os.path.basename(w)}: {info.duration:.1f}s")
print("[smoke] prepare 全链路通过")
