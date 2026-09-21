# -*- coding: utf-8 -*-
# 去混响模型基准：给干净人声人工加混响（ffmpeg aecho 多重回声），
# 各模型去混响后与原始干净人声算 SI-SDR（这次有真值，无循环偏差）。
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PATH"] = os.path.abspath("tools/ffmpeg-bin") + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import soundfile as sf

from test.bench_score import si_sdr  # noqa: E402  复用评分函数

CLEAN = "test/bench/ref_vocal_30s.wav"
REVERB = "test/bench/reverb_30s.wav"

MODELS = [
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",   # 当前默认
    "deverb_bs_roformer_8_384dim_10depth.ckpt",             # BS-Roformer-De-Reverb
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",  # anvuew 温和版
]

# 生成带混响的测试音频（三重回声模拟房间混响）
subprocess.run(["ffmpeg", "-y", "-i", CLEAN, "-af",
                "aecho=0.8:0.7:40|90|180:0.4|0.3|0.2", REVERB],
               check=True, capture_output=True)
print(f"人工混响音频已生成: {REVERB}")

from audio_separator.separator import Separator

clean, sr = sf.read(CLEAN, dtype="float32", always_2d=True)
clean = clean.mean(axis=1)

results = {}
for model in MODELS:
    tag = model.split("_sdr")[0].split(".")[0]
    out_dir = f"test/bench_dr/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== {model} =====")
    try:
        sep = Separator(output_dir=out_dir, output_format="WAV", model_file_dir="models")
        sep.load_model(model_filename=model)
        sep.separate(REVERB)
    except Exception as e:
        print(f"  [失败] {e}")
        continue
    dry = [f for f in glob.glob(os.path.join(out_dir, "*.wav"))
           if any(k in os.path.basename(f).lower() for k in ("noreverb", "(dry)"))]
    if not dry:
        dry = glob.glob(os.path.join(out_dir, "*.wav"))
    est, _ = sf.read(dry[0], dtype="float32", always_2d=True)
    est = est.mean(axis=1)
    m = min(len(est), len(clean))
    score = si_sdr(clean[:m], est[:m])
    results[model] = score
    print(f"  SI-SDR = {score:.2f} dB")

print("\n===== 去混响结果汇总（SI-SDR 越高越好）=====")
for name, s in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {s:7.2f} dB  {name}")
