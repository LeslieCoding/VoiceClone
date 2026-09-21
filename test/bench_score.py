# -*- coding: utf-8 -*-
# 修正版评分：只取真正的 vocals 主干文件，重新计算 SI-SDR
import glob
import os
import re

import numpy as np
import soundfile as sf


def si_sdr(ref, est):
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / (np.dot(noise, noise) + 1e-12) + 1e-12)


v, sr = sf.read("test/bench/ref_vocal_30s.wav", dtype="float32", always_2d=True)
v = v.mean(axis=1)
n = len(v)

results = {}
for d in sorted(glob.glob("test/bench/*/")):
    tag = os.path.basename(os.path.normpath(d))
    voc = [f for f in glob.glob(os.path.join(d, "*.wav"))
           if re.search(r"_\((vocals)\)_", os.path.basename(f), re.I)]
    if not voc:
        print(f"{tag}: 未找到 vocals 主干")
        continue
    est, _ = sf.read(voc[0], dtype="float32", always_2d=True)
    est = est.mean(axis=1)[:n]
    m = min(len(est), n)
    results[tag] = si_sdr(v[:m], est[:m])

print("===== 修正后结果（SI-SDR 越高越好）=====")
for name, s in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {s:7.2f} dB  {name}")
