# -*- coding: utf-8 -*-
# Tony001 音色评测：
#   1) 同人上限基线：参考切片 vs 其他源切片的平均 SECS（天花板，合成音不可能超过它太多）
#   2) 每条合成音 vs 参考切片的 SECS
#   3) 每条合成音 vs 全部源切片平均 embedding 的 SECS（更稳的整体音色分）
# 用法: venv/Scripts/python.exe test/tony_eval.py <参考切片.wav> <合成1.wav> [合成2.wav ...]
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import similarity


def main():
    args = [a.strip('"') for a in sys.argv[1:] if a.strip()]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    ref, synths = args[0], args[1:]

    slices = sorted(glob.glob("output/slicer_opt/Tony001_clean_s*.wav"))
    others = [s for s in slices if os.path.abspath(s) != os.path.abspath(ref)]
    print(f"源切片共 {len(slices)} 个，基线用 {len(others)} 个\n")

    ref_emb = similarity.speaker_embedding(ref)
    other_embs = [similarity.speaker_embedding(s) for s in others]

    # 1) 同人基线：ref vs 其他切片
    base_scores = [float(np.dot(ref_emb, e) / (np.linalg.norm(ref_emb) * np.linalg.norm(e) + 1e-9))
                   for e in other_embs]
    print(f"[同人上限基线] 参考切片 vs 其他源切片: "
          f"mean={np.mean(base_scores):.4f}  min={np.min(base_scores):.4f}  max={np.max(base_scores):.4f}")

    # 源切片平均 embedding（整体音色中心）
    center = np.mean([ref_emb] + other_embs, axis=0)
    center = center / (np.linalg.norm(center) + 1e-9)

    # 2) 合成音评测
    print("\n[合成音评测]")
    print(f"{'文件':<40} {'vs参考':>8} {'vs中心':>8}  判读")
    rows = []
    for s in synths:
        e = similarity.speaker_embedding(s)
        s_ref = float(np.dot(ref_emb, e) / (np.linalg.norm(ref_emb) * np.linalg.norm(e) + 1e-9))
        s_ctr = float(np.dot(center, e) / (np.linalg.norm(e) + 1e-9))
        rows.append((s, s_ref, s_ctr))
        print(f"{os.path.basename(s):<40} {s_ref:>8.4f} {s_ctr:>8.4f}  {similarity.verdict(min(s_ref, s_ctr))}")

    avg_ref = np.mean([r[1] for r in rows])
    avg_ctr = np.mean([r[2] for r in rows])
    print(f"\n平均: vs参考={avg_ref:.4f}  vs中心={avg_ctr:.4f}")
    print(f"基线: 同人切片 mean={np.mean(base_scores):.4f}")
    ok = avg_ref >= 0.70 or avg_ctr >= 0.70
    print(f"\n结论: {'达标（SECS≥0.70，音色合格）' if ok else '未达标（SECS<0.70，需要继续调参）'}")

    # F0 辅助对比
    print("\n[F0 基频] 参考切片 vs 合成音（均值越接近越好）")
    m, md = similarity.f0_stats(ref)
    print(f"  参考: mean={m:.1f}Hz median={md:.1f}Hz")
    for s, _, _ in rows:
        m, md = similarity.f0_stats(s)
        print(f"  {os.path.basename(s)}: mean={m:.1f}Hz median={md:.1f}Hz")


if __name__ == "__main__":
    main()
