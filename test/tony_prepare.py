# -*- coding: utf-8 -*-
# Tony001 数据准备驱动：跑 vc.py prepare，标注页启动后自动点「完成并关闭」。
# 用法: venv/Scripts/python.exe test/tony_prepare.py
import glob
import os
import queue
import re
import subprocess
import sys
import threading
import time

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

SRC = os.path.join(PROJECT_ROOT, "Voice Orginal", "TonyStark", "Tony001.wav")
assert os.path.isfile(SRC), f"音源不存在: {SRC}"

proc = subprocess.Popen(
    [PYTHON, "vc.py", "prepare", SRC], cwd=PROJECT_ROOT,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
    env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))

q = queue.Queue()
def reader():
    for line in proc.stdout:
        q.put(line)
        print(line, end="", flush=True)
threading.Thread(target=reader, daemon=True).start()

# 等标注页端口出现
deadline = time.time() + 1800
port = None
while time.time() < deadline:
    try:
        line = q.get(timeout=5)
    except queue.Empty:
        if proc.poll() is not None:
            break
        continue
    m = re.search(r"localhost:(\d+)", line)
    if m:
        port = int(m.group(1))
        break

assert port, "标注页没有启动，数据准备中途失败"
print(f"\n[tony] 标注页已在 {port} 端口启动，3 秒后自动点完成...")
time.sleep(3)

from gradio_client import Client
import annotate
entries = annotate.load_list(os.path.abspath("output/asr_opt/slicer_opt.list"))
print(f"[tony] 标注条目数: {len(entries)}")
texts = [e[3] for e in entries[:8]] + [""] * (8 - len(entries[:8]))
chks = [False] * 8
client = Client(f"http://127.0.0.1:{port}/")
client.predict(*texts, *chks, api_name="/finish")
print("[tony] 已点击完成并关闭")

proc.wait(timeout=600)

# 验证切片
stem = os.path.splitext(os.path.basename(SRC))[0]
slices = glob.glob(f"output/slicer_opt/{stem}_clean_s*.wav")
print(f"[tony] {stem} 切片 {len(slices)} 个")
assert len(slices) > 0, "没有产出切片！"
final_entries = annotate.load_list(os.path.abspath("output/asr_opt/slicer_opt.list"))
assert any(stem in e[0] for e in final_entries), "标注文件缺少 Tony001 条目"
print(f"[tony] 标注文件共 {len(final_entries)} 条，数据准备完成")
