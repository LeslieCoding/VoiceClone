# -*- coding: utf-8 -*-
# 端到端冒烟：向导多音源批量处理（2 个文件）——验证切片齐全、标注合并、只弹一次标注页。
# 自动点「完成并关闭」以免阻塞。
import glob
import os
import queue
import re
import shutil
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

# 1. 准备 N 个测试音源（默认 2 个，可用命令行参数指定，如 python test/smoke_multi.py 4）
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
src = os.path.join("test", "tony_30s.wav")
files = []
for i in range(N):
    dst = os.path.abspath(f"test/multi_{chr(97+i)}.wav")
    shutil.copyfile(src, dst)
    files.append(dst)

# 2. 清空产物目录
for d in ("output/slicer_opt", "output/asr_opt", "output/uvr5_opt", "output/dereverb_opt"):
    shutil.rmtree(d, ignore_errors=True)

# 3. 启动向导，喂入选项 3 + 两个带引号路径 + n（非纯人声）
proc = subprocess.Popen(
    [PYTHON, "vc.py"], cwd=PROJECT_ROOT,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
    env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
proc.stdin.write("3\n" + " ".join(f'"{f}"' for f in files) + "\nn\n")
proc.stdin.flush()

q = queue.Queue()
def reader():
    for line in proc.stdout:
        q.put(line)
        print(line, end="", flush=True)
threading.Thread(target=reader, daemon=True).start()

# 4. 等标注页出现，自动点「完成并关闭」
deadline = time.time() + 840
port = None
buf = []
while time.time() < deadline:
    try:
        line = q.get(timeout=5)
    except queue.Empty:
        if proc.poll() is not None:
            break
        continue
    buf.append(line)
    m = re.search(r"localhost:(\d+)", line)
    if m:
        port = int(m.group(1))
        break

assert port, "标注页没有启动，处理中途失败"
print(f"\n[smoke] 标注页已在 {port} 端口启动，3 秒后自动点完成...")
time.sleep(3)

from gradio_client import Client
import annotate
entries = annotate.load_list(os.path.abspath("output/asr_opt/slicer_opt.list"))
texts = [e[3] for e in entries[:8]] + [""] * (8 - len(entries[:8]))
chks = [False] * 8
client = Client(f"http://127.0.0.1:{port}/")
client.predict(*texts, *chks, api_name="/finish")
print("[smoke] 已点击完成并关闭")

# 5. 等回到主菜单，退出
time.sleep(6)
proc.stdin.write("0\n")
proc.stdin.flush()
proc.wait(timeout=30)

# 6. 验证：每个音源都有切片，标注文件合并包含全部
final_entries = annotate.load_list(os.path.abspath("output/asr_opt/slicer_opt.list"))
paths = [e[0] for e in final_entries]
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]
    n = len(glob.glob(f"output/slicer_opt/{stem}_clean_s*.wav"))
    print(f"[smoke] {stem} 切片 {n} 个")
    assert n > 0, f"{stem} 没有切片！"
    assert any(stem in p for p in paths), f"标注文件缺少 {stem} 条目"
print(f"[smoke] 标注文件共 {len(final_entries)} 条，{N} 个音源都在")
print(f"[smoke] {N} 个音源批量处理全链路通过")
