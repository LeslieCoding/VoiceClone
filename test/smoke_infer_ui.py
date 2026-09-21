# -*- coding: utf-8 -*-
# 推理网页边界用例冒烟：空文本、无参考音、中英混排长文本
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

from gradio_client import Client

client = Client("http://127.0.0.1:9874/")

MODEL = "F5TTS_v1_Base（默认，中英文，质量最好）"
wavs = sorted(glob.glob(os.path.join("output", "slicer_opt", "*.wav")))
REF = os.path.abspath(wavs[-1])  # 7.9s 那条
print(f"参考音: {REF}")

def call(ref, ref_text, gen_text):
    audio, status = client.predict(
        MODEL, ref, ref_text, gen_text, 32, 2.0, 1.0, -1, api_name="/synth")
    return audio, status

# 1. 空文本
_, status = call(REF, "", "")
assert "请输入" in status, status
print(f"[1] 空文本        -> 拦截正常: {status}")

# 2. 无参考音（None = 下拉框未选）
_, status = call(None, "", "你好")
assert "请先选择参考音频" in status, status
print(f"[2] 无参考音      -> 拦截正常: {status}")

# 3. 纯空格文本
_, status = call(REF, "", "   \n  ")
assert "请输入" in status, status
print(f"[3] 纯空格文本    -> 拦截正常: {status}")

# 4. 中英混排长文本（正常合成）
ref_text = "Is it better to be feared or respected? And I say, is it too much to ask for both?"
audio, status = call(REF, ref_text, "我不开心的时候，小布陪我说话。Hello world, this is a test. 心情一下子就好多了。")
assert audio and os.path.isfile(audio), status
import soundfile as sf
dur = sf.info(audio).duration
assert 1.0 < dur < 60.0, f"时长异常: {dur}s"
print(f"[4] 中英混排      -> 合成成功: {os.path.basename(audio)} ({dur:.1f}s)")

print("[smoke] 推理网页边界用例全部通过")
