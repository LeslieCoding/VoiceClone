# VoiceClone

零样本语音克隆流水线：给它一段音源，自动完成 人声分离 → 去混响 → 句子级切分 → 语音识别 → 标注确认，然后用 F5-TTS 零样本克隆合成任意文本的语音。无需训练，开箱即用。

## 快速开始

双击 `go-vc.bat`，按菜单操作：

1. **数据准备** —— 把音源文件（mp3/wav 均可，可以带背景音乐，支持一次拖入多个）交给它，全流程自动，只在标注确认环节弹出一个网页让你核对文本。
2. **推理合成** —— 打开推理网页，输入文本即可合成。
3. **一条龙** —— 数据准备 + 推理合成连着跑。

也可以命令行使用：

```bash
venv/Scripts/python.exe vc.py prepare 音源.wav      # 数据准备
venv/Scripts/python.exe vc.py infer                 # 终端推理
venv/Scripts/python.exe vc.py ui                    # 推理网页
venv/Scripts/python.exe vc.py all 音源.wav          # 一条龙
```

## 架构

```
音源文件
  │
  ├─ [1] 人声分离   MelBand RoFormer FV4（UVR 基准冠军，SI-SDR 5.70dB）
  ├─ [2] 去混响     anvuew dereverb（消除分离残留，这是合成电音的主要来源）
  ├─ [3] 句子切分   whisper 词级时间戳 + 停顿/音调打包 + 静音对齐
  │                 （完整句子不切断，音调相近的句子贴在一起，6~15 秒一段）
  ├─ [4] 语音识别   faster-whisper large-v3-turbo（中英双语）
  └─ [5] 标注确认   本地网页分页校对（唯一需要人工的环节）

推理合成
  └─ F5-TTS v1 Base（零样本克隆，无需训练）+ vocos 声码器（无电音）
```

## 测试

```bash
venv/Scripts/python.exe -m pytest tests/ -v    # 33 项单元测试
venv/Scripts/python.exe test/smoke_prepare.py  # 端到端冒烟
```

## 常见问题

- **合成报错/网页打不开**：浏览器若走系统代理，请在 Clash/v2ray 绕过列表加 `localhost;127.0.0.1`，或改用 `http://127.0.0.1:9873` 访问。
- **端口被占用**：标注页和推理页都会自动 +1 换端口，不会冲突。
- **首次运行慢**：模型已缓存在本地 `models/` 目录，无需重复下载。

## 版本

当前 v2.0.0，变更历史见 `CHANGELOG.md`。
