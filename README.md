# VoiceClone

语音克隆流水线（GPT-SoVITS v4 训练版）：给它一段音源，自动完成 人声分离 → 去混响 → 句子级切分 → 语音识别 → 标注确认 → **模型训练（SoVITS + GPT）** → 用训练出的专属权重合成任意文本。**训练出来的模型，音色才像本人。**

## 快速开始

双击 `go-vc.bat`，按菜单操作：

1. **推理合成（网页）** —— 选权重、选参考音、输文本，浏览器里完成。
2. **推理合成（终端）** —— 自动选最新权重，终端直接合成。
3. **模型训练** —— 格式化 + SoVITS 训练 + GPT 训练，终端实时看进度。
4. **数据准备** —— 把音源文件（可一次拖入多个）交给它，全流程自动，只在标注确认环节弹出网页核对文本。
5. **标注校对页** —— 只需重新校对时用。

也可以命令行使用：

```bash
venv/Scripts/python.exe vc.py prepare 音源.wav       # 数据准备（可多个文件）
venv/Scripts/python.exe vc.py train                  # 模型训练（自动格式化+两步训练）
venv/Scripts/python.exe vc.py infer --text "你好"     # 终端推理
venv/Scripts/python.exe vc.py ui                     # 推理网页
venv/Scripts/python.exe vc.py all 音源.wav           # 一条龙
```

## 架构

```
音源文件
  │
  ├─ [1] 人声分离   MelBand RoFormer FV4（UVR 基准冠军，SI-SDR 5.70dB）
  ├─ [2] 去混响     anvuew dereverb（消除分离残留——电音的主要来源）
  ├─ [3] 句子切分   whisper 词级时间戳 + 停顿/音调打包 + 静音对齐
  │                 （完整句子不切断，音调相近的句子贴在一起，6~15 秒一段）
  ├─ [4] 语音识别   faster-whisper large-v3-turbo（中英双语）
  ├─ [5] 标注确认   本地网页分页校对（唯一需要人工的环节）
  └─ [6] 质量门     自动剔除过短/静音/削波的坏切片（电音防线·训练前）

模型训练（gsv/ 内置 GPT-SoVITS v4 源码，含单卡训练补丁）
  ├─ 格式化        BERT 文本特征 → cnhubert 语音特征 → 语义 token
  ├─ SoVITS 训练   v4 LoRA（默认 8 轮，rank 32，batch 6）
  └─ GPT 训练      （默认 15 轮，batch 4）

推理合成
  └─ 训练出的 SoVITS/GPT 权重克隆音色（采样 64 步 + 输出后处理，电音防线·推理后）
```

## 电音防线（三层）

1. **训练前**：去混响 + 数据质量门，坏切片不进训练集。
2. **训练中**：显存碎片优化，batch 按 16G 显存调优。
3. **推理后**：采样步数 64（官方 32 有电音感）+ 输出统一后处理（去直流/去超高频噪点/软限幅/峰值归一）。

## 测试

```bash
venv/Scripts/python.exe -m pytest tests/ -q        # 44 项单元测试
venv/Scripts/python.exe test/smoke_prepare.py      # 数据准备端到端冒烟
venv/Scripts/python.exe test/smoke_multi.py 4      # 多音源批量处理冒烟
```

## 常见问题

- **合成报错/网页打不开**：浏览器若走系统代理，请在 Clash/v2ray 绕过列表加 `localhost;127.0.0.1`，或改用 `http://127.0.0.1:9873` 访问。
- **端口被占用**：标注页和推理页都会自动 +1 换端口，不会冲突。
- **参考音频报"3~10 秒范围外"**：v4 推理的硬性要求，网页端已自动隐藏超范围切片。
- **训练显存不够**：`vc.py train --sovits-batch 4 --gpt-batch 2` 再降一档。

## 版本

当前 v3.0.0，变更历史见 `CHANGELOG.md`。
