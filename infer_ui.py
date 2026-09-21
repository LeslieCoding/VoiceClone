# -*- coding: utf-8 -*-
# VoiceClone 推理合成网页（自研中文界面，替代 F5-TTS 官方演示页）：
# 只做一件事——选参考音、输文本、合成。没有语音聊天，没有制作人员，没有多余板块。
import os
import threading

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, ::1"

import gradio as gr

from annotate import find_free_port, load_list


def _wav_duration(path):
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        return 0.0


F5_MODELS = {
    "F5TTS_v1_Base（默认，中英文，质量最好）": "F5TTS_v1_Base",
    "E2TTS_Base（上一代，备用）": "E2TTS_Base",
}


def run_ui(list_path, tts_dir, port=9873):
    state = {"tts": None, "model": None}

    def get_entries():
        if os.path.isfile(list_path):
            return [e for e in load_list(list_path) if os.path.isfile(e[0])]
        return []

    def ref_choices():
        choices = []
        for i, (path, _, lang, text) in enumerate(get_entries()):
            dur = _wav_duration(path)
            tag = " ★推荐" if 8.0 <= dur <= 12.0 else ""
            choices.append((f"#{i+1}（{dur:.1f}秒{tag}）{os.path.basename(path)}", path))
        return choices

    def load_model(model_label):
        from f5_tts.api import F5TTS
        name = F5_MODELS[model_label]
        if state["model"] != name:
            state["tts"] = F5TTS(model=name)
            state["model"] = name
        return f"模型已就绪：{name}"

    def on_ref_change(ref_path):
        if not ref_path:
            return ""
        for path, _, _, text in get_entries():
            if path == ref_path:
                return text
        return ""

    def synth(model_label, ref_path, ref_text, gen_text, steps, cfg, speed, seed):
        try:
            if not ref_path:
                return None, "请先选择参考音频"
            if not gen_text.strip():
                return None, "请输入要合成的文本"
            load_model(model_label)
            import soundfile as sf
            # F5-TTS 内部 ODE 采样器偶发 "t must be strictly increasing" 断言（上游已知，
            # 与随机种子相关的数值边界），遇到时换种子自动重试，最多 3 次
            last_err = None
            for attempt in range(3):
                try:
                    # 用户指定种子则首轮使用；重试时换随机种子绕过数值边界
                    use_seed = int(seed) if (attempt == 0 and int(seed) >= 0) else None
                    wav, sr, _ = state["tts"].infer(
                        ref_file=ref_path, ref_text=ref_text.strip(), gen_text=gen_text.strip(),
                        nfe_step=int(steps), cfg_strength=float(cfg), speed=float(speed),
                        seed=use_seed,
                    )
                    break
                except AssertionError as e:
                    if "strictly increasing" in str(e):
                        last_err = e
                        print(f"[提示] 采样器偶发断言（第 {attempt+1}/3 次），换随机种子重试...", flush=True)
                        continue
                    raise
            else:
                raise last_err
            os.makedirs(tts_dir, exist_ok=True)
            idx = len([f for f in os.listdir(tts_dir) if f.endswith(".wav")]) + 1
            out_path = os.path.join(tts_dir, f"tts_{idx:04d}.wav")
            sf.write(out_path, wav, sr)
            return out_path, f"合成完成，已保存: {out_path}"
        except Exception as e:
            import traceback
            traceback.print_exc()  # 完整错误打到终端
            return None, f"合成出错：{type(e).__name__}: {e}\n（终端窗口里有详细错误信息，截图发我）"

    # 默认选中评分最高的参考音（8~12 秒优先，其次最长）
    def default_ref():
        choices = ref_choices()
        if not choices:
            return None
        starred = [c for c in choices if "★推荐" in c[0]]
        return (starred[0] if starred else choices[0])[1]

    with gr.Blocks(title="VoiceClone 语音合成") as demo:
        gr.Markdown("# VoiceClone 语音合成\n选一条参考音 → 输入文本 → 点合成。就这么简单。")

        with gr.Group():
            gr.Markdown("### 第一步：模型")
            model_dd = gr.Dropdown(choices=list(F5_MODELS.keys()),
                                   value=list(F5_MODELS.keys())[0],
                                   label="合成模型", interactive=True)

        with gr.Group():
            gr.Markdown("### 第二步：参考音频（克隆谁的音色）")
            with gr.Row():
                ref_dd = gr.Dropdown(choices=ref_choices(), value=default_ref(),
                                     label="从切片库选择（★推荐 = 8~12 秒最佳）", scale=4,
                                     interactive=True, allow_custom_value=False)
                refresh_btn = gr.Button("刷新列表", scale=1, min_width=80)
            ref_text_box = gr.Textbox(label="参考音频对应的文字（自动带出，可修改）", lines=2)

        with gr.Group():
            gr.Markdown("### 第三步：合成")
            gen_box = gr.Textbox(label="要合成的文本（支持多行）", lines=5,
                                 placeholder="在这里输入你想让它说的话……")
            with gr.Accordion("高级参数（一般用默认值即可）", open=False):
                steps_sl = gr.Slider(16, 128, value=32, step=8, label="采样步数（越大越精细越慢）")
                cfg_sl = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="CFG 强度")
                speed_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速")
                seed_num = gr.Number(value=-1, label="随机种子（-1 = 随机）")
            synth_btn = gr.Button("开始合成", variant="primary", size="lg")

        with gr.Group():
            gr.Markdown("### 结果")
            out_audio = gr.Audio(label="合成结果", type="filepath")
            status = gr.Markdown()

        ref_dd.change(on_ref_change, inputs=ref_dd, outputs=ref_text_box)
        refresh_btn.click(lambda: gr.update(choices=ref_choices()), outputs=ref_dd)
        synth_btn.click(synth,
                        inputs=[model_dd, ref_dd, ref_text_box, gen_box,
                                steps_sl, cfg_sl, speed_sl, seed_num],
                        outputs=[out_audio, status])
        demo.load(lambda: on_ref_change(default_ref()), outputs=ref_text_box)

    port = find_free_port(port)
    print(f"推理网页已启动: http://localhost:{port}（关闭终端窗口即可停止）")
    print("提示：如果点「开始合成」后网页弹出错误提示，多半是浏览器走了系统代理——")
    print("      请在 Clash/v2ray 的绕过列表里加上 localhost;127.0.0.1，或改用 http://127.0.0.1:%d 访问。" % port)
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=True,
                theme=gr.themes.Soft(primary_hue="blue"))
