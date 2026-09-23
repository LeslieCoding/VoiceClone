# -*- coding: utf-8 -*-
# VoiceClone 推理合成网页（GPT-SoVITS v4 训练权重版）：
# 选 SoVITS/GPT 权重 → 选参考音 → 输文本 → 合成。权重来自自己的训练产物，音色准。
import os
import threading

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, ::1"

import gradio as gr

from annotate import find_free_port, load_list
import gsv_backend as gb


def _wav_duration(path):
    import soundfile as sf
    try:
        return sf.info(path).duration
    except Exception:
        return 0.0


def run_ui(list_path, tts_dir, port=9873):
    state = {"tts": None, "key": None}

    def get_entries():
        if os.path.isfile(list_path):
            return [e for e in load_list(list_path) if os.path.isfile(e[0])]
        return []

    def ref_choices():
        choices, skipped = [], 0
        for i, (path, _, lang, text) in enumerate(get_entries()):
            dur = _wav_duration(path)
            if not (3.0 <= dur <= 10.0):
                skipped += 1  # v4 推理硬性要求参考音 3~10 秒
                continue
            tag = " ★推荐" if 5.0 <= dur <= 9.0 else ""
            choices.append((f"#{i+1}（{dur:.1f}秒{tag}）{os.path.basename(path)}", path))
        if skipped:
            print(f"提示：{skipped} 个切片超出 3~10 秒范围，已从参考音列表隐藏。", flush=True)
        return choices

    def weight_choices():
        sovits, gpts = gb.scan_weights()
        sovits_c = [(os.path.basename(p), p) for p in sovits]
        gpt_c = [(os.path.basename(p), p) for p in gpts]
        # 默认选最新
        return (sovits_c, sovits[-1] if sovits else None,
                gpt_c, gpts[-1] if gpts else None)

    def default_ref():
        choices = ref_choices()
        if not choices:
            return None
        starred = [c for c in choices if "★推荐" in c[0]]
        return (starred[0] if starred else choices[0])[1]

    def on_ref_change(ref_path):
        if not ref_path:
            return ""
        for path, _, _, text in get_entries():
            if path == ref_path:
                return text
        return ""

    def lang_of(ref_path):
        for path, _, lang, _ in get_entries():
            if path == ref_path:
                return (lang or "zh").lower()
        return "zh"

    def load_tts(sovits_path, gpt_path):
        key = (sovits_path, gpt_path)
        if state["key"] != key:
            state["tts"] = gb.GsvTTS(sovits_path, gpt_path)
            state["key"] = key
        return state["tts"]

    def synth(sovits_path, gpt_path, ref_path, ref_text, gen_text, steps, speed, seed):
        try:
            if not sovits_path or not gpt_path:
                return None, "请先训练出模型权重（终端菜单选 3），再刷新权重列表"
            if not ref_path:
                return None, "请先选择参考音频"
            if not gen_text.strip():
                return None, "请输入要合成的文本"
            tts = load_tts(sovits_path, gpt_path)
            import soundfile as sf
            sr, audio = tts.synth(gen_text.strip(), ref_path, ref_text.strip(),
                                  prompt_lang=lang_of(ref_path),
                                  sample_steps=int(steps), speed_factor=float(speed),
                                  seed=int(seed))
            os.makedirs(tts_dir, exist_ok=True)
            idx = len([f for f in os.listdir(tts_dir) if f.endswith(".wav")]) + 1
            out_path = os.path.join(tts_dir, f"tts_{idx:04d}.wav")
            sf.write(out_path, audio, sr)
            return out_path, f"合成完成，已保存: {out_path}"
        except Exception as e:
            import traceback
            traceback.print_exc()  # 完整错误打到终端
            return None, f"合成出错：{type(e).__name__}: {e}\n（终端窗口里有详细错误信息，截图发我）"

    def refresh_all():
        sovits_c, sovits_v, gpt_c, gpt_v = weight_choices()
        return (gr.update(choices=sovits_c, value=sovits_v),
                gr.update(choices=gpt_c, value=gpt_v),
                gr.update(choices=ref_choices()))

    with gr.Blocks(title="VoiceClone 语音合成") as demo:
        gr.Markdown("# VoiceClone 语音合成（GPT-SoVITS v4）\n选权重 → 选参考音 → 输文本 → 点合成。")

        with gr.Group():
            gr.Markdown("### 第一步：模型权重（自己训练的）")
            sovits_c, sovits_v, gpt_c, gpt_v = weight_choices()
            with gr.Row():
                sovits_dd = gr.Dropdown(choices=sovits_c, value=sovits_v,
                                        label="SoVITS 权重（默认最新）", scale=4, interactive=True)
                gpt_dd = gr.Dropdown(choices=gpt_c, value=gpt_v,
                                     label="GPT 权重（默认最新）", scale=4, interactive=True)
                refresh_btn = gr.Button("刷新列表", scale=1, min_width=90)

        with gr.Group():
            gr.Markdown("### 第二步：参考音频（决定语调和节奏）")
            with gr.Row():
                ref_dd = gr.Dropdown(choices=ref_choices(), value=default_ref(),
                                     label="从切片库选择（★推荐 = 5~9 秒最佳）", scale=4,
                                     interactive=True, allow_custom_value=False)
            ref_text_box = gr.Textbox(label="参考音频对应的文字（自动带出，可修改）", lines=2)

        with gr.Group():
            gr.Markdown("### 第三步：合成")
            gen_box = gr.Textbox(label="要合成的文本（支持多行）", lines=5,
                                 placeholder="在这里输入你想让它说的话……")
            with gr.Accordion("高级参数（一般用默认值即可）", open=False):
                steps_sl = gr.Slider(16, 256, value=128, step=8,
                                     label="采样步数（默认 128 最干净，调低更快但可能有杂音）")
                speed_sl = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="语速")
                seed_num = gr.Number(value=-1, label="随机种子（-1 = 随机）")
            synth_btn = gr.Button("开始合成", variant="primary", size="lg")

        with gr.Group():
            gr.Markdown("### 结果")
            out_audio = gr.Audio(label="合成结果", type="filepath")
            status = gr.Markdown()

        ref_dd.change(on_ref_change, inputs=ref_dd, outputs=ref_text_box)
        refresh_btn.click(refresh_all, outputs=[sovits_dd, gpt_dd, ref_dd])
        synth_btn.click(synth,
                        inputs=[sovits_dd, gpt_dd, ref_dd, ref_text_box, gen_box,
                                steps_sl, speed_sl, seed_num],
                        outputs=[out_audio, status])
        demo.load(lambda: on_ref_change(default_ref()), outputs=ref_text_box)

    port = find_free_port(port)
    print(f"推理网页已启动: http://localhost:{port}（关闭终端窗口即可停止）", flush=True)
    print("提示：如果点「开始合成」后网页弹出错误提示，多半是浏览器走了系统代理——", flush=True)
    print("      请在 Clash/v2ray 的绕过列表里加上 localhost;127.0.0.1，或改用 http://127.0.0.1:%d 访问。" % port, flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=True,
                theme=gr.themes.Soft(primary_hue="blue"))
