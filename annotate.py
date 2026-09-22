# -*- coding: utf-8 -*-
# 标注校对页（gradio）：分页显示，每页多条，每条 = 音频播放器 + 文本框 + 删除勾选框。
# 逻辑：勾选只是选中；「删除勾选的条目」立即删除并写盘；「保存本页」保存文本修改；
# 翻页自动保存文本；「完成并关闭」保存后退出（不做隐式删除）。
import os
import socket
import threading

LIST_COLS = 4  # path|speaker|lang|text
PAGE_SIZE = 8  # 每页显示条数


def find_free_port(start, tries=20):
    """从 start 开始找空闲端口，被占就 +1 依次尝试"""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"端口 {start}~{start+tries-1} 全部被占用")


def load_list(list_path):
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("|")]
            if len(parts) >= LIST_COLS and parts[0]:
                entries.append(parts[:LIST_COLS])
    return entries


def save_list(list_path, entries):
    with open(list_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write("|".join(e) + "\n")
    return len(entries)


def run_annotator(list_path, port=9871):
    """启动标注网页，阻塞直到用户点击「完成并关闭」"""
    import gradio as gr

    entries = load_list(list_path)
    if not entries:
        print(f"标注文件为空: {list_path}")
        return
    done_event = threading.Event()
    state = {"page": 0}

    def n_pages():
        return max(1, (len(entries) + PAGE_SIZE - 1) // PAGE_SIZE)

    def render_page(msg=""):
        """当前页的组件值：[info, 每行(row, audio, text, chk) 平铺, status]"""
        state["page"] = max(0, min(state["page"], n_pages() - 1))
        page = state["page"]
        start = page * PAGE_SIZE
        outs = []
        for r in range(PAGE_SIZE):
            i = start + r
            if i < len(entries):
                path, _, lang, text = entries[i]
                outs += [
                    gr.update(visible=True),
                    gr.update(value=path if os.path.isfile(path) else None,
                              label=f"#{i+1}/{len(entries)} | {lang}"),
                    gr.update(value=text),
                    gr.update(value=False),
                ]
            else:
                outs += [gr.update(visible=False), gr.update(value=None),
                         gr.update(value=""), gr.update(value=False)]
        info = f"**第 {page+1}/{n_pages()} 页**（当前共 {len(entries)} 条）"
        return [info] + outs + [msg]

    def apply_texts(texts):
        """把当前页的文本修改写回内存并落盘"""
        start = state["page"] * PAGE_SIZE
        for r in range(PAGE_SIZE):
            i = start + r
            if i < len(entries):
                entries[i][3] = str(texts[r]).strip()
        return save_list(list_path, entries)

    def delete_checked(*vals):
        """显式删除：立即移除勾选条目、写盘、刷新页面"""
        texts, chks = vals[:PAGE_SIZE], vals[PAGE_SIZE:]
        apply_texts(texts)  # 先保住本页文本修改
        start = state["page"] * PAGE_SIZE
        to_del = [start + r for r in range(PAGE_SIZE)
                  if chks[r] and start + r < len(entries)]
        if not to_del:
            return render_page("没有勾选任何条目，未执行删除。")
        for i in sorted(to_del, reverse=True):
            del entries[i]
        n = save_list(list_path, entries)
        return render_page(f"已删除 {len(to_del)} 条（第 {', '.join('#'+str(i+1) for i in to_del)} 条），剩余 {n} 条。")

    def nav(delta, *vals):
        apply_texts(vals[:PAGE_SIZE])  # 翻页自动保存文本
        state["page"] = max(0, min(state["page"] + delta, n_pages() - 1))
        return render_page(f"已自动保存，当前第 {state['page']+1}/{n_pages()} 页")

    def finish(*vals):
        n = apply_texts(vals[:PAGE_SIZE])
        # 延迟关闭服务器，让浏览器先收到并显示完成提示，否则会显示连接错误
        threading.Timer(3.0, done_event.set).start()
        return (f"✔ 完成：共 {n} 条已全部保存（勾选未点删除的条目仍然保留）。\n\n"
                "本页将在 3 秒后自动断开，你可以直接关闭这个浏览器标签页，回到终端继续。")

    with gr.Blocks(title="VoiceClone 标注校对") as demo:
        gr.Markdown(
            "## 标注校对\n"
            "1. 试听、直接修改文本（翻页会自动保存）；\n"
            "2. 识别错的条目 → 勾选「删除此条」→ 点 **「删除勾选条目并保存」**（立即生效，条目会消失）；\n"
            "3. 全部校对完点 **「完成并关闭」**（只保存，不删除）。"
        )
        info = gr.Markdown()
        rows, audios, texts, chks = [], [], [], []
        for _ in range(PAGE_SIZE):
            with gr.Row() as row:
                audio = gr.Audio(label="", type="filepath", scale=2)
                with gr.Column(scale=3):
                    text = gr.Textbox(label="", show_label=False, lines=2)
                    chk = gr.Checkbox(label="删除此条（勾选后点下方删除按钮生效）", value=False)
            rows.append(row)
            audios.append(audio)
            texts.append(text)
            chks.append(chk)
        with gr.Row():
            prev_btn = gr.Button("上一页")
            next_btn = gr.Button("下一页")
            del_btn = gr.Button("删除勾选条目并保存", variant="stop")
        status = gr.Markdown()
        finish_btn = gr.Button("完成并关闭", variant="primary")

        flat = [c for r in range(PAGE_SIZE) for c in (rows[r], audios[r], texts[r], chks[r])]
        page_outputs = [info] + flat + [status]
        edit_inputs = texts + chks

        demo.load(lambda: render_page(), outputs=page_outputs)
        prev_btn.click(lambda *v: nav(-1, *v), inputs=edit_inputs, outputs=page_outputs)
        next_btn.click(lambda *v: nav(1, *v), inputs=edit_inputs, outputs=page_outputs)
        del_btn.click(delete_checked, inputs=edit_inputs, outputs=page_outputs)
        finish_btn.click(finish, inputs=edit_inputs, outputs=status)

    port = find_free_port(port)
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=True,
                prevent_thread_lock=True, quiet=True)
    print(f"标注网页已启动: http://localhost:{port}", flush=True)
    print("在浏览器中校对，完成后点击页面底部的「完成并关闭」。", flush=True)
    try:
        while not done_event.wait(1.0):
            pass
    except KeyboardInterrupt:
        save_list(list_path, entries)
        print("检测到 Ctrl+C，已保存当前进度。")
    demo.close()
    print("标注网页已关闭。")
