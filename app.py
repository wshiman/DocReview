#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

try:
    from .pipeline import build_config, issues_to_rows, issues_to_table_rows, process_document, summarize_result
    from .services.parsers import parse_document
    from .services.temp_paths import configure_workspace_temp
except ImportError:  # pragma: no cover
    from pipeline import build_config, issues_to_rows, issues_to_table_rows, process_document, summarize_result
    from services.parsers import parse_document
    from services.temp_paths import configure_workspace_temp

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document Review Tool (DeepSeek + Gradio)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17930)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def on_file_change(uploaded_file: Any):
    if uploaded_file is None:
        return (
            gr.update(value="No file selected."),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            {},
        )

    file_path = getattr(uploaded_file, "name", None) or str(uploaded_file)
    suffix = Path(file_path).suffix.lower()
    if suffix not in {".pdf", ".doc", ".docx"}:
        return (
            gr.update(value="Unsupported file type. Upload .pdf/.doc/.docx"),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            {},
        )

    try:
        info = f"Loaded: {Path(file_path).name} | parser=deferred_until_run"
        state = {
            "input_file": str(file_path),
            "normalized_file": str(Path(file_path).resolve()),
            "source_type": "pdf" if suffix == ".pdf" else "docx",
            "doc_id": None,
            "parse_meta": {},
            "result": None,
            "word_helper_file": None,
        }

        return (
            gr.update(value=info),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            state,
        )
    except Exception as exc:
        return (
            gr.update(value=f"Failed to load file: {exc}"),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            {},
        )


def on_word_helper_change(state: Dict[str, Any], uploaded_file: Any):
    state = dict(state or {})
    if uploaded_file is None:
        state.pop("word_helper_file", None)
        state.pop("word_helper_normalized_file", None)
        state.pop("word_helper_meta", None)
        return gr.update(value=""), state

    file_path = getattr(uploaded_file, "name", None) or str(uploaded_file)
    suffix = Path(file_path).suffix.lower()
    if suffix not in {".doc", ".docx"}:
        return gr.update(value="仅支持 .doc/.docx"), state

    state["word_helper_file"] = str(file_path)
    state.pop("word_helper_normalized_file", None)
    state.pop("word_helper_meta", None)
    return gr.update(value=f"已选择 Word 辅助文件: {Path(file_path).name}，点击开始核查后解析。"), state


def on_run_review(
    state: Dict[str, Any],
    uploaded_file: Any,
    word_helper_file: Any,
    pdf_parse_backend: str,
    model: str,
    language_mode: str,
    window_size: int,
    max_retries: int,
    max_review_rounds: int,
    severity_threshold: str,
    enable_visual_structure: bool,
):
    if (not state or not state.get("normalized_file")) and uploaded_file is not None:
        file_path = getattr(uploaded_file, "name", None) or str(uploaded_file)
        suffix = Path(file_path).suffix.lower()
        if suffix in {".pdf", ".doc", ".docx"}:
            state = {
                "input_file": str(file_path),
                "normalized_file": str(Path(file_path).resolve()),
                "source_type": "pdf" if suffix == ".pdf" else "docx",
                "doc_id": None,
                "parse_meta": {},
                "result": None,
                "word_helper_file": None,
            }
            if word_helper_file is not None and state["source_type"] == "pdf":
                helper_path = getattr(word_helper_file, "name", None) or str(word_helper_file)
                state["word_helper_file"] = str(helper_path)

    if not state or not state.get("normalized_file"):
        return (
            gr.update(value="Please upload a file first."),
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            state or {},
        )

    file_path = state["normalized_file"]
    if word_helper_file is not None and state.get("source_type") == "pdf":
        state["word_helper_file"] = getattr(word_helper_file, "name", None) or str(word_helper_file)
    word_helper_path = (
        state.get("word_helper_normalized_file") or state.get("word_helper_file")
        if state.get("source_type") == "pdf"
        else None
    )
    config = build_config(
        model=model,
        language_mode=language_mode,
        window_size=window_size,
        max_retries=max_retries,
        max_review_rounds=max_review_rounds,
        severity_threshold=severity_threshold,
        enable_visual_structure=enable_visual_structure,
        pdf_parse_backend=pdf_parse_backend,
    )

    try:
        result = process_document(
            file_path=file_path,
            config=config,
            output_dir=OUTPUT_DIR,
            word_helper_path=word_helper_path,
        )
        rows = issues_to_rows(result)
        table_rows = issues_to_table_rows(rows)
        issue_ids = [row["issue_id"] for row in rows]

        state = dict(state)
        state["result"] = {
            "annotated_file": str(result.annotated_file_path) if result.annotated_file_path else None,
            "annotated_markdown": str(result.annotated_markdown_path) if result.annotated_markdown_path else None,
            "review_report": str(result.review_report_path) if result.review_report_path else None,
            "review_summary_markdown": str(result.summary_markdown_path) if result.summary_markdown_path else None,
            "issues": rows,
        }
        visual_bits = []
        helper_bits = []
        if enable_visual_structure:
            visual_bits.append(
                "视觉结构审查: "
                f"版面区域={result.stats.get('visual_region_count', 0)}, "
                f"截图/Qwen调用={result.stats.get('visual_review_region_count', 0)}, "
                f"视觉问题={result.stats.get('visual_issue_count', 0)}"
            )
            visual_error = str(result.stats.get("visual_structure_error") or "").strip()
            if visual_error:
                visual_bits.append(f"视觉审查错误: {visual_error}")
        else:
            visual_bits.append("视觉结构审查: 已关闭")
        if result.parse_result.source_type == "pdf":
            visual_bits.append(f"PDF解析: {result.stats.get('pdf_parse_backend') or config.pdf_parse_backend}")
        helper_file = state.get("word_helper_file")
        if result.parse_result.source_type == "docx":
            helper_bits.append(
                "Word 主审查: "
                f"已启用, "
                f"结构问题={result.stats.get('word_structure_issue_count', 0)}"
            )
        elif helper_file:
            helper_bits.append(
                "Word 辅助审查: "
                f"已启用 ({Path(helper_file).name}), "
                f"证据问题={result.stats.get('word_helper_issue_count', 0)}"
            )
            helper_error = str(result.stats.get("word_helper_error") or "").strip()
            if helper_error:
                helper_bits.append(f"Word 辅助错误: {helper_error}")
        else:
            helper_bits.append("Word 辅助审查: 未启用")
        status_text = summarize_result(result)
        if visual_bits:
            status_text += "\n" + "\n".join(visual_bits)
        if helper_bits:
            status_text += "\n" + "\n".join(helper_bits)

        annotated_file_value = str(result.annotated_file_path) if result.annotated_file_path else None
        annotated_markdown_value = str(result.annotated_markdown_path) if result.annotated_markdown_path else None
        if result.parse_result.source_type == "docx":
            annotated_markdown_value = None

        return (
            gr.update(value=status_text),
            gr.update(value=table_rows),
            gr.update(choices=issue_ids, value=issue_ids[0] if issue_ids else None),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=annotated_file_value),
            gr.update(value=annotated_markdown_value),
            gr.update(value=str(result.review_report_path) if result.review_report_path else None),
            gr.update(value=str(result.summary_markdown_path) if result.summary_markdown_path else None),
            state,
        )
    except Exception as exc:
        return (
            gr.update(value=f"Review failed: {exc}"),
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            state,
        )


def on_select_issue(state: Dict[str, Any], issue_id: Optional[str]):
    if not state or not state.get("result") or not issue_id:
        return gr.update(value=""), gr.update(value="")

    issues = state["result"]["issues"]
    issue = next((x for x in issues if x["issue_id"] == issue_id), None)
    if not issue:
        return gr.update(value="Issue not found."), gr.update(value="")

    detail = (
        f"Issue: {issue['issue_id']} ({issue.get('issue_no', '')})\n"
        f"Paragraph: {issue['para_id']}\n"
        f"Page: {issue['page_no']}\n"
        f"Category: {issue['category']}\n"
        f"Severity: {issue['severity']}\n"
        f"Confidence: {issue['confidence']}\n\n"
        f"Reason:\n{issue['reason']}\n\n"
        f"Suggestion:\n{issue['suggestion']}\n\n"
        f"Original Text:\n{issue.get('original_text', '')}\n"
        f"Modified Text:\n{issue.get('modified_text', '')}\n"
        f"Review Round:\n{issue.get('review_round', '')}\n"
        f"Review Input Text:\n{issue.get('review_input_text', '')}\n"
    )
    return gr.update(value=detail), gr.update(value=issue.get("snippet", ""))


def build_app():
    with gr.Blocks(title="DeepSeek 文档核查工具") as demo:
        gr.Markdown("# DeepSeek 文档核查工具")
        gr.Markdown("上传 PDF/Word，逐段核查并导出审查结果。")

        state = gr.State({})

        with gr.Column():
            upload = gr.File(label="上传 PDF/Word", file_types=[".pdf", ".doc", ".docx"])
            pdf_parse_backend = gr.Radio(
                choices=[("PaddleOCR + MinerU", "paddle"), ("Doc2X v3 JSON", "doc2x")],
                value="paddle",
                label="PDF 解析方式",
            )
            enable_word_helper = gr.Checkbox(value=False, label="Word 辅助审查")
            word_helper_upload = gr.File(label="上传 Word 辅助文件", file_types=[".doc", ".docx"], visible=False)

            with gr.Row():
                model = gr.Dropdown(
                    choices=["deepseek-v4-pro", "deepseek-v4-flash"],
                    value="deepseek-v4-pro",
                    label="DeepSeek 模型",
                )
                language_mode = gr.Dropdown(
                    choices=["follow_source", "zh", "en"],
                    value="zh",
                    label="输出语言",
                )

            with gr.Row():
                window_size = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="上下文窗口")
                max_retries = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="重试次数")
                max_review_rounds = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=1,
                    label="最大审查轮次（0=自动直到无错）",
                )
                severity_threshold = gr.Dropdown(
                    choices=["low", "medium", "high"],
                    value="low",
                    label="严重度阈值",
                )
            enable_visual_structure = gr.Checkbox(value=True, label="视觉结构审查")

            run_btn = gr.Button("开始核查", variant="primary")
            status = gr.Textbox(label="状态", lines=3)

            issue_table = gr.Dataframe(
                headers=["issue_id", "issue_no", "para_id", "page_no", "category", "severity", "confidence", "reason", "suggestion", "original_text", "modified_text", "review_round", "snippet"],
                datatype=["str", "str", "str", "number", "str", "str", "number", "str", "str", "str", "str", "number", "str"],
                label="问题列表",
                wrap=True,
            )

            issue_selector = gr.Dropdown(choices=[], label="定位问题", value=None)
            issue_detail = gr.Textbox(label="问题详情", lines=10)
            source_snippet = gr.Textbox(label="原文片段", lines=4)
            download_pdf = gr.File(label="下载含标注文档")
            download_md = gr.File(label="下载含标注Markdown包")
            download_report = gr.File(label="下载审查结果(JSON)")
            download_summary_md = gr.File(label="下载问题摘要(Markdown)")

        def _toggle_word_helper(enabled: bool, uploaded_file: Any):
            file_path = (getattr(uploaded_file, "name", None) or str(uploaded_file)) if uploaded_file is not None else ""
            suffix = Path(file_path).suffix.lower() if file_path else ""
            if suffix in {".doc", ".docx"}:
                return gr.update(visible=False, value=None)
            return gr.update(visible=enabled)

        def _toggle_outputs(uploaded_file: Any):
            file_path = (getattr(uploaded_file, "name", None) or str(uploaded_file)) if uploaded_file is not None else ""
            suffix = Path(file_path).suffix.lower() if file_path else ""
            if suffix in {".doc", ".docx"}:
                return gr.update(visible=True), gr.update(visible=False, value=None)
            return gr.update(visible=True), gr.update(visible=True)

        def _toggle_pdf_parse_backend(uploaded_file: Any):
            file_path = (getattr(uploaded_file, "name", None) or str(uploaded_file)) if uploaded_file is not None else ""
            suffix = Path(file_path).suffix.lower() if file_path else ""
            return gr.update(visible=suffix not in {".doc", ".docx"})

        upload.change(
            fn=on_file_change,
            inputs=[upload],
            outputs=[status, issue_table, issue_detail, source_snippet, download_pdf, download_md, download_report, download_summary_md, state],
        )

        upload.change(
            fn=_toggle_outputs,
            inputs=[upload],
            outputs=[download_pdf, download_md],
        )

        upload.change(
            fn=_toggle_pdf_parse_backend,
            inputs=[upload],
            outputs=[pdf_parse_backend],
        )

        upload.change(
            fn=_toggle_word_helper,
            inputs=[enable_word_helper, upload],
            outputs=[word_helper_upload],
        )

        enable_word_helper.change(
            fn=_toggle_word_helper,
            inputs=[enable_word_helper, upload],
            outputs=[word_helper_upload],
        )

        word_helper_upload.change(
            fn=on_word_helper_change,
            inputs=[state, word_helper_upload],
            outputs=[status, state],
        )

        run_btn.click(
            fn=on_run_review,
            inputs=[
                state,
                upload,
                word_helper_upload,
                pdf_parse_backend,
                model,
                language_mode,
                window_size,
                max_retries,
                max_review_rounds,
                severity_threshold,
                enable_visual_structure,
            ],
            outputs=[status, issue_table, issue_selector, issue_detail, source_snippet, download_pdf, download_md, download_report, download_summary_md, state],
        )

        issue_selector.change(
            fn=on_select_issue,
            inputs=[state, issue_selector],
            outputs=[issue_detail, source_snippet],
        )

    return demo


def main() -> None:
    args = parse_args()
    temp_root = configure_workspace_temp()
    app = build_app()
    allowed_paths = [
        str((APP_DIR / "outputs").resolve()),
        str((APP_DIR / ".runtime").resolve()),
        str(temp_root),
        # Keep /tmp readable for existing Gradio-uploaded files from older sessions.
        "/tmp",
    ]
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
