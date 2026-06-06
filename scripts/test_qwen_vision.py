#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.7-plus"


def _default_image() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "visual_regions"
    if not root.exists():
        return None
    for path in sorted(root.glob("*/*.png")):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _data_uri(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen/DashScope vision image input.")
    parser.add_argument("--image", type=Path, default=None, help="Local image path. Defaults to first outputs/visual_regions PNG.")
    parser.add_argument("--model", default=os.getenv("DOC_REVIEW_VISUAL_MODEL", DEFAULT_QWEN_MODEL))
    parser.add_argument("--base-url", default=os.getenv("DOC_REVIEW_VISUAL_BASE_URL", DEFAULT_QWEN_BASE_URL))
    parser.add_argument(
        "--api-key",
        default=os.getenv("DOC_REVIEW_VISUAL_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"),
    )
    parser.add_argument("--prompt", default="请只根据图片内容，用一句话说明你能看到的文字和排版结构。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("ERROR: missing API key. Set DOC_REVIEW_VISUAL_API_KEY, DASHSCOPE_API_KEY, or QWEN_API_KEY.")
        return 2

    image_path = args.image or _default_image()
    if image_path is None or not image_path.exists():
        print("ERROR: no image found. Pass --image /path/to/image.png or run visual crop generation first.")
        return 2

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"ERROR: openai SDK is not installed or cannot be imported: {exc}")
        print("Install with: python3 -m pip install openai")
        return 2

    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print(f"image={image_path} size={image_path.stat().st_size} bytes")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_uri(image_path)},
                        },
                        {
                            "type": "text",
                            "text": args.prompt,
                        },
                    ],
                }
            ],
            temperature=0.1,
        )
    except Exception as exc:
        print(f"VISION_CALL_FAILED: {type(exc).__name__}: {exc}")
        return 1

    message = response.choices[0].message
    print("VISION_CALL_OK")
    print(message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
