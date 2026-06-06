#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from doc_review_tool.services.review_summary import write_review_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build issue summary markdown from review JSON.")
    parser.add_argument("--review-json", required=True, type=Path)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_json = args.review_json.expanduser().resolve()
    if not review_json.exists():
        raise FileNotFoundError(f"review json not found: {review_json}")

    output = args.out.expanduser().resolve() if args.out else review_json.with_suffix(".summary.md")
    pdf_path = args.pdf.expanduser().resolve() if args.pdf else None
    title = args.title or f"{review_json.stem} 问题摘要报告"

    result = write_review_summary_markdown(
        review_json_path=review_json,
        output_path=output,
        pdf_path=pdf_path,
        title=title,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
