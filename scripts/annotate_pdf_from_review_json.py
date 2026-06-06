from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doc_review_tool.services.pdf_review_annotator import annotate_pdf_from_review_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate original PDF from review.json issues.")
    parser.add_argument("--pdf", required=True, help="Input original PDF path.")
    parser.add_argument("--review-json", required=True, help="Review JSON path containing issues.")
    parser.add_argument("--out", required=True, help="Output annotated PDF path.")
    args = parser.parse_args()

    report = annotate_pdf_from_review_json(args.pdf, args.review_json, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
