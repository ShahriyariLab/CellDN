"""Run CellDN on the sample plain-text file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from celldn import CellDNPipeline, write_predictions_json
from celldn.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
DEFAULT_INPUT = EXAMPLES_DIR / "sample_input.txt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "quickstart_text_predictions.json"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "quickstart_text_work"
ENV_NER_MODEL = os.environ.get("CELLDN_NER_MODEL", DEFAULT_NER_MODEL)
ENV_NEN_MODEL = os.environ.get("CELLDN_NEN_MODEL", DEFAULT_NEN_MODEL)


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the example script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--ner-model", default=ENV_NER_MODEL)
    parser.add_argument("--nen-model", default=ENV_NEN_MODEL)
    parser.add_argument("--task", choices=["ner", "end-to-end"], default="end-to-end")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--print-results", action="store_true")
    return parser


def main() -> int:
    """Run the example and save compact JSON output."""

    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    work_dir = Path(args.work_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {input_path}")

    pipeline = CellDNPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        output_dir=work_dir,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    predictions = pipeline.run_text(
        input_path.read_text(encoding="utf-8"),
        task=args.task,
        document_id="sample-text",
    )
    write_predictions_json(predictions, output_path)

    print(f"Input text: {input_path}")
    print(f"Predictions: {output_path}")
    if args.print_results:
        print(json.dumps(json.loads(output_path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
