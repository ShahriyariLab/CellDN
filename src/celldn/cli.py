"""Command line interface for CellDN."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path

from .io import (
    clean_pmid_list,
    collection_summary,
    convert_bioc_file,
    output_format_from_path,
)
from .ontology_download import OntologyDownloadError, download_cell_ontology
from .pipeline import (
    DEFAULT_BIOC_CHUNK_SIZE,
    DEFAULT_FILE_CHUNK_SIZE,
    DEFAULT_NEN_MODEL,
    DEFAULT_NER_MODEL,
    DEFAULT_PMID_CHUNK_SIZE,
    CellDNPipeline,
)
from .retrieval import RetrievalError, TextUnavailableError, combine_ids
from .runtime_logging import show_completion

TEXT_TASK_CHOICES = ["ner", "end-to-end", "e2e"]
BIOC_INPUT_FORMAT_CHOICES = ["auto", "bioc-xml", "bioc-json", "json"]
BIOC_OUTPUT_FORMAT_CHOICES = ["auto", "bioc-xml", "bioc-json"]
BIOC_RUN_TASK_CHOICES = ["ner", "end-to-end", "e2e"]
PMID_TASK_CHOICES = ["ner", "nen", "end-to-end", "e2e"]
FILE_TASK_CHOICES = ["ner", "nen", "end-to-end", "e2e"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``celldn`` command parser."""

    parser = argparse.ArgumentParser(
        prog="celldn",
        description="Cell-type recognition and Cell Ontology normalization for text, PDF, BioC, JSON, and PMID/PMCID inputs.",
    )
    parser.add_argument("--version", action="store_true", help="Print package version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    p_download = subparsers.add_parser("download-models", help="Download default Hugging Face model checkpoints.")
    p_download.add_argument("--ner-model", default=DEFAULT_NER_MODEL)
    p_download.add_argument("--nen-model", default=DEFAULT_NEN_MODEL)
    p_download.add_argument("--output-dir", default="models")
    p_download.add_argument("--trust-remote-code", action="store_true")
    p_download.set_defaults(func=_cmd_download_models)

    p_ontology = subparsers.add_parser(
        "download-ontology",
        aliases=["download-cl"],
        help="Download the latest official CL release as CellDN JSONL (no API key).",
    )
    p_ontology.add_argument(
        "--output-dir",
        default="resources",
        help="Destination folder, created if needed (default: ./resources).",
    )
    p_ontology.add_argument("--include-obsolete", action="store_true")
    p_ontology.add_argument("--timeout", type=float, default=30.0)
    p_ontology.add_argument("--max-retries", type=int, default=3)
    p_ontology.set_defaults(func=_cmd_download_ontology)

    p_text = subparsers.add_parser("predict-text", help="Run CellDN on plain text and write JSON.")
    _add_model_args(p_text)
    text_group = p_text.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text string to process.")
    text_group.add_argument("--input", "-i", help="Plain-text input file.")
    p_text.add_argument("--output", "-o", required=True, help="Output JSON file.")
    p_text.add_argument("--task", choices=TEXT_TASK_CHOICES, default="end-to-end")
    p_text.add_argument("--document-id", default=None)
    p_text.set_defaults(func=_cmd_predict_text)

    p_run_bioc = subparsers.add_parser(
        "run-bioc",
        help="Run NER, NEN, or end-to-end prediction on BioC XML, BioC JSON, or generic JSON input.",
    )
    _add_bioc_run_args(p_run_bioc, default_task="end-to-end", allow_nen=True)
    p_run_bioc.set_defaults(func=_cmd_run_bioc)

    p_run_files = subparsers.add_parser(
        "run-files",
        help="Process one input type per call: PDF-only, TXT-only, XML-only, or JSON-only.",
    )
    _add_model_args(p_run_files, include_output_dir=False)
    p_run_files.add_argument(
        "inputs",
        nargs="+",
        help="Files or directories containing one input type. Mixed PDF/TXT/XML/JSON batches are rejected.",
    )
    p_run_files.add_argument(
        "--results-dir",
        required=True,
        help="Directory for one result file per input file.",
    )
    p_run_files.add_argument("--task", choices=FILE_TASK_CHOICES, default="end-to-end")
    p_run_files.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_FILE_CHUNK_SIZE,
        help="Input files combined per batch.",
    )
    p_run_files.add_argument(
        "--passage-chunk-size",
        type=int,
        default=DEFAULT_BIOC_CHUNK_SIZE,
        help="BioC passages processed per inference chunk.",
    )
    p_run_files.add_argument("--input-format", choices=BIOC_INPUT_FORMAT_CHOICES, default="auto")
    p_run_files.add_argument("--output-format", choices=BIOC_OUTPUT_FORMAT_CHOICES, default="auto")
    p_run_files.add_argument(
        "--preserve-existing-annotations",
        action="store_true",
        help="Retain existing passage annotations alongside new CellDN predictions.",
    )
    p_run_files.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search input directories recursively (default: true).",
    )
    p_run_files.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace existing result files (default: true).",
    )
    p_run_files.set_defaults(func=_cmd_run_files)

    p_predict_pmid = subparsers.add_parser(
        "predict-pmid",
        help="Fetch PMID/PMCID content and run NER, NEN, or end-to-end prediction.",
    )
    _add_model_args(p_predict_pmid)
    _add_id_args(p_predict_pmid)
    p_predict_pmid.add_argument("--output", "-o", required=True, help="Output BioC XML/JSON path.")
    p_predict_pmid.add_argument("--task", choices=PMID_TASK_CHOICES, default="end-to-end")
    p_predict_pmid.add_argument("--text-source", choices=["abstract", "fulltext"], default="abstract")
    p_predict_pmid.add_argument("--source", choices=["ncbi", "europepmc"], default="ncbi")
    p_predict_pmid.add_argument("--output-format", choices=BIOC_OUTPUT_FORMAT_CHOICES, default="auto")
    p_predict_pmid.add_argument("--keep-intermediate-bioc", help="Save the retrieved BioC XML used for prediction.")
    p_predict_pmid.add_argument(
        "--preserve-existing-annotations",
        action="store_true",
        help="Retain existing passage annotations alongside new CellDN predictions.",
    )
    p_predict_pmid.add_argument(
        "--strict-fetch",
        action="store_true",
        help="Fail if any requested ID cannot be retrieved.",
    )
    p_predict_pmid.add_argument("--timeout", type=int, default=30)
    p_predict_pmid.add_argument("--pause", type=float, default=0.12)
    p_predict_pmid.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_PMID_CHUNK_SIZE,
        help="Publication identifiers retrieved per batch.",
    )
    p_predict_pmid.add_argument(
        "--passage-chunk-size",
        type=int,
        default=DEFAULT_BIOC_CHUNK_SIZE,
        help="BioC passages processed per inference chunk.",
    )
    p_predict_pmid.set_defaults(func=_cmd_predict_pmid)

    p_convert = subparsers.add_parser(
        "convert-bioc",
        help="Convert BioC XML, BioC JSON, or generic JSON input to BioC XML/JSON.",
    )
    p_convert.add_argument("input")
    p_convert.add_argument("output")
    p_convert.add_argument("--input-format", choices=BIOC_INPUT_FORMAT_CHOICES, default="auto")
    p_convert.add_argument("--output-format", choices=BIOC_OUTPUT_FORMAT_CHOICES, default="auto")
    p_convert.add_argument("--summary", action="store_true", help="Print collection counts after conversion.")
    p_convert.set_defaults(func=_cmd_convert_bioc)

    return parser


def _add_model_args(
    parser: argparse.ArgumentParser,
    *,
    include_output_dir: bool = True,
) -> None:
    """Add shared model/runtime options to a subcommand parser."""

    parser.add_argument("--ner-model", default=DEFAULT_NER_MODEL)
    parser.add_argument("--nen-model", default=DEFAULT_NEN_MODEL)
    parser.add_argument("--ontology-path")
    parser.add_argument("--abbreviations-path")
    parser.add_argument("--disable-abbreviations", action="store_true")
    if include_output_dir:
        parser.add_argument(
            "--output-dir",
            default="celldn_outputs",
            help="Working directory for intermediate prediction artifacts.",
        )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Optional one-batch NER warmup runs. The full corpus is never used for warmup.",
    )
    parser.add_argument("--inference-batch-size", type=int, default=16, help="Model inference batch size per device.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _add_bioc_run_args(parser: argparse.ArgumentParser, *, default_task: str, allow_nen: bool) -> None:
    """Add shared BioC prediction arguments to a subcommand parser."""

    _add_model_args(parser)
    parser.add_argument("input", help="Input BioC XML/BioC JSON/JSON file.")
    parser.add_argument("output", help="Output BioC XML/JSON file.")
    choices = BIOC_RUN_TASK_CHOICES + (["nen"] if allow_nen else [])
    parser.add_argument("--task", choices=choices, default=default_task)
    parser.add_argument("--input-format", choices=BIOC_INPUT_FORMAT_CHOICES, default="auto")
    parser.add_argument("--output-format", choices=BIOC_OUTPUT_FORMAT_CHOICES, default="auto")
    parser.add_argument("--ner-output-xml")
    parser.add_argument(
        "--preserve-existing-annotations",
        action="store_true",
        help="Retain existing passage annotations alongside new CellDN predictions.",
    )
    parser.add_argument(
        "--passage-chunk-size",
        type=int,
        default=DEFAULT_BIOC_CHUNK_SIZE,
        help="BioC passages processed per inference chunk.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)


def _add_id_args(parser: argparse.ArgumentParser) -> None:
    """Add PMID/PMCID input arguments to a subcommand parser."""

    parser.add_argument("ids", nargs="*", help="PMIDs/PMCIDs. Commas, spaces, and repeated values are accepted.")
    parser.add_argument(
        "--pmid",
        "--id",
        dest="inline_ids",
        action="append",
        default=[],
        help="PMID/PMCID value or comma-separated list. Can be repeated.",
    )
    parser.add_argument("--ids-file", help="Text file containing PMIDs/PMCIDs.")


def _pipeline_from_args(args: argparse.Namespace) -> CellDNPipeline:
    """Create a pipeline instance from parsed CLI arguments."""

    return CellDNPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        ontology_path=args.ontology_path,
        abbreviations_path=args.abbreviations_path,
        disable_abbreviations=args.disable_abbreviations,
        output_dir=getattr(args, "output_dir", "celldn_outputs"),
        warmup_runs=args.warmup_runs,
        batch_size=args.inference_batch_size,
        fp16=args.fp16,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose,
    )


def _ids_from_args(args: argparse.Namespace) -> list[str]:
    """Resolve PMIDs/PMCIDs from positional, repeated, and file-based inputs."""

    ids: list[str] = []
    ids.extend(clean_pmid_list(args.ids or []))
    ids.extend(clean_pmid_list(args.inline_ids or []))
    ids = combine_ids(ids, args.ids_file)
    if not ids:
        raise SystemExit("No PMIDs/PMCIDs provided. Use positional IDs, --pmid, or --ids-file.")
    return ids


def _read_text_arg(args: argparse.Namespace) -> str:
    """Return inline text or load it from the requested input file."""

    return args.text if args.text is not None else Path(args.input).expanduser().read_text(encoding="utf-8")


def _resolve_bioc_output_format(output_path: str, requested_format: str) -> str:
    """Honor an explicit format or infer one from the output path."""

    if requested_format == "auto":
        return output_format_from_path(output_path)
    return requested_format


def _model_dir_name(model_reference: str) -> str:
    """Derive a readable local directory name from a model reference."""

    text = str(model_reference).rstrip("/").rstrip("\\")
    if not text:
        return "model"
    return text.replace("\\", "/").split("/")[-1] or "model"


def _cmd_download_models(args: argparse.Namespace) -> int:
    """Download the default NER and NEN checkpoints for local reuse."""

    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    ner_dir = output_dir / _model_dir_name(args.ner_model)
    nen_dir = output_dir / _model_dir_name(args.nen_model)
    # snapshot_download stores model files only; trust_remote_code is used later by model loaders.
    ner_path = snapshot_download(args.ner_model, local_dir=ner_dir)
    nen_path = snapshot_download(args.nen_model, local_dir=nen_dir)
    manifest = {
        "ner_model": ner_path,
        "nen_model": nen_path,
    }
    (output_dir / "models.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"NER model saved to: {Path(ner_path).expanduser().resolve()}")
    print(f"NEN model saved to: {Path(nen_path).expanduser().resolve()}")
    show_completion(output_dir / "models.json")
    return 0


def _cmd_predict_text(args: argparse.Namespace) -> int:
    """Run text prediction and let the workflow own its save/status report."""

    pipe = _pipeline_from_args(args)
    pipe.run_text(
        _read_text_arg(args),
        task=args.task,
        document_id=args.document_id,
        output_path=args.output,
    )
    return 0


def _cmd_run_bioc(args: argparse.Namespace) -> int:
    """Run the BioC/JSON prediction subcommand."""

    pipe = _pipeline_from_args(args)
    pipe.run_bioc(
        args.input,
        args.output,
        task=args.task,
        input_format=args.input_format,
        output_format=args.output_format,
        ner_output_xml=args.ner_output_xml,
        preserve_existing_annotations=args.preserve_existing_annotations,
        overwrite=args.overwrite,
        passage_chunk_size=args.passage_chunk_size,
    )
    return 0


def _cmd_run_files(args: argparse.Namespace) -> int:
    """Process many files while reusing one pipeline instance."""

    pipe = _pipeline_from_args(args)
    outputs = pipe.run_files(
        args.inputs,
        args.results_dir,
        task=args.task,
        batch_size=args.batch_size,
        passage_chunk_size=args.passage_chunk_size,
        input_format=args.input_format,
        output_format=args.output_format,
        preserve_existing_annotations=args.preserve_existing_annotations,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )
    return 0 if outputs else 1


def _cmd_predict_pmid(args: argparse.Namespace) -> int:
    """Fetch PMID/PMCID content and run prediction in one step."""

    pipe = _pipeline_from_args(args)
    ids = _ids_from_args(args)
    output = pipe.run_pmids(
        ids,
        args.output,
        task=args.task,
        text_source=args.text_source,
        source=args.source,
        output_format=args.output_format,
        keep_intermediate_bioc=args.keep_intermediate_bioc,
        preserve_existing_annotations=args.preserve_existing_annotations,
        strict_fetch=args.strict_fetch,
        timeout=args.timeout,
        pause=args.pause,
        batch_size=args.batch_size,
        passage_chunk_size=args.passage_chunk_size,
    )
    if output is None:
        return 1
    return 0


def _cmd_download_ontology(args: argparse.Namespace) -> int:
    """Download a standalone CL resource without loading prediction models."""

    output = download_cell_ontology(
        output_dir=args.output_dir,
        include_obsolete=args.include_obsolete,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    show_completion(output)
    return 0


def _cmd_convert_bioc(args: argparse.Namespace) -> int:
    """Convert between supported BioC input and output formats."""

    output = convert_bioc_file(
        Path(args.input).expanduser(),
        Path(args.output).expanduser(),
        input_format=args.input_format,
        output_format=args.output_format,
    )
    if args.summary:
        print(json.dumps(collection_summary(output), indent=2))
    show_completion(output)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Run the ``celldn`` CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.version:
        try:
            print(version("celldn"))
        except Exception:  # noqa: BLE001
            print("celldn")
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except (OntologyDownloadError, TextUnavailableError, RetrievalError, OSError, ValueError) as exc:
        # Expected user, retrieval, and download errors need no Python traceback.
        message = " ".join(str(exc).split()) or "The operation could not be completed."
        print(f"Error: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
