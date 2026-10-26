"""Tests for the top-level CellDN command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_cli_help_lists_main_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from celldn.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: celldn" in captured.out
    assert "predict-text" in captured.out
    assert "run-bioc" in captured.out
    assert "run-files" in captured.out
    assert "run-pdfs" not in captured.out
    assert "predict-pdf" not in captured.out
    assert "download-models" in captured.out
    assert "fetch-pubmed" not in captured.out


@pytest.mark.parametrize("command", ["predict-pdf", "run-pdf", "run-pdfs"])
def test_removed_pdf_commands_are_not_aliases(command, capsys) -> None:
    from celldn.cli import build_parser

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, "--help"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_predict_text_parser_accepts_text_input(tmp_path: Path) -> None:
    from celldn.cli import build_parser

    output = tmp_path / "predictions.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-text",
            "--text",
            "The mesothelial cell was detected.",
            "--output",
            str(output),
        ]
    )

    assert args.command == "predict-text"
    assert args.text == "The mesothelial cell was detected."
    assert args.output == str(output)
    assert args.document_id is None
    assert args.inference_batch_size == 16


def test_run_bioc_parser_accepts_paths(tmp_path: Path) -> None:
    from celldn.cli import build_parser

    input_xml = tmp_path / "input.xml"
    output_xml = tmp_path / "normalized.xml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-bioc",
            str(input_xml),
            str(output_xml),
            "--task",
            "nen",
            "--ner-model",
            "local-ner",
            "--nen-model",
            "local-nen",
            "--inference-batch-size",
            "5",
        ]
    )

    assert args.command == "run-bioc"
    assert args.input == str(input_xml)
    assert args.output == str(output_xml)
    assert args.task == "nen"
    assert args.ner_model == "local-ner"
    assert args.nen_model == "local-nen"
    assert args.inference_batch_size == 5


def test_predict_text_cli_dispatch_can_be_tested_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from test_api_smoke import _install_fake_model_modules

    import celldn.cli as cli

    _install_fake_model_modules(monkeypatch)
    output = tmp_path / "predictions.json"
    exit_code = cli.main([
        "predict-text",
        "--text", "The mesothelial cell was detected.",
        "--output", str(output),
        "--output-dir", str(tmp_path / "work"),
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "Running CellDN NER...",
        "Running CellDN normalization...",
        f"Saved to: {output.resolve()}",
        "Process finished.",
    ]
    assert captured.err == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "cl_version": "2025-12-17",
        "annotations": [{
            "mention": "mesothelial cell",
            "entity_type": "cell_type",
            "span": {"begin": 4, "end": 20},
            "identifier": "CL:0000077",
            "label": "mesothelial cell",
        }],
    }


def test_download_models_uses_checkpoint_names_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import celldn.cli as cli

    def fake_snapshot_download(repo_id: str, *, local_dir: Path | str):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    import types

    fake_hf = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    output_dir = tmp_path / "models"
    exit_code = cli.main(["download-models", "--output-dir", str(output_dir)])

    assert exit_code == 0
    manifest = json.loads((output_dir / "models.json").read_text(encoding="utf-8"))
    assert manifest["ner_model"].endswith("CellExLink-bioformer16L")
    assert manifest["nen_model"].endswith("CellExLink-Sapbert")
    assert str(output_dir / "CellExLink-bioformer16L") in capsys.readouterr().out


@pytest.mark.parametrize("preserve", [False, True])
def test_run_bioc_cli_maps_passage_chunk_flag_to_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve: bool,
) -> None:
    import celldn.cli as cli

    captured: dict[str, object] = {}
    output = tmp_path / "output.xml"

    class FakePipeline:
        def run_bioc(self, input_path, output_path, **kwargs):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured.update(kwargs)
            return Path(output_path)

    monkeypatch.setattr(cli, "_pipeline_from_args", lambda args: FakePipeline())

    exit_code = cli.main(
        [
            "run-bioc",
            str(tmp_path / "input.xml"),
            str(output),
            "--passage-chunk-size",
            "17",
            *(["--preserve-existing-annotations"] if preserve else []),
        ]
    )

    assert exit_code == 0
    assert captured["passage_chunk_size"] == 17
    assert captured["preserve_existing_annotations"] is preserve
    assert "chunk_size" not in captured


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize(
    ("batch_flags", "expected_batch_size", "expected_inference_batch_size"),
    [
        ([], 32, 16),
        (["--batch-size", "7", "--inference-batch-size", "3"], 7, 3),
    ],
)
def test_run_files_cli_maps_batch_and_chunk_flags_to_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_flags: list[str],
    expected_batch_size: int,
    expected_inference_batch_size: int,
    preserve: bool,
) -> None:
    import celldn.cli as cli

    captured: dict[str, object] = {}
    model_options: dict[str, object] = {}
    result = tmp_path / "results" / "input.json"

    class FakePipeline:
        def run_files(self, input_paths, output_dir, **kwargs):
            captured["input_paths"] = input_paths
            captured["output_dir"] = output_dir
            captured.update(kwargs)
            return [result]

    def from_pretrained(**kwargs):
        model_options.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(cli.CellDNPipeline, "from_pretrained", from_pretrained)

    exit_code = cli.main(
        [
            "run-files",
            str(tmp_path / "input.txt"),
            "--results-dir",
            str(tmp_path / "results"),
            *batch_flags,
            "--passage-chunk-size",
            "11",
            *(["--preserve-existing-annotations"] if preserve else []),
        ]
    )

    assert exit_code == 0
    assert captured["batch_size"] == expected_batch_size
    assert model_options["batch_size"] == expected_inference_batch_size
    assert captured["passage_chunk_size"] == 11
    assert captured["preserve_existing_annotations"] is preserve
    assert "chunk_size" not in captured
    assert "bioc_chunk_size" not in captured


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize(
    ("batch_flags", "expected_batch_size", "expected_inference_batch_size"),
    [
        ([], 100, 16),
        (["--batch-size", "13", "--inference-batch-size", "5"], 13, 5),
    ],
)
def test_predict_pmid_cli_maps_batch_and_chunk_flags_to_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_flags: list[str],
    expected_batch_size: int,
    expected_inference_batch_size: int,
    preserve: bool,
) -> None:
    import celldn.cli as cli

    captured: dict[str, object] = {}
    model_options: dict[str, object] = {}
    output = tmp_path / "pmids.xml"

    class FakePipeline:
        def run_pmids(self, ids, output_path, **kwargs):
            captured["ids"] = ids
            captured["output_path"] = output_path
            captured.update(kwargs)
            return Path(output_path)

    def from_pretrained(**kwargs):
        model_options.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(cli.CellDNPipeline, "from_pretrained", from_pretrained)

    exit_code = cli.main(
        [
            "predict-pmid",
            "30243656",
            "--output",
            str(output),
            *batch_flags,
            "--passage-chunk-size",
            "19",
            *(["--preserve-existing-annotations"] if preserve else []),
        ]
    )

    assert exit_code == 0
    assert captured["ids"] == ["30243656"]
    assert captured["batch_size"] == expected_batch_size
    assert model_options["batch_size"] == expected_inference_batch_size
    assert captured["passage_chunk_size"] == 19
    assert captured["preserve_existing_annotations"] is preserve
    assert "chunk_size" not in captured
    assert "bioc_chunk_size" not in captured


@pytest.mark.parametrize("command", ["run-bioc", "run-files", "predict-pmid"])
def test_prediction_commands_offer_annotation_preservation_option(
    command: str, capsys: pytest.CaptureFixture[str],
) -> None:
    from celldn.cli import build_parser

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, "--help"])
    assert error.value.code == 0
    assert "--preserve-existing-annotations" in capsys.readouterr().out


def test_run_files_cli_rejects_mixed_formats_with_one_short_error(tmp_path, capsys):
    from celldn.cli import main

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "first.xml").write_text("Not read before validating types")
    (inputs / "second.json").write_text("Not read before validating types")
    output = tmp_path / "results"
    assert main([
        "run-files", str(inputs), "--results-dir", str(output),
        "--input-format", "bioc-json", "--output-format", "bioc-json",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Error: run_files accepts one input type per call; found JSON, XML. "
        "Use separate calls for PDF, TXT, XML, and JSON files.\n"
    )
    assert not output.exists()


def test_run_files_pdf_cli_maps_ontology_options_and_chunks(tmp_path, monkeypatch):
    import celldn.cli as cli

    captured = {}
    output_dir = tmp_path / "results"

    class FakePipeline:
        def run_files(self, input_paths, output_dir, **kwargs):
            captured.update(input_paths=input_paths, output_dir=output_dir, **kwargs)
            return [Path(output_dir) / "first.bioc.json", Path(output_dir) / "second.bioc.json"]

    def pipeline(args):
        captured["ontology_path"] = args.ontology_path
        return FakePipeline()

    monkeypatch.setattr(cli, "_pipeline_from_args", pipeline)
    assert cli.main([
        "run-files", "first.pdf", "second.pdf", "--results-dir", str(output_dir),
        "--ontology-path", "resources/custom.jsonl", "--batch-size", "3",
        "--passage-chunk-size", "7", "--no-recursive", "--no-overwrite",
        "--output-format", "bioc-json", "--task", "ner",
    ]) == 0
    assert captured == {
        "input_paths": ["first.pdf", "second.pdf"], "output_dir": str(output_dir),
        "ontology_path": "resources/custom.jsonl", "batch_size": 3, "passage_chunk_size": 7,
        "recursive": False, "overwrite": False, "input_format": "auto",
        "output_format": "bioc-json", "task": "ner",
        "preserve_existing_annotations": False,
    }
