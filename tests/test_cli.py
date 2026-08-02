"""Tests for the top-level CellExLink command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_cli_help_lists_main_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from cellexlink.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "predict-text" in captured.out
    assert "run-bioc" in captured.out
    assert "download-models" in captured.out
    assert "fetch-pubmed" not in captured.out


def test_predict_text_parser_accepts_text_input(tmp_path: Path) -> None:
    from cellexlink.cli import build_parser

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


def test_run_bioc_parser_accepts_paths(tmp_path: Path) -> None:
    from cellexlink.cli import build_parser

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
        ]
    )

    assert args.command == "run-bioc"
    assert args.input == str(input_xml)
    assert args.output == str(output_xml)
    assert args.task == "nen"
    assert args.ner_model == "local-ner"
    assert args.nen_model == "local-nen"


def test_predict_text_cli_dispatch_can_be_tested_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cellexlink.cli as cli

    output = tmp_path / "predictions.json"

    class FakePipeline:
        def run_text(
            self,
            text: str,
            *,
            task: str,
            document_id: str | None,
        ):
            assert "mesothelial cell" in text
            assert task == "end-to-end"
            assert document_id is None
            return [
                {
                    "document_id": document_id,
                    "passage_index": 0,
                    "mention": "mesothelial cell",
                    "start": 4,
                    "end": 20,
                    "entity_type": "cell_type",
                    "identifier": "CL:0000077",
                    "label": "mesothelial cell",
                    "score": 0.99,
                    "source": "unit-test",
                }
            ]

    monkeypatch.setattr(cli, "_pipeline_from_args", lambda args: FakePipeline())

    exit_code = cli.main(
        [
            "predict-text",
            "--text",
            "The mesothelial cell was detected.",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert str(output) in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "annotations": [
            {
                "mention": "mesothelial cell",
                "entity_type": "cell_type",
                "span": {"begin": 4, "end": 20},
                "identifier": "CL:0000077",
                "label": "mesothelial cell",
            }
        ]
    }


def test_download_models_uses_checkpoint_names_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cellexlink.cli as cli

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
