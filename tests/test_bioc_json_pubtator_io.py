from __future__ import annotations

import json
from pathlib import Path

from cellexlink.io import (
    collection_summary,
    convert_bioc_file,
    read_bioc_collection,
    read_bioc_annotations,
    write_bioc_collection,
)
from cellexlink.pipeline import CellExLinkPipeline


def test_bioc_json_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "input.bioc.json"
    source.write_text(
        json.dumps(
            {
                "source": "test",
                "key": "k",
                "documents": [
                    {
                        "id": "123",
                        "passages": [
                            {
                                "offset": 0,
                                "text": "CD8+ T cells expanded.",
                                "annotations": [
                                    {
                                        "id": "A1",
                                        "infons": {"type": "cell_type", "identifier": "CL:0000625"},
                                        "locations": [{"offset": 0, "length": 12}],
                                        "text": "CD8+ T cells",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    xml_path = tmp_path / "converted.xml"
    json_path = tmp_path / "roundtrip.json"

    convert_bioc_file(source, xml_path)
    convert_bioc_file(xml_path, json_path)

    assert collection_summary(xml_path) == {"documents": 1, "passages": 1, "annotations": 1}
    annotations = read_bioc_annotations(json_path)
    assert annotations[0].text == "CD8+ T cells"
    assert annotations[0].start == 0
    assert annotations[0].end == 12

def test_pipeline_prediction_reader_accepts_bioc_json(tmp_path: Path) -> None:
    bioc_json = tmp_path / "predictions.json"
    bioc_json.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "id": "D1",
                        "passages": [
                            {
                                "offset": 0,
                                "text": "B cells",
                                "annotations": [
                                    {
                                        "infons": {
                                            "type": "cell_type",
                                            "normalization_id_0": "CL:0000236",
                                            "normalization_identifier_name_0": "B cell",
                                            "normalization_confidence_score_0": "0.95",
                                        },
                                        "locations": [{"offset": 0, "length": 7}],
                                        "text": "B cells",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = CellExLinkPipeline.read_predictions_from_bioc(bioc_json)
    assert results[0].identifier == "CL:0000236"
    assert results[0].label == "B cell"
    assert results[0].score == 0.95


def test_bern2_style_json_can_be_read_as_bioc(tmp_path: Path) -> None:
    source = tmp_path / "bern2.json"
    source.write_text(
        json.dumps(
            {
                "pmid": "12345",
                "text": "EGFR was detected in T cells.",
                "denotations": [
                    {
                        "id": "T1",
                        "obj": "gene",
                        "span": {"begin": 0, "end": 4},
                        "text": "EGFR",
                    },
                    {
                        "id": "T2",
                        "obj": "cell_type",
                        "span": {"begin": 21, "end": 28},
                        "text": "T cells",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    collection = read_bioc_collection(source)
    assert collection.documents[0].id == "12345"
    assert collection.documents[0].passages[0].text == "EGFR was detected in T cells."
    assert collection.documents[0].passages[0].annotations[0].infons["type"] == "gene"
    assert collection.documents[0].passages[0].annotations[1].text == "T cells"


def test_hunflair2_style_json_can_be_read_as_bioc(tmp_path: Path) -> None:
    source = tmp_path / "hunflair2.json"
    source.write_text(
        json.dumps(
            {
                "document_id": "doc1",
                "text": "Aspirin reduces inflammation.",
                "entities": [
                    {
                        "text": "Aspirin",
                        "start_pos": 0,
                        "end_pos": 7,
                        "labels": [{"value": "chemical", "score": 0.97}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = read_bioc_annotations(source)
    assert annotations[0].document_id == "doc1"
    assert annotations[0].text == "Aspirin"
    assert annotations[0].label == "chemical"
    assert annotations[0].score == 0.97
