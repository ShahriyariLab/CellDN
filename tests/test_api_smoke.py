"""Model-free smoke tests for the CellExLink end-to-end pipeline."""

from __future__ import annotations

import shutil
import sys
import types
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


def _add_infon(element: ET.Element, key: str, value: str) -> None:
    infon = ET.SubElement(element, "infon", {"key": key})
    infon.text = value


def _install_fake_model_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    predictor_call: dict[str, object] | None = None,
    normalizer_call: dict[str, object] | None = None,
) -> None:
    """Install fake recognition/normalization modules so tests need no models."""
    fake_recognition_predict = types.ModuleType("cellexlink.recognition.predict")

    def fake_predict_ner(
        *,
        model_path,
        input_xml=None,
        input_file=None,
        output_dir,
        output_xml=None,
        warmup_runs=1,
        per_device_predict_batch_size=16,
        fp16=False,
        trust_remote_code=False,
        verbose=False,
        **kwargs,
    ) -> int:
        del model_path, warmup_runs, per_device_predict_batch_size, fp16, trust_remote_code, kwargs
        if predictor_call is not None:
            predictor_call["verbose"] = verbose

        mention = "mesothelial cell"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_xml is not None:
            tree = ET.parse(input_xml)
            root = tree.getroot()
            annotation_counter = 1
            for passage in root.findall(".//passage"):
                text_node = passage.find("text")
                offset_node = passage.find("offset")
                if text_node is None or not text_node.text:
                    continue
                passage_text = text_node.text
                passage_offset = int(offset_node.text) if offset_node is not None and offset_node.text else 0
                local_start = passage_text.find(mention)
                if local_start < 0:
                    continue

                annotation = ET.SubElement(passage, "annotation", {"id": f"T{annotation_counter}"})
                _add_infon(annotation, "type", "cell_type")
                ET.SubElement(
                    annotation,
                    "location",
                    {
                        "offset": str(passage_offset + local_start),
                        "length": str(len(mention)),
                    },
                )
                mention_node = ET.SubElement(annotation, "text")
                mention_node.text = mention
                annotation_counter += 1

            if output_xml is not None:
                Path(output_xml).parent.mkdir(parents=True, exist_ok=True)
                ET.indent(tree, space="  ")
                tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        elif input_file is not None:
            entries = json.loads(Path(input_file).read_text(encoding="utf-8"))
            predictions = []
            for entry in entries:
                text = entry["text"]
                local_start = text.find(mention)
                predicted_entities = []
                if local_start >= 0:
                    passage_offset = int(entry.get("passage_offset", 0))
                    predicted_entities.append(
                        {
                            "label": "cell_type",
                            "start_local": local_start,
                            "end_local": local_start + len(mention),
                            "start": passage_offset + local_start,
                            "end": passage_offset + local_start + len(mention),
                            "text": mention,
                        }
                    )
                predictions.append(
                    {
                        "id": entry["id"],
                        "document_id": entry.get("document_id"),
                        "passage_id": entry.get("passage_id"),
                        "passage_offset": entry.get("passage_offset", 0),
                        "text": text,
                        "predicted_entities": predicted_entities,
                    }
                )
            (output_dir / "predictions.json").write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:  # pragma: no cover - mirrors runtime guard
            raise ValueError("fake_predict_ner needs input_xml or input_file")

        return 0

    fake_recognition_predict.predict_ner = fake_predict_ner
    monkeypatch.setitem(sys.modules, "cellexlink.recognition.predict", fake_recognition_predict)

    fake_normalization_linker = types.ModuleType("cellexlink.normalization.linker")

    def fake_normalize_collection(collection, **kwargs):
        if normalizer_call is not None:
            normalizer_call.update(kwargs)
        del kwargs
        for document in collection.documents:
            for passage in document.passages:
                for annotation in passage.annotations:
                    annotation.infons["CellExLink-Sapbert_id_0"] = "CL:0000077"
                    annotation.infons["CellExLink-Sapbert_identifier_name_0"] = "mesothelial cell"
        return collection

    class _DummyNormalizationType:
        @classmethod
        def from_files(cls, **kwargs):
            del kwargs
            return cls()

    fake_normalization_linker.AMBIGUOUS_TOPN = 5
    fake_normalization_linker.DEFAULT_NEN_MODEL = "dummy-nen-model"
    fake_normalization_linker.DEFAULT_TOPN = 1
    fake_normalization_linker.CellOntologyLinker = _DummyNormalizationType
    fake_normalization_linker.MentionRecord = _DummyNormalizationType
    fake_normalization_linker.NormalizationCandidate = _DummyNormalizationType
    fake_normalization_linker.NormalizationResult = _DummyNormalizationType
    fake_normalization_linker.normalize_bioc = fake_normalize_collection
    fake_normalization_linker.normalize_collection = fake_normalize_collection
    monkeypatch.setitem(sys.modules, "cellexlink.normalization.linker", fake_normalization_linker)


def test_run_text_end_to_end_smoke_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline

    _install_fake_model_modules(monkeypatch)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    results = pipeline.run_text(
        "The mesothelial cell was detected in the sample.",
        task="end-to-end",
        document_id="doc-smoke",
        output_dir=tmp_path / "text-work",
    )

    assert len(results) == 1
    assert results[0].document_id == "doc-smoke"
    assert results[0].mention == "mesothelial cell"
    assert results[0].start == 4
    assert results[0].end == 20
    assert results[0].entity_type == "cell_type"
    assert results[0].identifier == "CL:0000077"
    assert results[0].label == "mesothelial cell"
    assert results[0].score is None


def test_run_text_omits_document_id_when_not_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline

    _install_fake_model_modules(monkeypatch)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    results = pipeline.run_text(
        "The mesothelial cell was detected in the sample.",
        task="end-to-end",
        output_dir=tmp_path / "text-work",
    )

    assert len(results) == 1
    assert results[0].document_id is None
    assert "document_id" not in results[0].to_dict()


def test_pipeline_passes_verbose_flag_to_predictor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline

    predictor_call: dict[str, object] = {}
    _install_fake_model_modules(monkeypatch, predictor_call=predictor_call)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
        verbose=True,
    )

    _ = pipeline.run_text(
        "The mesothelial cell was detected in the sample.",
        task="ner",
        output_dir=tmp_path / "text-work",
    )

    assert predictor_call["verbose"] is True


def test_pipeline_passes_verbose_flag_to_normalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import write_text_as_bioc_xml

    normalizer_call: dict[str, object] = {}
    _install_fake_model_modules(monkeypatch, normalizer_call=normalizer_call)

    input_xml = tmp_path / "input.xml"
    output_xml = tmp_path / "normalized.xml"
    write_text_as_bioc_xml(
        "The mesothelial cell was detected in the sample.",
        input_xml,
        document_id="doc-bioc-smoke",
    )

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
        verbose=False,
    )

    _ = pipeline.run_bioc(
        input_xml,
        output_xml,
        task="nen",
        input_format="bioc-xml",
        output_format="bioc-xml",
        output_dir=tmp_path / "work",
    )

    assert normalizer_call["verbose"] is False


def test_run_bioc_end_to_end_smoke_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import write_text_as_bioc_xml

    _install_fake_model_modules(monkeypatch)

    input_xml = tmp_path / "input.xml"
    output_xml = tmp_path / "normalized.xml"
    ner_xml = tmp_path / "ner.xml"
    write_text_as_bioc_xml(
        "The mesothelial cell was detected in the sample.",
        input_xml,
        document_id="doc-bioc-smoke",
    )

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    returned_path = pipeline.run_bioc(
        input_xml,
        output_xml,
        task="end-to-end",
        input_format="bioc-xml",
        output_format="bioc-xml",
        ner_output_xml=ner_xml,
        output_dir=tmp_path / "work",
    )

    assert returned_path == output_xml
    assert output_xml.is_file()
    assert ner_xml.is_file()

    results = pipeline.read_predictions_from_bioc(output_xml)
    assert len(results) == 1
    assert results[0].document_id == "doc-bioc-smoke"
    assert results[0].mention == "mesothelial cell"
    assert results[0].identifier == "CL:0000077"


def test_run_bioc_raises_for_missing_input(tmp_path: Path) -> None:
    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    with pytest.raises(FileNotFoundError):
        pipeline.run_bioc(
            tmp_path / "missing.xml",
            tmp_path / "out.xml",
        )


def test_run_bioc_generic_json_uses_shared_collection_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import read_bioc_collection

    _install_fake_model_modules(monkeypatch)

    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.bioc.json"
    input_json.write_text(
        json.dumps(
            {
                "document_id": "doc-generic",
                "text": "EGFR and mesothelial cell were detected.",
                "entities": [
                    {
                        "id": "T1",
                        "text": "EGFR",
                        "start_pos": 0,
                        "end_pos": 4,
                        "labels": [{"value": "gene", "score": 0.99}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    written = pipeline.run_bioc(
        input_json,
        output_json,
        task="end-to-end",
        input_format="json",
        output_format="bioc-json",
        preserve_existing_annotations=True,
        output_dir=tmp_path / "work",
    )

    assert written == output_json
    collection = read_bioc_collection(output_json)
    annotations = collection.documents[0].passages[0].annotations

    assert any(annotation.text == "EGFR" and annotation.infons.get("type") == "gene" for annotation in annotations)
    assert any(
        annotation.text == "mesothelial cell"
        and annotation.infons.get("type") == "cell_type"
        and annotation.infons.get("identifier") == "CL:0000077"
        and annotation.infons.get("label") == "mesothelial cell"
        and "CellExLink-Sapbert_id_0" not in annotation.infons
        for annotation in annotations
    )


def test_normalize_bioc_accepts_bioc_json_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink.io import (
        BioCAnnotation,
        BioCCollection,
        BioCDocument,
        BioCLocation,
        BioCPassage,
        read_bioc_collection,
        write_bioc_collection,
    )
    from cellexlink.normalization import linker

    input_json = tmp_path / "annotated.bioc.json"
    output_xml = tmp_path / "normalized.xml"

    collection = BioCCollection(
        source="test",
        date="2026-06-17",
        key="unit-test",
        documents=[
            BioCDocument(
                id="doc-json",
                passages=[
                    BioCPassage(
                        offset=0,
                        text="The mesothelial cell was detected in the sample.",
                        annotations=[
                            BioCAnnotation(
                                id="T1",
                                infons={"type": "cell_type"},
                                locations=[BioCLocation(offset=4, length=16)],
                                text="mesothelial cell",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    write_bioc_collection(collection, input_json, output_format="bioc-json")

    def fake_normalize_collection(collection, **kwargs):
        del kwargs
        for document in collection.documents:
            for passage in document.passages:
                for annotation in passage.annotations:
                    annotation.infons["CellExLink-Sapbert_id_0"] = "CL:0000077"
                    annotation.infons["CellExLink-Sapbert_identifier_name_0"] = "mesothelial cell"
        return collection

    monkeypatch.setattr(linker, "normalize_collection", fake_normalize_collection)

    written = linker.normalize_bioc(input_json, output_xml)

    assert written == output_xml
    assert output_xml.is_file()

    normalized = read_bioc_collection(output_xml, input_format="bioc-xml")
    annotations = normalized.documents[0].passages[0].annotations

    assert len(annotations) == 1
    assert annotations[0].text == "mesothelial cell"
    assert annotations[0].infons["CellExLink-Sapbert_id_0"] == "CL:0000077"


def test_pipeline_run_bioc_nen_accepts_bioc_json_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import (
        BioCAnnotation,
        BioCCollection,
        BioCDocument,
        BioCLocation,
        BioCPassage,
        read_bioc_collection,
        write_bioc_collection,
    )

    fake_normalization_linker = types.ModuleType("cellexlink.normalization.linker")

    def fake_normalize_collection(collection, **kwargs):
        del kwargs
        for document in collection.documents:
            for passage in document.passages:
                for annotation in passage.annotations:
                    annotation.infons["CellExLink-Sapbert_id_0"] = "CL:0000077"
                    annotation.infons["CellExLink-Sapbert_identifier_name_0"] = "mesothelial cell"
        return collection

    fake_normalization_linker.normalize_collection = fake_normalize_collection
    monkeypatch.setitem(sys.modules, "cellexlink.normalization.linker", fake_normalization_linker)

    input_json = tmp_path / "annotated.bioc.json"
    output_xml = tmp_path / "normalized.xml"
    collection = BioCCollection(
        source="test",
        date="2026-06-17",
        key="unit-test",
        documents=[
            BioCDocument(
                id="doc-json",
                passages=[
                    BioCPassage(
                        offset=0,
                        text="The mesothelial cell was detected in the sample.",
                        annotations=[
                            BioCAnnotation(
                                id="T1",
                                infons={"type": "cell_type"},
                                locations=[BioCLocation(offset=4, length=16)],
                                text="mesothelial cell",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    write_bioc_collection(collection, input_json, output_format="bioc-json")

    pipeline = CellExLinkPipeline.from_pretrained(
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    written = pipeline.run_bioc(
        input_json,
        output_xml,
        task="nen",
        input_format="bioc-json",
        output_format="bioc-xml",
    )

    assert written == output_xml
    normalized = read_bioc_collection(output_xml, input_format="bioc-xml")
    annotations = normalized.documents[0].passages[0].annotations
    assert len(annotations) == 1
    assert annotations[0].infons == {
        "type": "cell_type",
        "identifier": "CL:0000077",
        "label": "mesothelial cell",
    }

def test_write_predictions_json_groups_annotations_by_document(tmp_path: Path) -> None:
    from cellexlink import ExtractionResult, write_predictions_json

    output_path = tmp_path / "predictions.json"
    predictions = [
        ExtractionResult(
            document_id="doc0",
            passage_index=0,
            mention="mesothelial cell",
            start=4,
            end=20,
            entity_type="cell_type",
            identifier="CL:0000077",
            label="mesothelial cell",
            score=0.99,
            source="unit-test",
        ),
        ExtractionResult(
            document_id="doc0",
            passage_index=0,
            mention="SMC",
            start=25,
            end=28,
            entity_type="cell_type",
            identifier="CL:0000192",
            label="smooth muscle cell",
            score=0.95,
            source="unit-test",
        ),
    ]

    written = write_predictions_json(predictions, output_path)
    assert written == output_path

    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert payload["document_id"] == "doc0"
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][0]["mention"] == "mesothelial cell"
    assert payload["annotations"][0]["span"] == {"begin": 4, "end": 20}
    assert payload["annotations"][0]["identifier"] == "CL:0000077"
    assert payload["annotations"][0]["label"] == "mesothelial cell"
    assert "source" not in payload["annotations"][0]
    assert payload["annotations"][1]["mention"] == "SMC"
    assert payload["annotations"][1]["identifier"] == "CL:0000192"
    assert payload["annotations"][1]["label"] == "smooth muscle cell"
    assert "confidence_score" not in payload["annotations"][1]


def test_write_predictions_json_omits_document_id_when_missing(tmp_path: Path) -> None:
    from cellexlink import ExtractionResult, write_predictions_json

    output_path = tmp_path / "predictions.json"
    predictions = [
        ExtractionResult(
            document_id=None,
            passage_index=0,
            mention="mesothelial cell",
            start=4,
            end=20,
            entity_type="cell_type",
            identifier="CL:0000077",
            label="mesothelial cell",
            score=0.99,
            source="unit-test",
        )
    ]

    write_predictions_json(predictions, output_path)

    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert "document_id" not in payload
    assert payload["annotations"][0]["mention"] == "mesothelial cell"
    assert payload["annotations"][0]["identifier"] == "CL:0000077"
    assert payload["annotations"][0]["label"] == "mesothelial cell"
    assert "confidence_score" not in payload["annotations"][0]
    assert "source" not in payload["annotations"][0]


def test_write_predictions_json_strips_candidate_scores_and_sources(tmp_path: Path) -> None:
    from cellexlink import write_predictions_json

    output_path = tmp_path / "predictions.json"
    predictions = [
        {
            "document_id": "doc0",
            "passage_index": 0,
            "mention": "mesothelial cell",
            "start": 4,
            "end": 20,
            "entity_type": "cell_type",
            "identifier": "CL:0000077",
            "label": "mesothelial cell",
            "score": 0.99,
            "candidates": [
                {
                    "identifier": "CL:0000077",
                    "preferred_label": "mesothelial cell",
                    "final_score": 0.99,
                    "embedding_score": -0.02,
                    "source": "model_normal",
                    "ab3p_match_score": 1.0,
                }
            ],
            "infons": {
                "CellExLink-Sapbert_confidence_score_0": "0.99",
                "CellExLink-Sapbert_embedding_score_0": "-0.02",
                "CellExLink-Sapbert_match_source": "model_normal",
                "CellExLink-Sapbert_id_0": "CL:0000077",
            },
        }
    ]

    write_predictions_json(
        predictions,
        output_path,
        include_document_metadata=True,
    )

    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    annotation = payload["annotations"][0]
    assert "confidence_score" not in annotation
    assert annotation["identifier"] == "CL:0000077"
    assert annotation["label"] == "mesothelial cell"
    assert annotation["candidates"] == [
        {
            "identifier": "CL:0000077",
            "preferred_label": "mesothelial cell",
        }
    ]
    assert annotation["infons"] == {
        "CellExLink-Sapbert_id_0": "CL:0000077",
    }


def test_run_text_accepts_case_insensitive_ner_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline, RecognizedMention

    _install_fake_model_modules(monkeypatch)
    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    results = pipeline.run_text(
        "The mesothelial cell was detected.",
        task=" NER ",
    )

    assert len(results) == 1
    assert isinstance(results[0], RecognizedMention)
    assert results[0].mention == "mesothelial cell"


def test_run_text_rejects_nen_without_spans(tmp_path: Path) -> None:
    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(output_dir=tmp_path / "work")
    with pytest.raises(ValueError, match="NEN requires existing mention spans"):
        pipeline.run_text("T cells were detected.", task="nen")


def test_run_bioc_chunks_passages_but_uses_full_document_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import (
        BioCCollection,
        BioCDocument,
        BioCPassage,
        read_bioc_collection,
        write_bioc_collection,
    )

    normalizer_call: dict[str, object] = {}
    _install_fake_model_modules(monkeypatch, normalizer_call=normalizer_call)

    input_xml = tmp_path / "large.xml"
    output_xml = tmp_path / "large.out.xml"
    collection = BioCCollection(
        documents=[
            BioCDocument(
                id="doc-large",
                passages=[
                    BioCPassage(offset=0, text="A mesothelial cell was detected."),
                    BioCPassage(offset=100, text="Another mesothelial cell was detected."),
                ],
            )
        ]
    )
    write_bioc_collection(collection, input_xml, output_format="bioc-xml")

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )
    pipeline.run_bioc(
        input_xml,
        output_xml,
        passage_chunk_size=1,
    )

    document_context = normalizer_call["document_context"]
    assert isinstance(document_context, dict)
    full_text = next(iter(document_context.values()))
    assert "A mesothelial cell was detected." in full_text
    assert "Another mesothelial cell was detected." in full_text

    result = read_bioc_collection(output_xml)
    annotations = [
        annotation
        for passage in result.documents[0].passages
        for annotation in passage.annotations
    ]
    assert [annotation.id for annotation in annotations] == ["T1", "T2"]
    assert all(annotation.infons["identifier"] == "CL:0000077" for annotation in annotations)


def test_run_files_writes_one_result_per_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline

    _install_fake_model_modules(monkeypatch)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "first.txt").write_text(
        "The mesothelial cell was detected.",
        encoding="utf-8",
    )
    (inputs / "second.txt").write_text(
        "Another mesothelial cell was detected.",
        encoding="utf-8",
    )

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )
    outputs = pipeline.run_files(
        inputs,
        tmp_path / "results",
        batch_size=2,
        passage_chunk_size=1,
    )

    assert [path.name for path in outputs] == ["first.json", "second.json"]
    assert all(path.is_file() for path in outputs)
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["annotations"][0]["identifier"]
        == "CL:0000077"
        for path in outputs
    )


def test_run_files_rejects_legacy_chunk_keywords(tmp_path: Path) -> None:
    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'chunk_size'"):
        pipeline.run_files(
            tmp_path,
            tmp_path / "results",
            chunk_size=3,
        )


def test_run_pmids_chunks_identifiers_and_merges_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import (
        BioCCollection,
        BioCDocument,
        BioCPassage,
        read_bioc_collection,
        write_bioc_collection,
    )
    from cellexlink.retrieval import FetchReport
    import cellexlink.retrieval as retrieval

    _install_fake_model_modules(monkeypatch)
    fetch_calls: list[list[str]] = []

    def fake_fetch_pubmed_bioc(
        ids,
        output_path,
        *,
        text_source="abstract",
        output_format="bioc-xml",
        source="ncbi",
        timeout=30,
        pause=0.12,
        opener=None,
    ):
        del timeout, pause, opener
        id_list = [str(value) for value in ids]
        fetch_calls.append(id_list)
        collection = BioCCollection(
            source="test retrieval",
            documents=[
                BioCDocument(
                    id=record_id,
                    passages=[
                        BioCPassage(
                            offset=0,
                            text=f"PMID {record_id} contains a mesothelial cell.",
                        )
                    ],
                )
                for record_id in id_list
            ],
        )
        output = write_bioc_collection(
            collection,
            output_path,
            output_format=output_format,
        )
        return FetchReport(
            output_path=output,
            requested_ids=id_list,
            fetched_ids=id_list,
            source=source,
            text_source=text_source,
            output_format=output_format,
        )

    monkeypatch.setattr(retrieval, "fetch_pubmed_bioc", fake_fetch_pubmed_bioc)

    output_xml = tmp_path / "pmids.xml"
    intermediate_xml = tmp_path / "retrieved.xml"
    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )
    written = pipeline.run_pmids(
        ["1", "2", "3", "4", "5"],
        output_xml,
        batch_size=2,
        passage_chunk_size=1,
        keep_intermediate_bioc=intermediate_xml,
    )

    assert written == output_xml
    assert fetch_calls == [["1", "2"], ["3", "4"], ["5"]]
    result = read_bioc_collection(output_xml)
    assert result.source == "test retrieval"
    assert [document.id for document in result.documents] == ["1", "2", "3", "4", "5"]
    generated = [
        annotation
        for document in result.documents
        for passage in document.passages
        for annotation in passage.annotations
    ]
    assert [annotation.id for annotation in generated] == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    ]
    assert all(
        "__cellexlink_generated_annotation" not in annotation.infons
        for annotation in generated
    )
    assert all(
        document.passages[0].annotations[0].infons["identifier"] == "CL:0000077"
        for document in result.documents
    )
    retrieved = read_bioc_collection(intermediate_xml)
    assert [document.id for document in retrieved.documents] == ["1", "2", "3", "4", "5"]
