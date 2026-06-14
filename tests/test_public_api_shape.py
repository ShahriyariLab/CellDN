"""Lightweight tests for the public API shape.

These tests do not load model checkpoints.
"""

from __future__ import annotations


def test_public_api_imports() -> None:
    import cellexlink

    assert hasattr(cellexlink, "CellExLinkPipeline")
    assert hasattr(cellexlink, "ExtractionResult")
    assert hasattr(cellexlink, "MentionInput")


def test_pipeline_exposes_three_tasks() -> None:
    from cellexlink import CellExLinkPipeline

    pipe = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner",
        nen_model="dummy-nen",
    )

    # NER
    assert callable(pipe.recognize_text)
    assert callable(pipe.recognize_bioc)

    # NEN
    assert callable(pipe.normalize_mentions)
    assert callable(pipe.normalize_bioc)

    # End-to-end
    assert callable(pipe.extract_text)
    assert callable(pipe.extract_bioc)


def test_result_objects_have_dict_output() -> None:
    from cellexlink import ExtractionResult, MentionInput

    mention = MentionInput(text="SMC", start=0, end=3)
    result = ExtractionResult(
        document_id="doc0",
        passage_index=0,
        mention="SMC",
        start=0,
        end=3,
        entity_type="cell_type",
        cl_id="CL:0000192",
    )

    assert mention.to_dict()["text"] == "SMC"
    assert result.to_dict()["cl_id"] == "CL:0000192"
    assert result.length == 3
