"""Lightweight tests for the public API shape."""

from __future__ import annotations


def test_public_api_imports() -> None:
    import cellexlink

    assert hasattr(cellexlink, "CellExLinkPipeline")
    assert hasattr(cellexlink, "ExtractionResult")
    assert not hasattr(cellexlink, "fetch_pubmed_bioc")


def test_pipeline_exposes_the_compact_workflow_api() -> None:
    from cellexlink import CellExLinkPipeline

    pipe = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner",
        nen_model="dummy-nen",
    )

    assert callable(pipe.run_text)
    assert callable(pipe.run_bioc)
    assert callable(pipe.run_files)
    assert callable(pipe.run_pmids)

    assert not hasattr(pipe, "prepare")
    assert not hasattr(pipe, "fetch_bioc")
    assert not hasattr(pipe, "recognize_text")
    assert not hasattr(pipe, "extract_text")
    assert not hasattr(pipe, "extract_text_file")
    assert not hasattr(pipe, "recognize_bioc")
    assert not hasattr(pipe, "normalize_bioc")
    assert not hasattr(pipe, "extract_bioc")


def test_result_objects_have_dict_output() -> None:
    from cellexlink import ExtractionResult

    result = ExtractionResult(
        document_id="doc0",
        passage_index=0,
        mention="SMC",
        start=0,
        end=3,
        entity_type="cell_type",
        identifier="CL:0000192",
    )

    assert result.to_dict()["identifier"] == "CL:0000192"
    assert result.length == 3
