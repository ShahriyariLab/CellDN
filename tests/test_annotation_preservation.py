"""Model-free regressions for merging source annotations after prediction."""

from __future__ import annotations

from copy import deepcopy

import pytest
from test_api_smoke import _install_fake_model_modules

from celldn import CellDNPipeline
from celldn.io import (
    BioCAnnotation,
    BioCCollection,
    BioCDocument,
    BioCLocation,
    BioCPassage,
    read_bioc_collection,
    write_bioc_collection,
)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    # The fake normalizer modifies every annotation it receives. Exact source
    # equality below therefore also checks that originals bypass normalization.
    _install_fake_model_modules(monkeypatch)
    return CellDNPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )


def _annotated_collection(source: str = "PubTator") -> BioCCollection:
    documents = []
    for document_index in range(2):
        source_record = f"{source}-{document_index}"
        documents.append(BioCDocument(
            id="repeated-document-id",
            infons={"source_record": source_record},
            passages=[
                BioCPassage(
                    offset=0,
                    text="BRCA1 and BRCA1 identify a mesothelial cell.",
                    infons={"type": "title"},
                    annotations=[BioCAnnotation(
                        id="T1",
                        text="BRCA1 BRCA1",
                        infons={
                            "type": "Gene", "identifier": "672",
                            "source": "PubTator", "source_record": source_record,
                            "confidence": "0.99", "name": "BRCA1",
                            "CellDN_runtime_seconds": "original-runtime-value",
                        },
                        locations=[BioCLocation(0, 5), BioCLocation(10, 5)],
                    )],
                ),
                # Repeated offsets and IDs must not be used as passage keys.
                BioCPassage(
                    offset=0,
                    text="Cancer surrounds a mesothelial cell.",
                    infons={"type": "abstract"},
                    annotations=[BioCAnnotation(
                        id="T2",
                        text="Cancer",
                        infons={
                            "type": "Disease", "identifier": "MESH:D009369",
                            "source": "PubTator", "source_record": source_record,
                        },
                        locations=[BioCLocation(0, 6)],
                    )],
                ),
                BioCPassage(
                    offset=200,
                    text="BRCA1 in references mentions a mesothelial cell.",
                    infons={"section_type": "REF"},
                    annotations=[BioCAnnotation(
                        id="REF", text="BRCA1", infons={"type": "Gene"},
                        locations=[BioCLocation(200, 5)],
                    )],
                ),
                BioCPassage(
                    offset=400,
                    text="Cancer and a mesothelial cell in excluded material.",
                    infons={"type": "abstract", "annotatable": "false"},
                    annotations=[BioCAnnotation(
                        id="EXCLUDED", text="Cancer", infons={"type": "Disease"},
                        locations=[BioCLocation(400, 6)],
                    )],
                ),
            ],
        ))
    return BioCCollection(source=source, documents=documents)


def _assert_predictions(result, original, *, preserve: bool, task: str = "end-to-end"):
    assert len(result.documents) == len(original.documents)
    for document, source_document in zip(result.documents, original.documents, strict=True):
        assert document.id == source_document.id
        assert document.infons == source_document.infons
        assert len(document.passages) == 2
        for passage, source_passage in zip(
            document.passages, source_document.passages[:2], strict=True,
        ):
            assert passage.text == source_passage.text
            assert passage.offset == source_passage.offset
            originals = [
                annotation.to_json() for annotation in passage.annotations
                if annotation.infons.get("type") in {"Gene", "Disease"}
            ]
            expected = [annotation.to_json() for annotation in source_passage.annotations]
            assert originals == (expected if preserve else [])
            generated = [
                annotation for annotation in passage.annotations
                if annotation.infons.get("type") == "cell_type"
            ]
            assert len(generated) == 1
            assert len(passage.annotations) == (2 if preserve else 1)
            assert generated[0].text == "mesothelial cell"
            assert generated[0].locations == [BioCLocation(
                passage.offset + passage.text.index("mesothelial cell"),
                len("mesothelial cell"),
            )]
            if task == "end-to-end":
                assert generated[0].infons["identifier"] == "CL:0000077"
            else:
                assert "identifier" not in generated[0].infons
            assert not any(key.startswith("__celldn") for key in generated[0].infons)
        annotation_ids = [
            annotation.id for passage in document.passages for annotation in passage.annotations
        ]
        assert len(annotation_ids) == len(set(annotation_ids))
        if preserve:
            original_ids = {
                annotation.id for passage in source_document.passages[:2]
                for annotation in passage.annotations
            }
            assert original_ids.issubset(annotation_ids)


@pytest.mark.parametrize("preserve", [None, False, True])
@pytest.mark.parametrize("task", ["ner", "end-to-end"])
@pytest.mark.parametrize("output_format,suffix", [("bioc-xml", ".xml"), ("bioc-json", ".json")])
def test_bioc_preserves_source_annotations_after_prediction_across_chunks(
    pipeline, tmp_path, preserve, task, output_format, suffix,
):
    original = _annotated_collection()
    source = write_bioc_collection(original, tmp_path / f"source{suffix}", output_format=output_format)
    options = {} if preserve is None else {"preserve_existing_annotations": preserve}
    documents = []
    for chunk_size in (1, 128):
        output = tmp_path / f"result-{chunk_size}{suffix}"
        assert pipeline.run_bioc(
            source, output, task=task, passage_chunk_size=chunk_size,
            output_format=output_format, **options,
        ) == output
        result = read_bioc_collection(output)
        _assert_predictions(result, original, preserve=preserve is True, task=task)
        documents.append([document.to_json() for document in result.documents])
    assert documents[0] == documents[1]
    assert read_bioc_collection(source).to_json() == original.to_json()


@pytest.mark.parametrize("chunk_size", [1, 128])
def test_intermediate_ner_output_contains_only_generated_annotations(pipeline, tmp_path, chunk_size):
    original = _annotated_collection()
    source = write_bioc_collection(original, tmp_path / "source.xml")
    output = tmp_path / "result.xml"
    ner_output = tmp_path / "ner.xml"
    pipeline.run_bioc(
        source, output, preserve_existing_annotations=True,
        passage_chunk_size=chunk_size, ner_output_xml=ner_output,
    )
    _assert_predictions(read_bioc_collection(output), original, preserve=True)
    _assert_predictions(read_bioc_collection(ner_output), original, preserve=False, task="ner")


@pytest.mark.parametrize("preserve", [False, True])
def test_file_batches_preserve_annotations_without_crossing_source_boundaries(
    pipeline, tmp_path, preserve,
):
    originals = [_annotated_collection("first"), _annotated_collection("second")]
    sources = [
        write_bioc_collection(original, tmp_path / f"source-{index}.xml")
        for index, original in enumerate(originals)
    ]
    documents = []
    for batch_size, chunk_size in ((1, 1), (2, 128)):
        outputs = pipeline.run_files(
            sources, tmp_path / f"results-{batch_size}",
            batch_size=batch_size, passage_chunk_size=chunk_size,
            preserve_existing_annotations=preserve,
        )
        assert len(outputs) == len(sources)
        batch_documents = []
        for output, original in zip(outputs, originals, strict=True):
            result = read_bioc_collection(output)
            _assert_predictions(result, original, preserve=preserve)
            batch_documents.append([document.to_json() for document in result.documents])
        documents.append(batch_documents)
    assert documents[0] == documents[1]


@pytest.mark.parametrize("preserve", [False, True])
def test_pmid_batches_preserve_pubtator_annotations_after_final_merge(
    pipeline, monkeypatch, tmp_path, preserve,
):
    import celldn.retrieval as retrieval

    ids = ["11", "22", "33"]
    originals = []
    for record_id in ids:
        document = _annotated_collection(record_id).documents[0]
        document.id = record_id
        originals.append(document)
    # These originals appear only in the final retrieval chunk. Their IDs
    # must be reserved before any earlier generated annotation is finalized.
    originals[-1].passages[0].annotations[0].id = "T3"
    originals[-1].passages[1].annotations[0].id = "T4"
    expected = BioCCollection(documents=originals)
    fetch_calls = []

    def fetch(id_chunk, output_path, **kwargs):
        requested = list(id_chunk)
        fetch_calls.append(requested)
        collection = BioCCollection(documents=[
            deepcopy(document) for document in originals if document.id in requested
        ])
        output = write_bioc_collection(collection, output_path)
        return retrieval.FetchReport(
            output_path=output, requested_ids=requested, fetched_ids=requested,
            source=kwargs["source"], text_source=kwargs["text_source"],
            output_format=kwargs["output_format"],
        )

    monkeypatch.setattr(retrieval, "fetch_pubmed_bioc", fetch)
    output = tmp_path / "pmids.xml"
    assert pipeline.run_pmids(
        ids, output, batch_size=2, passage_chunk_size=1,
        preserve_existing_annotations=preserve,
    ) == output
    assert fetch_calls == [["11", "22"], ["33"]]
    result = read_bioc_collection(output)
    _assert_predictions(result, expected, preserve=preserve)
    if preserve:
        original_ids = {
            annotation.id for document in originals for passage in document.passages[:2]
            for annotation in passage.annotations
        }
        generated_ids = {
            annotation.id for document in result.documents for passage in document.passages
            for annotation in passage.annotations if annotation.infons.get("type") == "cell_type"
        }
        assert generated_ids.isdisjoint(original_ids)


@pytest.mark.parametrize("chunk_size", [1, 128])
def test_nen_only_preservation_does_not_duplicate_supplied_mentions(pipeline, tmp_path, chunk_size):
    original = _annotated_collection()
    for document in original.documents:
        document.passages = document.passages[:2]
        for index, passage in enumerate(document.passages):
            passage.annotations = [BioCAnnotation(
                id=f"T{index + 1}", text="mesothelial cell", infons={"type": "cell_type"},
                locations=[BioCLocation(
                    passage.offset + passage.text.index("mesothelial cell"),
                    len("mesothelial cell"),
                )],
            )]
    source = write_bioc_collection(original, tmp_path / "mentions.xml")
    documents = []
    for preserve in (None, False, True):
        options = {} if preserve is None else {"preserve_existing_annotations": preserve}
        output = tmp_path / f"normalized-{preserve}.xml"
        pipeline.run_bioc(source, output, task="nen", passage_chunk_size=chunk_size, **options)
        result = read_bioc_collection(output)
        for document in result.documents:
            for index, passage in enumerate(document.passages):
                assert len(passage.annotations) == 1
                annotation = passage.annotations[0]
                assert annotation.id == f"T{index + 1}"
                assert annotation.infons["identifier"] == "CL:0000077"
        documents.append([document.to_json() for document in result.documents])
    assert documents[0] == documents[1] == documents[2]
