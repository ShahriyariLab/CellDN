"""Offline coverage of official CL JSON conversion, provenance, and failures."""

from __future__ import annotations

import hashlib
import io
import json
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import celldn.ontology_download as download
from celldn.normalization.ontology import load_cell_ontology_terms


def _node(number: int, label: str, *, synonyms=None, deprecated=False) -> dict:
    return {
        "id": f"http://purl.obolibrary.org/obo/CL_{number:07d}",
        "lbl": label,
        "type": "CLASS",
        "meta": {
            "synonyms": [] if synonyms is None else synonyms,
            "deprecated": deprecated,
        },
    }


def _synonym(name: str, scope: str = "hasExactSynonym") -> dict:
    return {"pred": scope, "val": name}


def _graph(nodes=None, *, version="2026-09-15") -> dict:
    return {
        "graphs": [{
            "id": "http://purl.obolibrary.org/obo/cl.json",
            "meta": {
                "version": f"http://purl.obolibrary.org/obo/cl/releases/{version}/cl.json",
                "basicPropertyValues": [{
                    "pred": "http://www.w3.org/2002/07/owl#versionInfo", "val": version,
                }],
            },
            "nodes": [_node(77, "mesothelial cell")] if nodes is None else nodes,
            "edges": [],
        }],
    }


def _response(monkeypatch, payload):
    calls = []
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return payload, data

    monkeypatch.setattr(download, "_request_json", request)
    return calls, data


def test_conversion_preserves_all_synonym_scopes_and_five_field_schema():
    node = _node(77, "mesothelial cell", synonyms=[
        _synonym("mesotheliocyte"), _synonym("mesotheliocyte"),
        _synonym("Shared name", "hasRelatedSynonym"),
        _synonym("broad alias", "hasBroadSynonym"),
        _synonym("narrow alias", "hasNarrowSynonym"),
        _synonym(""), _synonym("α cell"),
    ])
    converted = download.convert_obograph_node(node)
    assert converted == {
        "norm_concept_id": "CL:0000077",
        "norm_concept_id_underscore": "CL_0000077",
        "norm_preferred_label": "mesothelial cell",
        "synonyms": ["Shared name", "broad alias", "mesotheliocyte", "narrow alias", "α cell"],
        "namespace": "CL",
    }
    assert len(node["meta"]["synonyms"]) == 7
    other = download.convert_obograph_node(_node(78, "another cell", synonyms=[_synonym("Shared name")]))
    assert other["synonyms"] == ["Shared name"]


def test_conversion_filters_imports_non_classes_and_deprecated_nodes():
    assert download.convert_obograph_node({"id": "http://purl.obolibrary.org/obo/UBERON_0001062"}) is None
    property_node = _node(77, "cell property")
    property_node["type"] = "PROPERTY"
    assert download.convert_obograph_node(property_node) is None
    deprecated = _node(77, "obsolete example", deprecated=True)
    assert download.convert_obograph_node(deprecated) is None
    assert download.convert_obograph_node(deprecated, include_obsolete=True) is not None
    current = _node(78, "current cell")
    current.pop("meta")
    assert download.convert_obograph_node(current)["synonyms"] == []


@pytest.mark.parametrize("change", [
    {"id": None}, {"id": "CL_bad"}, {"lbl": " "}, {"lbl": "\ud800"},
    {"meta": []}, {"meta": {"deprecated": "maybe"}},
    {"meta": {"synonyms": "bad"}}, {"meta": {"synonyms": [7]}},
    {"meta": {"synonyms": [{"val": 7}]}},
    {"meta": {"synonyms": [{"val": "\ud800"}]}},
])
def test_malformed_cl_node_is_rejected(change):
    node = _node(77, "mesothelial cell")
    node.update(change)
    with pytest.raises(download.OntologyDownloadError):
        download.convert_obograph_node(node)


def test_default_export_is_compatible_and_records_exact_version_without_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BIOPORTAL_API_KEY", raising=False)
    payload = _graph([
        _node(78, "another cell", synonyms=[_synonym("shared")]),
        {"id": "http://purl.obolibrary.org/obo/UBERON_0001062"},
        _node(77, "mesothelial cell", synonyms=[_synonym("shared"), _synonym("mesotheliocyte")]),
        _node(79, "old cell", deprecated=True),
        {"id": "CL:0000080", "type": "PROPERTY"},
    ])
    calls, source_data = _response(monkeypatch, payload)
    result = download.download_cell_ontology()
    assert result == tmp_path / "resources" / "cell_ontology_v2026-09-15.jsonl"
    rows = [json.loads(line) for line in result.read_text().splitlines()]
    assert [row["norm_concept_id"] for row in rows] == ["CL:0000077", "CL:0000078"]
    terms, concepts = load_cell_ontology_terms(result)
    assert set(concepts) == {"CL:0000077", "CL:0000078"}
    assert sum(entry.raw_name == "shared" for entry in terms) == 2
    metadata = json.loads(result.with_suffix(".metadata.json").read_text())
    assert metadata["ontology"] == "CL"
    assert metadata["version"] == "2026-09-15"
    assert metadata["version_iri"].endswith("/2026-09-15/cl.json")
    assert metadata["concept_count"] == 2
    assert metadata["synonym_count"] == 3
    assert metadata["counts"]["skipped_non_cl_nodes"] == 1
    assert metadata["counts"]["skipped_non_class_nodes"] == 1
    assert metadata["counts"]["skipped_obsolete_classes"] == 1
    assert metadata["retrieved_at_utc"].endswith("Z")
    assert metadata["sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert metadata["source_sha256"] == hashlib.sha256(source_data).hexdigest()
    assert metadata["source_url"] == download.OFFICIAL_CL_URL
    assert calls == [(download.OFFICIAL_CL_URL, {"timeout": 30.0, "max_retries": 3})]


def test_deprecated_terms_can_be_included_explicitly(tmp_path, monkeypatch):
    _response(monkeypatch, _graph([_node(77, "cell"), _node(78, "obsolete cell", deprecated=True)]))
    output = download.download_cell_ontology(output_dir=tmp_path, include_obsolete=True)
    assert len(output.read_text().splitlines()) == 2
    assert json.loads(output.with_suffix(".metadata.json").read_text())["filtering"]["include_obsolete"] is True


@pytest.mark.parametrize("failure", [
    "no_graphs", "wrong_graph", "multiple_main", "nodes_missing", "empty", "non_cl",
    "malformed", "version_missing", "wrong_version_iri", "version_conflict", "bad_date",
    "duplicate_label", "duplicate_deprecated",
])
def test_invalid_download_never_replaces_existing_exports(tmp_path, monkeypatch, failure):
    target = tmp_path / "cell_ontology_v2026-09-15.jsonl"
    sidecar = target.with_suffix(".metadata.json")
    target.write_text("original ontology\n")
    sidecar.write_text('{"original": true}\n')
    payload = _graph()
    graph = payload["graphs"][0]
    if failure == "no_graphs":
        payload = {"error": "wrong endpoint"}
    elif failure == "wrong_graph":
        graph["id"] = "http://purl.obolibrary.org/obo/go.json"
    elif failure == "multiple_main":
        payload["graphs"].append(dict(graph))
    elif failure == "nodes_missing":
        graph.pop("nodes")
    elif failure == "empty":
        graph["nodes"] = []
    elif failure == "non_cl":
        graph["nodes"] = [{"id": "UBERON:0001062", "type": "CLASS"}]
    elif failure == "malformed":
        graph["nodes"].append({"id": "CL:0000078", "type": "CLASS", "lbl": "cell", "meta": {"synonyms": {}}})
    elif failure == "version_missing":
        graph["meta"].pop("version")
    elif failure == "wrong_version_iri":
        graph["meta"]["version"] = "http://purl.obolibrary.org/obo/go/releases/2026-09-15/go.json"
    elif failure == "version_conflict":
        graph["meta"]["basicPropertyValues"][0]["val"] = "2026-09-16"
    elif failure == "bad_date":
        payload = _graph(version="2026-02-30")
    elif failure == "duplicate_label":
        graph["nodes"].append(_node(77, "different label"))
    elif failure == "duplicate_deprecated":
        graph["nodes"].append(_node(77, "obsolete cell", deprecated=True))
    _response(monkeypatch, payload)
    with pytest.raises(download.OntologyDownloadError):
        download.download_cell_ontology(output_dir=tmp_path)
    assert target.read_text() == "original ontology\n"
    assert sidecar.read_text() == '{"original": true}\n'
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([target.name, sidecar.name])


def test_duplicate_nodes_merge_aliases_deterministically(tmp_path, monkeypatch):
    _response(monkeypatch, _graph([
        _node(77, "cell", synonyms=[_synonym("b")]),
        _node(77, "cell", synonyms=[_synonym("a"), _synonym("b")]),
    ]))
    output = download.download_cell_ontology(output_dir=tmp_path)
    assert json.loads(output.read_text())["synonyms"] == ["a", "b"]
    assert json.loads(output.with_suffix(".metadata.json").read_text())["counts"]["duplicate_class_records"] == 1


@pytest.mark.parametrize("existing", ["jsonl", "metadata", "both"])
def test_existing_export_is_replaced_automatically(tmp_path, monkeypatch, existing):
    target = tmp_path / "cell_ontology_v2026-09-15.jsonl"
    metadata_path = target.with_suffix(".metadata.json")
    if existing in {"jsonl", "both"}:
        target.write_text("old\n")
    if existing in {"metadata", "both"}:
        metadata_path.write_text("{}\n")
    _response(monkeypatch, _graph())
    assert download.download_cell_ontology(output_dir=tmp_path) == target
    assert json.loads(target.read_text())["norm_concept_id"] == "CL:0000077"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_repeated_download_refreshes_same_release_and_metadata(tmp_path, monkeypatch):
    _response(monkeypatch, _graph([_node(77, "original label")]))
    first = download.download_cell_ontology(output_dir=tmp_path)
    original_hash = json.loads(first.with_suffix(".metadata.json").read_text())["sha256"]
    calls, source_data = _response(monkeypatch, _graph([
        _node(77, "updated label"), _node(78, "additional cell"),
    ]))
    second = download.download_cell_ontology(output_dir=tmp_path)
    assert first == second
    rows = [json.loads(line) for line in second.read_text().splitlines()]
    assert [row["norm_preferred_label"] for row in rows] == ["updated label", "additional cell"]
    metadata = json.loads(second.with_suffix(".metadata.json").read_text())
    assert metadata["version"] == "2026-09-15"
    assert metadata["concept_count"] == 2
    assert metadata["sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()
    assert metadata["sha256"] != original_hash
    assert metadata["source_sha256"] == hashlib.sha256(source_data).hexdigest()
    assert len(calls) == 1
    assert len(list(tmp_path.iterdir())) == 2


def test_downloader_has_no_overwrite_option(tmp_path):
    with pytest.raises(TypeError, match="overwrite"):
        download.download_cell_ontology(output_dir=tmp_path, overwrite=True)


def test_failed_final_write_restores_original_metadata_and_data(tmp_path, monkeypatch):
    target = tmp_path / "cell_ontology_v2026-09-15.jsonl"
    sidecar = target.with_suffix(".metadata.json")
    target.write_text("old ontology\n")
    sidecar.write_text('{"old": true}\n')
    _response(monkeypatch, _graph())
    original_replace = download.os.replace

    def fail_data_commit(source, destination):
        if destination == target:
            raise OSError("disk full")
        return original_replace(source, destination)

    monkeypatch.setattr(download.os, "replace", fail_data_commit)
    with pytest.raises(download.OntologyDownloadError, match="Could not save"):
        download.download_cell_ontology(output_dir=tmp_path)
    assert target.read_text() == "old ontology\n"
    assert sidecar.read_text() == '{"old": true}\n'
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("options", [
    {"timeout": 0}, {"timeout": float("inf")}, {"timeout": True},
    {"max_retries": -1}, {"max_retries": 11}, {"max_retries": 1.5},
    {"include_obsolete": 1},
])
def test_invalid_options_fail_before_network_or_directory_creation(tmp_path, monkeypatch, options):
    calls, _ = _response(monkeypatch, _graph())
    destination = tmp_path / "not-created"
    with pytest.raises(download.OntologyDownloadError):
        download.download_cell_ontology(output_dir=destination, **options)
    assert not calls
    assert not destination.exists()


def test_output_directory_errors_are_concise(tmp_path, monkeypatch):
    calls, _ = _response(monkeypatch, _graph())
    destination = tmp_path / "file"
    destination.write_text("keep")
    with pytest.raises(download.OntologyDownloadError, match="Could not create"):
        download.download_cell_ontology(output_dir=destination)
    assert calls == []
    assert destination.read_text() == "keep"


def test_http_request_is_public_and_does_not_send_api_key(monkeypatch):
    data = json.dumps(_graph()).encode()
    monkeypatch.setenv("BIOPORTAL_API_KEY", "unrelated-secret")

    def open_public(request, *, timeout):
        assert request.full_url == download.OFFICIAL_CL_URL
        assert request.get_header("Authorization") is None
        assert "unrelated-secret" not in repr(request.headers)
        assert timeout == 5
        return io.BytesIO(data)

    monkeypatch.setattr(download, "_open_url", open_public)
    payload, raw = download._request_json(download.OFFICIAL_CL_URL, timeout=5, max_retries=0)
    assert payload == _graph()
    assert raw == data


def test_transient_http_error_is_retried_with_bounded_delay(monkeypatch):
    calls = []
    sleeps = []
    data = json.dumps(_graph()).encode()

    def open_retry(request, *, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "busy", {"Retry-After": "999"}, None)
        return io.BytesIO(data)

    monkeypatch.setattr(download, "_open_url", open_retry)
    monkeypatch.setattr(download.time, "sleep", sleeps.append)
    payload, _ = download._request_json(download.OFFICIAL_CL_URL, timeout=3, max_retries=1)
    assert payload == _graph()
    assert len(calls) == 2
    assert sleeps == [30]


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError(), IncompleteRead(b"{"), HTTPError("url", 404, "missing", {}, None)])
def test_network_failures_are_controlled(monkeypatch, failure):
    def fail(request, *, timeout):
        raise failure

    monkeypatch.setattr(download, "_open_url", fail)
    with pytest.raises(download.OntologyDownloadError):
        download._request_json(download.OFFICIAL_CL_URL, timeout=3, max_retries=0)


@pytest.mark.parametrize("data", [b"{", b"<html>Unavailable</html>", b"\xff"])
def test_invalid_json_is_reported_cleanly(monkeypatch, data):
    monkeypatch.setattr(download, "_open_url", lambda *args, **kwargs: io.BytesIO(data))
    with pytest.raises(download.OntologyDownloadError, match="invalid JSON"):
        download._request_json(download.OFFICIAL_CL_URL, timeout=3, max_retries=0)


def test_truncated_body_is_detected_even_when_prefix_is_valid_json(monkeypatch):
    class TruncatedResponse(io.BytesIO):
        headers = {"Content-Length": "500"}

    monkeypatch.setattr(download, "_open_url", lambda *args, **kwargs: TruncatedResponse(b"{}"))
    with pytest.raises(download.OntologyDownloadError, match="Could not download"):
        download._request_json(download.OFFICIAL_CL_URL, timeout=3, max_retries=0)


def test_oversized_download_is_reported_cleanly(monkeypatch):
    monkeypatch.setattr(download, "_MAX_SOURCE_BYTES", 5)
    monkeypatch.setattr(download, "_open_url", lambda *args, **kwargs: io.BytesIO(b"0123456789"))
    with pytest.raises(download.OntologyDownloadError, match="size limit"):
        download._request_json(download.OFFICIAL_CL_URL, timeout=3, max_retries=0)


@pytest.mark.parametrize("destination", ["http://example.com/cl.json", "https://user:pass@example.com/cl.json", "https://[bad/cl.json"])
def test_redirect_downgrades_and_malformed_urls_are_rejected(destination):
    handler = download._HTTPSRedirectHandler()
    with pytest.raises(download.OntologyDownloadError, match="redirect"):
        handler.redirect_request(Request(download.OFFICIAL_CL_URL), None, 302, "Found", {}, destination)
