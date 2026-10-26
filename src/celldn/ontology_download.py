"""Prepare CellDN's JSONL resource from the public official Cell Ontology release.

The CL maintainers publish OBOGraph JSON at the official GitHub release URL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import date, datetime, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

OFFICIAL_CL_URL = "https://github.com/obophenotype/cell-ontology/releases/latest/download/cl.json"
_CL_ID = re.compile(r"(?:https?://purl\.obolibrary\.org/obo/)?CL[:_](\d{7})\Z")
_CL_GRAPH_ID = re.compile(r"https?://purl\.obolibrary\.org/obo/cl\.(?:json|owl|obo)\Z")
_CL_VERSION_IRI = re.compile(
    r"https?://purl\.obolibrary\.org/obo/cl/releases/([A-Za-z0-9][A-Za-z0-9._-]{0,79})/cl\.(?:json|owl|obo)\Z"
)
_VERSION_INFO = "http://www.w3.org/2002/07/owl#versionInfo"
_TRANSIENT_HTTP = {429, 500, 502, 503, 504}
_MAX_SOURCE_BYTES = 128 * 1024 * 1024


class OntologyDownloadError(RuntimeError):
    """The official Cell Ontology export could not be completed safely."""


def _valid_text(value: Any, *, description: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise OntologyDownloadError(f"The official CL release contains an invalid {description}.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise OntologyDownloadError("The official CL release contains invalid Unicode text.") from None
    return value


def _cl_identifier(node: Any) -> str | None:
    if not isinstance(node, dict):
        raise OntologyDownloadError("The official CL release contains a node that is not a JSON object.")
    raw_id = _valid_text(node.get("id"), description="node identifier")
    match = _CL_ID.fullmatch(raw_id)
    if match:
        return f"CL:{match.group(1)}"
    if raw_id.startswith(
        ("CL:", "CL_", "http://purl.obolibrary.org/obo/CL_", "https://purl.obolibrary.org/obo/CL_")
    ):
        raise OntologyDownloadError("The official CL release contains a malformed CL identifier.")
    return None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("meta", {})
    if not isinstance(metadata, dict):
        raise OntologyDownloadError("The official CL release contains invalid class metadata.")
    return metadata


def _deprecated(node: dict[str, Any]) -> bool:
    value = _node_metadata(node).get("deprecated", False)
    if not isinstance(value, bool):
        raise OntologyDownloadError("The official CL release contains an invalid deprecated flag.")
    return value


def convert_obograph_node(
    node: dict[str, Any], *, include_obsolete: bool = False
) -> dict[str, Any] | None:
    """Convert one OBOGraph CL class to CellDN's existing five-field JSONL schema.

    Imported non-CL nodes, non-class nodes, and (by default) explicitly deprecated
    classes are omitted. All synonym scopes in the official node are retained;
    spelling and case are preserved, with empty strings and exact duplicates
    removed. The input object is not changed.
    """

    identifier = _cl_identifier(node)
    if identifier is None or node.get("type") != "CLASS":
        return None
    if _deprecated(node) and not include_obsolete:
        return None
    label = _valid_text(node.get("lbl"), description=f"preferred label for {identifier}")
    synonyms = _node_metadata(node).get("synonyms", [])
    if not isinstance(synonyms, list):
        raise OntologyDownloadError(f"The official CL release contains invalid synonyms for {identifier}.")
    names: set[str] = set()
    for synonym in synonyms:
        if not isinstance(synonym, dict):
            raise OntologyDownloadError(f"The official CL release contains an invalid synonym for {identifier}.")
        name = _valid_text(synonym.get("val"), description=f"synonym for {identifier}", allow_empty=True)
        if name.strip():
            names.add(name)
    return {
        "norm_concept_id": identifier,
        "norm_concept_id_underscore": identifier.replace(":", "_"),
        "norm_preferred_label": label,
        "synonyms": sorted(names),
        "namespace": "CL",
    }


def _version_token(value: Any) -> str:
    value = _valid_text(value, description="release version").strip()
    if value.startswith("v") and len(value) > 1 and value[1].isdigit():
        value = value[1:]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise OntologyDownloadError("The official CL release contains an unsupported release version.")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            date.fromisoformat(value)
        except ValueError:
            raise OntologyDownloadError("The official CL release contains an invalid release date.") from None
    return value


def _cl_graph(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("graphs"), list):
        raise OntologyDownloadError("The official CL download is not an OBOGraph JSON document.")
    graphs = payload["graphs"]
    if any(not isinstance(graph, dict) for graph in graphs):
        raise OntologyDownloadError("The official CL download contains an invalid ontology graph.")
    matches = [
        graph for graph in graphs
        if isinstance(graph.get("id"), str) and _CL_GRAPH_ID.fullmatch(graph["id"])
    ]
    if len(matches) != 1:
        raise OntologyDownloadError("The download must contain exactly one main Cell Ontology graph.")
    graph = matches[0]
    if not isinstance(graph.get("nodes"), list):
        raise OntologyDownloadError("The official CL graph has no valid node list.")
    return graph


def _release_metadata(graph: dict[str, Any]) -> tuple[str, str]:
    metadata = graph.get("meta")
    if not isinstance(metadata, dict):
        raise OntologyDownloadError("The official CL graph has no release metadata.")
    version_iri = _valid_text(metadata.get("version"), description="CL version IRI")
    match = _CL_VERSION_IRI.fullmatch(version_iri)
    if match is None:
        raise OntologyDownloadError("The official CL graph has no supported CL release version IRI.")
    version = _version_token(match.group(1))
    annotations = metadata.get("basicPropertyValues", [])
    if not isinstance(annotations, list) or any(not isinstance(item, dict) for item in annotations):
        raise OntologyDownloadError("The official CL graph has invalid release annotations.")
    for item in annotations:
        if item.get("pred") == _VERSION_INFO and _version_token(item.get("val")) != version:
            raise OntologyDownloadError("The official CL release contains conflicting version metadata.")
    return version, version_iri


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    """Follow the official release's HTTPS redirects without a downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        try:
            parsed = urlsplit(newurl)
            valid = (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.port in (None, 443)
                and parsed.username is None
                and parsed.password is None
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise OntologyDownloadError("The official CL download returned an invalid HTTPS redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(request: Request, *, timeout: float):
    return build_opener(_HTTPSRedirectHandler()).open(request, timeout=timeout)


def _request_json(url: str, *, timeout: float, max_retries: int) -> tuple[Any, bytes]:
    """Download a bounded JSON document with bounded retries and concise errors."""

    for attempt in range(max_retries + 1):
        request = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "CellDN official Cell Ontology downloader",
        })
        retry_after = None
        try:
            with _open_url(request, timeout=timeout) as response:
                data = response.read(_MAX_SOURCE_BYTES + 1)
                headers = getattr(response, "headers", None)
                content_length = headers.get("Content-Length") if headers else None
            if len(data) > _MAX_SOURCE_BYTES:
                raise OntologyDownloadError("The official CL JSON download exceeds the supported size limit.")
            if content_length is not None:
                try:
                    expected_size = int(content_length)
                except (TypeError, ValueError):
                    expected_size = -1
                if expected_size < 0 or expected_size != len(data):
                    raise HTTPException("Incomplete CL download")
            try:
                return json.loads(data), data
            except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                raise OntologyDownloadError("The official CL download contains invalid JSON.") from None
        except HTTPError as exc:
            status = exc.code
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            if status not in _TRANSIENT_HTTP or attempt == max_retries:
                raise OntologyDownloadError(
                    f"Official CL download failed (HTTP {status}). Try again later."
                ) from None
        except (URLError, OSError, HTTPException):
            if attempt == max_retries:
                raise OntologyDownloadError(
                    "Could not download the official CL release. Check your connection and try again."
                ) from None
        delay = min(2**attempt, 8)
        if retry_after is not None:
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds):
                    delay = min(max(seconds, 0), 30)
            except (TypeError, ValueError):
                pass
        time.sleep(delay)
    raise AssertionError("unreachable")


def _write_temp(path: Path, payload: bytes) -> Path:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return temporary


def _write_export(path: Path, data: bytes, provenance: dict[str, Any]) -> None:
    """Stage both files before publishing JSONL; restore metadata on write failure."""

    metadata_path = path.with_suffix(".metadata.json")
    staged: list[Path] = []
    metadata_committed = False
    original_metadata: bytes | None = None
    try:
        if metadata_path.exists():
            original_metadata = metadata_path.read_bytes()
        staged.append(_write_temp(path, data))
        metadata_data = (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        staged.append(_write_temp(metadata_path, metadata_data))
        os.replace(staged[1], metadata_path)
        metadata_committed = True
        os.replace(staged[0], path)
    except OSError:
        if metadata_committed:
            try:
                if original_metadata is None:
                    metadata_path.unlink(missing_ok=True)
                else:
                    restored = _write_temp(metadata_path, original_metadata)
                    try:
                        os.replace(restored, metadata_path)
                    finally:
                        restored.unlink(missing_ok=True)
            except OSError:
                # Readers verify the JSONL hash before trusting sidecar metadata.
                pass
        raise OntologyDownloadError(
            "Could not save the ontology export. Check output permissions and available disk space."
        ) from None
    finally:
        for temporary in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def download_cell_ontology(
    *,
    output_dir: str | Path = "resources",
    include_obsolete: bool = False,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> Path:
    """Download the latest official CL JSON and save a compatible JSONL resource.

    No API key is needed. The default output directory is ``resources`` under
    the working directory and is created when needed. The returned absolute
    JSONL path can be passed as ``CellDNPipeline(ontology_path=...)``.

    Only CL class nodes are included. Deprecated classes are excluded unless
    ``include_obsolete=True``. Preferred labels and all listed synonym scopes
    are retained. A companion ``.metadata.json`` records the embedded official
    release version, source, SHA-256 hashes, filtering, and counts. The version
    is read from the downloaded ontology, never invented from today's date.

    Downloading the same release again replaces its JSONL and metadata files.
    The bundled resource and a pipeline's selected ``ontology_path`` are never changed.
    """

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise OntologyDownloadError("timeout must be a positive number of seconds.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
        raise OntologyDownloadError("max_retries must be an integer from 0 to 10.")
    if not isinstance(include_obsolete, bool):
        raise OntologyDownloadError("include_obsolete must be a boolean value.")
    try:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, ValueError):
        raise OntologyDownloadError("Could not create the requested ontology output directory.") from None

    payload, source_data = _request_json(OFFICIAL_CL_URL, timeout=timeout, max_retries=max_retries)
    graph = _cl_graph(payload)
    version, version_iri = _release_metadata(graph)
    output = directory / f"cell_ontology_v{version}.jsonl"

    records: dict[str, dict[str, Any]] = {}
    seen_deprecated: dict[str, bool] = {}
    counts = {
        "source_nodes": len(graph["nodes"]),
        "skipped_non_cl_nodes": 0,
        "skipped_non_class_nodes": 0,
        "skipped_obsolete_classes": 0,
        "duplicate_class_records": 0,
    }
    for node in graph["nodes"]:
        identifier = _cl_identifier(node)
        if identifier is None:
            counts["skipped_non_cl_nodes"] += 1
            continue
        if node.get("type") != "CLASS":
            counts["skipped_non_class_nodes"] += 1
            continue
        deprecated = _deprecated(node)
        if identifier in seen_deprecated and seen_deprecated[identifier] != deprecated:
            raise OntologyDownloadError(f"The official CL release contains conflicting deprecated flags for {identifier}.")
        seen_deprecated[identifier] = deprecated
        if deprecated and not include_obsolete:
            counts["skipped_obsolete_classes"] += 1
            continue
        converted = convert_obograph_node(node, include_obsolete=include_obsolete)
        assert converted is not None
        if identifier in records:
            previous = records[identifier]
            if previous["norm_preferred_label"] != converted["norm_preferred_label"]:
                raise OntologyDownloadError(f"The official CL release contains conflicting labels for {identifier}.")
            previous["synonyms"] = sorted(set(previous["synonyms"]) | set(converted["synonyms"]))
            counts["duplicate_class_records"] += 1
        else:
            records[identifier] = converted
    if not records:
        raise OntologyDownloadError("The official CL release contains no usable CL concepts; no export was written.")

    ordered_records = [records[identifier] for identifier in sorted(records)]
    data = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered_records).encode("utf-8")
    provenance = {
        "schema": "celldn.cell_ontology_export.v2",
        "ontology": "CL",
        "version": version,
        "version_iri": version_iri,
        "source": "Official Cell Ontology GitHub release",
        "source_url": OFFICIAL_CL_URL,
        "source_sha256": hashlib.sha256(source_data).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "jsonl_file": output.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "concept_count": len(ordered_records),
        "synonym_count": sum(len(record["synonyms"]) for record in ordered_records),
        "counts": counts,
        "filtering": {
            "namespace": "CL",
            "node_type": "CLASS",
            "include_obsolete": include_obsolete,
            "synonym_scopes": "all scopes supplied by the official release",
            "synonyms": "original strings; sorted and deduplicated within each concept",
        },
    }
    _write_export(output, data, provenance)
    return output


__all__ = ["OFFICIAL_CL_URL", "OntologyDownloadError", "convert_obograph_node", "download_cell_ontology"]
