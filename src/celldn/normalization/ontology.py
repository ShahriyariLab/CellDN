"""Cell Ontology aliases and version provenance for CellDN normalization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path

from .stemmer import plural_normalize_text

DEFAULT_ONTOLOGY_FILENAME = "cell_ontology_v2025-12-17.jsonl"
UNKNOWN_ONTOLOGY_VERSION = "unknown"


@dataclass(slots=True)
class TermEntry:
    """One searchable ontology alias."""

    name: str
    raw_name: str
    identifier: str
    preferred_label: str
    is_preferred: bool = False


@dataclass(slots=True)
class ConceptMetadata:
    """Metadata grouped by Cell Ontology concept identifier."""

    preferred_label: str
    synonyms: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    namespace: str = ""


def default_ontology_path() -> Path:
    """Return the packaged Cell Ontology JSONL resource path."""

    candidate = resources.files("celldn").joinpath(
        "resources", DEFAULT_ONTOLOGY_FILENAME
    )
    return Path(str(candidate))


def _version_token(value: object) -> str | None:
    """Read a release identifier without mistaking a retrieval date for one."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    dated = re.fullmatch(r"v?(\d{4}-\d{2}-\d{2})", value)
    if dated is None:
        dated = re.fullmatch(
            r"https?://purl\.obolibrary\.org/obo/cl/releases/(\d{4}-\d{2}-\d{2})/[^/]+",
            value,
        )
    if dated is not None:
        try:
            return date.fromisoformat(dated.group(1)).isoformat()
        except ValueError:
            return None
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,79}", value):
        return value[1:] if re.match(r"v\d", value) else value
    return None


def _version_for_snapshot(path: Path, payload: bytes) -> str:
    """Accept companion metadata only when it describes these exact bytes.

    The resource filename is deliberately not a provenance source: a user can
    rename or edit a custom JSONL file without changing its name. Invalid or
    missing provenance does not prevent normalization; it is reported honestly
    as ``unknown``.
    """

    metadata_path = path.with_suffix(".metadata.json")
    try:
        metadata = json.loads(metadata_path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        return UNKNOWN_ONTOLOGY_VERSION
    if not isinstance(metadata, dict) or metadata.get("ontology") != "CL":
        return UNKNOWN_ONTOLOGY_VERSION
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        return UNKNOWN_ONTOLOGY_VERSION
    if checksum.lower() != hashlib.sha256(payload).hexdigest():
        return UNKNOWN_ONTOLOGY_VERSION

    # v2 official-release exports have top-level version metadata. Support the
    # existing v1 BioPortal export sidecars for users who already downloaded one.
    submission = metadata.get("submission")
    submission = submission if isinstance(submission, dict) else {}
    candidates = [
        _version_token(metadata.get("version")),
        _version_token(metadata.get("version_iri")),
        _version_token(submission.get("version")),
        _version_token(submission.get("versionIRI")),
    ]
    versions = {candidate for candidate in candidates if candidate is not None}
    if len(versions) == 1:
        return versions.pop()
    return UNKNOWN_ONTOLOGY_VERSION


def _read_ontology_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology JSONL file does not exist: {path}")
    return path.read_bytes()


def get_ontology_version(ontology_path: str | Path) -> str:
    """Return the resource's verified version, or ``unknown``.

    A sibling ``.metadata.json`` must identify CL and carry a SHA-256 matching
    the JSONL file. A dated filename alone is never treated as proof of the
    ontology version. The packaged resource includes this metadata. An unreadable
    ontology file raises the same I/O error as the ontology loader.
    """

    path = Path(ontology_path)
    return _version_for_snapshot(path, _read_ontology_bytes(path))


def _as_namespace_set(namespace_filter: str | Iterable[str] | None) -> set[str] | None:
    if namespace_filter is None:
        return None
    if isinstance(namespace_filter, str):
        return {namespace_filter}
    return {str(item) for item in namespace_filter}


def _parse_cell_ontology_terms(
    path: Path,
    payload: bytes,
    *,
    namespace_filter: str | Iterable[str] | None,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata]]:
    namespaces = _as_namespace_set(namespace_filter)
    term_entries: list[TermEntry] = []
    concept_metadata: dict[str, ConceptMetadata] = {}

    for line_no, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Bad JSON on line {line_no} in {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object on line {line_no} in {path}")

        identifier = record.get("norm_concept_id")
        preferred_label = record.get("norm_preferred_label")
        synonyms = record.get("synonyms", []) or []
        namespace = record.get("namespace", "") or ""

        if not identifier or not preferred_label:
            continue
        if namespaces is not None and namespace not in namespaces:
            continue
        if not isinstance(synonyms, list):
            raise ValueError(
                f"Expected 'synonyms' to be a list on line {line_no} in {path}"
            )

        if identifier not in concept_metadata:
            concept_metadata[identifier] = ConceptMetadata(
                preferred_label=str(preferred_label),
                namespace=str(namespace),
            )

        meta = concept_metadata[identifier]
        meta.names.add(str(preferred_label))
        term_entries.append(
            TermEntry(
                name=plural_normalize_text(preferred_label),
                raw_name=str(preferred_label),
                identifier=str(identifier),
                preferred_label=str(preferred_label),
                is_preferred=True,
            )
        )

        for synonym in synonyms:
            if not synonym:
                continue
            synonym = str(synonym)
            meta.synonyms.add(synonym)
            meta.names.add(synonym)
            term_entries.append(
                TermEntry(
                    name=plural_normalize_text(synonym),
                    raw_name=synonym,
                    identifier=str(identifier),
                    preferred_label=str(preferred_label),
                    is_preferred=False,
                )
            )

    return term_entries, concept_metadata


def load_cell_ontology_terms_with_version(
    ontology_path: str | Path,
    *,
    namespace_filter: str | Iterable[str] | None = None,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata], str]:
    """Load aliases and their version from the same JSONL byte snapshot.

    Linkers retain this version with the loaded aliases; replacing the file
    afterward cannot retroactively relabel an existing in-memory inventory.
    """

    path = Path(ontology_path)
    payload = _read_ontology_bytes(path)
    terms, concepts = _parse_cell_ontology_terms(
        path, payload, namespace_filter=namespace_filter
    )
    return terms, concepts, _version_for_snapshot(path, payload)


def load_cell_ontology_terms(
    ontology_path: str | Path,
    *,
    namespace_filter: str | Iterable[str] | None = None,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata]]:
    """Load ontology aliases, preserving the existing two-value return API."""

    path = Path(ontology_path)
    return _parse_cell_ontology_terms(
        path, _read_ontology_bytes(path), namespace_filter=namespace_filter
    )


# Backward-compatible short name used by some scripts.
load_terms = load_cell_ontology_terms


__all__ = [
    "DEFAULT_ONTOLOGY_FILENAME",
    "UNKNOWN_ONTOLOGY_VERSION",
    "TermEntry",
    "ConceptMetadata",
    "default_ontology_path",
    "get_ontology_version",
    "load_cell_ontology_terms",
    "load_cell_ontology_terms_with_version",
    "load_terms",
]
