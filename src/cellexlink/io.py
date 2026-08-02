"""BioC and JSON I/O utilities for CellExLink.

It can be used by tests, converters, fetchers, and CLI
commands without loading the CellExLink NER/NEN checkpoints.

BioC XML, BioC JSON, and generic JSON inputs are all parsed into the same
in-memory BioC collection and passage-record objects. Downstream code can then
operate on those shared records instead of branching on the original file
format.

"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree as ET

PathLike = str | os.PathLike[str]
_GENERATED_ANNOTATION_KEY = "__cellexlink_generated_annotation"

BIOC_XML_FORMATS = {"xml", "bioc-xml", "bioc_xml"}
BIOC_JSON_FORMATS = {"bioc-json", "bioc_json"}
GENERIC_JSON_FORMATS = {"json", "generic-json", "generic_json"}
SUPPORTED_DOCUMENT_FORMATS = BIOC_XML_FORMATS | BIOC_JSON_FORMATS | GENERIC_JSON_FORMATS | {"auto"}


# ---------------------------------------------------------------------------
# Public lightweight prediction/record dataclasses retained from CellExLink 0.1
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PredictedEntity:
    """A predicted entity with absolute BioC offsets."""

    document_id: str = ""
    passage_id: int = 0
    start: int = 0
    end: int = 0
    label: str = "cell_type"
    text: str = ""
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))


@dataclass(slots=True)
class EntitySpan:
    """An entity span relative to a passage unless converted to absolute."""

    start: int
    end: int
    label: str = "cell_type"
    text: str = ""
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def to_absolute(
        self,
        passage_offset: int,
        *,
        document_id: str = "",
        passage_id: int = 0,
    ) -> PredictedEntity:
        """Return a :class:`PredictedEntity` using absolute BioC offsets."""

        offset = int(passage_offset)
        return PredictedEntity(
            document_id=document_id,
            passage_id=passage_id,
            start=offset + int(self.start),
            end=offset + int(self.end),
            label=self.label,
            text=self.text,
            score=self.score,
            infons=dict(self.infons),
        )


@dataclass(slots=True)
class PassageRecord:
    """A BioC passage and its optional entity annotations."""

    record_id: int
    document_id: str
    passage_id: int
    passage_offset: int
    text: str
    entities: list[EntitySpan] = field(default_factory=list)
    infons: dict[str, str] = field(default_factory=dict)
    source_path: str = ""


# ---------------------------------------------------------------------------
# In-memory BioC representation for XML/JSON conversion
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BioCLocation:
    offset: int
    length: int

    def to_json(self) -> dict[str, int]:
        return {"offset": int(self.offset), "length": int(self.length)}


@dataclass(slots=True)
class BioCAnnotation:
    id: str = ""
    infons: dict[str, str] = field(default_factory=dict)
    locations: list[BioCLocation] = field(default_factory=list)
    text: str = ""

    def primary_location(self) -> BioCLocation | None:
        return self.locations[0] if self.locations else None

    def duplicate_key(self) -> tuple[Any, ...]:
        location = self.primary_location()
        return (
            location.offset if location else None,
            location.length if location else None,
            self.text,
            self.infons.get("type") or self.infons.get("label") or self.infons.get("entity_type"),
            self.infons.get("identifier") or self.infons.get("cl_id") or self.infons.get("MESH") or "",
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "infons": dict(self.infons),
            "locations": [location.to_json() for location in self.locations],
            "text": self.text,
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [], "")}


@dataclass(slots=True)
class BioCPassage:
    infons: dict[str, str] = field(default_factory=dict)
    offset: int = 0
    text: str = ""
    annotations: list[BioCAnnotation] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "infons": dict(self.infons),
            "offset": int(self.offset),
            "text": self.text,
            "annotations": [annotation.to_json() for annotation in self.annotations],
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


@dataclass(slots=True)
class BioCDocument:
    id: str = ""
    infons: dict[str, str] = field(default_factory=dict)
    passages: list[BioCPassage] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "infons": dict(self.infons),
            "passages": [passage.to_json() for passage in self.passages],
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [], "")}


@dataclass(slots=True)
class BioCCollection:
    source: str = "CellExLink"
    date: str = ""
    key: str = "cell-type-extraction"
    infons: dict[str, str] = field(default_factory=dict)
    documents: list[BioCDocument] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "date": self.date,
            "key": self.key,
            "infons": dict(self.infons),
            "documents": [document.to_json() for document in self.documents],
        }
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def canonical_document_format(fmt: str | None, *, path: PathLike | None = None) -> str:
    """Return ``bioc-xml``, ``bioc-json``, ``json``, or auto-detected format."""

    fmt_norm = (fmt or "auto").strip().lower()
    if fmt_norm in {"auto", ""}:
        if path is None:
            raise ValueError("format='auto' requires a file path or explicit format")
        return detect_document_format(path)
    if fmt_norm in BIOC_XML_FORMATS:
        return "bioc-xml"
    if fmt_norm in BIOC_JSON_FORMATS:
        return "bioc-json"
    if fmt_norm in GENERIC_JSON_FORMATS:
        return "json"
    raise ValueError(
        f"Unsupported document format: {fmt!r}. "
        "Use one of: auto, bioc-xml, bioc-json, json."
    )


def canonical_bioc_format(fmt: str | None, *, path: PathLike | None = None) -> str:
    """Return ``bioc-xml`` or ``bioc-json`` for output formats."""

    resolved = (
        canonical_document_format(fmt, path=path)
        if (fmt or "auto") == "auto"
        else canonical_document_format(fmt)
    )
    if resolved == "json":
        raise ValueError("JSON is supported as input; output must be bioc-xml or bioc-json.")
    return resolved


def detect_document_format(path: PathLike) -> str:
    """Detect BioC XML, BioC JSON, or generic JSON."""

    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".bioc.json"):
        return "bioc-json"
    if suffixes.endswith(".json"):
        return _detect_json_file_format(p)
    if suffixes.endswith((".bioc.xml", ".xml")):
        return "bioc-xml"

    with p.open("r", encoding="utf-8") as handle:
        sample = handle.read(4096).lstrip("\ufeff\n\r\t ")
    if not sample:
        raise ValueError(f"Cannot detect format of empty file: {p}")
    if sample.startswith("<"):
        return "bioc-xml"
    if sample.startswith("{") or sample.startswith("["):
        return _detect_json_file_format(p)
    raise ValueError(f"Could not auto-detect input format for {p}. Pass --input-format explicitly.")


def _detect_json_file_format(path: Path) -> str:
    payload = _load_json_file(path)
    return "bioc-json" if _looks_like_bioc_json_obj(payload) else "json"


def _looks_like_bioc_json_obj(obj: Any) -> bool:
    obj = _unwrap_bioc_json_variant(obj)
    if isinstance(obj, list):
        return all(isinstance(item, Mapping) and ("passages" in item or "passage" in item) for item in obj)
    if not isinstance(obj, Mapping):
        return False
    if "collection" in obj and isinstance(obj["collection"], Mapping):
        obj = obj["collection"]
    return any(key in obj for key in ("documents", "document", "passages", "passage"))


def output_format_from_path(path: PathLike, *, default: str = "bioc-xml") -> str:
    """Infer BioC output format from extension, falling back to ``default``."""

    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith((".json", ".bioc.json")):
        return "bioc-json"
    if suffixes.endswith((".xml", ".bioc.xml")):
        return "bioc-xml"
    return canonical_bioc_format(default)


# ---------------------------------------------------------------------------
# BioC XML/JSON readers and writers
# ---------------------------------------------------------------------------
def read_bioc_collection(path: PathLike, *, input_format: str = "auto") -> BioCCollection:
    """Read BioC XML, BioC JSON, or generic JSON into a BioC collection."""

    fmt = canonical_document_format(input_format, path=path)
    p = Path(path)
    if fmt == "bioc-xml":
        return read_bioc_xml(p)
    if fmt == "bioc-json":
        return read_bioc_json(p)
    if fmt == "json":
        return read_generic_json(p)
    raise AssertionError(fmt)


def read_bioc_collection_from_string(text: str, *, input_format: str) -> BioCCollection:
    """Read a BioC collection from a string payload."""

    fmt = canonical_document_format(input_format)
    if fmt == "bioc-xml":
        root = ET.fromstring(text.encode("utf-8"))
        return _collection_from_xml_root(root)
    if fmt == "bioc-json":
        return _collection_from_json_obj(json.loads(text))
    if fmt == "json":
        return _collection_from_generic_json_obj(json.loads(text))
    raise AssertionError(fmt)


def write_bioc_collection(
    collection: BioCCollection,
    output_path: PathLike,
    *,
    output_format: str = "auto",
) -> Path:
    """Write a BioC collection as XML or JSON."""

    output = Path(output_path)
    fmt = (
        output_format_from_path(output)
        if output_format in (None, "", "auto")
        else canonical_bioc_format(output_format)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "bioc-xml":
        return write_bioc_xml(collection, output)
    if fmt == "bioc-json":
        return write_bioc_json(collection, output)
    raise AssertionError(fmt)


def merge_bioc_files(
    input_paths: Iterable[PathLike],
    output_path: PathLike,
    *,
    input_format: str = "auto",
    output_format: str = "auto",
    renumber_generated_annotations: bool = False,
) -> Path:
    """Merge BioC files while keeping only one input chunk in memory.

    Documents are written in input-file order. The first collection header is
    preserved, stable CellExLink metadata is retained, normalization mention
    counts are summed, and timing fields are omitted. Temporary generated
    annotation markers can be finalized while the documents are streamed.
    """

    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("At least one BioC input file is required.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")

    first_header = read_bioc_collection(paths[0], input_format=input_format)
    header = BioCCollection(
        source=first_header.source,
        date=first_header.date,
        key=first_header.key,
    )
    del first_header
    unique_mentions = 0
    saw_unique_mentions = False

    for path in paths:
        collection = read_bioc_collection(path, input_format=input_format)
        if not header.source and collection.source:
            header.source = collection.source
        if not header.date and collection.date:
            header.date = collection.date
        if not header.key and collection.key:
            header.key = collection.key
        for key, value in collection.infons.items():
            key_text = str(key)
            key_folded = key_text.casefold()
            if "elapsed" in key_folded or "runtime" in key_folded:
                continue
            if key_text == "CellExLink_normalization_unique_mentions":
                try:
                    unique_mentions += int(value)
                    saw_unique_mentions = True
                except (TypeError, ValueError):
                    pass
                continue
            header.infons.setdefault(key_text, str(value))

    if saw_unique_mentions:
        header.infons["CellExLink_normalization_unique_mentions"] = str(
            unique_mentions
        )

    annotation_index = 1

    def _documents() -> Iterator[BioCDocument]:
        nonlocal annotation_index
        for path in paths:
            collection = read_bioc_collection(path, input_format=input_format)
            for document in collection.documents:
                if renumber_generated_annotations:
                    annotation_index = _finalize_generated_annotation_ids(
                        document,
                        start=annotation_index,
                    )
                yield document

    return write_bioc_document_stream(
        header,
        _documents(),
        output_path,
        output_format=output_format,
    )


def _finalize_generated_annotation_ids(
    document: BioCDocument,
    *,
    start: int,
) -> int:
    """Renumber marked annotations and return the next available index."""

    annotation_index = start
    for passage in document.passages:
        for annotation in passage.annotations:
            if annotation.infons.pop(_GENERATED_ANNOTATION_KEY, None) is None:
                continue
            annotation.id = f"T{annotation_index}"
            annotation_index += 1
    return annotation_index


def write_bioc_document_stream(
    header: BioCCollection,
    documents: Iterable[BioCDocument],
    output_path: PathLike,
    *,
    output_format: str = "auto",
) -> Path:
    """Write BioC documents incrementally using collection-level metadata."""

    output = Path(output_path)
    fmt = (
        output_format_from_path(output)
        if output_format in (None, "", "auto")
        else canonical_bioc_format(output_format)
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if fmt == "bioc-xml":
            _write_bioc_xml_stream(header, documents, temporary_path)
        elif fmt == "bioc-json":
            _write_bioc_json_stream(header, documents, temporary_path)
        else:  # pragma: no cover - canonical_bioc_format guards this.
            raise AssertionError(fmt)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def _write_bioc_xml_stream(
    header: BioCCollection,
    documents: Iterable[BioCDocument],
    output: Path,
) -> None:
    """Write a BioC XML collection one document at a time."""

    root = ET.Element("collection")
    for key, value in header.infons.items():
        _add_infon(root, key, value)
    ET.SubElement(root, "source").text = header.source
    ET.SubElement(root, "date").text = header.date
    ET.SubElement(root, "key").text = header.key
    serialized_header = ET.tostring(
        root,
        encoding="unicode",
        short_empty_elements=False,
    )
    prefix, separator, _ = serialized_header.rpartition("</collection>")
    if not separator:  # pragma: no cover - root is always a collection.
        raise RuntimeError("Could not serialize BioC collection header")

    with output.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n')
        handle.write(prefix)
        for document in documents:
            element = _document_to_xml_element(document)
            ET.indent(element, space="  ", level=1)
            handle.write("\n")
            handle.write(ET.tostring(element, encoding="unicode"))
        handle.write("\n</collection>\n")


def _document_to_xml_element(document: BioCDocument) -> ET.Element:
    """Convert one in-memory document to a BioC XML element."""

    document_element = ET.Element("document")
    for key, value in document.infons.items():
        _add_infon(document_element, key, value)
    ET.SubElement(document_element, "id").text = document.id
    for passage in document.passages:
        passage_element = ET.SubElement(document_element, "passage")
        for key, value in passage.infons.items():
            _add_infon(passage_element, key, value)
        ET.SubElement(passage_element, "offset").text = str(int(passage.offset))
        ET.SubElement(passage_element, "text").text = passage.text
        for index, annotation in enumerate(passage.annotations):
            annotation_id = annotation.id or f"A{index}"
            annotation_element = ET.SubElement(
                passage_element,
                "annotation",
                {"id": str(annotation_id)},
            )
            for key, value in annotation.infons.items():
                _add_infon(annotation_element, key, value)
            for location in annotation.locations:
                ET.SubElement(
                    annotation_element,
                    "location",
                    {
                        "offset": str(int(location.offset)),
                        "length": str(int(location.length)),
                    },
                )
            ET.SubElement(annotation_element, "text").text = annotation.text
    return document_element


def _write_bioc_json_stream(
    header: BioCCollection,
    documents: Iterable[BioCDocument],
    output: Path,
) -> None:
    """Write a BioC JSON collection one document at a time."""

    header_payload = header.to_json()
    header_payload.pop("documents", None)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        items = list(header_payload.items())
        for index, (key, value) in enumerate(items):
            handle.write("  ")
            handle.write(json.dumps(str(key), ensure_ascii=False))
            handle.write(": ")
            handle.write(json.dumps(value, ensure_ascii=False, indent=2))
            handle.write(",\n")
        handle.write('  "documents": [')
        first = True
        for document in documents:
            if first:
                handle.write("\n")
                first = False
            else:
                handle.write(",\n")
            payload = json.dumps(
                document.to_json(),
                ensure_ascii=False,
                indent=2,
            )
            handle.write("    ")
            handle.write(payload.replace("\n", "\n    "))
        if not first:
            handle.write("\n  ")
        handle.write("]\n}\n")


def read_bioc_xml(path: PathLike) -> BioCCollection:
    return _collection_from_xml_root(ET.parse(path).getroot())


def _collection_from_xml_root(root: ET.Element) -> BioCCollection:
    source = _text_of(root, "source", "")
    date = _text_of(root, "date", "")
    key = _text_of(root, "key", "")
    collection = BioCCollection(source=source, date=date, key=key, infons=read_infons(root))

    for document_element in root.findall("document"):
        document = BioCDocument(
            id=_text_of(document_element, "id", ""),
            infons=read_infons(document_element),
        )
        for passage_element in document_element.findall("passage"):
            passage = BioCPassage(
                infons=read_infons(passage_element),
                offset=_int_text_of(passage_element, "offset", 0),
                text=_text_of(passage_element, "text", ""),
            )
            for annotation_element in passage_element.findall("annotation"):
                locations = [
                    BioCLocation(
                        offset=_safe_int(location.attrib.get("offset"), default=0) or 0,
                        length=_safe_int(location.attrib.get("length"), default=0) or 0,
                    )
                    for location in annotation_element.findall("location")
                ]
                passage.annotations.append(
                    BioCAnnotation(
                        id=annotation_element.attrib.get("id", ""),
                        infons=read_infons(annotation_element),
                        locations=locations,
                        text=_text_of(annotation_element, "text", ""),
                    )
                )
            document.passages.append(passage)
        collection.documents.append(document)
    return collection


def write_bioc_xml(collection: BioCCollection, output_xml: PathLike) -> Path:
    output = Path(output_xml)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("collection")
    for key, value in collection.infons.items():
        _add_infon(root, key, value)
    ET.SubElement(root, "source").text = collection.source
    ET.SubElement(root, "date").text = collection.date
    ET.SubElement(root, "key").text = collection.key

    for document in collection.documents:
        document_element = ET.SubElement(root, "document")
        for key, value in document.infons.items():
            _add_infon(document_element, key, value)
        ET.SubElement(document_element, "id").text = document.id
        for passage in document.passages:
            passage_element = ET.SubElement(document_element, "passage")
            for key, value in passage.infons.items():
                _add_infon(passage_element, key, value)
            ET.SubElement(passage_element, "offset").text = str(int(passage.offset))
            ET.SubElement(passage_element, "text").text = passage.text
            for index, annotation in enumerate(passage.annotations):
                annotation_id = annotation.id or f"A{index}"
                annotation_element = ET.SubElement(passage_element, "annotation", {"id": str(annotation_id)})
                for key, value in annotation.infons.items():
                    _add_infon(annotation_element, key, value)
                for location in annotation.locations:
                    ET.SubElement(
                        annotation_element,
                        "location",
                        {"offset": str(int(location.offset)), "length": str(int(location.length))},
                    )
                ET.SubElement(annotation_element, "text").text = annotation.text

    _indent_tree(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def read_bioc_json(path: PathLike) -> BioCCollection:
    return _collection_from_json_obj(_load_json_file(path))


def _collection_from_json_obj(obj: Any) -> BioCCollection:
    """Parse common BioC JSON variants."""

    obj = _unwrap_bioc_json_variant(obj)
    if isinstance(obj, list):
        obj = {"documents": obj}
    if not isinstance(obj, Mapping):
        raise ValueError("BioC JSON payload must be an object or a list of documents.")

    if "collection" in obj and isinstance(obj["collection"], Mapping):
        obj = obj["collection"]
    elif "passages" in obj or "passage" in obj:
        obj = {"documents": [obj]}

    collection = BioCCollection(
        source=str(obj.get("source") or ""),
        date=str(obj.get("date") or ""),
        key=str(obj.get("key") or ""),
        infons=_coerce_infons(obj.get("infons") or obj.get("infon") or {}),
    )

    raw_documents = _mapping_list(obj.get("documents") or obj.get("document") or [])
    for doc_obj in raw_documents:
        document = BioCDocument(
            id=str(doc_obj.get("id") or doc_obj.get("pmid") or doc_obj.get("pmcid") or ""),
            infons=_coerce_infons(doc_obj.get("infons") or doc_obj.get("infon") or {}),
        )
        raw_passages = _mapping_list(doc_obj.get("passages") or doc_obj.get("passage") or [])
        for passage_obj in raw_passages:
            passage = BioCPassage(
                infons=_coerce_infons(passage_obj.get("infons") or passage_obj.get("infon") or {}),
                offset=_safe_int(passage_obj.get("offset"), default=0) or 0,
                text=str(passage_obj.get("text") or ""),
            )
            raw_annotations = _mapping_list(passage_obj.get("annotations") or passage_obj.get("annotation") or [])
            for annotation_obj in raw_annotations:
                raw_locations = _mapping_list(annotation_obj.get("locations") or annotation_obj.get("location") or [])
                locations = [
                    BioCLocation(
                        offset=_safe_int(location.get("offset"), default=0) or 0,
                        length=_safe_int(location.get("length"), default=0) or 0,
                    )
                    for location in raw_locations
                    if isinstance(location, Mapping)
                ]
                passage.annotations.append(
                    BioCAnnotation(
                        id=str(annotation_obj.get("id") or ""),
                        infons=_coerce_infons(annotation_obj.get("infons") or annotation_obj.get("infon") or {}),
                        locations=locations,
                        text=str(annotation_obj.get("text") or ""),
                    )
                )
            document.passages.append(passage)
        collection.documents.append(document)
    return collection


def _unwrap_bioc_json_variant(obj: Any) -> Any:
    """Normalize wrapped BioC JSON payloads into the common parser shapes."""

    if not isinstance(obj, Mapping):
        return obj

    pubtator_payload = obj.get("PubTator3")
    if isinstance(pubtator_payload, list):
        return {"documents": pubtator_payload}
    if isinstance(pubtator_payload, Mapping):
        return pubtator_payload

    return obj


def read_generic_json(path: PathLike) -> BioCCollection:
    return _collection_from_generic_json_obj(_load_json_file(path))


def _collection_from_generic_json_obj(obj: Any) -> BioCCollection:
    """Parse generic document JSON such as BERN2 or HunFlair2-style outputs."""

    collection_source = ""
    raw_documents: list[Any]

    if isinstance(obj, list):
        raw_documents = list(obj)
    elif isinstance(obj, Mapping):
        if "documents" in obj and isinstance(obj["documents"], list) and not _looks_like_bioc_json_obj(obj):
            raw_documents = list(obj["documents"])
            collection_source = str(obj.get("source") or "")
        else:
            raw_documents = [obj]
            collection_source = str(obj.get("source") or "")
    else:
        raise ValueError("JSON payload must be an object or a list of objects.")

    collection = BioCCollection(source=collection_source, date="", key="generic JSON")
    for index, doc_obj in enumerate(raw_documents, start=1):
        if not isinstance(doc_obj, Mapping):
            continue
        document = _generic_json_document_to_bioc(doc_obj, default_id=f"doc{index}")
        collection.documents.append(document)
    return collection


def _generic_json_document_to_bioc(doc_obj: Mapping[str, Any], *, default_id: str) -> BioCDocument:
    document_id = str(
        doc_obj.get("id")
        or doc_obj.get("document_id")
        or doc_obj.get("doc_id")
        or doc_obj.get("pmid")
        or doc_obj.get("pmcid")
        or default_id
    )
    document = BioCDocument(id=document_id, infons=_coerce_infons(doc_obj.get("infons") or {}))

    title = str(doc_obj.get("title") or "")
    abstract = str(doc_obj.get("abstract") or doc_obj.get("abstract_text") or "")
    text = str(doc_obj.get("text") or doc_obj.get("content") or doc_obj.get("body") or "")
    if not text:
        sentences = doc_obj.get("sentences")
        if isinstance(sentences, list):
            text = " ".join(
                str(
                    sentence.get("text")
                    if isinstance(sentence, Mapping)
                    else sentence or ""
                )
                for sentence in sentences
            ).strip()
    if not text:
        text = f"{title} {abstract}".strip() if title and abstract else title or abstract

    passage = BioCPassage(
        infons=_coerce_infons(doc_obj.get("passage_infons") or {}),
        offset=0,
        text=text,
    )

    raw_annotations = _mapping_list(
        doc_obj.get("entities")
        or doc_obj.get("annotations")
        or doc_obj.get("denotations")
        or doc_obj.get("spans")
        or []
    )
    for annotation_index, annotation_obj in enumerate(raw_annotations, start=1):
        start, end = _generic_json_span(annotation_obj)
        label, score = _generic_json_label_score(annotation_obj)
        annotation_text = str(annotation_obj.get("text") or annotation_obj.get("mention") or "")
        if not annotation_text and start is not None and end is not None and text:
            annotation_text = text[max(0, start):max(0, end)]

        infons = _coerce_infons(annotation_obj.get("infons") or {})
        if label and "type" not in infons:
            infons["type"] = label
        if score is not None and "score" not in infons:
            infons["score"] = score

        locations: list[BioCLocation] = []
        if start is not None and end is not None and end >= start:
            locations.append(BioCLocation(offset=int(start), length=int(end) - int(start)))

        passage.annotations.append(
            BioCAnnotation(
                id=str(annotation_obj.get("id") or f"A{annotation_index}"),
                infons=infons,
                locations=locations,
                text=annotation_text,
            )
        )

    document.passages.append(passage)
    return document


def _generic_json_span(annotation_obj: Mapping[str, Any]) -> tuple[int | None, int | None]:
    span = annotation_obj.get("span")
    if isinstance(span, Mapping):
        start = _safe_int(span.get("begin"), default=None)
        if start is None:
            start = _safe_int(span.get("start"), default=None)
        if start is None:
            start = _safe_int(span.get("offset"), default=None)
        end = _safe_int(span.get("end"), default=None)
        if end is None:
            length = _safe_int(span.get("length"), default=None)
            if start is not None and length is not None:
                end = start + length
        return start, end

    start = _safe_int(annotation_obj.get("start"), default=None)
    if start is None:
        start = _safe_int(annotation_obj.get("begin"), default=None)
    if start is None:
        start = _safe_int(annotation_obj.get("start_pos"), default=None)
    if start is None:
        start = _safe_int(annotation_obj.get("offset"), default=None)

    end = _safe_int(annotation_obj.get("end"), default=None)
    if end is None:
        end = _safe_int(annotation_obj.get("end_pos"), default=None)
    if end is None:
        length = _safe_int(annotation_obj.get("length"), default=None)
        if start is not None and length is not None:
            end = start + length

    return start, end


def _generic_json_label_score(annotation_obj: Mapping[str, Any]) -> tuple[str, float | None]:
    labels = annotation_obj.get("labels")
    if isinstance(labels, list) and labels:
        first = labels[0]
        if isinstance(first, Mapping):
            label = str(first.get("value") or first.get("label") or first.get("name") or "")
            score = _safe_float(first.get("score"), default=None)
            return label, score

    label = str(
        annotation_obj.get("obj")
        or annotation_obj.get("type")
        or annotation_obj.get("entity_type")
        or annotation_obj.get("label")
        or ""
    )
    score = _safe_float(
        annotation_obj.get("score") if annotation_obj.get("score") is not None else annotation_obj.get("confidence"),
        default=None,
    )
    return label, score


def write_bioc_json(collection: BioCCollection, output_json: PathLike) -> Path:
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(collection.to_json(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output

# ---------------------------------------------------------------------------
# Conversion and temporary XML helpers
# ---------------------------------------------------------------------------
def convert_bioc_file(
    input_path: PathLike,
    output_path: PathLike,
    *,
    input_format: str = "auto",
    output_format: str = "auto",
) -> Path:
    """Convert BioC XML/BioC JSON/JSON input to BioC XML or BioC JSON."""

    collection = read_bioc_collection(input_path, input_format=input_format)
    return write_bioc_collection(collection, output_path, output_format=output_format)


@contextmanager
def bioc_file_as_xml(input_path: PathLike, *, input_format: str = "auto") -> Iterator[Path]:
    """Yield a BioC XML path for any supported document input format.

    If ``input_path`` is already BioC XML it is yielded unchanged.  Otherwise a
    temporary XML copy is created and deleted when the context exits.
    """

    fmt = canonical_document_format(input_format, path=input_path)
    p = Path(input_path)
    if fmt == "bioc-xml":
        yield p
        return
    with tempfile.TemporaryDirectory(prefix="cellexlink_input_bioc_") as tmp:
        xml_path = Path(tmp) / "input.xml"
        convert_bioc_file(p, xml_path, input_format=fmt, output_format="bioc-xml")
        yield xml_path


def ensure_bioc_xml_file(
    input_path: PathLike,
    output_xml: PathLike,
    *,
    input_format: str = "auto",
) -> Path:
    """Create a BioC XML copy of any supported input file."""

    return convert_bioc_file(input_path, output_xml, input_format=input_format, output_format="bioc-xml")


def merge_bioc_collections(collections: Iterable[BioCCollection]) -> BioCCollection:
    """Merge several BioC collections into one without modifying predictions."""

    merged = BioCCollection(source="CellExLink", date="", key="merged BioC collection")
    for collection in collections:
        if not merged.source and collection.source:
            merged.source = collection.source
        if not merged.date and collection.date:
            merged.date = collection.date
        if not merged.key and collection.key:
            merged.key = collection.key
        merged.documents.extend(collection.documents)
    return merged


def merge_existing_annotations(
    prediction_path: PathLike,
    original_path: PathLike,
    output_path: PathLike,
    *,
    prediction_format: str = "auto",
    original_format: str = "auto",
    output_format: str = "auto",
    keep_duplicates: bool = False,
) -> Path:
    """Copy annotations from ``original_path`` into ``prediction_path`` when missing.

    This is useful for BERN2-style outputs: CellExLink can add cell-type
    annotations while preserving gene/disease/chemical/species annotations that
    were present before prediction.
    """

    predictions = read_bioc_collection(prediction_path, input_format=prediction_format)
    original = read_bioc_collection(original_path, input_format=original_format)
    _merge_annotations_into_collection(predictions, original, keep_duplicates=keep_duplicates)
    return write_bioc_collection(predictions, output_path, output_format=output_format)


def merge_annotations_into_collection(
    target: BioCCollection,
    source: BioCCollection,
    *,
    keep_duplicates: bool = False,
) -> BioCCollection:
    """Merge annotations from one in-memory collection into another."""

    _merge_annotations_into_collection(target, source, keep_duplicates=keep_duplicates)
    return target


def _merge_annotations_into_collection(
    target: BioCCollection,
    source: BioCCollection,
    *,
    keep_duplicates: bool = False,
) -> None:
    source_by_doc = {document.id: document for document in source.documents}
    for target_doc in target.documents:
        source_doc = source_by_doc.get(target_doc.id)
        if source_doc is None:
            continue
        for passage_index, target_passage in enumerate(target_doc.passages):
            if passage_index >= len(source_doc.passages):
                continue
            source_passage = source_doc.passages[passage_index]
            existing_keys = {annotation.duplicate_key() for annotation in target_passage.annotations}
            next_id = len(target_passage.annotations) + 1
            for annotation in source_passage.annotations:
                if not keep_duplicates and annotation.duplicate_key() in existing_keys:
                    continue
                new_annotation = BioCAnnotation(
                    id=annotation.id or f"S{next_id}",
                    infons=dict(annotation.infons),
                    locations=[BioCLocation(location.offset, location.length) for location in annotation.locations],
                    text=annotation.text,
                )
                if new_annotation.id and any(
                    existing.id == new_annotation.id
                    for existing in target_passage.annotations
                ):
                    new_annotation.id = f"S{next_id}"
                target_passage.annotations.append(new_annotation)
                existing_keys.add(new_annotation.duplicate_key())
                next_id += 1


# ---------------------------------------------------------------------------
# Existing XML-oriented CellExLink helpers retained for compatibility
# ---------------------------------------------------------------------------
def _coerce_paths(paths: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _text_of(parent: ET.Element, tag: str, default: str = "") -> str:
    child = parent.find(tag)
    if child is None or child.text is None:
        return default
    return child.text


def _int_text_of(parent: ET.Element, tag: str, default: int = 0) -> int:
    value = _text_of(parent, tag, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _indent_tree(root: ET.Element) -> None:
    ET.indent(root, space="  ")


def read_infons(element: ET.Element) -> dict[str, str]:
    """Read BioC ``infon`` children from an XML element."""

    infons: dict[str, str] = {}
    for infon in element.findall("infon"):
        key = infon.attrib.get("key", "")
        if not key:
            continue
        infons[key] = infon.text or ""
    return infons


def write_text_as_bioc_xml(
    text: str,
    output_xml: PathLike,
    *,
    document_id: str = "doc0",
    passage_offset: int = 0,
    source: str = "CellExLink",
    key: str = "cell-type-extraction",
    passage_type: str | None = None,
) -> Path:
    """Write plain text as a one-document, one-passage BioC XML file."""

    collection = _single_passage_collection(
        text,
        document_id=document_id,
        passage_offset=passage_offset,
        source=source,
        key=key,
        passage_type=passage_type,
    )
    return write_bioc_xml(collection, output_xml)


def write_text_as_bioc_json(
    text: str,
    output_json: PathLike,
    *,
    document_id: str = "doc0",
    passage_offset: int = 0,
    source: str = "CellExLink",
    key: str = "cell-type-extraction",
    passage_type: str | None = None,
) -> Path:
    """Write plain text as a one-document, one-passage BioC JSON file."""

    collection = _single_passage_collection(
        text,
        document_id=document_id,
        passage_offset=passage_offset,
        source=source,
        key=key,
        passage_type=passage_type,
    )
    return write_bioc_json(collection, output_json)


def _annotation_to_entity_span(annotation: BioCAnnotation, passage_offset: int) -> EntitySpan:
    infons = dict(annotation.infons)
    label = infons.get("type") or infons.get("label") or infons.get("entity_type") or ""
    text = annotation.text
    location = annotation.primary_location()
    if location is None:
        absolute_start = passage_offset
        length = len(text)
    else:
        absolute_start = int(location.offset)
        length = int(location.length)
    relative_start = absolute_start - int(passage_offset)
    relative_end = relative_start + length
    score: float | None = None
    if "score" in infons:
        score = _safe_float(infons["score"], default=None)
    return EntitySpan(
        start=relative_start,
        end=relative_end,
        label=label,
        text=text,
        score=score,
        infons=infons,
    )


def iter_bioc_passage_records(
    input_xml: PathLike | Sequence[PathLike],
    *,
    include_entities: bool = True,
    input_format: str = "auto",
) -> Iterator[PassageRecord]:
    """Yield passage records from one or more BioC XML/JSON files."""

    record_id = 0
    for path in _coerce_paths(input_xml):
        collection = read_bioc_collection(path, input_format=input_format)
        for record in iter_collection_passage_records(
            collection,
            include_entities=include_entities,
            source_path=str(path),
            start_record_id=record_id,
        ):
            yield record
            record_id = record.record_id + 1


def iter_collection_passage_records(
    collection: BioCCollection,
    *,
    include_entities: bool = True,
    source_path: str = "",
    start_record_id: int = 0,
) -> Iterator[PassageRecord]:
    """Yield passage records from an in-memory BioC collection."""

    record_id = start_record_id
    for document in collection.documents:
        for passage_id, passage in enumerate(document.passages):
            entities = []
            if include_entities:
                entities = [
                    _annotation_to_entity_span(annotation, passage.offset)
                    for annotation in passage.annotations
                ]
            yield PassageRecord(
                record_id=record_id,
                document_id=document.id,
                passage_id=passage_id,
                passage_offset=passage.offset,
                text=passage.text,
                entities=entities,
                infons=dict(passage.infons),
                source_path=source_path,
            )
            record_id += 1


def read_bioc_annotations(
    input_xml: PathLike | Sequence[PathLike],
    *,
    input_format: str = "auto",
) -> list[PredictedEntity]:
    """Read all BioC annotations as entities with absolute offsets."""

    annotations: list[PredictedEntity] = []
    for record in iter_bioc_passage_records(input_xml, include_entities=True, input_format=input_format):
        for entity in record.entities:
            annotations.append(
                entity.to_absolute(
                    record.passage_offset,
                    document_id=record.document_id,
                    passage_id=record.passage_id,
                )
            )
    return annotations


def _remove_existing_annotations(root: ET.Element) -> None:
    for passage in root.findall(".//passage"):
        for annotation in list(passage.findall("annotation")):
            passage.remove(annotation)


def _group_predictions_by_passage(
    predicted_entities: Iterable[PredictedEntity],
) -> dict[tuple[str, int], list[PredictedEntity]]:
    grouped: dict[tuple[str, int], list[PredictedEntity]] = {}
    for entity in predicted_entities:
        key = (entity.document_id, int(entity.passage_id))
        grouped.setdefault(key, []).append(entity)
    for entities in grouped.values():
        entities.sort(key=lambda item: (int(item.start), int(item.end), str(item.label), str(item.text)))
    return grouped


def apply_predicted_entities_to_collection(
    collection: BioCCollection,
    predicted_entities: Iterable[PredictedEntity],
    *,
    clear_existing: bool = True,
) -> BioCCollection:
    """Apply predicted entities to an in-memory BioC collection."""

    grouped = _group_predictions_by_passage(predicted_entities)
    annotation_index = 1

    for document in collection.documents:
        for passage_id, passage in enumerate(document.passages):
            if clear_existing:
                passage.annotations = []

            candidates = list(grouped.get((document.id, passage_id), []))
            if document.id:
                candidates.extend(grouped.get(("", passage_id), []))

            for entity in candidates:
                passage.annotations.append(
                    BioCAnnotation(
                        id=f"T{annotation_index}",
                        infons={
                            "type": str(entity.label),
                            **{
                                str(key): "" if value is None else str(value)
                                for key, value in entity.infons.items()
                                if str(key) != "type"
                            },
                        },
                        locations=[
                            BioCLocation(
                                offset=int(entity.start),
                                length=max(0, int(entity.end) - int(entity.start)),
                            )
                        ],
                        text=entity.text,
                    )
                )
                annotation_index += 1

    return collection


def _add_infon(parent: ET.Element, key: str, value: Any) -> None:
    ET.SubElement(parent, "infon", {"key": str(key)}).text = "" if value is None else str(value)


def write_predictions_to_bioc_xml(
    input_xml: PathLike,
    output_xml: PathLike,
    *,
    predicted_entities: Iterable[PredictedEntity],
    clear_existing: bool = True,
) -> Path:
    """Write predicted entities into a BioC XML file."""

    collection = read_bioc_collection(input_xml, input_format="bioc-xml")
    apply_predicted_entities_to_collection(
        collection,
        predicted_entities,
        clear_existing=clear_existing,
    )
    return write_bioc_collection(collection, output_xml, output_format="bioc-xml")


def collection_summary(
    input_xml: PathLike | Sequence[PathLike],
    *,
    input_format: str = "auto",
) -> dict[str, int]:
    """Return simple counts for BioC XML/JSON files."""

    documents = 0
    passages = 0
    annotations = 0
    for path in _coerce_paths(input_xml):
        collection = read_bioc_collection(path, input_format=input_format)
        documents += len(collection.documents)
        for document in collection.documents:
            passages += len(document.passages)
            annotations += sum(len(passage.annotations) for passage in document.passages)
    return {"documents": documents, "passages": passages, "annotations": annotations}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _load_json_file(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _single_passage_collection(
    text: str,
    *,
    document_id: str,
    passage_offset: int,
    source: str,
    key: str,
    passage_type: str | None,
) -> BioCCollection:
    return BioCCollection(
        source=source,
        date="",
        key=key,
        documents=[
            BioCDocument(
                id=document_id,
                passages=[
                    BioCPassage(
                        infons=({"type": passage_type} if passage_type is not None else {}),
                        offset=int(passage_offset),
                        text=text,
                    )
                ],
            )
        ],
    )


def _coerce_infons(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): "" if val is None else str(val) for key, val in value.items()}
    if isinstance(value, list):
        infons: dict[str, str] = {}
        for item in value:
            if isinstance(item, Mapping):
                key = item.get("key") or item.get("name")
                val = item.get("value") or item.get("text") or item.get("content") or ""
                if key is not None:
                    infons[str(key)] = str(val)
        return infons
    return {}


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_pmid_list(ids: Iterable[str] | str) -> list[str]:
    """Normalize a comma/space/newline separated PMID list."""

    if isinstance(ids, str):
        candidates = re.split(r"[\s,;]+", ids.strip())
    else:
        candidates = []
        for item in ids:
            candidates.extend(re.split(r"[\s,;]+", str(item).strip()))
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


__all__ = [
    "BIOC_JSON_FORMATS",
    "BIOC_XML_FORMATS",
    "SUPPORTED_DOCUMENT_FORMATS",
    "BioCAnnotation",
    "BioCCollection",
    "BioCDocument",
    "BioCLocation",
    "BioCPassage",
    "EntitySpan",
    "PassageRecord",
    "PathLike",
    "PredictedEntity",
    "bioc_file_as_xml",
    "canonical_bioc_format",
    "canonical_document_format",
    "clean_pmid_list",
    "collection_summary",
    "convert_bioc_file",
    "detect_document_format",
    "ensure_bioc_xml_file",
    "iter_bioc_passage_records",
    "merge_bioc_collections",
    "merge_bioc_files",
    "merge_existing_annotations",
    "output_format_from_path",
    "read_bioc_annotations",
    "read_bioc_collection",
    "read_bioc_collection_from_string",
    "read_infons",
    "write_bioc_collection",
    "write_bioc_document_stream",
    "write_bioc_json",
    "write_bioc_xml",
    "write_predictions_to_bioc_xml",
    "write_text_as_bioc_json",
    "write_text_as_bioc_xml",
]
