"""Render CellDN annotations as highlighted HTML in notebook environments.

This module supports two common notebook workflows:

1. Plain text plus pipeline results:
       render_cell_type_annotations(text, results)
2. BioC output directly:
       render_cell_type_annotations("predicted.bioc.xml")

For BioC inputs, absolute annotation offsets are converted into passage-local
offsets automatically before rendering.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

from IPython.display import HTML

CELL_TYPE_COLOR = "#ddd6fe"


def _find_first_infon_value_by_suffix(infons: dict[str, Any], suffix: str) -> Any:
    """Return the first infon value whose key ends with the requested suffix."""

    for key, value in infons.items():
        if str(key).endswith(suffix):
            return value
    return None


def _as_annotation_dict(annotation: Any) -> dict[str, Any] | None:
    """Convert a result object or dict-like annotation into a plain dict."""

    if isinstance(annotation, dict):
        return annotation

    to_dict = getattr(annotation, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted

    return None


def get_annotation_start_end(annotation: dict[str, Any]) -> tuple[int | None, int | None]:
    """Support both flat and nested span representations."""

    if "start" in annotation and "end" in annotation:
        return annotation["start"], annotation["end"]

    span = annotation.get("span", {})
    if isinstance(span, dict) and "begin" in span and "end" in span:
        return span["begin"], span["end"]

    return None, None


def _normalize_annotations(annotations: Iterable[Any]) -> list[dict[str, Any]]:
    """Keep only annotations that can be represented as dictionaries."""

    normalized: list[dict[str, Any]] = []
    for annotation in annotations:
        item = _as_annotation_dict(annotation)
        if item is not None:
            normalized.append(item)
    return normalized


def build_highlighted_text_html(text: str, annotations: Iterable[Any]) -> str:
    """Render cell-type mentions as inline highlighted spans."""

    normalized_annotations = []

    for annotation in _normalize_annotations(annotations):
        if annotation.get("entity_type") != "cell_type":
            continue

        start, end = get_annotation_start_end(annotation)
        if start is None or end is None:
            continue

        start = int(start)
        end = int(end)

        if start < 0 or end > len(text) or start >= end:
            continue

        normalized_annotations.append(
            {
                "start": start,
                "end": end,
                "mention": annotation.get("mention", text[start:end]),
                "identifier": annotation.get("identifier"),
                "label": annotation.get("label"),
            }
        )

    normalized_annotations.sort(key=lambda annotation: annotation["start"])

    html_parts: list[str] = []
    last = 0

    for annotation in normalized_annotations:
        start = annotation["start"]
        end = annotation["end"]

        if start < last:
            continue

        html_parts.append(escape(text[last:start]))

        mention_text = escape(text[start:end])
        label = escape(str(annotation.get("label", "")))
        identifier = escape(str(annotation.get("identifier", "")))
        tooltip = f"{label} | {identifier}" if label or identifier else "cell_type"

        html_parts.append(
            f"""
        <span title="{tooltip}" style="
            background-color: {CELL_TYPE_COLOR};
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: 600;
        ">{mention_text}</span>
        """
        )
        last = end

    html_parts.append(escape(text[last:]))
    return "".join(html_parts)


def _build_section_html(title: str, body_html: str) -> str:
    """Wrap one rendered block with a small section heading."""

    return f"""
    <section style="margin-bottom: 24px;">
        <div style="
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 8px;
        ">{escape(title)}</div>
        <div>{body_html}</div>
    </section>
    """


def build_full_html(body_html: str) -> str:
    """Wrap rendered spans in a readable notebook container."""

    return f"""
    <div style="
        font-family: Arial, sans-serif;
        font-size: 18px;
        line-height: 1.7;
        max-width: 1200px;
    ">
        {body_html}
    </div>
    """


def _coerce_bioc_collection(source: Any) -> Any:
    """Accept either a BioC collection object or a file path and return a collection."""

    if hasattr(source, "documents"):
        return source

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.exists():
            from ..io import read_bioc_collection

            return read_bioc_collection(source_path)

    raise TypeError(
        "When annotations are omitted, source must be a BioC path or BioC collection."
    )


def _annotation_from_bioc(annotation: Any, *, passage_offset: int) -> dict[str, Any] | None:
    """Convert one BioC annotation into the display format used by this module."""

    infons = getattr(annotation, "infons", {})
    if not isinstance(infons, dict):
        infons = dict(infons or {})

    locations = getattr(annotation, "locations", [])
    if not locations:
        return None

    location = locations[0]
    absolute_start = int(getattr(location, "offset", 0))
    length = int(getattr(location, "length", 0))
    absolute_end = absolute_start + length

    entity_type = infons.get("type") or infons.get("entity_type") or infons.get("label")
    identifier = (
        _find_first_infon_value_by_suffix(infons, "_id_0")
        or infons.get("identifier")
        or infons.get("cl_id")
    )
    label = (
        _find_first_infon_value_by_suffix(infons, "_identifier_name_0")
        or infons.get("label")
        or infons.get("name")
        or infons.get("cl_label")
    )

    return {
        "start": absolute_start - int(passage_offset),
        "end": absolute_end - int(passage_offset),
        "mention": getattr(annotation, "text", ""),
        "entity_type": entity_type,
        "identifier": identifier,
        "label": label,
    }


def _render_bioc_collection_html(collection: Any) -> str:
    """Render every annotated BioC passage into one HTML string."""

    sections: list[str] = []

    for document in getattr(collection, "documents", []):
        for passage_index, passage in enumerate(getattr(document, "passages", [])):
            annotations = []
            for annotation in getattr(passage, "annotations", []):
                item = _annotation_from_bioc(
                    annotation,
                    passage_offset=getattr(passage, "offset", 0),
                )
                if item is not None:
                    annotations.append(item)

            passage_text = getattr(passage, "text", "")
            highlighted_text_html = build_highlighted_text_html(passage_text, annotations)
            if highlighted_text_html == escape(passage_text):
                continue

            passage_type = getattr(passage, "infons", {}).get("type", f"passage {passage_index}")
            document_id = getattr(document, "id", "") or "document"
            title = f"{document_id} - {passage_type}"
            sections.append(_build_section_html(title, highlighted_text_html))

    if not sections:
        sections.append(_build_section_html("No cell-type annotations", ""))

    return "".join(sections)


def render_cell_type_annotations(source: Any, annotations: Iterable[Any] | None = None) -> HTML:
    """Return notebook-ready HTML for plain-text results or a BioC file/collection."""

    if annotations is None:
        collection = _coerce_bioc_collection(source)
        final_html = build_full_html(_render_bioc_collection_html(collection))
        return HTML(final_html)

    highlighted_text_html = build_highlighted_text_html(str(source), annotations)
    final_html = build_full_html(_build_section_html("Annotations", highlighted_text_html))
    return HTML(final_html)
