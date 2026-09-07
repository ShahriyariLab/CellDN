"""Render PubTator3 and CellDN BioC JSON annotations.

This module provides small notebook-friendly helpers for loading a BioC JSON
file and rendering its passages as highlighted HTML. It is designed for merged
outputs where PubTator3 annotations are preserved and CellDN cell-type
annotations are added on top.

Display behavior:
    - CellDN ``cell_type`` mentions are highlighted in lavender.
    - Other PubTator3 entities are highlighted in pale yellow.
    - When two annotations overlap, CellDN annotations take priority.

Expected input:
    A BioC JSON dictionary with a top-level ``documents`` list, where each
    document contains ``passages`` and each passage may contain
    ``annotations`` with ``infons`` and ``locations`` fields.
"""

import json
from html import escape
from pathlib import Path

from IPython.display import HTML


# =========================
# Colors
# =========================
CELLDN_COLOR = "#ddd6fe"   # CellDN cell_type annotations
PUBTATOR3_COLOR = "#fef3c7"    # PubTator3 annotations: Gene, Disease, Species, etc.


# =========================
# Helper functions
# =========================
def load_bioc_json(path):
    """Load a BioC JSON file into a Python dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_ann_type(annotation):
    """Read the annotation type from BioC infons."""

    infons = annotation.get("infons", {})
    return str(infons.get("type", "")).strip()


def is_cell_type(annotation):
    """Return True when the annotation is a CellDN cell-type mention."""

    ann_type = get_ann_type(annotation).lower()
    ann_type = ann_type.replace("-", "_").replace(" ", "_")
    return ann_type == "cell_type"


def get_identifier(annotation):
    """Return the best available normalized identifier for an annotation."""

    infons = annotation.get("infons", {})
    return str(
        infons.get("identifier")
        or infons.get("normalized_id")
        or ""
    )


def get_label(annotation):
    """Return the best available human-readable label for an annotation."""

    infons = annotation.get("infons", {})
    return str(
        infons.get("label")
        or infons.get("name")
        or ""
    )


def get_local_span(annotation, passage_text, passage_offset):
    """
    BioC offsets are usually document-level offsets.
    Passage text starts at passage["offset"], so we subtract passage_offset.

    This function also tries raw offsets as a fallback.
    """
    locations = annotation.get("locations", [])

    if not locations:
        return None, None

    loc = locations[0]

    if "offset" not in loc or "length" not in loc:
        return None, None

    raw_start = int(loc["offset"])
    raw_end = raw_start + int(loc["length"])
    mention = annotation.get("text", "")

    candidates = [
        (raw_start - passage_offset, raw_end - passage_offset),
        (raw_start, raw_end),
    ]

    for start, end in candidates:
        if 0 <= start < end <= len(passage_text):
            if mention and passage_text[start:end] == mention:
                return start, end

    for start, end in candidates:
        if 0 <= start < end <= len(passage_text):
            return start, end

    return None, None


def collect_passage_annotations(passage):
    """Convert one passage's BioC annotations into display-ready records."""

    passage_text = passage.get("text", "")
    passage_offset = int(passage.get("offset", 0))

    collected = []

    for annotation in passage.get("annotations", []):
        start, end = get_local_span(
            annotation=annotation,
            passage_text=passage_text,
            passage_offset=passage_offset,
        )

        if start is None or end is None:
            continue

        ann_type = get_ann_type(annotation)

        if is_cell_type(annotation):
            source = "CellDN"
            color = CELLDN_COLOR
            priority = 2
        else:
            source = "PubTator3"
            color = PUBTATOR3_COLOR
            priority = 1

        collected.append({
            "start": start,
            "end": end,
            "text": passage_text[start:end],
            "type": ann_type,
            "source": source,
            "color": color,
            "priority": priority,
            "identifier": get_identifier(annotation),
            "label": get_label(annotation),
        })

    return collected


def remove_overlaps(annotations):
    """
    Keeps non-overlapping annotations.
    If CellDN and PubTator3 overlap, CellDN wins.

    Example:
      PubTator3:   B lymphoma
      CellDN:  B lymphoma cells

    Result:
      B lymphoma cells highlighted as CellDN.
    """
    sorted_annotations = sorted(
        annotations,
        key=lambda ann: (
            -ann["priority"],
            -(ann["end"] - ann["start"]),
            ann["start"],
        )
    )

    selected = []

    for ann in sorted_annotations:
        has_overlap = any(
            not (
                ann["end"] <= chosen["start"]
                or ann["start"] >= chosen["end"]
            )
            for chosen in selected
        )

        if not has_overlap:
            selected.append(ann)

    return sorted(selected, key=lambda ann: ann["start"])


def highlight_text(text, annotations):
    """Apply HTML highlights to a passage after overlap filtering."""

    annotations = remove_overlaps(annotations)

    html_parts = []
    last = 0

    for ann in annotations:
        start = ann["start"]
        end = ann["end"]

        html_parts.append(escape(text[last:start]))

        tooltip_parts = [
            ann["source"],
            ann["type"],
        ]

        if ann["label"]:
            tooltip_parts.append(ann["label"])

        if ann["identifier"]:
            tooltip_parts.append(ann["identifier"])

        tooltip = escape(" | ".join(tooltip_parts), quote=True)
        mention = escape(text[start:end])

        html_parts.append(f"""
            <span title="{tooltip}" style="
                background-color: {ann['color']};
                padding: 2px 5px;
                border-radius: 4px;
                font-weight: 600;
            ">{mention}</span>
        """)

        last = end

    html_parts.append(escape(text[last:]))

    return "".join(html_parts)


def make_legend():
    """Return the HTML legend used above the rendered document."""

    return f"""
    <div style="
        display: flex;
        gap: 24px;
        align-items: center;
        margin-bottom: 18px;
        font-size: 16px;
    ">
        <div>
            <span style="
                background-color: {CELLDN_COLOR};
                padding: 3px 14px;
                border-radius: 4px;
            ">&nbsp;</span>
            CellDN cell_type
        </div>

        <div>
            <span style="
                background-color: {PUBTATOR3_COLOR};
                padding: 3px 14px;
                border-radius: 4px;
            ">&nbsp;</span>
            PubTator3 entities
        </div>
    </div>
    """


def render_celldn_bioc(bioc_data):
    """Render a BioC JSON dictionary as notebook HTML."""

    html_blocks = []

    documents = bioc_data.get("documents", [])

    for document in documents:
        pmid = document.get("id", "")

        html_blocks.append(f"""
            <div style="
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 16px;
            ">
                PMID: {escape(str(pmid))}
            </div>
        """)

        for passage in document.get("passages", []):
            passage_type = passage.get("infons", {}).get("type", "")
            passage_text = passage.get("text", "")

            annotations = collect_passage_annotations(passage)
            highlighted_passage = highlight_text(passage_text, annotations)

            if passage_type == "title":
                style = """
                    font-size: 21px;
                    font-weight: 700;
                    line-height: 1.7;
                    margin-bottom: 16px;
                """
            else:
                style = """
                    font-size: 18px;
                    font-weight: 400;
                    line-height: 1.8;
                    margin-bottom: 22px;
                """

            html_blocks.append(f"""
                <div style="{style}">
                    {highlighted_passage}
                </div>
            """)

    full_html = f"""
    <div style="
        font-family: Arial, sans-serif;
        max-width: 1200px;
        color: #222;
    ">
        {make_legend()}
        {''.join(html_blocks)}
    </div>
    """

    return HTML(full_html)

def render_celldn_bioc_file(path):
    """Load a BioC JSON file from disk and render it as notebook HTML."""

    return render_celldn_bioc(load_bioc_json(path))


__all__ = [
    "load_bioc_json",
    "render_celldn_bioc",
    "render_celldn_bioc_file",
]
