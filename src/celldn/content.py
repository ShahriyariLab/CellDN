"""Shared article-content rules for retrieval, PDF, NER, and normalization.

Excluded sections and passages marked non-annotatable are removed before model
processing. Retained passages keep their original offsets.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .io import BioCCollection, BioCPassage

# Exact front-matter labels, shared by structured input and PDF layout cleanup.
# In PDFs their continuation is bounded by the physical region rather than by
# the next column, so an adjacent unlabelled abstract remains scientific text.
_FRONTMATTER_SECTIONS = frozenset({
    "edited by", "editor", "editors", "handling editor", "academic editor",
    "reviewed by", "reviewer", "reviewers", "peer reviewed by",
    "correspondence", "corresponding author", "corresponding authors",
    "correspondence to", "address for correspondence", "contact information",
    "author information", "authors information", "author details",
    "author affiliations", "affiliation", "affiliations",
    "keywords", "keyword", "key words", "index terms",
    "received", "received on", "accepted", "accepted on", "published", "published on",
    "publication date", "article history", "doi", "email", "e mail",
})

# These are section labels, not words to search for inside scientific prose.
_EXCLUDED_SECTIONS = frozenset({
    "ref", "refs", "reference", "references", "reference list", "references and notes",
    "ref list", "bibliography", "bibliographic references", "literature cited",
    "ack", "acks", "acknowledgment", "acknowledgments", "acknowledgement",
    "acknowledgements", "acknowledgment section", "acknowledgements and funding",
    "acknowledgments and funding", "funding and acknowledgements",
    "funding and acknowledgments", "fund", "funding", "funding information",
    "funding statement", "funding sources", "sources of funding", "financial support",
    "financial disclosure", "financial disclosures", "funding and disclosure",
    "funding and disclosures", "funding disclosure", "grant support",
    "comp int", "competing interest", "competing interests", "competing interests statement",
    "competing interests declaration", "conflict", "conflicts", "conflict of interest",
    "conflicts of interest", "conflict of interests", "conflicts of interests",
    "conflict of interest statement", "conflicts of interest statement",
    "declaration of competing interest", "declaration of competing interests",
    "declaration of interest", "declaration of interests", "disclosure",
    "disclosures", "disclosure statement", "declarations", "author declarations",
    "auth cont", "author contribution", "author contributions", "authors contributions",
    "contribution", "contributions", "contrib", "contrib group", "credit authorship contribution statement",
    "credit author statement", "credit authorship statement", "authors roles", "author roles",
    "data availability", "data availability statement", "data avail", "data sharing statement",
    "availability of data and materials", "availability of data", "data and code availability",
    "code availability", "code availability statement", "availability of supporting data",
    "ethics statement", "ethical statement", "ethics approval and consent to participate",
    "ethics approval", "ethical approval", "consent for publication", "patient consent",
    "informed consent statement", "institutional review board statement",
    "publisher note", "publishers note", "publisher s note", "disclaimer", "disclaimers",
    "copyright", "copyright statement", "license", "licence",
})

_KEPT_SECTIONS = frozenset({
    "title", "abstract", "summary", "author summary", "background", "introduction", "intro",
    "objective", "objectives", "aim", "aims", "purpose", "materials", "methods", "method",
    "materials and methods", "material and methods", "patients and methods",
    "subjects and methods", "experimental procedures", "experimental methods",
    "experimental section", "methodology", "results", "result", "findings",
    "results and discussion", "results discussion", "discussion", "discuss",
    "discussion and conclusions", "conclusion", "conclusions", "concluding remarks", "concl",
    "limitations", "study limitations", "future directions", "perspectives",
    "key points", "key messages", "highlights", "significance", "significance statement",
    "appendix", "appendices", "supplementary material", "supplementary materials",
    "supporting information", "supplementary information", "abbreviations", "abbreviation list",
})

_SECTION_KEYS = ("section_type", "section", "section_title", "section_name", "heading")
_HEADING_TYPES = frozenset({"section title", "section heading", "heading", "subtitle", "title 1", "title 2", "title 3"})
_GENERIC_TYPES = frozenset({"", "paragraph", "body", "text", "body text", "fulltext", "p", "article", "other", "unknown", "unassigned", "general"})
_SCIENTIFIC_BLOCK_TYPES = frozenset({"caption", "figure", "fig", "table", "table caption", "figure caption"})
_NUMBERED_PREFIX = re.compile(r"^(?:(?:\d+(?:\.\d+)*[.)]?)|(?:[ivxlcdm]+[.)]))\s+", re.I)


def _normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = _NUMBERED_PREFIX.sub("", text)
    text = re.sub(r"[_/–—-]+", " ", text)
    text = text.replace("’", "'").replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def is_excluded_section(value: Any) -> bool:
    """Whether a section label denotes references or administrative material."""

    return _normalize_label(value) in _EXCLUDED_SECTIONS | _FRONTMATTER_SECTIONS


def is_frontmatter_section(value: Any) -> bool:
    """Whether an exact label denotes author, editorial, or indexing metadata."""

    return _normalize_label(value) in _FRONTMATTER_SECTIONS


def classify_section_heading(text: str) -> str | None:
    """Recognize a common heading or an explicitly labelled administrative line.

    Numbered headings and excluded labels before a colon, period, or dash are
    accepted. Scientific sentences merely containing a section-name word are
    not headings. Unknown headings can use ``type=section_title``.
    """

    if not isinstance(text, str):
        return None
    stripped = unicodedata.normalize("NFKC", text).strip()
    if len(stripped) <= 140:
        label = _normalize_label(stripped)
        if is_excluded_section(label) or label in _KEPT_SECTIONS:
            return label
    # Administrative statements often start on their heading's line. Require
    # punctuation after an exact label; "Funding mechanisms influence T cells"
    # and "References to T cells were checked" remain scientific prose.
    unnumbered = _NUMBERED_PREFIX.sub("", stripped)
    match = re.match(r"^(.{1,100}?)(?:\s*[:–—]\s*|\.\s+|\s+-\s+)(?=\S)", unnumbered)
    if match and is_excluded_section(match.group(1)):
        return _normalize_label(match.group(1))
    return None


def _infons(passage: BioCPassage | Mapping[str, Any]) -> Mapping[str, Any]:
    values = passage.get("infons", {}) if isinstance(passage, Mapping) else passage.infons
    return values if isinstance(values, Mapping) else {}


def _text(passage: BioCPassage | Mapping[str, Any]) -> str:
    value = passage.get("text", "") if isinstance(passage, Mapping) else passage.text
    return str(value or "")


def _explicitly_false(infons: Mapping[str, Any]) -> bool:
    return str(infons.get("annotatable", "true")).strip().casefold() in {"false", "0", "no"}


def _labelled_exclusion(text: str) -> str | None:
    """Recognize standalone headings or administrative labels before a colon."""

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    heading = classify_section_heading(first_line)
    if heading and is_excluded_section(heading):
        return heading
    prefix, separator, _ = first_line.partition(":")
    if separator and is_excluded_section(prefix):
        return _normalize_label(prefix)
    return None


def is_passage_annotatable(passage: BioCPassage | Mapping[str, Any]) -> bool:
    """Whether a passage is eligible based on its own flag and section labels.

    Use ``filter_collection_content`` on the complete document as well, because
    unlabelled paragraphs can inherit an excluded preceding section heading.
    """

    infons = _infons(passage)
    if _explicitly_false(infons):
        return False
    if any(is_excluded_section(infons.get(key)) for key in (*_SECTION_KEYS, "type")):
        return False
    # A real article title or abstract is not a section heading even if very short.
    kind = _normalize_label(infons.get("type"))
    if kind in {"title", "abstract"}:
        return True
    return _labelled_exclusion(_text(passage)) is None


def filter_collection_content(collection: BioCCollection) -> BioCCollection:
    """Remove excluded passages in-place while keeping original retained offsets.

    Structured section labels take priority. With unlabelled text, an excluded
    heading starts a section that ends at the next recognized or structurally
    marked scientific heading. State is reset for each document.
    """

    for document in collection.documents:
        retained = []
        excluded = False
        for passage in document.passages:
            infons = _infons(passage)
            kind = _normalize_label(infons.get("type"))
            section_values = [infons[key] for key in _SECTION_KEYS if infons.get(key)]
            explicit_exclusion = any(is_excluded_section(value) for value in section_values)
            explicit_exclusion = explicit_exclusion or is_excluded_section(kind)
            heading = classify_section_heading(passage.text)
            labelled_exclusion = None if kind in {"title", "abstract"} else _labelled_exclusion(passage.text)

            if explicit_exclusion or labelled_exclusion:
                excluded = True
            elif any(_normalize_label(value) not in _GENERIC_TYPES for value in section_values):
                # A named section identifies new context; a generic BODY/TEXT
                # label does not end references identified by an earlier heading.
                excluded = False
            elif kind in {"title", "abstract"} | _SCIENTIFIC_BLOCK_TYPES:
                excluded = False
            elif kind in _HEADING_TYPES or heading is not None:
                excluded = False
            elif kind not in _GENERIC_TYPES and kind in _KEPT_SECTIONS:
                excluded = False

            if not excluded and is_passage_annotatable(passage):
                retained.append(passage)
        document.passages = retained
    return collection


__all__ = [
    "classify_section_heading", "filter_collection_content",
    "is_excluded_section", "is_frontmatter_section", "is_passage_annotatable",
]
