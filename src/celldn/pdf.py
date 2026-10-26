"""Extract selectable scientific PDF text using real glyph geometry.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

from .content import (
    classify_section_heading,
    filter_collection_content,
    is_excluded_section,
    is_frontmatter_section,
)
from .io import BioCCollection, BioCDocument, BioCPassage


class PDFExtractionError(ValueError):
    """A PDF cannot supply usable article text."""


@dataclass(slots=True)
class _Line:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    size: float
    bold: bool = False
    region: int = 0
    kind: str = "paragraph"


@dataclass(slots=True)
class _Page:
    number: int
    width: float
    height: float
    lines: list[_Line]


@dataclass(slots=True)
class _SidebarContinuation:
    """Evidence that a small-font administrative region continues on the next page."""

    side: str
    x0: float
    x1: float
    font_ratio: float


_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table)\s+\d+[A-Za-z]?(?:\b|[.:|])", re.I)
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?[*.]?$", re.I)


def _clean_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).replace("\u00ad", "").replace("\x00", "").split())


def _metadata_label(text: str) -> str | None:
    heading = classify_section_heading(text)
    return heading if heading and is_frontmatter_section(heading) else None


def _body_size(lines: list[_Line]) -> float:
    """Estimate the prose font by character mass instead of heading count."""

    counts: Counter[float] = Counter()
    for line in lines:
        counts[round(line.size, 1)] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 11.0


def _join_words(words: list[dict[str, Any]]) -> _Line:
    words.sort(key=lambda word: float(word["x0"]))
    text = ""
    previous: dict[str, Any] | None = None
    for word in words:
        value = _clean_text(str(word["text"]))
        gap = float(word["x0"]) - float(previous["x1"]) if previous else 0.0
        # Font changes can split a word (CD4+, Greek letters, superscripts).
        # Actual inter-word spaces, including tight journal typesetting, scale
        # with the font; a fixed pixel tolerance merges words in small fonts.
        separated = previous is not None and gap > min(float(word["size"]), float(previous["size"])) * 0.08
        text += (" " if separated else "") + value
        previous = word
    sizes = [float(word["size"]) for word in words]
    substantive = [word for word in words if any(char.isalpha() for char in str(word["text"]))]
    return _Line(
        text=_clean_text(text), x0=min(float(word["x0"]) for word in words),
        top=min(float(word["top"]) for word in words),
        x1=max(float(word["x1"]) for word in words),
        bottom=max(float(word["bottom"]) for word in words), size=median(sizes),
        bold=bool(substantive) and all(
            any(marker in str(word.get("fontname", "")).casefold() for marker in ("bold", "black", "heavy", "demi"))
            for word in substantive
        ),
    )


def _page_words(page: Any) -> list[dict[str, Any]]:
    """Read positioned glyphs directly, without image-rendering dependencies."""

    from pdfminer.layout import LTChar, LTContainer

    glyphs: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], list[tuple[float, float]]] = {}

    def collect(item: Any) -> None:
        if isinstance(item, LTChar):
            x0, x1 = float(item.x0), float(item.x1)
            top, bottom = float(page.height - item.y1), float(page.height - item.y0)
            size = float(item.size)
            if not all(math.isfinite(value) for value in (x0, x1, top, bottom, size)):
                raise PDFExtractionError("PDF text has invalid layout coordinates.")
            text = item.get_text()
            font = item.fontname
            if isinstance(font, bytes):
                font = font.decode("utf-8", errors="replace")
            # Overpainted glyphs are common in bold/scanned-text overlays. Keep
            # one occurrence of the same glyph within half a PDF point.
            cell_x, cell_y = math.floor(x0 / 0.5), math.floor(top / 0.5)
            prefix = (text, font, size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if any(abs(x0 - x) <= 0.5 and abs(top - y) <= 0.5
                           for x, y in seen.get((*prefix, cell_x + dx, cell_y + dy), ())):
                        return
            seen.setdefault((*prefix, cell_x, cell_y), []).append((x0, top))
            glyphs.append({"text": text, "fontname": font, "size": size,
                           "x0": x0, "x1": x1, "top": top, "bottom": bottom})
        elif isinstance(item, LTContainer):
            for child in item:
                collect(child)

    collect(page)
    fonts: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for glyph in glyphs:
        fonts.setdefault((glyph["fontname"], glyph["size"]), []).append(glyph)

    words: list[dict[str, Any]] = []

    def append_word(chars: list[dict[str, Any]]) -> None:
        if chars:
            words.append({
                "text": "".join(char["text"] for char in chars),
                "fontname": chars[0]["fontname"], "size": chars[0]["size"],
                "x0": min(char["x0"] for char in chars),
                "x1": max(char["x1"] for char in chars),
                "top": min(char["top"] for char in chars),
                "bottom": max(char["bottom"] for char in chars),
            })

    for chars in fonts.values():
        rows: list[list[dict[str, Any]]] = []
        for char in sorted(chars, key=lambda value: value["top"]):
            if rows and char["top"] - rows[-1][-1]["top"] <= 3:
                rows[-1].append(char)
            else:
                rows.append([char])
        for row in rows:
            word: list[dict[str, Any]] = []
            for char in sorted(row, key=lambda value: value["x0"]):
                if char["text"].isspace():
                    append_word(word)
                    word = []
                    continue
                if word and char["x0"] - word[-1]["x1"] > 0.12 * word[-1]["size"]:
                    append_word(word)
                    word = []
                word.append(char)
            append_word(word)
    return words


def _extract_page(page: Any, number: int) -> _Page:
    """Build article lines from the PDF's actual glyph positions and fonts."""

    width, height = float(page.width), float(page.height)
    words = _page_words(page)
    valid_words = []
    for word in words:
        if not _clean_text(str(word.get("text", ""))):
            continue
        values = [float(word[key]) for key in ("x0", "x1", "top", "bottom", "size")]
        if not all(math.isfinite(value) for value in values):
            raise PDFExtractionError(f"PDF text has invalid layout coordinates on page {number}.")
        valid_words.append(word)

    rows: list[tuple[float, list[dict[str, Any]]]] = []
    for word in sorted(valid_words, key=lambda value: (float(value["top"]), float(value["x0"]))):
        middle = (float(word["top"]) + float(word["bottom"])) / 2
        candidates = [
            row for row in rows[-6:]
            if abs(middle - row[0]) <= max(float(word["size"]), median(float(item["size"]) for item in row[1])) * 0.5
        ]
        if candidates:
            min(candidates, key=lambda row: abs(middle - row[0]))[1].append(word)
        else:
            rows.append((middle, [word]))

    lines: list[_Line] = []
    for _, row in rows:
        row.sort(key=lambda word: float(word["x0"]))
        groups: list[list[dict[str, Any]]] = [[]]
        gap_limit = max(6.0, median(float(word["size"]) for word in row))
        for word in row:
            if groups[-1] and float(word["x0"]) - float(groups[-1][-1]["x1"]) > gap_limit:
                groups.append([])
            groups[-1].append(word)
        lines.extend(_join_words(group) for group in groups)
    return _Page(number, width, height, lines)


def _margin_signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.casefold())


def _without_repeated_margins(pages: list[_Page]) -> list[_Page]:
    """Remove repeated running matter only near the top/bottom of the page."""

    def near_edge(line: _Line, page: _Page) -> bool:
        return line.bottom < page.height * 0.12 or line.top > page.height * 0.90

    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({
            _margin_signature(line.text) for line in page.lines
            if near_edge(line, page) and classify_section_heading(line.text) is None
        })
    for page in pages:
        page.lines = [line for line in page.lines if not (
            near_edge(line, page) and classify_section_heading(line.text) is None and (
                counts[_margin_signature(line.text)] >= 2
                or _PAGE_NUMBER.fullmatch(line.text)
                or re.fullmatch(r"(?:https?://|www\.)\S+", line.text, re.I)
            )
        )]
    return pages


def _gaps(lines: list[_Line], start: str, end: str) -> list[tuple[float, float]]:
    intervals = sorted((getattr(line, start), getattr(line, end)) for line in lines)
    if not intervals:
        return []
    boundary = intervals[0][1]
    gaps = []
    for low, high in intervals[1:]:
        if low > boundary:
            gaps.append((boundary, low))
        boundary = max(boundary, high)
    return gaps


def _partition_regions(lines: list[_Line]) -> list[list[_Line]]:
    """Recursively split whitespace: columns first, then horizontal layout bands.

    A title or wide abstract bridges the body-column gutter, so a page-wide
    two-column guess is unsafe. Horizontal cuts first isolate those spanning
    bands whenever there is no continuous vertical gutter. Gaps and thresholds
    depend on the text's own font size, not a particular publisher's coordinates.
    """

    if not lines:
        return []
    size = _body_size(lines)
    horizontal = sorted(_gaps(lines, "top", "bottom"), key=lambda gap: gap[1] - gap[0], reverse=True)
    # A large full-width band (often a figure) separates body columns from
    # captions. Finish both columns above it before reading either below it.
    if horizontal and horizontal[0][1] - horizontal[0][0] > size * 2.0:
        low, high = horizontal[0]
        middle = (low + high) / 2
        above = [line for line in lines if line.bottom < middle]
        below = [line for line in lines if line.top > middle]
        if above and below:
            return _partition_regions(above) + _partition_regions(below)
    vertical = sorted(_gaps(lines, "x0", "x1"), key=lambda gap: gap[1] - gap[0], reverse=True)
    for low, high in vertical:
        if high - low < max(6.0, size * 0.75):
            continue
        middle = (low + high) / 2
        left = [line for line in lines if line.x1 < middle]
        right = [line for line in lines if line.x0 > middle]
        if len(left) < 2 or len(right) < 2:
            continue
        overlap = min(max(line.bottom for line in left), max(line.bottom for line in right)) - max(
            min(line.top for line in left), min(line.top for line in right),
        )
        if overlap > size:
            return _partition_regions(left) + _partition_regions(right)
    horizontal = sorted(_gaps(lines, "top", "bottom"), key=lambda gap: gap[1] - gap[0], reverse=True)
    if horizontal and horizontal[0][1] - horizontal[0][0] > max(7.0, size * 0.9):
        low, high = horizontal[0]
        middle = (low + high) / 2
        above = [line for line in lines if line.bottom < middle]
        below = [line for line in lines if line.top > middle]
        if above and below:
            return _partition_regions(above) + _partition_regions(below)
    return [sorted(lines, key=lambda line: (line.top, line.x0))]


def _styled_heading(line: _Line, body_size: float) -> bool:
    words = line.text.split()
    if len(words) > 22 or len(line.text) > 180 or re.search(r"[.!?;,]$", line.text):
        return False
    if not line.bold or line.size < body_size * 0.95:
        return False
    # A numbered citation in References must never reopen scientific content.
    return not re.match(r"^\s*\d+[.)]\s+", line.text) and not _metadata_label(line.text)


def _coalesce_headings(lines: list[_Line]) -> list[_Line]:
    """Recognize a short heading wrapped onto two physical lines."""

    result: list[_Line] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if index + 1 < len(lines) and classify_section_heading(line.text) is None:
            following = lines[index + 1]
            combined = f"{line.text} {following.text}"
            if (
                len(line.text.split()) <= 6 and len(following.text.split()) <= 6
                and abs(line.x0 - following.x0) <= max(line.size, following.size)
                and following.top - line.bottom <= max(line.size, following.size)
                and classify_section_heading(combined) is not None
            ):
                result.append(replace(
                    line, text=combined, x1=max(line.x1, following.x1),
                    bottom=following.bottom,
                ))
                index += 2
                continue
        result.append(line)
        index += 1
    return result


def _without_metadata_sidebars(
    page: _Page,
    regions: list[list[_Line]],
    previous: list[_SidebarContinuation],
) -> tuple[list[list[_Line]], list[_SidebarContinuation]]:
    """Bound administrative state to a geometrically separate small-font column.

    Horizontal whitespace can divide one sidebar into several regions. An
    excluded section in that sidebar must neither leak into the main column nor
    lose its own continuation at a region boundary. Across pages, continuation
    requires matching normalized geometry/typography and an unfinished sentence
    at the bottom of the preceding page. A new scientific heading resets it.
    """

    body_size = _body_size(page.lines)
    wide_regions = [
        region for region in regions
        if max(line.x1 for line in region) - min(line.x0 for line in region) >= page.width * 0.45
        and _body_size(region) >= body_size * 0.95
    ]
    groups: list[tuple[str, list[int]]] = []
    for index, region in enumerate(regions):
        low, high = min(line.x0 for line in region), max(line.x1 for line in region)
        top, bottom = min(line.top for line in region), max(line.bottom for line in region)
        if high - low >= page.width * 0.38 or _body_size(region) > body_size * 0.9:
            continue
        side = None
        for main in wide_regions:
            main_low, main_high = min(line.x0 for line in main), max(line.x1 for line in main)
            # Whitespace may split the main column too. Its paragraph gaps do
            # not turn intervening lines of the separate sidebar into prose.
            aligned_main = [
                line for block in wide_regions
                if abs(min(item.x0 for item in block) - main_low) <= body_size * 2
                for line in block
            ]
            overlap = min(bottom, max(line.bottom for line in aligned_main)) - max(top, min(line.top for line in aligned_main))
            if overlap <= 0:
                continue
            if high < main_low:
                side = "left"
            elif low > main_high:
                side = "right"
            if side:
                break
        if side is None:
            continue
        for group_side, indices in groups:
            existing = [line for region_index in indices for line in regions[region_index]]
            group_low, group_high = min(line.x0 for line in existing), max(line.x1 for line in existing)
            if group_side == side and min(high, group_high) - max(low, group_low) > min(high - low, group_high - group_low) * 0.4:
                indices.append(index)
                break
        else:
            groups.append((side, [index]))

    removed: set[int] = set()
    following: list[_SidebarContinuation] = []
    page_bottom = max((line.bottom for line in page.lines), default=0.0)
    for side, indices in groups:
        lines = sorted(
            [line for index in indices for line in regions[index]],
            key=lambda line: (line.top, line.x0),
        )
        low = min(line.x0 for line in lines) / page.width
        high = max(line.x1 for line in lines) / page.width
        size = _body_size(lines)
        font_ratio = size / body_size
        inherited = lines[0].top < page.height * 0.25 and any(
            state.side == side and abs(state.x0 - low) < 0.03
            and abs(state.x1 - high) < 0.04
            and abs(state.font_ratio - font_ratio) < 0.12
            for state in previous
        )
        excluded = inherited
        for line in lines:
            heading = classify_section_heading(line.text)
            if heading and is_excluded_section(heading):
                excluded = True
            elif (heading and not is_excluded_section(heading)) or _styled_heading(line, size):
                excluded = False
            if excluded:
                removed.add(id(line))
        # A complete metadata paragraph at mid-page is not evidence that the
        # same geometric column will remain administrative on the next page.
        if (
            excluded and lines[-1].bottom >= page.height * 0.8
            and page_bottom - lines[-1].bottom <= size * 2
            and not re.search(r"[.!?][\"'’”)]?$", lines[-1].text)
        ):
            following.append(_SidebarContinuation(side, low, high, font_ratio))
    retained = [[line for line in region if id(line) not in removed] for region in regions]
    return [region for region in retained if region], following


def _without_local_metadata(page: _Page, regions: list[list[_Line]]) -> list[_Line]:
    """Discard labelled front matter and continuation within its own region.

    Metadata does not consume an adjacent abstract, and Keywords does not carry
    exclusion into a following body column. Wider prose is retained if evidence
    for a metadata continuation ends at a clear heading or physical paragraph.
    """

    kept: list[_Line] = []
    body_size = _body_size(page.lines)
    for region_index, region in enumerate(regions):
        region = _coalesce_headings(region)
        excluded = False
        previous: _Line | None = None
        labels = [classify_section_heading(line.text) for line in region]
        sidebar = (
            max(line.x1 for line in region) - min(line.x0 for line in region) < page.width * 0.38
            and sum(bool(label and is_excluded_section(label)) for label in labels) >= 2
        )
        for line in region:
            label = _metadata_label(line.text)
            heading = classify_section_heading(line.text)
            scientific_heading = (heading is not None and not is_excluded_section(heading)) or _styled_heading(line, body_size)
            paragraph_gap = previous is not None and line.top - previous.bottom > max(previous.size, line.size) * 1.4
            if scientific_heading or (paragraph_gap and not sidebar) or (
                previous is not None and line.size > previous.size * 1.2
                and len(line.text.split()) >= 2 and not label
            ):
                excluded = False
            if label or (sidebar and heading is not None and is_excluded_section(heading)):
                excluded = True
            if not excluded:
                kept.append(replace(line, region=region_index))
            previous = line
    return kept


def _looks_like_prose(lines: list[_Line]) -> bool:
    text = " ".join(line.text for line in lines)
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 12:
        return False
    ordinary_words = sum(word[0].islower() for word in words)
    return ordinary_words >= len(words) * 0.6 and (len(words) >= 24 or bool(re.search(r"[.!?](?:\s|$)", text)))


def _without_author_frontmatter(page: _Page) -> _Page:
    """Preserve a prominent title and an explicit or confidently inferred abstract.

    An abstract need not contain an 'Abstract' heading. We identify a prose block
    after a large title using typography/paragraph evidence; ambiguous preambles
    are retained rather than deleting potentially scientific text.
    """

    lines = page.lines
    if not lines:
        return page
    body_size = _body_size(lines)
    candidates = [
        line for line in lines if line.top < page.height * 0.45
        and line.size >= body_size * 1.25
        and len(line.text.split()) >= 2
        and classify_section_heading(line.text) is None
        and _metadata_label(line.text) is None
    ]
    if not candidates:
        return page
    largest = max(line.size for line in candidates)
    title_lines = [line for line in candidates if line.size >= largest * 0.95]
    title_lines.sort(key=lambda line: (line.top, line.x0))
    # Only a contiguous group at the first large title is part of that title.
    title_group = [title_lines[0]]
    for line in title_lines[1:]:
        if line.top - title_group[-1].bottom > largest * 1.4:
            break
        title_group.append(line)
    title_bottom = max(line.bottom for line in title_group)
    after = [line for line in lines if line.top >= title_bottom - 0.1]
    title = replace(
        title_group[0], text=" ".join(line.text for line in title_group),
        x1=max(line.x1 for line in title_group), bottom=title_bottom, kind="title",
    )
    explicit = next((index for index, line in enumerate(after) if classify_section_heading(line.text) in {"abstract", "summary"}), None)
    start: int | None = explicit
    abstract_start: int | None = None
    if start is None:
        for index, line in enumerate(after):
            heading = classify_section_heading(line.text)
            if (heading and not is_excluded_section(heading)) or _styled_heading(line, body_size):
                # Ignore the common bold author byline: it is not all capitals
                # and usually shares the title's upper preamble band.
                if heading or line.text.isupper():
                    start = index
                    break
            if line.bold or line.size < body_size * 0.85:
                continue
            run = [line]
            for following in after[index + 1:index + 5]:
                if following.region != line.region or following.bold or abs(following.size - line.size) > line.size * 0.12:
                    break
                run.append(following)
            if _looks_like_prose(run):
                start = index
                abstract_start = index
                break
    if start is None:
        # A clear title still lets us discard the publication banner above it.
        page.lines = [title, *after]
        return page
    retained = after[start:]
    if abstract_start is not None:
        abstract_region = after[abstract_start].region
        for index, line in enumerate(retained):
            if line.region != abstract_region or classify_section_heading(line.text) is not None or _styled_heading(line, body_size):
                break
            retained[index] = replace(line, kind="abstract")
    page.lines = [title, *retained]
    return page


def _paragraphs(pages: list[_Page]) -> list[BioCPassage]:
    passages: list[BioCPassage] = []
    offset = 0
    excluded = False

    def append(text: str, page_number: int, kind: str) -> None:
        nonlocal offset
        if text:
            passages.append(BioCPassage(
                infons={"type": kind, "page_number": str(page_number)},
                offset=offset, text=text,
            ))
            offset += len(text) + 1

    def flush(buffer: list[str], page_number: int, kind: str) -> None:
        text = ""
        for value in buffer:
            text += ("" if not text or text.endswith("-") else " ") + value
        append(text, page_number, kind)
        buffer.clear()

    for page in pages:
        size = _body_size(page.lines)
        buffer: list[str] = []
        buffer_kind = "paragraph"
        previous: _Line | None = None

        for line in page.lines:
            heading = classify_section_heading(line.text)
            is_heading = heading is not None or (_styled_heading(line, size) and not excluded)
            if line.kind == "title" or is_heading:
                flush(buffer, page.number, buffer_kind)
                append(line.text, page.number, "title" if line.kind == "title" else "section_title")
                if heading is not None:
                    excluded = is_excluded_section(heading)
                previous = line
                continue
            kind = line.kind
            if (
                kind == "paragraph" and buffer_kind in {"figure_caption", "table_caption"}
                and buffer and previous is not None and line.region == previous.region
                and line.top - previous.bottom <= max(previous.size, line.size) * 0.8
            ):
                kind = buffer_kind
            if _CAPTION.match(line.text):
                kind = "figure_caption" if not line.text.casefold().startswith("table") else "table_caption"
            if previous is not None and (
                line.region != previous.region or line.top - previous.bottom > max(previous.size, line.size) * 0.8
                or kind != buffer_kind or _CAPTION.match(line.text)
            ):
                flush(buffer, page.number, buffer_kind)
            if not buffer:
                buffer_kind = kind
            buffer.append(line.text)
            previous = line
        flush(buffer, page.number, buffer_kind)
    return passages


def read_pdf_collection(path: str | Path, *, document_id: str | None = None) -> BioCCollection:
    """Extract a local selectable-text PDF and apply shared article exclusions.

    Real glyph positions determine reading order and metadata boundaries. Titles,
    labelled/unlabelled abstracts, article prose, and figure/table captions are
    retained. Author/editor/reviewer/correspondence blocks and keyword lists are
    removed before model processing. Arbitrary PDF layouts remain heuristic;
    scans require OCR. No document is sent to an external service.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise PDFExtractionError(f"PDF file not found: {source}")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFExtractionError("PDF support requires pypdf. Install it with: pip install pypdf") from exc
    try:
        from pdfminer.high_level import extract_pages
    except ImportError as exc:
        raise PDFExtractionError("PDF support requires pdfminer.six. Install it with: pip install pdfminer.six") from exc

    try:
        with source.open("rb") as stream:
            if not stream.read(1024).lstrip().startswith(b"%PDF-"):
                raise PDFExtractionError(f"Invalid PDF file: {source.name}")
            stream.seek(0)
            reader = PdfReader(stream, strict=False)
            if reader.is_encrypted and not reader.decrypt(""):
                raise PDFExtractionError(f"PDF is password-protected: {source.name}")
        pages = [_extract_page(page, index + 1)
                 for index, page in enumerate(extract_pages(source, password=""))]
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"Unable to read PDF: {source.name}") from exc

    if not any(any(character.isalpha() for character in line.text) for page in pages for line in page.lines):
        raise PDFExtractionError(f"PDF text not available: {source.name}. Image-only PDFs require OCR.")
    sidebars: list[_SidebarContinuation] = []
    for page in _without_repeated_margins(pages):
        regions = [_coalesce_headings(region) for region in _partition_regions(page.lines)]
        regions, sidebars = _without_metadata_sidebars(page, regions, sidebars)
        page.lines = _without_local_metadata(page, regions)
    if pages:
        _without_author_frontmatter(pages[0])
    collection = BioCCollection(
        source="PDF", key="PDF selectable text converted to BioC",
        documents=[BioCDocument(
            id=document_id if document_id is not None else source.stem,
            infons={"source_file": source.name, "input_format": "pdf", "offsets": "extracted_text"},
            passages=_paragraphs(pages),
        )],
    )
    filter_collection_content(collection)
    if not any(
        any(character.isalpha() for character in passage.text)
        for document in collection.documents for passage in document.passages
    ):
        raise PDFExtractionError(f"PDF contains no usable article text: {source.name}")
    return collection


__all__ = ["PDFExtractionError", "read_pdf_collection"]
