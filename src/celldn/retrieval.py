"""Retrieve validated article text with per-request retries and per-ID fallback.

The preferred provider and its alternate are tried for the same content mode
before a full-text request falls back to an abstract. Retrieval is independent
of prediction chunk sizes; only validated article text enters the output.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from celldn.content import filter_collection_content, is_excluded_section
from celldn.io import (
    BioCCollection,
    BioCDocument,
    BioCPassage,
    PathLike,
    canonical_bioc_format,
    clean_pmid_list,
    merge_bioc_collections,
    read_bioc_collection_from_string,
    write_bioc_collection,
)

UrlOpener = Callable[[Request, int], Any]
_Result = TypeVar("_Result")

NCBI_BIOC_PUBMED = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi"
NCBI_BIOC_PMCOA = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
NCBI_ID_CONVERTER = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EUROPEPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_MAX_ATTEMPTS = 3
_MAX_RETRY_WAIT = 30.0
_MIN_REQUEST_INTERVAL = 0.34
_REQUEST_LOCK = Lock()
_last_request_started: float | None = None
_host_retry_after: dict[str, float] = {}


class RetrievalError(RuntimeError):
    """An expected request or invalid-provider-response failure."""

    def __init__(self, message: str, *, retryable: bool = False, retry_after: float = 0.0):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class _UnavailableText(RetrievalError):
    """A valid provider response confirms that the requested text is absent."""


class TextUnavailableError(RuntimeError):
    """No requested article text could be retrieved after the allowed attempts."""


@dataclass(slots=True)
class FetchReport:
    """Retrieval outcomes, with request failures distinct from missing content.

    ``source`` is the preferred provider. ``text_sources`` records the actual
    content mode obtained for each ID. ``attempt_errors`` keeps failed provider
    details by mode even when a later provider or abstract fallback succeeds.
    """

    output_path: Path
    requested_ids: list[str]
    fetched_ids: list[str]
    failed_ids: dict[str, str] = field(default_factory=dict)
    source: str = "ncbi"
    text_source: str = "abstract"
    output_format: str = "bioc-xml"
    fallback_ids: list[str] = field(default_factory=list)
    text_sources: dict[str, str] = field(default_factory=dict)
    attempt_errors: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every requested record was fetched successfully."""

        return not self.failed_ids

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly version of the fetch summary."""

        return {
            "output_path": str(self.output_path),
            "requested_ids": list(self.requested_ids),
            "fetched_ids": list(self.fetched_ids),
            "failed_ids": dict(self.failed_ids),
            "source": self.source,
            "text_source": self.text_source,
            "output_format": self.output_format,
            "fallback_ids": list(self.fallback_ids),
            "text_sources": dict(self.text_sources),
            "attempt_errors": {
                record_id: dict(errors) for record_id, errors in self.attempt_errors.items()
            },
        }


class _ArticleClient:
    """Bounded HTTP retries and metadata reuse within one retrieval call."""

    def __init__(self, *, timeout: int, pause: float, opener: UrlOpener | None):
        self.timeout = timeout
        self.interval = max(pause, _MIN_REQUEST_INTERVAL)
        self.opener = opener
        self.europe_metadata: dict[str, dict[str, Any]] = {}

    def get(self, url: str, parse: Callable[[str], _Result]) -> _Result:
        retry_delay = 0.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _pace_request(url, self.interval, retry_delay)
            try:
                return parse(_urlopen_text(url, timeout=self.timeout, opener=self.opener))
            except _UnavailableText:
                raise
            except RetrievalError as exc:
                _remember_retry_after(url, exc.retry_after)
                if (
                    not exc.retryable
                    or attempt == _MAX_ATTEMPTS
                    or exc.retry_after > _MAX_RETRY_WAIT
                ):
                    suffix = "attempt" if attempt == 1 else "attempts"
                    raise RetrievalError(
                        f"request failed after {attempt} {suffix}: {exc}"
                    ) from exc
                retry_delay = max(0.5 * 2 ** (attempt - 1), exc.retry_after)
        raise AssertionError("HTTP retry loop did not return or raise")


def _remember_retry_after(url: str, seconds: float) -> None:
    """Keep the server's host deadline even when this request has no retries left."""

    if seconds <= 0:
        return
    host = urlsplit(url).netloc
    with _REQUEST_LOCK:
        _host_retry_after[host] = max(
            _host_retry_after.get(host, 0.0), time.monotonic() + seconds
        )


def _pace_request(url: str, interval: float, retry_delay: float) -> None:
    """Respect request spacing and host cooldowns across modes, IDs, and calls."""

    global _last_request_started
    host = urlsplit(url).netloc
    with _REQUEST_LOCK:
        now = time.monotonic()
        cooldown = max(0.0, _host_retry_after.get(host, 0.0) - now)
        if cooldown > _MAX_RETRY_WAIT:
            raise RetrievalError(
                f"Retry-After for {host} requires another {cooldown:.0f} seconds; "
                "provider deferred without making a request"
            )
        delay = max(retry_delay, cooldown)
        if _last_request_started is not None:
            delay = max(delay, interval - (now - _last_request_started))
        if delay > 0:
            time.sleep(delay)
        _last_request_started = time.monotonic()


def read_ids_file(path: PathLike) -> list[str]:
    """Read PMIDs/PMCIDs separated by commas, spaces, or newlines."""

    return clean_pmid_list(Path(path).read_text(encoding="utf-8"))


def combine_ids(ids: Iterable[str] | str | None = None, ids_file: PathLike | None = None) -> list[str]:
    """Combine inline IDs and a file into a de-duplicated list."""

    combined: list[str] = []
    if ids is not None:
        combined.extend(clean_pmid_list(ids))
    if ids_file is not None:
        combined.extend(read_ids_file(ids_file))
    return clean_pmid_list(combined)


def fetch_pubmed_bioc(
    ids: Iterable[str] | str,
    output_path: PathLike,
    *,
    text_source: str = "abstract",
    output_format: str = "bioc-xml",
    source: str = "ncbi",
    timeout: int = 30,
    pause: float = 0.12,
    opener: UrlOpener | None = None,
) -> FetchReport:
    """Retrieve each ID independently and write its validated article text.

    ``source`` selects the preferred provider; the other provider is tried for
    the same content mode before falling back from full text to an abstract.
    Transient network, HTTP, and malformed-response failures get at most three
    attempts per request. A valid response with no usable text is not retried.
    Every request is spaced by at least 0.34 seconds, or ``pause`` if larger.

    Reports distinguish unavailable text from exhausted request failures and
    retain successful records in input order. Configuration and programming
    errors propagate to the caller instead of being reported as missing text.
    """

    id_list = clean_pmid_list(ids)
    if not id_list:
        raise ValueError("At least one PMID/PMCID is required.")
    normalized_text_source = _canonical_text_source(text_source)
    normalized_source = _canonical_source(source)
    fmt = canonical_bioc_format(output_format)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive number.")
    if not math.isfinite(pause) or pause < 0:
        raise ValueError("pause must be a non-negative number.")
    client = _ArticleClient(timeout=timeout, pause=pause, opener=opener)
    output = Path(output_path).expanduser()
    report = FetchReport(
        output_path=output,
        requested_ids=id_list,
        fetched_ids=[],
        source=normalized_source,
        text_source=normalized_text_source,
        output_format=fmt,
    )
    collections: list[BioCCollection] = []
    modes = ("fulltext", "abstract") if normalized_text_source == "fulltext" else ("abstract",)
    providers = (normalized_source, "europepmc" if normalized_source == "ncbi" else "ncbi")

    for record_id in id_list:
        try:
            normalized_id = _normalize_record_id(record_id)
        except RetrievalError as exc:
            report.failed_ids[record_id] = f"Invalid identifier: {exc}"
            continue
        errors: dict[str, str] = {}
        selected: BioCCollection | None = None
        for mode in modes:
            provider_errors: list[str] = []
            for provider in providers:
                try:
                    if provider == "ncbi":
                        selected = _fetch_ncbi_bioc_record(
                            normalized_id, text_source=mode, output_format=fmt, client=client
                        )
                    else:
                        selected = _fetch_europepmc_record(
                            normalized_id, text_source=mode, client=client
                        )
                except _UnavailableText as exc:
                    provider_errors.append(f"{provider}: unavailable: {exc}")
                except RetrievalError as exc:
                    detail = str(exc)
                    if not detail.startswith("request failed"):
                        detail = f"request failed: {detail}"
                    provider_errors.append(f"{provider}: {detail}")
                else:
                    break
            if provider_errors:
                errors[mode] = "; ".join(provider_errors)
            if selected is None:
                continue
            for document in selected.documents:
                document.infons["celldn_requested_id"] = record_id
                document.infons["celldn_requested_text_source"] = normalized_text_source
                document.infons["celldn_text_source"] = mode
            collections.append(selected)
            report.fetched_ids.append(record_id)
            report.text_sources[record_id] = mode
            if mode != normalized_text_source:
                report.fallback_ids.append(record_id)
            break
        if errors:
            report.attempt_errors[record_id] = errors
        if selected is None:
            report.failed_ids[record_id] = "; ".join(
                f"{mode}: {detail}" for mode, detail in errors.items()
            )

    merged = merge_bioc_collections(collections) if collections else BioCCollection(
        source="CellDN retrieval", key="empty retrieval result"
    )
    write_bioc_collection(merged, output, output_format=fmt)
    return report


def _fetch_ncbi_bioc_record(
    record_id: str, *, text_source: str, output_format: str, client: _ArticleClient
) -> BioCCollection:
    """Fetch BioC and verify both the returned identifier and requested text."""

    if text_source == "abstract" and record_id.startswith("PMC"):
        record_id = _resolve_pmcid_to_pmid(record_id, client=client)
    base = NCBI_BIOC_PUBMED if text_source == "abstract" else NCBI_BIOC_PMCOA
    export_fmt = "BioC_json" if output_format == "bioc-json" else "BioC_xml"
    url = f"{base}/{export_fmt}/{record_id}/unicode"

    def parse(payload: str) -> BioCCollection:
        if payload.lstrip().lower().startswith("[error]") and "no result can be found" in payload.lower():
            raise _UnavailableText("NCBI returned no matching record")
        try:
            if output_format == "bioc-json":
                data = json.loads(payload)
                groups = data if isinstance(data, list) else [data]
                if not all(
                    isinstance(item, dict) and isinstance(item.get("documents"), list)
                    for item in groups
                ):
                    raise ValueError("expected BioC collections with document lists")
                if any(
                    not isinstance(document, dict)
                    or not isinstance(document.get("passages", []), list)
                    for item in groups for document in item["documents"]
                ):
                    raise ValueError("expected BioC documents with passage lists")
                collection = merge_bioc_collections(
                    read_bioc_collection_from_string(json.dumps(item), input_format=output_format)
                    for item in groups
                )
            else:
                if _local_name(ET.fromstring(payload.encode("utf-8")).tag) != "collection":
                    raise ValueError("expected a BioC collection")
                collection = read_bioc_collection_from_string(payload, input_format=output_format)
        except (ET.ParseError, ValueError, LookupError) as exc:
            raise RetrievalError(f"NCBI returned invalid BioC content: {exc}", retryable=True) from exc
        if collection.documents:
            matching = [
                document for document in collection.documents
                if _ncbi_document_matches(document, record_id)
            ]
            if not matching:
                raise RetrievalError(
                    "NCBI returned a different article identifier", retryable=True
                )
            collection.documents = matching
        filter_collection_content(collection)
        _validate_collection_text(collection, text_source=text_source)
        return collection

    return client.get(url, parse)


def _ncbi_document_matches(document: BioCDocument, record_id: str) -> bool:
    """NCBI full-text document IDs can be PMC numbers; compare its PMID infons."""

    values = [document.infons, *(passage.infons for passage in document.passages)]
    pmc = record_id.startswith("PMC")
    keys = ("pmcid", "article-id_pmc") if pmc else ("pmid", "article-id_pmid")
    declared = {
        str(infons[key]).strip().upper()
        for infons in values for key in keys if infons.get(key)
    }
    if pmc:
        declared = {value if value.startswith("PMC") else f"PMC{value}" for value in declared}
    if declared:
        return record_id in declared
    identifier = str(document.id).strip().upper()
    return identifier == record_id or (pmc and identifier == record_id[3:])


def _resolve_pmcid_to_pmid(pmcid: str, *, client: _ArticleClient) -> str:
    """Resolve PMC IDs explicitly; their numeric suffix is never a PubMed ID."""

    params = {"ids": pmcid, "idtype": "pmcid", "format": "json", "tool": "CellDN"}

    def parse(payload: str) -> str:
        metadata = _parse_json_object(payload, service="NCBI ID converter")
        records = metadata.get("records")
        if not isinstance(records, list):
            raise RetrievalError("NCBI ID converter returned an invalid records list", retryable=True)
        for row in records:
            if not isinstance(row, dict):
                continue
            if pmcid not in {
                str(row.get("pmcid") or "").upper(),
                str(row.get("requested-id") or "").upper(),
            }:
                continue
            pmid = str(row.get("pmid") or "").strip()
            if re.fullmatch(r"[0-9]+", pmid):
                return pmid
        raise _UnavailableText(f"NCBI ID converter found no corresponding PMID for {pmcid}")

    return client.get(f"{NCBI_ID_CONVERTER}?{urlencode(params)}", parse)


def _fetch_europepmc_record(
    record_id: str, *, text_source: str, client: _ArticleClient
) -> BioCCollection:
    row = _fetch_europepmc_metadata(record_id, client=client)
    document_id = str(row.get("pmid") or row.get("id") or record_id)
    if text_source == "abstract":
        title = _plain_text(row.get("title"))
        abstract = _plain_text(row.get("abstractText"))
        if not _has_usable_text(abstract):
            raise _UnavailableText("Europe PMC record has no usable abstract")
        passages = []
        if title:
            passages.append(BioCPassage(infons={"type": "title"}, offset=0, text=title))
        passages.append(BioCPassage(
            infons={"type": "abstract"}, offset=len(title) + 1 if title else 0, text=abstract
        ))
        collection = BioCCollection(
            source="Europe PMC",
            key="Europe PMC title/abstract",
            documents=[BioCDocument(
                id=document_id, infons={"source": str(row.get("source") or "MED")}, passages=passages
            )],
        )
        filter_collection_content(collection)
        _validate_collection_text(collection, text_source="abstract")
        return collection
    pmcid = str(row.get("pmcid") or "").strip().upper()
    if not pmcid and record_id.startswith("PMC"):
        pmcid = record_id
    if not pmcid:
        raise _UnavailableText("Europe PMC record has no PMCID; full text is unavailable")
    if re.fullmatch(r"PMC[0-9]+", pmcid) is None:
        raise RetrievalError("Europe PMC record contained an invalid PMCID")
    return client.get(
        f"{EUROPEPMC_REST}/{pmcid}/fullTextXML",
        lambda payload: _jats_xml_to_bioc(payload, document_id=document_id, pmcid=pmcid),
    )


def _fetch_europepmc_metadata(record_id: str, *, client: _ArticleClient) -> dict[str, Any]:
    if record_id in client.europe_metadata:
        return client.europe_metadata[record_id]
    query = f"PMCID:{record_id}" if record_id.startswith("PMC") else f"EXT_ID:{record_id} AND SRC:MED"
    params = {"query": query, "format": "json", "resultType": "core"}

    def parse(payload: str) -> dict[str, Any]:
        metadata = _parse_json_object(payload, service="Europe PMC")
        results = metadata.get("resultList")
        if not isinstance(results, dict) or not isinstance(results.get("result"), list):
            raise RetrievalError("Europe PMC search returned an invalid result list", retryable=True)
        rows = results["result"]
        if not rows:
            raise _UnavailableText("Europe PMC search returned no matching record")
        matches = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if record_id.startswith("PMC"):
                identifier = row.get("pmcid") or (row.get("id") if row.get("source") == "PMC" else "")
            else:
                identifier = row.get("pmid") or (row.get("id") if row.get("source") == "MED" else "")
            if str(identifier).strip().upper() == record_id:
                matches.append(row)
        if not matches:
            raise RetrievalError("Europe PMC returned a different article identifier", retryable=True)
        return max(matches, key=lambda row: (
            _has_usable_text(_plain_text(row.get("abstractText"))), row.get("source") == "MED"
        ))

    row = client.get(f"{EUROPEPMC_REST}/search?{urlencode(params)}", parse)
    client.europe_metadata[record_id] = row
    return row


def _jats_xml_to_bioc(xml_text: str, *, document_id: str, pmcid: str) -> BioCCollection:
    """Convert JATS text with the same content rules as NCBI BioC and PDF.

    Section context is retained while visiting XML blocks. A paragraph is
    consumed once, so nested inline text does not become duplicate passages.
    Images and external supplementary attachments are not downloaded.
    """

    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except (ET.ParseError, ValueError, LookupError) as exc:
        raise RetrievalError(f"Europe PMC returned invalid full-text XML: {exc}", retryable=True) from exc
    for node in root.iter():
        node.tag = _local_name(node.tag)
    if root.tag != "article":
        article = root.find(".//article")
        if article is None:
            raise RetrievalError("Europe PMC fullTextXML did not contain a JATS article", retryable=True)
        root = article
    if str(root.get("annotatable", "true")).strip().casefold() in {"false", "0", "no"}:
        raise _UnavailableText("Europe PMC fullTextXML contained no usable article body")
    passages: list[BioCPassage] = []
    offset = 0

    def add(text: str, kind: str, section: str = "") -> None:
        nonlocal offset
        text = " ".join(text.split())
        if not text:
            return
        infons = {"type": kind}
        if section:
            infons["section_type"] = section
        passages.append(BioCPassage(infons=infons, offset=offset, text=text))
        offset += len(text) + 1

    def text_of(node: ET.Element) -> str:
        return _jats_text(node)

    title = _first_text(root, "./front//article-title")
    add(title, "title", "title")
    abstract_parts = _texts(root, "./front//abstract//p") or _texts(root, "./front//abstract")
    add(" ".join(abstract_parts), "abstract", "abstract")

    excluded_tags = {
        "ack": "acknowledgments", "ref-list": "references", "ref": "references",
        "funding-group": "funding", "funding-statement": "funding",
        "contrib-group": "author contributions", "author-notes": "author contributions",
        "permissions": "copyright", "copyright-statement": "copyright", "license": "license",
    }

    def visit(node: ET.Element, section: str = "body", kind: str = "paragraph") -> None:
        tag = node.tag
        if tag in excluded_tags:
            section = excluded_tags[tag]
        elif not is_excluded_section(section):
            declared_type = node.get("sec-type") or node.get("fn-type") or ""
            if declared_type and is_excluded_section(declared_type):
                section = declared_type
            elif tag in {"sec", "app", "notes", "statement"}:
                heading_node = node.find("title")
                heading_text = text_of(heading_node) if heading_node is not None else ""
                section = heading_text or declared_type or section
        # The shared label predicate applies to structural sections, including
        # excluded ancestors with seemingly scientific nested subsection titles.
        if is_excluded_section(section):
            return
        if str(node.get("annotatable", "true")).strip().casefold() in {"false", "0", "no"}:
            return

        if tag == "fig":
            for child in node:
                visit(child, section, "caption")
            return
        if tag == "table-wrap":
            for child in node:
                visit(child, section, "table")
            return
        if tag == "caption":
            # Captions can contain plain text instead of paragraph elements.
            add(text_of(node), "caption", section)
            return
        if tag == "tr":
            cells = [text_of(cell) for cell in node if cell.tag in {"td", "th"}]
            add(" | ".join(cells), "table", section)
            return
        if tag in {"p", "list-item", "def-item"}:
            add(text_of(node), kind, section)
            return
        if tag == "title":
            add(text_of(node), "section_title", section)
            return
        if tag == "label":
            # Figure/table labels alone are not scientific content.
            return
        for child in node:
            visit(child, section, kind)

    # Validate the article body itself before adding appendices and captions.
    # A title, abstract, or caption without body text is not a full-text article.
    body = root.find("body")
    if body is not None:
        visit(body)
    collection = BioCCollection(
        source="Europe PMC",
        key="Europe PMC fullTextXML converted to BioC",
        documents=[BioCDocument(id=document_id, infons={"pmcid": pmcid}, passages=passages)],
    )
    filter_collection_content(collection)
    try:
        _validate_collection_text(collection, text_source="fulltext")
    except _UnavailableText as exc:
        raise _UnavailableText("Europe PMC fullTextXML contained no usable article body") from exc
    passages = collection.documents[0].passages
    for container_name in ("back", "floats-group"):
        container = root.find(container_name)
        if container is not None:
            visit(container)
    filter_collection_content(collection)
    return collection


def _jats_text(node: ET.Element) -> str:
    """Flatten inline XML without inserting spaces inside words such as CD8+."""

    block_tags = {"p", "title", "list-item", "def-item", "td", "th", "tr", "break"}

    def collect(element: ET.Element) -> str:
        if str(element.get("annotatable", "true")).strip().casefold() in {"false", "0", "no"}:
            return ""
        parts = [element.text or ""]
        for child in element:
            text = collect(child)
            parts.append(f" {text} " if child.tag in block_tags else text)
            parts.append(child.tail or "")
        return "".join(parts)

    return " ".join(collect(node).split())


def _texts(root: ET.Element, xpath: str) -> list[str]:
    """Collect text blocks only when the node and every ancestor allow annotation."""

    nodes = root.findall(xpath)
    parents = {child: parent for parent in root.iter() for child in parent}
    values: list[str] = []
    for node in nodes:
        ancestor: ET.Element | None = node
        while ancestor is not None:
            if str(ancestor.get("annotatable", "true")).strip().casefold() in {"false", "0", "no"}:
                break
            ancestor = parents.get(ancestor)
        else:
            text = _jats_text(node)
            if text:
                values.append(text)
    return values


def _first_text(root: ET.Element, xpath: str) -> str:
    """Return the first text block matched by ``_texts`` or an empty string."""

    values = _texts(root, xpath)
    return values[0] if values else ""


def _local_name(tag: str) -> str:
    """Strip an XML namespace prefix from a tag name when present."""

    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _urlopen_text(url: str, *, timeout: int, opener: UrlOpener | None = None) -> str:
    """Make one HTTP request; the article client handles retries and pacing."""

    request = Request(url, headers={"User-Agent": "CellDN/1.1.0 (+https://github.com/ShahriyariLab/CellDN)"})
    try:
        response = opener(request, timeout) if opener is not None else urlopen(request, timeout=timeout)  # noqa: S310
        with response:
            raw = response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except (OSError, HTTPException):
            detail = ""
        finally:
            exc.close()
        detail = " ".join(detail.split())[:160]
        if exc.code in {404, 410}:
            raise _UnavailableText(f"HTTP {exc.code} for {url}") from exc
        raise RetrievalError(
            f"HTTP {exc.code} for {url}: {detail}",
            retryable=exc.code in {408, 425, 429, 500, 502, 503, 504},
            retry_after=_retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None),
        ) from exc
    except URLError as exc:
        raise RetrievalError(f"Network error for {url}: {exc.reason}", retryable=True) from exc
    except (OSError, HTTPException) as exc:
        raise RetrievalError(f"Network error for {url}: {exc}", retryable=True) from exc
    payload = raw.decode("utf-8", errors="replace")
    if not payload.strip():
        raise RetrievalError(f"Empty response from {url}", retryable=True)
    return payload


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            delay = (timestamp - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def _parse_json_object(payload: str, *, service: str) -> dict[str, Any]:
    try:
        metadata = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"{service} returned invalid JSON: {exc}", retryable=True) from exc
    if not isinstance(metadata, dict):
        raise RetrievalError(f"{service} returned an invalid JSON object", retryable=True)
    return metadata


def _normalize_record_id(record_id: str) -> str:
    value = record_id.strip().upper()
    if re.fullmatch(r"(?:PMC)?[0-9]+", value) is None:
        raise RetrievalError("Expected a numeric PMID or a PMCID such as PMC6821189")
    return value


def _has_usable_text(text: str) -> bool:
    """Reject missing, whitespace-only, or markup-only passage content."""

    return any(character.isalnum() for character in _plain_text(text))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _validate_collection_text(collection: BioCCollection, *, text_source: str) -> None:
    """Require actual abstract/body prose, not titles, headings, or captions."""

    def usable(passage: BioCPassage) -> bool:
        if not _has_usable_text(passage.text):
            return False
        kind = str(passage.infons.get("type") or "").lower().replace("-", "_")
        section = str(passage.infons.get("section_type") or "").lower().replace("-", "_")
        if "title" in kind or kind in {"heading", "section_heading"} or section == "title":
            return False
        is_abstract = "abstract" in kind or "abstract" in section
        if text_source == "abstract":
            return is_abstract
        if is_abstract or any(token in kind for token in ("caption", "table", "figure", "footnote")):
            return False
        if is_excluded_section(kind) or is_excluded_section(section):
            return False
        return kind in {
            "paragraph", "body", "text", "body_text", "fulltext", "p",
            "introduction", "methods", "results", "discussion", "conclusion", "conclusions",
        } or section in {
            "intro", "introduction", "methods", "materials_and_methods", "results",
            "discuss", "discussion", "concl", "conclusion", "conclusions", "body",
        }

    documents = [
        document for document in collection.documents if any(usable(p) for p in document.passages)
    ]
    if not documents:
        label = "abstract" if text_source == "abstract" else "full-text article body"
        raise _UnavailableText(f"Retrieval returned no usable {label}")
    collection.documents = documents


def _canonical_source(source: str) -> str:
    value = source.strip().lower().replace("_", "-")
    if value == "ncbi":
        return value
    if value in {"europepmc", "europe-pmc"}:
        return "europepmc"
    raise ValueError("source must be 'ncbi' or 'europepmc'.")


def _canonical_text_source(text_source: str) -> str:
    """Normalize user text-source aliases into ``abstract`` or ``fulltext``."""

    value = text_source.strip().lower().replace("-", "").replace("_", "")
    if value in {"abstract", "abstracts", "titleabstract", "titleabstracts"}:
        return "abstract"
    if value in {"fulltext", "full", "pmc", "pmcoa"}:
        return "fulltext"
    raise ValueError("text_source must be 'abstract' or 'fulltext'.")


__all__ = [
    "FetchReport",
    "RetrievalError",
    "TextUnavailableError",
    "combine_ids",
    "fetch_pubmed_bioc",
    "read_ids_file",
]
