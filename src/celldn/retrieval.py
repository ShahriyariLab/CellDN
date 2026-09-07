"""Fetch PubMed/PMC/Europe PMC content as BioC for CellDN.

The prediction pipeline is intentionally separated from retrieval.  These
helpers only download/convert literature text into BioC XML/JSON;
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

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

NCBI_BIOC_PUBMED = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi"
NCBI_BIOC_PMCOA = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
EUROPEPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@dataclass(slots=True)
class FetchReport:
    """Summary returned by retrieval helpers."""

    output_path: Path
    requested_ids: list[str]
    fetched_ids: list[str]
    failed_ids: dict[str, str] = field(default_factory=dict)
    source: str = "ncbi"
    text_source: str = "abstract"
    output_format: str = "bioc-xml"

    @property
    def ok(self) -> bool:
        """Return ``True`` when every requested record was fetched successfully."""

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
        }


def read_ids_file(path: PathLike) -> list[str]:
    """Read PMIDs/PMCIDs from a text file; commas, spaces, and newlines all work."""

    return clean_pmid_list(Path(path).read_text(encoding="utf-8"))


def combine_ids(ids: Iterable[str] | str | None = None, ids_file: PathLike | None = None) -> list[str]:
    """Combine inline IDs and a file of IDs into a de-duplicated list."""

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
    """Fetch PubMed abstracts or PMC OA full text and write a BioC XML/JSON file.
    """

    id_list = clean_pmid_list(ids)
    if not id_list:
        raise ValueError("At least one PMID/PMCID is required.")

    normalized_text_source = _canonical_text_source(text_source)
    normalized_source = source.strip().lower().replace("_", "-")
    fmt = canonical_bioc_format(output_format)
    output = Path(output_path)

    collections: list[BioCCollection] = []
    fetched_ids: list[str] = []
    failed_ids: dict[str, str] = {}

    for record_id in id_list:
        try:
            if normalized_source == "ncbi":
                collection = _fetch_ncbi_bioc_record(
                    record_id,
                    text_source=normalized_text_source,
                    output_format=fmt,
                    timeout=timeout,
                    opener=opener,
                )
            elif normalized_source in {"europepmc", "europe-pmc"}:
                collection = _fetch_europepmc_record(
                    record_id,
                    text_source=normalized_text_source,
                    timeout=timeout,
                    opener=opener,
                )
            else:
                raise ValueError("source must be 'ncbi' or 'europepmc'.")
            if not collection.documents:
                raise ValueError("retrieval returned no BioC documents")
            collections.append(collection)
            fetched_ids.append(record_id)
        except Exception as exc:  # noqa: BLE001 - return a report of failed IDs
            failed_ids[record_id] = str(exc)
        if pause > 0:
            time.sleep(pause)

    if collections:
        merged = merge_bioc_collections(collections)
    else:
        merged = BioCCollection(source=f"CellDN retrieval: {source}", key="empty retrieval result")
    write_bioc_collection(merged, output, output_format=fmt)
    return FetchReport(
        output_path=output,
        requested_ids=id_list,
        fetched_ids=fetched_ids,
        failed_ids=failed_ids,
        source=normalized_source,
        text_source=normalized_text_source,
        output_format=fmt,
    )


def _fetch_ncbi_bioc_record(
    record_id: str,
    *,
    text_source: str,
    output_format: str,
    timeout: int,
    opener: UrlOpener | None,
) -> BioCCollection:
    """Fetch one PubMed or PMC Open Access record from the NCBI BioC API."""

    export_fmt = "BioC_json" if output_format == "bioc-json" else "BioC_xml"
    base = NCBI_BIOC_PUBMED if text_source == "abstract" else NCBI_BIOC_PMCOA
    url = f"{base}/{export_fmt}/{record_id}/unicode"
    payload = _urlopen_text(url, timeout=timeout, opener=opener)
    return read_bioc_collection_from_string(payload, input_format=output_format)


def _fetch_europepmc_record(
    record_id: str,
    *,
    text_source: str,
    timeout: int,
    opener: UrlOpener | None,
) -> BioCCollection:
    """Fetch one record from Europe PMC as abstract text or converted full text."""

    if text_source == "abstract":
        metadata = _fetch_europepmc_metadata(record_id, timeout=timeout, opener=opener)
        result = metadata.get("resultList", {}).get("result", [])
        if not result:
            raise ValueError("Europe PMC search returned no result")
        row = result[0]
        title = str(row.get("title") or "")
        abstract = str(row.get("abstractText") or "")
        pmid = str(row.get("pmid") or row.get("id") or record_id)
        text = f"{title} {abstract}".strip() if title and abstract else title or abstract
        if not text:
            raise ValueError("Europe PMC record has no title or abstract text")
        return BioCCollection(
            source="Europe PMC",
            key="Europe PMC title/abstract",
            documents=[
                BioCDocument(
                    id=pmid,
                    infons={"source": str(row.get("source") or "MED")},
                    passages=[BioCPassage(infons={"type": "title_abstract"}, offset=0, text=text)],
                )
            ],
        )

    metadata = _fetch_europepmc_metadata(record_id, timeout=timeout, opener=opener)
    result = metadata.get("resultList", {}).get("result", [])
    if not result:
        raise ValueError("Europe PMC search returned no result")
    row = result[0]
    pmcid = str(row.get("pmcid") or "")
    if not pmcid:
        raise ValueError("Europe PMC result has no PMCID; fullTextXML is unavailable")
    url = f"{EUROPEPMC_REST}/{pmcid}/fullTextXML"
    payload = _urlopen_text(url, timeout=timeout, opener=opener)
    return _jats_xml_to_bioc(payload, document_id=str(row.get("pmid") or record_id), pmcid=pmcid)


def _fetch_europepmc_metadata(record_id: str, *, timeout: int, opener: UrlOpener | None) -> dict[str, Any]:
    """Fetch one Europe PMC metadata record as JSON."""

    query = f"EXT_ID:{record_id} AND (SRC:MED OR SRC:PMC)"
    url = f"{EUROPEPMC_REST}/search?{urlencode({'query': query, 'format': 'json', 'resultType': 'core'})}"
    payload = _urlopen_text(url, timeout=timeout, opener=opener)
    return json.loads(payload)


def _jats_xml_to_bioc(xml_text: str, *, document_id: str, pmcid: str) -> BioCCollection:
    """Convert Europe PMC full-text JATS XML into a simple BioC collection."""

    root = ET.fromstring(xml_text.encode("utf-8"))
    passages: list[BioCPassage] = []
    offset = 0

    title = _first_text(root, ".//article-title")
    if title:
        passages.append(BioCPassage(infons={"type": "title"}, offset=offset, text=title))
        offset += len(title) + 1

    abstract_parts = _texts(root, ".//abstract//p") or _texts(root, ".//abstract")
    abstract = " ".join(part for part in abstract_parts if part).strip()
    if abstract:
        passages.append(BioCPassage(infons={"type": "abstract"}, offset=offset, text=abstract))
        offset += len(abstract) + 1

    for paragraph in _texts(root, ".//body//p"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        passages.append(BioCPassage(infons={"type": "paragraph"}, offset=offset, text=paragraph))
        offset += len(paragraph) + 1

    if not passages:
        raise ValueError("Europe PMC fullTextXML contained no usable text passages")

    return BioCCollection(
        source="Europe PMC",
        key="Europe PMC fullTextXML converted to BioC",
        documents=[BioCDocument(id=document_id, infons={"pmcid": pmcid}, passages=passages)],
    )


def _texts(root: ET.Element, xpath: str) -> list[str]:
    """Collect non-empty text blocks for an XPath, retrying without namespaces."""

    nodes = root.findall(xpath)
    values: list[str] = []
    for node in nodes:
        text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
        if text:
            values.append(text)
    if values:
        return values
    # Retry without namespaces by matching element local names.
    local_name = xpath.rsplit("/", 1)[-1]
    if local_name.startswith(".//"):
        local_name = local_name[3:]
    local_name = local_name.replace("//", "/").split("/")[-1]
    for node in root.iter():
        if _local_name(node.tag) == local_name:
            text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
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
    """Fetch a URL and return its decoded text payload."""

    request = Request(url, headers={"User-Agent": "CellDN/0.2 (+https://github.com/ShahriyariLab/CellDN)"})
    try:
        response = opener(request, timeout) if opener is not None else urlopen(request, timeout=timeout)  # noqa: S310
        with response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc
    return raw.decode("utf-8", errors="replace")


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
    "combine_ids",
    "fetch_pubmed_bioc",
    "read_ids_file",
]
