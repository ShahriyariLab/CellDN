"""Lightweight checks for package data shipped with CellDN.

These tests do not load Bioformer, SapBERT, Ab3P, or any model checkpoints.
They only verify that the small packaged resources needed by the normalizer are
present and parseable after installation.
"""
from __future__ import annotations

import csv
import json
from itertools import islice

from celldn.normalization.abbreviations import (
    ABBREVIATION_HEADER,
    classify_abbreviation_path,
    default_abbreviations_path,
    load_abbreviation_identifier_lookup,
)
from celldn.normalization.ontology import (
    default_ontology_path,
    load_cell_ontology_terms,
)


def test_abbreviation_tsv_resource_exists_and_is_parseable() -> None:
    """The packaged abbreviation TSV should exist and have the expected schema."""
    path = default_abbreviations_path()

    assert path.is_file(), f"Missing packaged abbreviation resource: {path}"
    assert classify_abbreviation_path(path) == "short_form_identifier_tsv"

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames is not None
        assert reader.fieldnames[:2] == ABBREVIATION_HEADER

        rows = [row for row in islice(reader, 5) if row]

    assert rows, "abbreviations.tsv should contain data rows after the header"
    for row in rows:
        assert row.get("short_form", "").strip()
        assert row.get("matched_cl_id", "").strip()

    lookup = load_abbreviation_identifier_lookup(path, verbose=False)
    assert lookup.all_keys, "abbreviation lookup should contain at least one key"
    assert lookup.direct_lookup or lookup.ambiguous_candidates


def test_ontology_jsonl_resource_exists_and_records_are_parseable() -> None:
    """The packaged Cell Ontology JSONL should exist and contain valid records."""
    path = default_ontology_path()

    assert path.is_file(), f"Missing packaged ontology resource: {path}"

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) == 5:
                break

    assert records, "Cell Ontology JSONL should contain at least one record"

    required_fields = {"norm_concept_id", "norm_preferred_label", "synonyms", "namespace"}
    for record in records:
        assert required_fields.issubset(record)
        assert str(record["norm_concept_id"]).startswith("CL:")
        assert str(record["norm_preferred_label"]).strip()
        assert isinstance(record["synonyms"], list)

    term_entries, concept_metadata = load_cell_ontology_terms(path)
    assert term_entries, "ontology loader should produce searchable alias entries"
    assert concept_metadata, "ontology loader should produce concept metadata"
    assert all(entry.identifier.startswith("CL:") for entry in term_entries[:20])
