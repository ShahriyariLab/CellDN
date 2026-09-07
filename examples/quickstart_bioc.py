"""Quick start for CellDN BioC XML input/output.

This example demonstrates the BioC XML workflow through ``run_bioc``:

1. ``task="ner"``: recognition only
2. ``task="nen"``: normalization of existing mention annotations
3. ``task="end-to-end"``: recognition followed by normalization
"""

from __future__ import annotations

from pathlib import Path

from celldn import CellDNPipeline
from celldn.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL


def main() -> None:
    output_dir = Path("outputs/quickstart_bioc_work")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = CellDNPipeline.from_pretrained(
        ner_model=DEFAULT_NER_MODEL,
        nen_model=DEFAULT_NEN_MODEL,
        ontology_path="src/celldn/resources/cell_ontology_v2025-12-17.jsonl",
        abbreviations_path="src/celldn/resources/abbreviations.tsv",
    )

    ner_output = output_dir / "sample.ner.xml"
    normalized_from_existing = output_dir / "sample.normalized_from_ner.xml"
    end_to_end_output = output_dir / "sample.end_to_end.xml"
    end_to_end_ner = output_dir / "sample.end_to_end.ner.xml"

    pipe.run_bioc(
        "examples/sample_input.xml",
        ner_output,
        task="ner",
    )

    pipe.run_bioc(
        ner_output,
        normalized_from_existing,
        task="nen",
    )

    pipe.run_bioc(
        "examples/sample_input.xml",
        end_to_end_output,
        task="end-to-end",
        ner_output_xml=end_to_end_ner,
    )

    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
