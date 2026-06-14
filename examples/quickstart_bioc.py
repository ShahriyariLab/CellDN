"""Quick start for CellExLink BioC XML input/output.

This example demonstrates the BioC XML workflow:

1. recognize_bioc(): NER only
2. normalize_bioc(): NEN only on existing mention annotations
3. extract_bioc(): end-to-end NER + NEN
"""

from __future__ import annotations

from pathlib import Path

from cellexlink import CellExLinkPipeline
from cellexlink.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL


def main() -> None:
    output_dir = Path("outputs/quickstart_bioc_work")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = CellExLinkPipeline.from_pretrained(
        ner_model=DEFAULT_NER_MODEL,
        nen_model=DEFAULT_NEN_MODEL,
        ontology_path="src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl",
        abbreviations_path="src/cellexlink/resources/abbreviations.tsv",
    )

    ner_output = output_dir / "sample.ner.xml"
    normalized_from_existing = output_dir / "sample.normalized_from_ner.xml"
    end_to_end_output = output_dir / "sample.end_to_end.xml"
    end_to_end_ner = output_dir / "sample.end_to_end.ner.xml"

    pipe.recognize_bioc(
        input_xml="examples/sample_input.xml",
        output_xml=ner_output,
    )

    pipe.normalize_bioc(
        input_xml=ner_output,
        output_xml=normalized_from_existing,
    )

    pipe.extract_bioc(
        input_xml="examples/sample_input.xml",
        output_xml=end_to_end_output,
        ner_output_xml=end_to_end_ner,
    )

    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
