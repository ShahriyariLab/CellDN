"""Helpers for moving BioC passages into and out of the NER prediction flow.

The pipeline uses these helpers to:
- convert BioC XML input into passage-level JSON records for prediction
- turn predicted JSON entities back into BioC annotations
- write updated BioC XML output after NER
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..io import (
    BioCCollection,
    PassageRecord,
    PredictedEntity,
    PathLike,
    apply_predicted_entities_to_collection,
    iter_bioc_passage_records,
    read_bioc_collection,
    write_bioc_collection,
)

LOGGER = logging.getLogger(__name__)


def iter_input_files(paths: Sequence[PathLike]) -> Iterator[Path]:
    """Yield files from a mixed list of files and directories."""

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                yield child
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")


def iter_passage_records(srcs: Sequence[PathLike]) -> Iterator[PassageRecord]:
    """Stream BioC XML passages as shared passage records."""

    record_id = 0
    for src in iter_input_files(srcs):
        LOGGER.info("Reading BioC XML: %s", src)
        for record in iter_bioc_passage_records(
            [src],
            include_entities=False,
            input_format="bioc-xml",
        ):
            yield PassageRecord(
                record_id=record_id,
                document_id=record.document_id,
                passage_id=record.passage_id,
                passage_offset=record.passage_offset,
                text=record.text,
                entities=[],
                infons=dict(record.infons),
                source_path=record.source_path,
            )
            record_id += 1


def convert_bioc_to_json(srcs: Sequence[PathLike], dest: PathLike) -> int:
    """Convert BioC XML files into JSON records for NER prediction."""

    return write_passage_records_json(iter_passage_records(srcs), dest)


def write_passage_records_json(records: Iterable[PassageRecord], dest: PathLike) -> int:
    """Write shared passage records as JSON prediction inputs."""

    output_path = Path(dest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with output_path.open("w", encoding="utf-8") as sink:
        sink.write("[\n")
        first = True
        for record in records:
            payload: dict[str, Any] = {
                "id": record.record_id,
                "document_id": record.document_id,
                "passage_id": record.passage_id,
                "passage_offset": record.passage_offset,
                "text": record.text,
            }
            if not first:
                sink.write(",\n")
            sink.write(json.dumps(payload, ensure_ascii=False))
            first = False
            num_records += 1
        if num_records:
            sink.write("\n")
        sink.write("]\n")

    LOGGER.info("Wrote %d passage records to %s.", num_records, output_path)
    return num_records


def prediction_entries_to_predicted_entities(
    prediction_entries: Sequence[dict[str, Any]],
) -> list[PredictedEntity]:
    """Convert prediction JSON entries to absolute-offset entities."""

    entities: list[PredictedEntity] = []
    for entry in prediction_entries:
        document_id = str(entry.get("document_id", ""))
        raw_passage_id = entry.get("passage_id")
        passage_id = int(raw_passage_id) if raw_passage_id is not None else None
        for entity in entry.get("predicted_entities", []) or []:
            entities.append(
                PredictedEntity(
                    document_id=document_id,
                    passage_id=int(passage_id or 0),
                    label=str(entity["label"]),
                    start=int(entity["start"]),
                    end=int(entity["end"]),
                    text=str(entity.get("text", "")),
                )
            )
    return entities


def apply_prediction_entries_to_collection(
    collection: BioCCollection,
    prediction_entries: Sequence[dict[str, Any]],
    *,
    clear_existing: bool = True,
) -> BioCCollection:
    """Attach NER prediction entries to an in-memory BioC collection."""

    return apply_predicted_entities_to_collection(
        collection,
        prediction_entries_to_predicted_entities(prediction_entries),
        clear_existing=clear_existing,
    )


def write_predictions_to_bioc_xml(
    input_xml_path: PathLike,
    output_xml_path: PathLike,
    prediction_entries: Sequence[dict[str, Any]],
) -> None:
    """Write predicted NER annotations back into the original BioC XML structure."""

    collection = read_bioc_collection(input_xml_path, input_format="bioc-xml")
    apply_prediction_entries_to_collection(collection, prediction_entries, clear_existing=True)
    write_bioc_collection(collection, output_xml_path, output_format="bioc-xml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the BioC-to-JSON converter."""

    parser = argparse.ArgumentParser(
        description="Convert BioC XML into JSON passage records for CellExLink NER prediction."
    )
    parser.add_argument("inputs", nargs="+", help="Input BioC XML files or directories.")
    parser.add_argument("--output", required=True, help="Destination JSON file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    """Run the small BioC conversion CLI utility."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    count = convert_bioc_to_json(args.inputs, args.output)
    print(f"Wrote {count} passage records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
