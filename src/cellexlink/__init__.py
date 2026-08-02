"""CellExLink public API."""

from .pipeline import (
    DEFAULT_BIOC_CHUNK_SIZE,
    DEFAULT_FILE_CHUNK_SIZE,
    DEFAULT_NEN_MODEL,
    DEFAULT_NER_MODEL,
    DEFAULT_PMID_CHUNK_SIZE,
    CellExLinkPipeline,
    ExtractionResult,
    RecognizedMention,
    write_predictions_json,
)

__all__ = [
    "DEFAULT_BIOC_CHUNK_SIZE",
    "DEFAULT_FILE_CHUNK_SIZE",
    "DEFAULT_NEN_MODEL",
    "DEFAULT_NER_MODEL",
    "DEFAULT_PMID_CHUNK_SIZE",
    "CellExLinkPipeline",
    "ExtractionResult",
    "RecognizedMention",
    "write_predictions_json",
]

try:
    from .mics.output_display import render_cell_type_annotations
except ImportError:  # pragma: no cover - notebook helper is optional.
    pass
else:
    __all__.append("render_cell_type_annotations")
