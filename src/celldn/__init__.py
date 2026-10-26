"""CellDN public API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("celldn")
except PackageNotFoundError:  # pragma: no cover - source-tree fallback.
    __version__ = "1.1.0"

from .ontology_download import OntologyDownloadError, download_cell_ontology
from .pdf import PDFExtractionError, read_pdf_collection
from .pipeline import (
    DEFAULT_BIOC_CHUNK_SIZE,
    DEFAULT_FILE_CHUNK_SIZE,
    DEFAULT_NEN_MODEL,
    DEFAULT_NER_MODEL,
    DEFAULT_PMID_CHUNK_SIZE,
    CellDNPipeline,
    ExtractionResult,
    ExtractionResults,
    RecognizedMention,
    write_predictions_json,
)
from .retrieval import TextUnavailableError

__all__ = [
    "DEFAULT_BIOC_CHUNK_SIZE",
    "DEFAULT_FILE_CHUNK_SIZE",
    "DEFAULT_NEN_MODEL",
    "DEFAULT_NER_MODEL",
    "DEFAULT_PMID_CHUNK_SIZE",
    "CellDNPipeline",
    "download_cell_ontology",
    "OntologyDownloadError",
    "TextUnavailableError",
    "PDFExtractionError",
    "read_pdf_collection",
    "ExtractionResult",
    "ExtractionResults",
    "RecognizedMention",
    "write_predictions_json",
    "__version__",
]

try:
    from .mics.output_display import (
        render_cell_type_annotations as render_cell_type_annotations,
    )
except ImportError:  # pragma: no cover - notebook helper is optional.
    pass
else:
    __all__.append("render_cell_type_annotations")
