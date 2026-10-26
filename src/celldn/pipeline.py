"""High-level Python API for CellDN workflows."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from .content import filter_collection_content, is_passage_annotatable
from .runtime_logging import configure_external_runtime, show_completion, show_status, status_scope

if TYPE_CHECKING:
    from .retrieval import FetchReport

PathLike = str | Path
_T = TypeVar("_T")

DEFAULT_NER_MODEL = "almire/CellExLink-bioformer16L"
DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"
DEFAULT_BIOC_CHUNK_SIZE = 128
DEFAULT_FILE_CHUNK_SIZE = 32
DEFAULT_PMID_CHUNK_SIZE = 100

_INTERNAL_DOCUMENT_KEY = "__celldn_internal_document_key"
_GENERATED_ANNOTATION_KEY = "__celldn_generated_annotation"


def _compact_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass object to a compact public dictionary."""

    data = asdict(obj)
    return {
        key: value
        for key, value in data.items()
        if value is not None and value != {} and value != []
    }


def _result_repr(item: Any) -> str:
    """Show meaningful public fields once, retaining additional metadata."""

    values = _compact_dict(item)
    return f"{type(item).__name__}(" + ", ".join(
        f"{key}={value!r}" for key, value in values.items()
    ) + ")"


def _extra_result_infons(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retain infons that are not already represented by named result fields."""

    infons = record.get("infons", {})
    if not isinstance(infons, Mapping):
        return {}
    aliases = {
        "entity_type": {"type", "entity_type"},
        "identifier": {"identifier", "cl_id"},
        "label": {"label", "name", "cl_label"},
        "score": {"score", "confidence_score"},
        "source": {"source"},
    }
    blocked = set().union(*(
        names for field_name, names in aliases.items()
        if record.get(field_name) is not None
    ))
    if record.get("entity_type") is not None and infons.get("label") == record["entity_type"]:
        blocked.add("label")
    suffixes = []
    for field_name, candidates in (
        ("identifier", ("_id_0",)),
        ("label", ("_identifier_name_0",)),
        ("score", ("_confidence_score_0", "_identifier_score_0")),
        ("source", ("_match_source",)),
    ):
        if record.get(field_name) is not None:
            suffixes.extend(candidates)
    return {
        str(key): value for key, value in infons.items()
        if str(key) not in blocked and not str(key).endswith(tuple(suffixes))
        and not str(key).startswith("__celldn_")
    }


def _resource_signature(path: Path) -> tuple[Any, ...]:
    """Invalidate a cached linker if its resource or provenance is replaced."""

    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return (None,)



@dataclass(slots=True, repr=False)
class RecognizedMention:
    """One NER-only cell-type mention prediction."""

    document_id: str | None
    passage_index: int
    mention: str
    start: int | None = None
    end: int | None = None
    entity_type: str | None = "cell_type"
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.infons = _extra_result_infons(asdict(self))

    def __repr__(self) -> str:
        return _result_repr(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-ready mention dictionary."""

        return _public_result_dict(self)

    @property
    def length(self) -> int | None:
        """Return the mention length when offsets are available."""

        if self.start is None or self.end is None:
            return None
        return self.end - self.start


@dataclass(slots=True, repr=False)
class ExtractionResult:
    """One end-to-end CellDN prediction."""

    document_id: str | None
    passage_index: int
    mention: str
    start: int | None = None
    end: int | None = None
    entity_type: str | None = None
    identifier: str | None = None
    label: str | None = None
    score: float | None = None
    source: str | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.infons = _extra_result_infons(asdict(self))

    def __repr__(self) -> str:
        return _result_repr(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-ready extraction dictionary."""

        return _public_result_dict(self)

    @property
    def length(self) -> int | None:
        """Return the mention length when offsets are available."""

        if self.start is None or self.end is None:
            return None
        return self.end - self.start


class ExtractionResults(list[ExtractionResult]):
    """List-compatible predictions retaining CL provenance even with no hits.

    The selected ``cl_version`` belongs to this container. Individual
    annotations contain their cell-type predictions only.
    """

    def __init__(
        self,
        items: Iterable[ExtractionResult] = (),
        *,
        cl_version: str | None = None,
    ) -> None:
        super().__init__(items)
        self.cl_version = cl_version

    def __repr__(self) -> str:
        items = list.__repr__(self)
        if self.cl_version is not None:
            return f"ExtractionResults({items}, cl_version={self.cl_version!r})"
        return f"ExtractionResults({items})"


@dataclass(slots=True, frozen=True)
class _BatchPlan:
    """One input file and its planned output path."""

    index: int
    input_path: Path
    output_path: Path


@dataclass(slots=True, frozen=True)
class _BatchDocumentRef:
    """Map one temporary batch document back to its source document."""

    internal_id: str
    original_id: str


@dataclass(slots=True)
class _LoadedBatchSource:
    """Loaded source collection and its temporary document mapping."""

    plan: _BatchPlan
    original_collection: Any
    document_refs: list[_BatchDocumentRef]


@dataclass(slots=True, frozen=True)
class _PassageRef:
    """Location of one passage in an input collection."""

    document_index: int
    passage_index: int


@dataclass(slots=True, frozen=True)
class _ChunkDocumentMap:
    """Map one chunk document back to its original passages."""

    document_index: int
    passage_indices: tuple[int, ...]


@dataclass(slots=True)
class _CollectionRun:
    """Processed collection and optional NER intermediate."""

    result: Any
    ner: Any | None = None


@dataclass(slots=True)
class CellDNPipeline:
    """High-level CellDN pipeline.

    Public workflows are :meth:`run_text`, :meth:`run_bioc`,
    :meth:`run_files`, and :meth:`run_pmids`.
    Model components load lazily on
    first use and remain attached to the pipeline for later calls. No public
    ``prepare()`` step is required.
    """

    ner_model: PathLike = DEFAULT_NER_MODEL
    nen_model: PathLike = DEFAULT_NEN_MODEL
    ontology_path: PathLike | None = None
    abbreviations_path: PathLike | None = None
    disable_abbreviations: bool = False
    output_dir: PathLike = "celldn_outputs"
    warmup_runs: int = 0
    batch_size: int = 16
    fp16: bool = False
    trust_remote_code: bool = False
    verbose: bool = False

    last_fetch_report: FetchReport | None = field(default=None, init=False, repr=False)
    last_cl_version: str | None = field(default=None, init=False, repr=False)

    _ner_predictor: Any | None = field(default=None, init=False, repr=False)
    _nen_linker: Any | None = field(default=None, init=False, repr=False)
    _ner_cache_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _nen_cache_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _prepare_lock: Any = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate lightweight settings before models are loaded."""

        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @classmethod
    def from_pretrained(
        cls,
        ner_model: PathLike = DEFAULT_NER_MODEL,
        nen_model: PathLike = DEFAULT_NEN_MODEL,
        **kwargs: Any,
    ) -> CellDNPipeline:
        """Create a pipeline from local paths or Hugging Face model IDs."""

        return cls(ner_model=ner_model, nen_model=nen_model, **kwargs)

    def _current_ner_cache_key(self) -> tuple[Any, ...]:
        """Return settings that determine the reusable NER predictor."""

        return (str(self.ner_model), self.trust_remote_code)

    def _current_nen_cache_key(self) -> tuple[Any, ...]:
        """Return settings that determine the reusable NEN linker."""

        from celldn.normalization.ontology import default_ontology_path

        ontology_path = Path(self.ontology_path or default_ontology_path()).expanduser().resolve()
        return (
            str(self.nen_model),
            str(ontology_path),
            _resource_signature(ontology_path),
            _resource_signature(ontology_path.with_suffix(".metadata.json")),
            str(self.abbreviations_path) if self.abbreviations_path is not None else None,
            self.disable_abbreviations,
            self.trust_remote_code,
        )

    def _get_ner_predictor(self) -> Any | None:
        """Return the retained NER predictor, creating it on first use."""

        module = importlib.import_module("celldn.recognition.predict")
        predictor_class = getattr(module, "NERPredictor", None)
        if predictor_class is None:
            return None

        cache_key = self._current_ner_cache_key()
        with self._prepare_lock:
            if self._ner_predictor is None or self._ner_cache_key != cache_key:
                self._ner_predictor = predictor_class(
                    model_path=self.ner_model,
                    warmup_runs=self.warmup_runs,
                    per_device_predict_batch_size=self.batch_size,
                    fp16=self.fp16,
                    trust_remote_code=self.trust_remote_code,
                    verbose=self.verbose,
                )
                self._ner_cache_key = cache_key

            self._ner_predictor.warmup_runs = self.warmup_runs
            self._ner_predictor.per_device_predict_batch_size = self.batch_size
            self._ner_predictor.fp16 = self.fp16
            self._ner_predictor.verbose = self.verbose
            return self._ner_predictor

    def _get_nen_linker(self) -> Any | None:
        """Return the retained NEN linker and static ontology resources."""

        module = importlib.import_module("celldn.normalization.linker")
        linker_class = getattr(module, "CellOntologyLinker", None)
        if linker_class is None:
            return None

        from celldn.normalization.abbreviations import default_abbreviations_path
        from celldn.normalization.ontology import default_ontology_path

        # Public workflows reset last_cl_version. Once normalization starts,
        # keep the same loaded inventory for every chunk of that run.
        if self.last_cl_version is not None and self._nen_linker is not None:
            return self._nen_linker
        cache_key = self._current_nen_cache_key()
        with self._prepare_lock:
            if self._nen_linker is None or self._nen_cache_key != cache_key:
                ontology_path = self.ontology_path or default_ontology_path()
                abbreviations_path: PathLike | None = self.abbreviations_path
                if abbreviations_path is None and not self.disable_abbreviations:
                    default_path = default_abbreviations_path()
                    abbreviations_path = default_path if default_path.is_file() else None

                self._nen_linker = linker_class.from_files(
                    ontology_path=ontology_path,
                    model_path=self.nen_model,
                    abbreviations_path=abbreviations_path,
                    disable_abbreviations=self.disable_abbreviations,
                    batch_size=self.batch_size,
                    trust_remote_code=self.trust_remote_code,
                    verbose=self.verbose,
                )
                self._nen_cache_key = cache_key

            self._nen_linker.batch_size = self.batch_size
            self._nen_linker.verbose = self.verbose
            return self._nen_linker

    def _recognize_collection(
        self,
        collection: Any,
        *,
        output_dir: PathLike,
    ) -> Any:
        """Run NER on an in-memory BioC collection."""

        from celldn.io import iter_collection_passage_records
        from celldn.recognition.bioc import (
            apply_prediction_entries_to_collection,
            write_passage_records_json,
        )

        working_collection = filter_collection_content(deepcopy(collection))
        records = list(
            iter_collection_passage_records(
                working_collection,
                include_entities=False,
            )
        )
        for document in working_collection.documents:
            for passage in document.passages:
                passage.annotations = []

        if not records:
            return working_collection

        show_status("Running CellDN NER...", verbose=self.verbose)
        configure_external_runtime(verbose=self.verbose)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="celldn_collection_ner_") as tmp:
            test_file = Path(tmp) / "test.hf.json"
            write_passage_records_json(records, test_file)
            predictor = self._get_ner_predictor()
            if predictor is None:
                prediction_module = importlib.import_module(
                    "celldn.recognition.predict"
                )
                result = prediction_module.predict_ner(
                    model_path=self.ner_model,
                    input_file=test_file,
                    output_dir=output_dir,
                    warmup_runs=self.warmup_runs,
                    per_device_predict_batch_size=self.batch_size,
                    fp16=self.fp16,
                    trust_remote_code=self.trust_remote_code,
                    verbose=self.verbose,
                )
            else:
                result = predictor.predict(
                    input_file=test_file,
                    output_dir=output_dir,
                )
            if isinstance(result, int) and result != 0:
                raise RuntimeError(
                    "CellDN NER failed with exit code "
                    f"{result}. Check the logs in: {output_dir}"
                )
            predictions_path = output_dir / "predictions.json"
            prediction_entries = json.loads(
                predictions_path.read_text(encoding="utf-8")
            )

        apply_prediction_entries_to_collection(
            working_collection,
            prediction_entries,
            clear_existing=True,
        )
        for document in working_collection.documents:
            for passage in document.passages:
                for annotation in passage.annotations:
                    annotation.infons[_GENERATED_ANNOTATION_KEY] = "true"
        return working_collection

    def _normalize_collection(
        self,
        collection: Any,
        *,
        document_context: Mapping[str, str] | None = None,
    ) -> Any:
        """Run NEN while sharing static resources and refreshing context."""

        normalization_module = importlib.import_module(
            "celldn.normalization.linker"
        )
        normalize_collection = normalization_module.normalize_collection

        show_status("Running CellDN normalization...", verbose=self.verbose)
        configure_external_runtime(verbose=self.verbose)
        kwargs: dict[str, Any] = {
            "collection": filter_collection_content(deepcopy(collection)),
            "model_path": self.nen_model,
            "disable_abbreviations": self.disable_abbreviations,
            "batch_size": self.batch_size,
            "trust_remote_code": self.trust_remote_code,
            "verbose": self.verbose,
            "document_context": document_context,
        }
        linker = self._get_nen_linker()
        if linker is not None:
            kwargs["linker"] = linker
        if self.ontology_path is not None:
            kwargs["cell_types"] = self.ontology_path
        if self.abbreviations_path is not None:
            kwargs["abbreviations"] = self.abbreviations_path

        result = normalize_collection(**kwargs)
        version = getattr(linker, "ontology_version", None)
        if version is None:
            from celldn.normalization.ontology import default_ontology_path, get_ontology_version

            version = get_ontology_version(self.ontology_path or default_ontology_path())
        # One selected resource supplies the version for the whole run.
        result.infons["cl_version"] = str(version)
        self.last_cl_version = str(version)
        return result

    def _process_collection(
        self,
        input_collection: Any,
        *,
        task: str,
        run_dir: PathLike,
        preserve_existing_annotations: bool = False,
        document_context: Mapping[str, str] | None = None,
    ) -> _CollectionRun:
        """Run one task on an in-memory collection without final serialization."""

        input_collection = filter_collection_content(deepcopy(input_collection))
        resolved_task = _canonical_task(task)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        ner_collection: Any | None = None

        if resolved_task == "ner":
            result_collection = self._recognize_collection(
                input_collection,
                output_dir=run_dir / "ner",
            )
        elif resolved_task == "nen":
            result_collection = self._normalize_collection(
                input_collection,
                document_context=document_context,
            )
        else:
            ner_collection = self._recognize_collection(
                input_collection,
                output_dir=run_dir / "ner",
            )
            ner_collection.infons.pop("cl_version", None)
            result_collection = self._normalize_collection(
                ner_collection,
                document_context=document_context,
            )

        if resolved_task == "ner":
            result_collection.infons.pop("cl_version", None)
            self.last_cl_version = None
        else:
            self.last_cl_version = result_collection.infons.get("cl_version")
            _compact_normalization_infons_in_collection(result_collection)
        _remove_runtime_infons_in_collection(result_collection)
        if ner_collection is not None:
            _remove_runtime_infons_in_collection(ner_collection)

        if preserve_existing_annotations and resolved_task != "nen":
            _restore_passage_annotations(result_collection, input_collection)

        return _CollectionRun(result=result_collection, ner=ner_collection)

    def _run_collection(
        self,
        input_collection: Any,
        *,
        task: str,
        run_dir: PathLike,
        preserve_existing_annotations: bool = False,
        ner_output_path: PathLike | None = None,
        document_context: Mapping[str, str] | None = None,
        finalize_generated_annotations: bool = True,
    ) -> Any:
        """Process one complete collection and optionally finalize annotation IDs."""

        from celldn.io import write_bioc_collection

        run = self._process_collection(
            input_collection,
            task=task,
            run_dir=run_dir,
            preserve_existing_annotations=preserve_existing_annotations,
            document_context=document_context,
        )
        if finalize_generated_annotations:
            _renumber_generated_annotations(run.result)
            if run.ner is not None:
                _renumber_generated_annotations(run.ner)

        if ner_output_path is not None and run.ner is not None:
            write_bioc_collection(
                run.ner,
                ner_output_path,
                output_format="bioc-xml",
            )
        return run.result

    def _run_collection_chunked(
        self,
        input_collection: Any,
        *,
        task: str,
        run_dir: PathLike,
        chunk_size: int,
        preserve_existing_annotations: bool = False,
        ner_output_path: PathLike | None = None,
        finalize_generated_annotations: bool = True,
    ) -> Any:
        """Process a collection in bounded passage chunks."""

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        from celldn.io import write_bioc_collection

        # Filter complete documents before chunking so excluded-section state
        # and abbreviation context never depend on a chunk boundary.
        input_collection = filter_collection_content(deepcopy(input_collection))
        resolved_task = _canonical_task(task)
        passage_refs = [
            _PassageRef(document_index, passage_index)
            for document_index, document in enumerate(input_collection.documents)
            for passage_index, _ in enumerate(document.passages)
        ]
        document_ids = [document.id for document in input_collection.documents]
        # Recognition maps predictions by document ID. Repeated source IDs
        # need the unique internal IDs used by the chunk path even for a small
        # collection, so annotations cannot spill into another document.
        if len(passage_refs) <= chunk_size and (
            not passage_refs or len(set(document_ids)) == len(document_ids)
        ):
            return self._run_collection(
                input_collection,
                task=resolved_task,
                run_dir=run_dir,
                preserve_existing_annotations=preserve_existing_annotations,
                ner_output_path=ner_output_path,
                finalize_generated_annotations=finalize_generated_annotations,
            )

        document_keys = [
            f"__celldn_document_{index:08d}"
            for index in range(len(input_collection.documents))
        ]
        document_context = _document_context_by_key(
            input_collection,
            document_keys,
        )
        result_collection = deepcopy(input_collection)
        result_collection.infons.pop("cl_version", None)
        ner_collection = (
            _empty_annotation_copy(input_collection)
            if resolved_task == "end-to-end" and ner_output_path is not None
            else None
        )
        unique_mentions: set[tuple[int, str]] = set()

        for chunk_number, refs in enumerate(
            _iter_chunks(passage_refs, chunk_size),
            start=1,
        ):
            chunk_collection, document_maps = _build_collection_chunk(
                input_collection,
                refs,
                document_keys,
            )
            run = self._process_collection(
                chunk_collection,
                task=resolved_task,
                run_dir=Path(run_dir) / f"chunk_{chunk_number:05d}",
                preserve_existing_annotations=preserve_existing_annotations,
                document_context=document_context,
            )
            _copy_chunk_passages(
                run.result,
                result_collection,
                document_maps,
            )
            _copy_processing_infons(
                run.result,
                result_collection,
                include_unique_count=False,
            )

            if resolved_task == "nen":
                unique_mentions.update(
                    _unique_mentions_from_chunk(chunk_collection, document_maps)
                )
            elif resolved_task == "end-to-end" and run.ner is not None:
                unique_mentions.update(
                    _unique_mentions_from_chunk(run.ner, document_maps)
                )

            if ner_collection is not None and run.ner is not None:
                _copy_chunk_passages(
                    run.ner,
                    ner_collection,
                    document_maps,
                )
                _copy_processing_infons(
                    run.ner,
                    ner_collection,
                    include_unique_count=False,
                )

        if resolved_task in {"nen", "end-to-end"}:
            result_collection.infons[
                "CellDN_normalization_unique_mentions"
            ] = str(len(unique_mentions))

        _remove_runtime_infons_in_collection(
            result_collection,
            generated_annotations_only=preserve_existing_annotations and resolved_task != "nen",
        )
        if finalize_generated_annotations:
            _renumber_generated_annotations(result_collection)
        if ner_collection is not None:
            if finalize_generated_annotations:
                _renumber_generated_annotations(ner_collection)
            _remove_runtime_infons_in_collection(ner_collection)
            write_bioc_collection(
                ner_collection,
                ner_output_path,
                output_format="bioc-xml",
            )

        return result_collection

    # ------------------------------------------------------------------
    # Public workflows
    # ------------------------------------------------------------------
    @status_scope()
    def run_text(
        self,
        text: str,
        *,
        task: str = "end-to-end",
        document_id: str | None = None,
        output_dir: PathLike | None = None,
        output_path: PathLike | None = None,
    ) -> list[RecognizedMention] | list[ExtractionResult]:
        """Run NER or end-to-end extraction on one text string.

        Results are returned in memory. Optionally write compact JSON with
        ``output_path``; the final status is printed only after that save.
        """

        self.last_cl_version = None
        self._validate_text(text)
        resolved_task = _canonical_task(task)
        if resolved_task == "nen":
            raise ValueError(
                "Plain text supports task='ner' or task='end-to-end'; "
                "NEN requires existing mention spans in structured input."
            )

        results: list[RecognizedMention] | list[ExtractionResult]
        if not text.strip():
            if resolved_task == "end-to-end":
                from celldn.io import BioCCollection

                # Empty text still uses the selected ontology, as an empty
                # file batch does. Normalization records its inventory version
                # without loading either prediction encoder.
                results = _extraction_results_from_collection(
                    self._normalize_collection(BioCCollection(source="CellDN", key="plain text"))
                )
            else:
                results = []
        else:
            from celldn.io import BioCCollection, BioCDocument, BioCPassage

            collection = BioCCollection(
                source="CellDN",
                key="plain text",
                documents=[BioCDocument(
                    id=document_id or "doc0",
                    passages=[BioCPassage(offset=0, text=text)],
                )],
            )
            run_dir = Path(output_dir or self.output_dir).expanduser() / f"text_{resolved_task}"
            result_collection = self._run_collection(
                collection,
                task=resolved_task,
                run_dir=run_dir,
            )
            results = (
                _recognized_mentions_from_collection(result_collection)
                if resolved_task == "ner"
                else _extraction_results_from_collection(result_collection)
            )
            if document_id is None:
                for result in results:
                    result.document_id = None

        written = (
            write_predictions_json(results, output_path, include_document_metadata=False)
            if output_path is not None else None
        )
        show_completion(written)
        return results

    @status_scope()
    def run_bioc(
        self,
        input_path: PathLike,
        output_path: PathLike,
        *,
        task: str = "end-to-end",
        input_format: str = "auto",
        output_format: str = "auto",
        preserve_existing_annotations: bool = False,
        ner_output_xml: PathLike | None = None,
        output_dir: PathLike | None = None,
        overwrite: bool = True,
        passage_chunk_size: int = DEFAULT_BIOC_CHUNK_SIZE,
    ) -> Path:
        """Run NER, NEN, or end-to-end prediction on structured input.

        ``passage_chunk_size`` is the maximum number of BioC passages
        processed in one inference unit. All retained document text remains
        available to NEN for document-specific abbreviation resolution.
        Excluded sections and non-annotatable passages are removed first.
        For NER and end-to-end tasks, ``preserve_existing_annotations=True``
        keeps annotations on retained passages alongside CellDN predictions.
        Originals are restored after prediction so NEN only sees new mentions.
        NEN-only tasks update supplied annotations without merging extra copies.
        """

        self.last_cl_version = None

        from celldn.io import (
            canonical_bioc_format,
            output_format_from_path,
            read_bioc_collection,
            write_bioc_collection,
        )

        resolved_task = _canonical_task(task)
        input_path = Path(input_path).expanduser()
        output_path = Path(output_path).expanduser()
        if ner_output_xml is not None:
            ner_output_xml = Path(ner_output_xml).expanduser()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        if passage_chunk_size < 1:
            raise ValueError("passage_chunk_size must be >= 1")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_format = (
            output_format_from_path(output_path)
            if output_format in (None, "", "auto")
            else canonical_bioc_format(output_format)
        )
        run_dir = Path(output_dir or self.output_dir).expanduser() / "bioc"
        input_collection = read_bioc_collection(
            input_path,
            input_format=input_format,
        )
        result_collection = self._run_collection_chunked(
            input_collection,
            task=resolved_task,
            run_dir=run_dir,
            chunk_size=passage_chunk_size,
            preserve_existing_annotations=preserve_existing_annotations,
            ner_output_path=ner_output_xml,
        )

        written = write_bioc_collection(
            result_collection,
            output_path,
            output_format=final_format,
        )
        show_completion(written)
        return written

    @status_scope()
    def run_files(
        self,
        input_paths: PathLike | Iterable[PathLike],
        output_dir: PathLike,
        *,
        task: str = "end-to-end",
        batch_size: int = DEFAULT_FILE_CHUNK_SIZE,
        passage_chunk_size: int = DEFAULT_BIOC_CHUNK_SIZE,
        input_format: str = "auto",
        output_format: str = "auto",
        preserve_existing_annotations: bool = False,
        recursive: bool = True,
        overwrite: bool = True,
    ) -> list[Path]:
        """Process a single file, file list, or directory of one input type.

        Files and directory contents must share one case-insensitive extension
        type. Mixed types are rejected before loading models or writing output.
        XML/JSON files can each contain several BioC documents. Unreadable PDFs
        are reported briefly and skipped; each successful input produces a file.

        ``batch_size`` controls files per batch and ``passage_chunk_size`` bounds
        the passages per inference unit. Model components are reused throughout.
        ``preserve_existing_annotations`` retains annotations on processed
        structured-input passages alongside new NER or end-to-end predictions.
        """

        from celldn.io import canonical_bioc_format

        self.last_cl_version = None
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if passage_chunk_size < 1:
            raise ValueError("passage_chunk_size must be >= 1")

        resolved_task = _canonical_task(task)
        results_dir = Path(output_dir).expanduser()
        paths = _expand_input_paths(
            input_paths,
            recursive=recursive,
            excluded_directory=results_dir,
        )
        if not paths:
            raise ValueError("No supported input files were found")
        input_type = _batch_input_type(paths)
        if resolved_task == "nen" and input_type in {"txt", "pdf"}:
            raise ValueError(
                "task='nen' requires structured inputs with existing mention spans; "
                "plain-text and PDF files support only 'ner' or 'end-to-end'."
            )
        if input_type == "txt":
            final_format = "compact-json"
        elif output_format in (None, "", "auto"):
            final_format = "bioc-json" if input_type == "json" else "bioc-xml"
        else:
            final_format = canonical_bioc_format(output_format)

        plans = _plan_batch_outputs(paths, output_dir=results_dir, output_format=final_format)
        if not overwrite:
            existing = [plan.output_path for plan in plans if plan.output_path.exists()]
            if existing:
                raise FileExistsError(f"Output file already exists: {existing[0]}")

        outputs: list[Path] = []
        for chunk_number, chunk in enumerate(_iter_chunks(plans, batch_size), start=1):
            with tempfile.TemporaryDirectory(prefix=f"celldn_batch_{chunk_number:05d}_") as tmp:
                outputs.extend(self._run_file_chunk(
                    chunk,
                    task=resolved_task,
                    input_type=input_type,
                    input_format=input_format,
                    output_format=final_format,
                    preserve_existing_annotations=preserve_existing_annotations,
                    run_dir=Path(tmp),
                    passage_chunk_size=passage_chunk_size,
                ))

        show_completion(outputs, output_dir=results_dir, skipped=len(plans) - len(outputs))
        return outputs

    def _run_file_chunk(
        self,
        plans: Sequence[_BatchPlan],
        *,
        task: str,
        input_type: str,
        input_format: str,
        output_format: str,
        preserve_existing_annotations: bool,
        run_dir: Path,
        passage_chunk_size: int,
    ) -> list[Path]:
        """Process one homogeneous file chunk with one reader and writer."""

        from celldn.io import (
            BioCCollection,
            BioCDocument,
            BioCPassage,
            read_bioc_collection,
            write_bioc_collection,
        )

        if input_type == "txt":
            def read_source(plan: _BatchPlan) -> Any:
                path = plan.input_path
                text = path.read_text(encoding="utf-8")
                documents = [BioCDocument(
                    id=path.stem,
                    passages=[BioCPassage(offset=0, text=text)],
                )] if text.strip() else []
                return BioCCollection(source="CellDN", key="cell-type-extraction", documents=documents)

            def write_result(collection: Any, path: Path) -> Path:
                predictions = (
                    _recognized_mentions_from_collection(collection) if task == "ner"
                    else _extraction_results_from_collection(collection)
                )
                return write_predictions_json(predictions, path, include_document_metadata=False)
        else:
            if input_type == "pdf":
                from celldn.pdf import PDFExtractionError, read_pdf_collection

                def read_source(plan: _BatchPlan) -> Any:
                    show_status("Extracting PDF text...", verbose=self.verbose)
                    try:
                        return read_pdf_collection(plan.input_path)
                    except PDFExtractionError as exc:
                        print(str(exc), file=sys.stderr)
                        return None
            else:
                def read_source(plan: _BatchPlan) -> Any:
                    return read_bioc_collection(plan.input_path, input_format=input_format)

            def write_result(collection: Any, path: Path) -> Path:
                return write_bioc_collection(collection, path, output_format=output_format)

        combined = BioCCollection(source="CellDN", key="cell-type-extraction")
        loaded_sources: list[_LoadedBatchSource] = []
        for plan in plans:
            original_collection = read_source(plan)
            if original_collection is None:
                continue
            filter_collection_content(original_collection)
            document_refs: list[_BatchDocumentRef] = []
            for document_index, original_document in enumerate(
                original_collection.documents
            ):
                internal_id = (
                    f"__celldn_batch_{plan.index:08d}_{document_index:08d}"
                )
                document = deepcopy(original_document)
                document.id = internal_id
                document.infons[_INTERNAL_DOCUMENT_KEY] = internal_id
                combined.documents.append(document)
                document_refs.append(
                    _BatchDocumentRef(
                        internal_id=internal_id,
                        original_id=original_document.id,
                    )
                )

            loaded_sources.append(
                _LoadedBatchSource(
                    plan=plan,
                    original_collection=original_collection,
                    document_refs=document_refs,
                )
            )

        if not loaded_sources:
            return []

        if combined.documents:
            processed = self._run_collection_chunked(
                combined,
                task=task,
                run_dir=run_dir,
                chunk_size=passage_chunk_size,
                preserve_existing_annotations=preserve_existing_annotations,
                finalize_generated_annotations=False,
            )
        else:
            processed = (
                self._normalize_collection(combined)
                if task in {"nen", "end-to-end"} else combined
            )

        processed_by_id = {document.id: document for document in processed.documents}
        written: list[Path] = []

        for source in loaded_sources:
            result_collection = deepcopy(source.original_collection)
            result_collection.infons.pop("cl_version", None)
            result_collection.documents = []
            for document_ref in source.document_refs:
                processed_document = processed_by_id.get(document_ref.internal_id)
                if processed_document is None:
                    raise RuntimeError(
                        "A document was lost while splitting a batch result: "
                        f"{document_ref.internal_id}"
                    )
                document = deepcopy(processed_document)
                document.id = document_ref.original_id
                document.infons.pop(_INTERNAL_DOCUMENT_KEY, None)
                result_collection.documents.append(document)

            _copy_processing_infons(
                processed,
                result_collection,
                include_unique_count=False,
            )
            if task in {"nen", "end-to-end"}:
                result_collection.infons[
                    "CellDN_normalization_unique_mentions"
                ] = str(
                    _count_unique_normalization_mentions(
                        result_collection,
                        generated_only=task == "end-to-end",
                    )
                )
            _remove_runtime_infons_in_collection(
                result_collection,
                generated_annotations_only=preserve_existing_annotations and task != "nen",
            )
            _renumber_generated_annotations(result_collection)

            written.append(write_result(result_collection, source.plan.output_path))

        return written

    @status_scope()
    def run_pmids(
        self,
        ids: Iterable[str] | str,
        output_path: PathLike,
        *,
        task: str = "end-to-end",
        text_source: str = "abstract",
        source: str = "ncbi",
        output_format: str = "auto",
        keep_intermediate_bioc: PathLike | None = None,
        preserve_existing_annotations: bool = False,
        output_dir: PathLike | None = None,
        overwrite: bool = True,
        strict_fetch: bool = False,
        timeout: int = 30,
        pause: float = 0.12,
        batch_size: int = DEFAULT_PMID_CHUNK_SIZE,
        passage_chunk_size: int = DEFAULT_BIOC_CHUNK_SIZE,
    ) -> Path | None:
        """Retrieve and process PMID/PMCID records in bounded chunks.

        Full-text requests fall back to abstracts when full text is unavailable.
        If no record can be retrieved, print a short message and return ``None``
        without replacing the requested output. A partially available batch processes its available
        records. ``strict_fetch=True`` instead raises ``TextUnavailableError``
        when any record remains unavailable after fallback.

        ``last_fetch_report`` records successful IDs, failures, fallback IDs, and
        the actual text source of each retrieved record. ``batch_size`` limits
        identifiers per retrieval chunk; ``passage_chunk_size`` limits passages
        processed in one inference chunk.
        ``preserve_existing_annotations`` keeps any annotations supplied by the
        retrieved BioC records on retained passages alongside new predictions.
        """

        self.last_cl_version = None

        from celldn.io import (
            canonical_bioc_format,
            clean_pmid_list,
            merge_bioc_files,
            output_format_from_path,
            read_bioc_collection,
            write_bioc_collection,
        )
        from celldn.retrieval import (
            FetchReport,
            TextUnavailableError,
            _canonical_text_source,
            fetch_pubmed_bioc,
        )

        self.last_fetch_report = None
        resolved_task = _canonical_task(task)
        text_source = _canonical_text_source(text_source)
        id_list = clean_pmid_list(ids)
        if not id_list:
            raise ValueError("At least one PMID/PMCID is required.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if passage_chunk_size < 1:
            raise ValueError("passage_chunk_size must be >= 1")

        output_path = Path(output_path).expanduser()
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        intermediate_path = (
            Path(keep_intermediate_bioc).expanduser()
            if keep_intermediate_bioc is not None
            else None
        )
        if intermediate_path is not None and intermediate_path.resolve() == output_path.resolve():
            raise ValueError("keep_intermediate_bioc must differ from output_path")

        final_format = (
            output_format_from_path(output_path)
            if output_format in (None, "", "auto")
            else canonical_bioc_format(output_format)
        )
        combined_report = FetchReport(
            output_path=output_path,
            requested_ids=id_list,
            fetched_ids=[],
            source=source,
            text_source=text_source,
            output_format=final_format,
        )
        self.last_fetch_report = combined_report
        base_run_dir = Path(output_dir or self.output_dir).expanduser() / "pmids"

        def short_ids(values: Iterable[str]) -> str:
            items = list(values)
            label = ", ".join(items[:3])
            return label + (f" (+{len(items) - 3} more)" if len(items) > 3 else "")

        with tempfile.TemporaryDirectory(prefix="celldn_pmids_") as tmp:
            tmp_dir = Path(tmp)
            retrieved_paths: list[Path] = []
            processed_paths: list[Path] = []

            for chunk_number, id_chunk in enumerate(
                _iter_chunks(id_list, batch_size),
                start=1,
            ):
                retrieved_path = tmp_dir / f"retrieved_{chunk_number:05d}.xml"
                report = fetch_pubmed_bioc(
                    id_chunk,
                    retrieved_path,
                    text_source=text_source,
                    output_format="bioc-xml",
                    source=source,
                    timeout=timeout,
                    pause=pause,
                )
                combined_report.source = report.source
                combined_report.text_source = report.text_source
                combined_report.fetched_ids.extend(report.fetched_ids)
                combined_report.failed_ids.update(report.failed_ids)
                combined_report.fallback_ids.extend(report.fallback_ids)
                combined_report.text_sources.update(report.text_sources)
                combined_report.attempt_errors.update(report.attempt_errors)

                abstract_fallbacks = [
                    record_id for record_id in report.fetched_ids
                    if text_source == "fulltext" and report.text_sources.get(record_id) == "abstract"
                ]
                if abstract_fallbacks:
                    print(
                        f"Full text unavailable for {short_ids(abstract_fallbacks)}; using abstract.",
                        file=sys.stderr,
                    )
                if report.failed_ids:
                    message = f"Not retrievable: {short_ids(report.failed_ids)}."
                    if strict_fetch:
                        raise TextUnavailableError(message)
                    print(message, file=sys.stderr)
                if not report.fetched_ids:
                    continue

                retrieved_paths.append(retrieved_path)
                processed_path = tmp_dir / f"processed_{chunk_number:05d}.xml"
                retrieved_collection = read_bioc_collection(
                    retrieved_path,
                    input_format="bioc-xml",
                )
                processed_collection = self._run_collection_chunked(
                    retrieved_collection,
                    task=resolved_task,
                    run_dir=base_run_dir / f"chunk_{chunk_number:05d}",
                    chunk_size=passage_chunk_size,
                    preserve_existing_annotations=preserve_existing_annotations,
                    finalize_generated_annotations=False,
                )
                write_bioc_collection(
                    processed_collection,
                    processed_path,
                    output_format="bioc-xml",
                )
                processed_paths.append(processed_path)

            if not processed_paths:
                show_completion([], skipped=len(combined_report.failed_ids))
                return None

            merge_bioc_files(
                processed_paths,
                output_path,
                input_format="bioc-xml",
                output_format=final_format,
                renumber_generated_annotations=True,
            )
            if intermediate_path is not None:
                merge_bioc_files(
                    retrieved_paths,
                    intermediate_path,
                    input_format="bioc-xml",
                    output_format="bioc-xml",
                )

        show_completion(output_path, skipped=len(combined_report.failed_ids))
        return output_path

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------
    @staticmethod
    def read_recognized_mentions_from_bioc(
        bioc_path: PathLike,
        *,
        input_format: str = "auto",
    ) -> list[RecognizedMention]:
        """Read NER-only predictions from BioC XML/JSON."""

        from celldn.io import read_bioc_collection

        collection = read_bioc_collection(bioc_path, input_format=input_format)
        return _recognized_mentions_from_collection(collection)

    @staticmethod
    def read_predictions_from_bioc(
        bioc_path: PathLike,
        *,
        input_format: str = "auto",
    ) -> list[ExtractionResult]:
        """Read normalized predictions from BioC XML/JSON."""

        from celldn.io import read_bioc_collection

        collection = read_bioc_collection(bioc_path, input_format=input_format)
        return _extraction_results_from_collection(collection)

    @staticmethod
    def _validate_text(text: str) -> None:
        """Require plain-text inputs to be strings."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")


# ----------------------------------------------------------------------
# Collection chunk helpers
# ----------------------------------------------------------------------
def _restore_passage_annotations(target: Any, source: Any) -> None:
    """Combine original annotations with predictions on the same passages.

    Both collections have already been filtered and retain the same document
    and passage order. Match positions rather than IDs or offsets, which may
    repeat. Restore originals after NEN and metadata compaction so their fields
    are not changed by CellDN. Generated IDs are finalized after all chunks
    have been assembled and can then avoid every preserved annotation ID.
    """

    for target_document, source_document in zip(
        target.documents, source.documents, strict=True,
    ):
        for target_passage, source_passage in zip(
            target_document.passages, source_document.passages, strict=True,
        ):
            if (target_passage.offset, target_passage.text) != (
                source_passage.offset, source_passage.text,
            ):
                raise RuntimeError("Passage changed while restoring input annotations")
            target_passage.annotations = (
                deepcopy(source_passage.annotations) + target_passage.annotations
            )


def _empty_annotation_copy(collection: Any) -> Any:
    """Copy a collection while removing all passage annotations."""

    copied = deepcopy(collection)
    copied.infons.pop("cl_version", None)
    for document in copied.documents:
        for passage in document.passages:
            passage.annotations = []
    return copied


def _document_context_by_key(
    collection: Any,
    document_keys: Sequence[str],
) -> dict[str, str]:
    """Build full document text used for abbreviation resolution."""

    context: dict[str, str] = {}
    for document_index, document in enumerate(collection.documents):
        parts = [
            passage.text
            for passage in document.passages
            if _passage_is_annotatable(passage) and passage.text
        ]
        context[document_keys[document_index]] = "\n".join(parts)
    return context


def _build_collection_chunk(
    collection: Any,
    passage_refs: Sequence[_PassageRef],
    document_keys: Sequence[str],
) -> tuple[Any, list[_ChunkDocumentMap]]:
    """Copy selected passages into a small collection with unique IDs."""

    from celldn.io import BioCCollection, BioCDocument

    grouped: dict[int, list[int]] = {}
    for ref in passage_refs:
        grouped.setdefault(ref.document_index, []).append(ref.passage_index)

    chunk = BioCCollection(
        source=collection.source,
        date=collection.date,
        key=collection.key,
        infons=deepcopy(collection.infons),
    )
    maps: list[_ChunkDocumentMap] = []

    for document_index, passage_indices in grouped.items():
        source_document = collection.documents[document_index]
        internal_id = document_keys[document_index]
        infons = deepcopy(source_document.infons)
        infons[_INTERNAL_DOCUMENT_KEY] = internal_id
        chunk.documents.append(
            BioCDocument(
                id=internal_id,
                infons=infons,
                passages=[
                    deepcopy(source_document.passages[passage_index])
                    for passage_index in passage_indices
                ],
            )
        )
        maps.append(
            _ChunkDocumentMap(
                document_index=document_index,
                passage_indices=tuple(passage_indices),
            )
        )

    return chunk, maps


def _copy_chunk_passages(
    chunk_collection: Any,
    target_collection: Any,
    document_maps: Sequence[_ChunkDocumentMap],
) -> None:
    """Place processed chunk passages back in their original positions."""

    if len(chunk_collection.documents) != len(document_maps):
        raise RuntimeError("Chunk document count changed during processing")

    for chunk_document, document_map in zip(
        chunk_collection.documents,
        document_maps,
        strict=True,
    ):
        if len(chunk_document.passages) != len(document_map.passage_indices):
            raise RuntimeError("Chunk passage count changed during processing")
        target_document = target_collection.documents[document_map.document_index]
        for chunk_passage, passage_index in zip(
            chunk_document.passages,
            document_map.passage_indices,
            strict=True,
        ):
            target_document.passages[passage_index] = deepcopy(chunk_passage)


def _unique_mentions_from_chunk(
    collection: Any,
    document_maps: Sequence[_ChunkDocumentMap],
) -> set[tuple[int, str]]:
    """Return the unique normalization inputs represented by one chunk."""

    unique: set[tuple[int, str]] = set()
    for document, document_map in zip(
        collection.documents,
        document_maps,
        strict=True,
    ):
        for passage in document.passages:
            if not _passage_is_annotatable(passage):
                continue
            for annotation in passage.annotations:
                if annotation.infons.get("type") == "cell_vague":
                    continue
                text = (annotation.text or "").strip()
                if text:
                    unique.add(
                        (
                            document_map.document_index,
                            _plural_normalize_text(text),
                        )
                    )
    return unique


def _count_unique_normalization_mentions(
    collection: Any,
    *,
    generated_only: bool,
) -> int:
    """Count unique mentions using the same text normalization as NEN."""

    unique: set[tuple[int, str]] = set()
    for document_index, document in enumerate(
        getattr(collection, "documents", [])
    ):
        for passage in getattr(document, "passages", []):
            if not _passage_is_annotatable(passage):
                continue
            for annotation in getattr(passage, "annotations", []):
                if generated_only and _GENERATED_ANNOTATION_KEY not in annotation.infons:
                    continue
                if annotation.infons.get("type") == "cell_vague":
                    continue
                text = (annotation.text or "").strip()
                if text:
                    unique.add((document_index, _plural_normalize_text(text)))
    return len(unique)


def _plural_normalize_text(text: str) -> str:
    """Apply the normalization used to deduplicate NEN inputs."""

    from celldn.normalization.stemmer import plural_normalize_text

    return plural_normalize_text(text)


def _passage_is_annotatable(passage: Any) -> bool:
    """Use the same content predicate in both model stages."""

    return is_passage_annotatable(passage)


def _renumber_generated_annotations(collection: Any) -> None:
    """Assign generated IDs without changing or colliding with original IDs."""

    from celldn.io import _finalize_generated_annotation_ids

    reserved_ids = {
        annotation.id
        for document in collection.documents
        for passage in document.passages
        for annotation in passage.annotations
        if _GENERATED_ANNOTATION_KEY not in annotation.infons
    }

    annotation_index = 1
    for document in collection.documents:
        annotation_index = _finalize_generated_annotation_ids(
            document,
            start=annotation_index,
            reserved_ids=reserved_ids,
        )


# ----------------------------------------------------------------------
# Batch helpers
# ----------------------------------------------------------------------
def _expand_input_paths(
    input_paths: PathLike | Iterable[PathLike],
    *,
    recursive: bool,
    excluded_directory: Path | None = None,
) -> list[Path]:
    """Expand files and directories into a stable, duplicate-free list."""

    if isinstance(input_paths, (str, Path)):
        raw_paths: list[PathLike] = [input_paths]
    else:
        raw_paths = list(input_paths)

    supported_suffixes = {".txt", ".pdf", ".xml", ".json"}
    expanded: list[Path] = []
    seen: set[Path] = set()
    excluded_resolved = (
        excluded_directory.resolve() if excluded_directory is not None else None
    )

    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_file() or (path.suffix.casefold() == ".pdf" and not path.is_dir()):
            candidates = [path]
        elif path.is_dir():
            directory_resolved = path.resolve()
            excluded_child = (
                excluded_resolved
                if excluded_resolved is not None
                and excluded_resolved != directory_resolved
                and excluded_resolved.is_relative_to(directory_resolved)
                else None
            )
            iterator = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                child
                for child in iterator
                if child.is_file()
                and child.suffix.casefold() in supported_suffixes
                and (
                    excluded_child is None
                    or not child.resolve().is_relative_to(excluded_child)
                )
            )
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            expanded.append(candidate)

    return expanded


def _known_input_stem(path: Path) -> str:
    """Remove supported document suffixes from one filename."""

    name = path.name
    for suffix in (".bioc.json", ".bioc.xml", ".json", ".xml", ".txt", ".pdf"):
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)] or "output"
    return path.stem or "output"


def _batch_input_type(paths: Sequence[Path]) -> str:
    """Require one supported extension type for the complete discovered batch."""

    kinds = {path.suffix.casefold().lstrip(".") for path in paths}
    unsupported = kinds - {"pdf", "txt", "xml", "json"}
    if unsupported:
        raise ValueError("Unsupported input type. Use PDF, TXT, XML, or JSON files.")
    if len(kinds) != 1:
        found = ", ".join(sorted(kind.upper() for kind in kinds))
        raise ValueError(
            f"run_files accepts one input type per call; found {found}. "
            "Use separate calls for PDF, TXT, XML, and JSON files."
        )
    return next(iter(kinds))


def _plan_batch_outputs(
    paths: Sequence[Path],
    *,
    output_dir: Path,
    output_format: str,
) -> list[_BatchPlan]:
    """Choose collision-safe paths using the batch's single output format."""

    suffix = {"compact-json": ".json", "bioc-json": ".bioc.json", "bioc-xml": ".xml"}[output_format]
    used: set[Path] = set()
    input_files = {path.resolve() for path in paths}
    plans: list[_BatchPlan] = []
    for index, input_path in enumerate(paths):
        stem = _known_input_stem(input_path)
        candidate = output_dir / f"{stem}{suffix}"
        if candidate.resolve() in input_files:
            candidate = output_dir / f"{stem}.celldn{suffix}"
        collision_index = 2
        while candidate.resolve() in used or candidate.resolve() in input_files:
            candidate = output_dir / f"{stem}__{collision_index}{suffix}"
            collision_index += 1
        used.add(candidate.resolve())
        plans.append(_BatchPlan(index=index, input_path=input_path, output_path=candidate))
    return plans


def _iter_chunks(items: Sequence[_T], chunk_size: int) -> Iterator[Sequence[_T]]:
    """Yield consecutive chunks from a sequence."""

    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _copy_processing_infons(
    source_collection: Any,
    target_collection: Any,
    *,
    include_unique_count: bool,
) -> None:
    """Copy stable CellDN metadata while excluding runtime fields."""

    for key, value in getattr(source_collection, "infons", {}).items():
        key_text = str(key)
        key_folded = key_text.casefold()
        if not key_text.startswith("CellDN_") and key_text != "cl_version":
            continue
        if "elapsed" in key_folded or "runtime" in key_folded:
            continue
        if not include_unique_count and key_text.endswith("_unique_mentions"):
            continue
        target_collection.infons[key_text] = str(value)


def _remove_runtime_infons_in_collection(
    collection: Any,
    *,
    generated_annotations_only: bool = False,
) -> None:
    """Remove CellDN timing metadata from a result collection."""

    def _remove_from(infons: Any) -> None:
        if not isinstance(infons, dict):
            return
        for key in list(infons):
            key_text = str(key)
            key_folded = key_text.casefold()
            if key_text.startswith("CellDN_") and (
                "elapsed" in key_folded or "runtime" in key_folded
            ):
                del infons[key]

    _remove_from(getattr(collection, "infons", None))
    for document in getattr(collection, "documents", []):
        _remove_from(getattr(document, "infons", None))
        for passage in getattr(document, "passages", []):
            _remove_from(getattr(passage, "infons", None))
            for annotation in getattr(passage, "annotations", []):
                if generated_annotations_only and _GENERATED_ANNOTATION_KEY not in annotation.infons:
                    continue
                _remove_from(getattr(annotation, "infons", None))


def _recognized_mentions_from_collection(collection: Any) -> list[RecognizedMention]:
    """Convert an in-memory NER collection into public result objects."""

    results: list[RecognizedMention] = []
    for document in collection.documents:
        for passage_index, passage in enumerate(document.passages):
            for annotation in passage.annotations:
                location = annotation.primary_location()
                start = int(location.offset) if location is not None else None
                end = (
                    int(location.offset) + int(location.length)
                    if location is not None
                    else None
                )
                infons = dict(annotation.infons)
                results.append(
                    RecognizedMention(
                        document_id=document.id,
                        passage_index=passage_index,
                        mention=annotation.text,
                        start=start,
                        end=end,
                        entity_type=(
                            infons.get("type")
                            or infons.get("entity_type")
                            or infons.get("label")
                        ),
                        score=_safe_float(
                            infons.get("confidence_score") or infons.get("score")
                        ),
                        infons=infons,
                    )
                )
    return results


def _extraction_results_from_collection(collection: Any) -> ExtractionResults:
    """Convert a normalized collection into public extraction results."""

    results = ExtractionResults(cl_version=collection.infons.get("cl_version"))
    for document in collection.documents:
        for passage_index, passage in enumerate(document.passages):
            for annotation in passage.annotations:
                location = annotation.primary_location()
                start = int(location.offset) if location is not None else None
                end = (
                    int(location.offset) + int(location.length)
                    if location is not None
                    else None
                )
                infons = dict(annotation.infons)
                entity_type = infons.get("type") or infons.get("entity_type")
                if (
                    entity_type is None
                    and "identifier" not in infons
                    and _find_first_infon_value_by_suffix(infons, "_id_0") is None
                ):
                    entity_type = infons.get("label")
                identifier = (
                    _find_first_infon_value_by_suffix(infons, "_id_0")
                    or infons.get("identifier")
                    or infons.get("cl_id")
                )
                label = (
                    _find_first_infon_value_by_suffix(
                        infons,
                        "_identifier_name_0",
                    )
                    or infons.get("label")
                    or infons.get("name")
                    or infons.get("cl_label")
                )
                score = _safe_float(
                    _find_first_infon_value_by_suffix(
                        infons,
                        "_confidence_score_0",
                    )
                    or _find_first_infon_value_by_suffix(
                        infons,
                        "_identifier_score_0",
                    )
                    or infons.get("confidence_score")
                    or infons.get("score")
                )
                source = (
                    _find_first_infon_value_by_suffix(infons, "_match_source")
                    or infons.get("source")
                )
                results.append(
                    ExtractionResult(
                        document_id=document.id,
                        passage_index=passage_index,
                        mention=annotation.text,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                        identifier=identifier,
                        label=label,
                        score=score,
                        source=source,
                        infons=infons,
                    )
                )
    return results


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _canonical_task(task: str) -> str:
    value = task.strip().lower()
    if value == "e2e":
        value = "end-to-end"
    if value not in {"ner", "nen", "end-to-end"}:
        raise ValueError("task must be one of: ner, nen, end-to-end")
    return value


def _find_first_infon_value_by_suffix(infons: dict[str, str], suffix: str) -> str | None:
    for key, value in infons.items():
        if key.endswith(suffix):
            return value
    return None


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable(item: Any) -> dict[str, Any]:
    if hasattr(item, "__dataclass_fields__"):
        return {
            key: value
            for key, value in asdict(item).items()
            if value is not None and value != {} and value != []
        }
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return {
            key: value
            for key, value in item.items()
            if value is not None and value != {} and value != []
        }
    raise TypeError(f"Object is not JSON serializable by CellDN: {type(item)!r}")


def _annotation_from_prediction(
    item: Any,
    *,
    include_document_metadata: bool,
) -> tuple[str | None, dict[str, Any]]:
    record = _jsonable(item)
    raw_document_id = record.get("document_id")
    document_id = str(raw_document_id) if raw_document_id not in (None, "") else None
    if not include_document_metadata:
        document_id = None

    annotation: dict[str, Any] = {}
    if record.get("mention") is not None:
        annotation["mention"] = record["mention"]
    if record.get("entity_type") is not None:
        annotation["entity_type"] = record["entity_type"]
    if record.get("start") is not None and record.get("end") is not None:
        annotation["span"] = {
            "begin": int(record["start"]),
            "end": int(record["end"]),
        }
    if include_document_metadata and record.get("passage_index") is not None:
        annotation["passage_index"] = int(record["passage_index"])
    identifier = record.get("identifier")
    if identifier is None:
        identifier = record.get("cl_id")
    if identifier is not None:
        annotation["identifier"] = identifier

    label = record.get("label")
    if label is None:
        label = record.get("name")
    if label is None:
        label = record.get("cl_label")
    if label is not None:
        annotation["label"] = label
    if record.get("normalized_text") is not None:
        annotation["normalized_text"] = record["normalized_text"]
    if record.get("candidates") not in (None, []):
        candidates = _sanitize_output_candidates(record["candidates"])
        if candidates:
            annotation["candidates"] = candidates
    if record.get("infons") not in (None, {}):
        infons = _sanitize_output_infons(_extra_result_infons(record))
        if infons:
            annotation["infons"] = infons

    return document_id, annotation


def _public_result_dict(item: Any) -> dict[str, Any]:
    """Return the same compact public shape used by JSON export."""

    return _annotation_from_prediction(
        item,
        include_document_metadata=False,
    )[1]


def _sanitize_output_candidates(candidates: Any) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return sanitized

    blocked_keys = {
        "score",
        "confidence_score",
        "embedding_score",
        "final_score",
        "source",
        "ab3p_match_score",
    }
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        sanitized.append(
            {
                key: value
                for key, value in item.items()
                if value is not None and key not in blocked_keys
            }
        )
    return sanitized


def _sanitize_output_infons(infons: Any) -> dict[str, Any]:
    if not isinstance(infons, Mapping):
        return {}

    blocked_fragments = (
        "_confidence_score_",
        "_identifier_score_",
        "_embedding_score_",
        "_match_source",
        "_ab3p_match_score",
    )
    blocked_keys = {"score", "confidence_score", "source"}
    return {
        str(key): value
        for key, value in infons.items()
        if value is not None
        and str(key) not in blocked_keys
        and not any(fragment in str(key) for fragment in blocked_fragments)
    }


def _compact_normalization_infons_in_collection(collection: Any) -> None:
    documents = getattr(collection, "documents", None)
    if not isinstance(documents, list):
        return

    for document in documents:
        for passage in getattr(document, "passages", []):
            for annotation in getattr(passage, "annotations", []):
                annotation.infons = _compact_normalization_infons(
                    getattr(annotation, "infons", {}),
                )


def _compact_normalization_infons(infons: Any) -> dict[str, str]:
    if not isinstance(infons, Mapping):
        return {}

    compact = dict(_sanitize_output_infons(infons))

    identifier = _find_first_infon_value_by_suffix(compact, "_id_0") or compact.get("identifier")
    if identifier:
        compact["identifier"] = identifier

    label = (
        _find_first_infon_value_by_suffix(compact, "_identifier_name_0")
        or compact.get("label")
        or compact.get("name")
    )
    if label:
        compact["label"] = label

    blocked_fragments = (
        "_id_",
        "_identifier_name_",
        "_matched_alias_",
        "_normalized_text",
        "_abbreviation_method",
        "_expanded_long_form",
        "_ab3p_method",
        "_ab3p_matched_key",
    )
    return {
        str(key): "" if value is None else str(value)
        for key, value in compact.items()
        if str(key) != "name"
        and not any(fragment in str(key) for fragment in blocked_fragments)
    }


def write_predictions_json(
    predictions: Iterable[Any],
    output_path: PathLike,
    *,
    include_document_metadata: bool = True,
    cl_version: str | None = None,
) -> Path:
    """Write compact annotations with the selected version once per file.

    The version comes from ``ExtractionResults.cl_version`` or the explicit
    ``cl_version`` argument. Individual annotations supply no provenance.
    """

    output_path = Path(output_path).expanduser()
    selected_version = (
        cl_version if cl_version is not None else getattr(predictions, "cl_version", None)
    )
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for prediction in predictions:
        document_id, annotation = _annotation_from_prediction(
            prediction,
            include_document_metadata=include_document_metadata,
        )
        grouped.setdefault(document_id, []).append(annotation)

    payloads: list[dict[str, Any]] = []
    for document_id, document_annotations in grouped.items():
        payload: dict[str, Any] = {"annotations": document_annotations}
        if document_id is not None:
            payload["document_id"] = document_id
        payloads.append(payload)

    final_payload: dict[str, Any] | list[dict[str, Any]]
    if len(payloads) == 1:
        final_payload = payloads[0]
    elif len(payloads) > 1:
        final_payload = {"documents": payloads}
    elif selected_version is not None:
        final_payload = {"annotations": []}
    else:
        final_payload = []
    if selected_version is not None:
        assert isinstance(final_payload, dict)
        final_payload = {"cl_version": str(selected_version), **final_payload}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "DEFAULT_BIOC_CHUNK_SIZE",
    "DEFAULT_FILE_CHUNK_SIZE",
    "DEFAULT_NEN_MODEL",
    "DEFAULT_NER_MODEL",
    "DEFAULT_PMID_CHUNK_SIZE",
    "CellDNPipeline",
    "ExtractionResult",
    "ExtractionResults",
    "RecognizedMention",
    "write_predictions_json",
]
