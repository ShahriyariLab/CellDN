"""Shared preprocessing and reconstruction helpers for CellExLink NER.

This module contains the reusable pieces of the NER runtime: dataset loading,
column inference, tokenizer-based preprocessing with overflow windows, Trainer
construction, and reconstruction of final entity spans from token-level model
predictions.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import datasets
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast, Trainer, TrainingArguments

from ..runtime_logging import configure_external_runtime

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100
INTERNAL_EXAMPLE_INDEX_COLUMN = "__example_index__"
GENERATED_JSON_FILENAMES = {
    "test": "test.hf.json",
}


@dataclass(slots=True)
class DatasetColumns:
    """Column names used by the offset-based NER prediction pipeline."""

    text: str
    entities: Optional[str]
    document_id: Optional[str]
    passage_id: Optional[str]
    passage_offset: Optional[str]
    record_id: Optional[str]


@dataclass(slots=True)
class LabelSchema:
    """BIO label mappings for token classification."""

    label_list: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    def encode(self, label: str) -> int:
        try:
            return self.label_to_id[str(label)]
        except KeyError as exc:
            raise KeyError(f"Unknown label {label!r}. Known labels: {self.label_list}") from exc


class TokenClassificationBatchCollator(DataCollatorForTokenClassification):
    """Drop metadata fields before padding/model dispatch."""

    ignored_feature_keys = {
        "sample_index",
        "offset_mapping",
        "special_tokens_mask",
        "text",
        "document_id",
        "passage_id",
        "passage_offset",
        "id",
    }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        sanitized = [
            {key: value for key, value in feature.items() if key not in self.ignored_feature_keys}
            for feature in features
        ]
        return super().__call__(sanitized)


def configure_logging(
    prediction_args: TrainingArguments,
    *,
    verbose: bool,
) -> None:
    """Configure logging consistently with Hugging Face Trainer."""
    import sys

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = prediction_args.get_process_log_level() if verbose else logging.ERROR

    logging.getLogger().setLevel(log_level)
    LOGGER.setLevel(log_level)
    configure_external_runtime(verbose=verbose, log_level=log_level)

    if verbose:
        LOGGER.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed prediction: %s, fp16: %s",
            prediction_args.local_rank,
            prediction_args.device,
            prediction_args.n_gpu,
            prediction_args.parallel_mode.value == "distributed",
            prediction_args.fp16,
        )
        LOGGER.info("Prediction arguments: %s", prediction_args)


def load_raw_datasets(data_files: dict[str, str], cache_dir: Optional[str]) -> DatasetDict:
    """Load JSON/CSV data files into a DatasetDict."""
    if not data_files:
        raise ValueError("No dataset files were provided.")

    first_path = next(iter(data_files.values()))
    extension = first_path.rsplit(".", 1)[-1].lower()
    if extension not in {"json", "csv"}:
        raise ValueError(f"Unsupported dataset extension: {extension}. Use JSON or CSV.")

    return load_dataset(extension, data_files=data_files, cache_dir=cache_dir)


def choose_reference_split(raw_datasets: DatasetDict) -> str:
    """Pick the dataset split used for column inference and preprocessing."""

    if "test" in raw_datasets:
        return "test"
    try:
        return next(iter(raw_datasets.keys()))
    except StopIteration as exc:
        raise ValueError("No dataset splits were loaded.") from exc


def infer_columns(
    raw_datasets: DatasetDict,
    text_column_name: Optional[str],
    entities_column_name: Optional[str],
) -> DatasetColumns:
    """Infer text/entity/metadata columns from the loaded records."""
    reference_split = choose_reference_split(raw_datasets)
    column_names = raw_datasets[reference_split].column_names

    if text_column_name is not None:
        text_column = text_column_name
    elif "text" in column_names:
        text_column = "text"
    else:
        text_column = column_names[0]

    if entities_column_name is not None:
        entities_column = entities_column_name
    elif "entities" in column_names:
        entities_column = "entities"
    else:
        entities_column = None

    return DatasetColumns(
        text=text_column,
        entities=entities_column,
        document_id="document_id" if "document_id" in column_names else None,
        passage_id="passage_id" if "passage_id" in column_names else None,
        passage_offset="passage_offset" if "passage_offset" in column_names else None,
        record_id="id" if "id" in column_names else None,
    )


def canonicalize_id2label_mapping(id2label: dict[Any, str]) -> dict[int, str]:
    """Coerce model config label keys to integers and values to strings."""

    return {int(index): str(label) for index, label in id2label.items()}


def build_label_schema_from_model(config: transformers.PretrainedConfig) -> LabelSchema:
    """Read BIO label mappings from a saved model config."""
    if not getattr(config, "id2label", None):
        raise ValueError("Prediction requires label mappings in the saved model config.")

    canonical = canonicalize_id2label_mapping(config.id2label)
    label_list = [canonical[index] for index in range(len(canonical))]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    return LabelSchema(
        label_list=label_list,
        label_to_id=label_to_id,
        id_to_label={index: label for label, index in label_to_id.items()},
    )


def select_subset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    """Optionally keep only the first ``limit`` examples from a dataset."""

    if limit is None:
        return dataset
    return dataset.select(range(min(len(dataset), limit)))


def attach_example_indices(dataset: Dataset) -> Dataset:
    """Attach stable example indices so overflow chunks can be merged later."""

    if INTERNAL_EXAMPLE_INDEX_COLUMN in dataset.column_names:
        return dataset
    return dataset.add_column(INTERNAL_EXAMPLE_INDEX_COLUMN, list(range(len(dataset))))


def full_tokenize_texts(texts: Sequence[str], tokenizer: PreTrainedTokenizerFast) -> list[list[tuple[int, int]]]:
    """Tokenize full texts once and return raw offset mappings without truncation."""

    encodings = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    return [[tuple(map(int, offset)) for offset in sample_offsets] for sample_offsets in encodings["offset_mapping"]]


def tokenize_prediction_examples(
    examples: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    text_column: str,
    example_index_column: str,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
) -> dict[str, list[Any]]:
    """Tokenize unlabeled examples while preserving source example indices."""
    texts = [str(text) for text in examples[text_column]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        padding=padding,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["sample_index"] = [int(examples[example_index_column][index]) for index in overflow_to_sample_mapping]
    return tokenized


def preprocess_prediction_dataset(
    raw_dataset: Dataset,
    *,
    columns: DatasetColumns,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    num_proc: Optional[int],
    overwrite_cache: bool,
    limit: Optional[int],
) -> tuple[Dataset, Dataset]:
    """Prepare raw prediction examples and their tokenized overflow chunks."""

    raw_subset = attach_example_indices(select_subset(raw_dataset, limit))
    tokenized = raw_subset.map(
        lambda batch: tokenize_prediction_examples(
            batch,
            tokenizer=tokenizer,
            text_column=columns.text,
            example_index_column=INTERNAL_EXAMPLE_INDEX_COLUMN,
            max_seq_length=max_seq_length,
            stride=stride,
            padding=padding,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        remove_columns=raw_subset.column_names,
        desc="Tokenizing prediction dataset with raw-text offsets",
    )
    return raw_subset, tokenized


def build_trainer(
    *,
    model: transformers.PreTrainedModel,
    prediction_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForTokenClassification,
    compute_metrics: Optional[Callable[[Any], dict[str, float]]],
) -> Trainer:
    """Create a prediction runner while supporting old and new Transformers constructor names."""
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": prediction_args,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def reconstruct_entities_from_offsets(
    *,
    text: str,
    full_offsets: Sequence[tuple[int, int]],
    predicted_label_ids_by_offset: dict[tuple[int, int], int],
    label_schema: LabelSchema,
    passage_offset: int,
) -> list[dict[str, Any]]:
    """Convert BIO token predictions into entity spans."""
    entities: list[dict[str, Any]] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is None or active_start is None or active_end is None:
            return
        entities.append(
            {
                "label": active_label,
                "start_local": int(active_start),
                "end_local": int(active_end),
                "start": passage_offset + int(active_start),
                "end": passage_offset + int(active_end),
                "text": text[int(active_start) : int(active_end)],
            }
        )
        active_label = None
        active_start = None
        active_end = None

    for token_start, token_end in full_offsets:
        if token_end <= token_start:
            continue

        label_id = predicted_label_ids_by_offset.get((token_start, token_end), label_schema.encode("O"))
        tag = label_schema.id_to_label[int(label_id)]

        if tag == "O":
            close_active()
            continue

        if "-" in tag:
            prefix, entity_label = tag.split("-", 1)
        else:
            prefix, entity_label = "B", tag

        starts_new = prefix == "B" or active_label is None or active_label != entity_label
        if starts_new:
            close_active()
            active_label = entity_label
            active_start = token_start
            active_end = token_end
        else:
            active_end = token_end

    close_active()
    return entities


def reconstruct_prediction_outputs(
    *,
    raw_predict_dataset: Dataset,
    tokenized_predict_dataset: Dataset,
    prediction_logits: Sequence[Sequence[Sequence[float]]],
    label_schema: LabelSchema,
    tokenizer: PreTrainedTokenizerFast,
    columns: DatasetColumns,
) -> list[dict[str, Any]]:
    """Aggregate overflow-window logits and reconstruct full-passage entities."""
    aggregated_logits: dict[int, dict[tuple[int, int], list[float]]] = {}
    aggregated_counts: dict[int, dict[tuple[int, int], int]] = {}

    for chunk_index, chunk_logits in enumerate(prediction_logits):
        sample_index = int(tokenized_predict_dataset[chunk_index]["sample_index"])
        offset_mapping = tokenized_predict_dataset[chunk_index]["offset_mapping"]
        special_tokens_mask = tokenized_predict_dataset[chunk_index]["special_tokens_mask"]

        sample_logits = aggregated_logits.setdefault(sample_index, {})
        sample_counts = aggregated_counts.setdefault(sample_index, {})

        for token_logits, offset, is_special in zip(chunk_logits, offset_mapping, special_tokens_mask):
            token_start, token_end = int(offset[0]), int(offset[1])
            if is_special or token_end <= token_start:
                continue

            key = (token_start, token_end)
            if key not in sample_logits:
                sample_logits[key] = [float(value) for value in token_logits]
                sample_counts[key] = 1
            else:
                sample_logits[key] = [old + float(new) for old, new in zip(sample_logits[key], token_logits)]
                sample_counts[key] += 1

    outputs: list[dict[str, Any]] = []
    for raw_index, example in enumerate(raw_predict_dataset):
        text = str(example[columns.text])
        full_offsets = full_tokenize_texts([text], tokenizer=tokenizer)[0]
        sample_logits = aggregated_logits.get(raw_index, {})
        sample_counts = aggregated_counts.get(raw_index, {})

        predicted_label_ids_by_offset: dict[tuple[int, int], int] = {}
        for offset in full_offsets:
            if offset not in sample_logits:
                continue
            count = max(sample_counts.get(offset, 1), 1)
            averaged = [value / count for value in sample_logits[offset]]
            best_label = max(range(len(averaged)), key=lambda index: averaged[index])
            predicted_label_ids_by_offset[offset] = int(best_label)

        passage_offset = int(example[columns.passage_offset]) if columns.passage_offset and columns.passage_offset in example else 0
        predicted_entities = reconstruct_entities_from_offsets(
            text=text,
            full_offsets=full_offsets,
            predicted_label_ids_by_offset=predicted_label_ids_by_offset,
            label_schema=label_schema,
            passage_offset=passage_offset,
        )

        entry: dict[str, Any] = {
            "id": example[columns.record_id] if columns.record_id and columns.record_id in example else raw_index,
            "text": text,
            "passage_offset": passage_offset,
            "predicted_entities": predicted_entities,
        }
        if columns.document_id and columns.document_id in example:
            entry["document_id"] = example[columns.document_id]
        if columns.passage_id and columns.passage_id in example:
            entry["passage_id"] = example[columns.passage_id]
        outputs.append(entry)

    return outputs


def save_prediction_outputs(output_dir: str, prediction_entries: list[dict[str, Any]]) -> None:
    """Write predictions.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(prediction_entries, handle, ensure_ascii=False, indent=2)
