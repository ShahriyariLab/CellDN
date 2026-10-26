"""Helpers for quieting or enabling third-party ML runtime logs.

These functions centralize logging and progress-bar behavior for libraries such
as Transformers, Datasets, and Hugging Face Hub.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_STATUS_MESSAGES: ContextVar[set[str] | None] = ContextVar("celldn_status_messages", default=None)


def configure_external_runtime(*, verbose: bool, log_level: int | None = None) -> None:
    """Set logging verbosity and progress bars for third-party ML libraries."""

    resolved_level = log_level if log_level is not None else (
        logging.INFO if verbose else logging.ERROR
    )

    try:
        import datasets
    except ImportError:  # pragma: no cover - optional dependency
        datasets = None
    else:
        datasets.utils.logging.set_verbosity(resolved_level)
        if verbose:
            datasets.enable_progress_bar()
        else:
            datasets.disable_progress_bar()

    try:
        import transformers
    except ImportError:  # pragma: no cover - optional dependency
        transformers = None
    else:
        transformers.utils.logging.set_verbosity(resolved_level)
        if verbose:
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
            transformers.utils.logging.enable_progress_bar()
        else:
            transformers.utils.logging.disable_progress_bar()

    try:
        from huggingface_hub import utils as huggingface_hub_utils
    except ImportError:  # pragma: no cover - optional dependency
        huggingface_hub_utils = None
    else:
        if verbose:
            huggingface_hub_utils.enable_progress_bars()
        else:
            huggingface_hub_utils.disable_progress_bars()

    for logger_name in (
        "datasets",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(resolved_level)


@contextmanager
def status_scope() -> Iterator[None]:
    """Limit each short stage message to once per public workflow invocation.

    Context-local state keeps independent threads and tasks separate. Nested
    helpers share the outer workflow, and errors always release its state.
    """

    current = _STATUS_MESSAGES.get()
    if current is not None:
        yield
        return
    token = _STATUS_MESSAGES.set(set())
    try:
        yield
    finally:
        _STATUS_MESSAGES.reset(token)


def show_status(message: str, *, verbose: bool) -> None:
    """Print one short stage line in the default concise mode."""

    if verbose:
        return
    seen = _STATUS_MESSAGES.get()
    if seen is not None:
        if message in seen:
            return
        seen.add(message)
    print(message, file=sys.stdout, flush=True)


def show_completion(
    output_paths: str | Path | Iterable[str | Path] | None = None,
    *,
    output_dir: str | Path | None = None,
    skipped: int = 0,
) -> None:
    """Report written results and completion without printing predictions.

    Call only after all requested writes succeed. A partially available batch
    states its skipped count explicitly; an entirely failed batch never claims
    success. A workflow returning in-memory results needs no saved-file line.
    """

    if isinstance(output_paths, (str, Path)):
        paths = [Path(output_paths)]
    else:
        paths = [Path(path) for path in output_paths or ()]
    noun = "input" if skipped == 1 else "inputs"
    if len(paths) == 1:
        print(f"Saved to: {paths[0].resolve()}", flush=True)
    elif paths:
        destination = Path(output_dir) if output_dir is not None else paths[0].parent
        print(
            f"Saved {len(paths)} result files to: {destination.resolve()}",
            flush=True,
        )
    elif skipped:
        print(f"No output files saved ({skipped} skipped {noun}).", file=sys.stderr, flush=True)

    if skipped:
        print(f"Process finished with {skipped} skipped {noun}.", flush=True)
    else:
        print("Process finished.", flush=True)
