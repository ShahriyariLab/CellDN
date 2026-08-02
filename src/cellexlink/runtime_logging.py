"""Helpers for quieting or enabling third-party ML runtime logs.

These functions centralize logging and progress-bar behavior for libraries such
as Transformers, Datasets, and Hugging Face Hub.
"""

from __future__ import annotations

import logging
import sys


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


def show_status(message: str, *, verbose: bool) -> None:
    """Print a short user-facing status line only in quiet mode."""

    if verbose:
        return
    print(message, file=sys.stdout, flush=True)
