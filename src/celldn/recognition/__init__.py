"""
Recognition module for CellDN.

This subpackage contains the cell-type named entity recognition (NER)
component used before Cell Ontology normalization.
"""

from __future__ import annotations

from ..io import EntitySpan, PassageRecord
from .bioc import convert_bioc_to_json, write_predictions_to_bioc_xml

__all__ = [
    "EntitySpan",
    "PassageRecord",
    "convert_bioc_to_json",
    "write_predictions_to_bioc_xml",
    "predict_ner",
]


def __getattr__(name: str):
    """Keep BioC conversion and empty-input handling independent of model imports."""

    if name == "predict_ner":
        from .predict import predict_ner

        return predict_ner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
