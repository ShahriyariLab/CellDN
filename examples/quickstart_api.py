"""Minimal CellDN Python API example."""

from celldn import CellDNPipeline
from celldn.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL

TEXT = (
    "The stromal vascular fraction contained mesothelial cells, smooth muscle "
    "cells (SMCs), endothelial cells (ECs), and macrophages."
)

pipe = CellDNPipeline.from_pretrained(
    ner_model=DEFAULT_NER_MODEL,
    nen_model=DEFAULT_NEN_MODEL,
)

print("NER")
for result in pipe.run_text(TEXT, task="ner"):
    print(result.to_dict())

print("END TO END")
for result in pipe.run_text(TEXT, task="end-to-end"):
    print(result.to_dict())
