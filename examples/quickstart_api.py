"""Minimal CellExLink Python API example.

Run after installing CellExLink and downloading model checkpoints.
"""

from cellexlink import CellExLinkPipeline
from cellexlink.pipeline import DEFAULT_NER_MODEL, DEFAULT_NEN_MODEL


TEXT = ("The stromal vascular fraction contained mesothelial cells, smooth muscle cells (SMCs), endothelial cells (ECs), and macrophages. Mesotheliocytes and SMC clusters formed a separate population.")

pipe = CellExLinkPipeline.from_pretrained(
    ner_model=DEFAULT_NER_MODEL,
    nen_model=DEFAULT_NEN_MODEL,
)

# 1. NER only
ner_results = pipe.recognize_text(TEXT)
print("NER")
for result in ner_results:
    print(result.to_dict())

# 2. NEN only: normalize known mentions without running NER
nen_results = pipe.normalize_mentions(
    ["mesothelial cell", "SMC"],
    document_text=TEXT,
)
print("NEN")
for result in nen_results:
    print(result.to_dict())

# 3. End-to-end extraction: NER followed by NEN
e2e_results = pipe.extract_text(TEXT)
print("END TO END")
for result in e2e_results:
    print(result.to_dict())
