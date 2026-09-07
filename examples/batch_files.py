"""Process a directory of CellDN inputs with automatic model reuse."""

from __future__ import annotations

from celldn import CellDNPipeline


pipeline = CellDNPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

# Models load on the first chunk and remain available for later chunks.
outputs = pipeline.run_files(
    "corpus",
    "outputs/corpus_results",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)

for output in outputs:
    print(output)
