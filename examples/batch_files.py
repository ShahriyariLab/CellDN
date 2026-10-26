"""Process a directory of CellDN inputs with automatic model reuse."""

from __future__ import annotations

from celldn import CellDNPipeline

pipeline = CellDNPipeline.from_pretrained()


outputs = pipeline.run_files(
    "papers",
    "outputs/paper_results",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)

# The pipeline summarizes the saved count and destination.
# outputs contains the exact paths for further processing.
