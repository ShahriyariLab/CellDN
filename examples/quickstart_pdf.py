"""Run CellDN on a local PDF through the file-processing API.

Replace article.pdf with your input path before running this example.
"""

from __future__ import annotations

from celldn import CellDNPipeline

pipeline = CellDNPipeline.from_pretrained()
outputs = pipeline.run_files(
    "article.pdf",
    "outputs/",
    task="end-to-end",
    passage_chunk_size=128,
)

# outputs contains outputs/article.xml on success, or is empty for an unusable PDF.
