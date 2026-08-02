# Usage

## Python

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="almire/CellExLink-bioformer16L",
    nen_model="almire/CellExLink-Sapbert",
)
```

### Text

```python
results = pipe.run_text(
    "Macrophages accumulated near the tumor.",
    task="end-to-end",
)
```

Use `task="ner"` when only mention spans are needed.

### Structured input

```python
pipe.run_bioc(
    "input.xml",
    "output.xml",
    task="end-to-end",
    passage_chunk_size=128,
)
```

Supported tasks are `ner`, `nen`, and `end-to-end`.

### Many files

```python
outputs = pipe.run_files(
    "input_directory/",
    "result_directory/",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)
```

One result file is written for every input file.

### Publication identifiers

```python
pipe.run_pmids(
    ["30243656", "PMC1234567"],
    "results.xml",
    batch_size=100,
    passage_chunk_size=128,
)
```

Large identifier lists are chunked internally and merged into one output file.

## Command line

```bash
cellexlink predict-text \
  --text "Macrophages accumulated near the tumor." \
  --output predictions.json
```

```bash
cellexlink run-bioc input.xml output.xml \
  --task end-to-end \
  --chunk-size 128
```

```bash
cellexlink run-files input_directory/ \
  --results-dir result_directory/ \
  --chunk-size 32 \
  --bioc-chunk-size 128
```

```bash
cellexlink predict-pmid \
  --ids-file examples/PMID_list.txt \
  --output results.xml \
  --chunk-size 100 \
  --bioc-chunk-size 128
```

