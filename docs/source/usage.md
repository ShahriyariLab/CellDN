# Usage

## Python

```python
from celldn import CellDNPipeline

pipe = CellDNPipeline.from_pretrained(
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

### Structured input

```python
pipe.run_bioc(
    "input.xml",
    "output.xml",
    task="end-to-end"
    )
```

Supported tasks are `ner`, `nen`, and `end-to-end`.

### PDF input

```python
outputs = pipe.run_files(
    "article.pdf",
    "outputs/",
    task="end-to-end"
    )
```

### PDF lists

```python
outputs = pipe.run_files(
    ["first.pdf", "second.pdf"],
    "outputs/pdfs/",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)
```

### Directory

```python
outputs = pipe.run_files(
    "input_directory/",
    "result_directory/",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)
```

### Publication identifiers

```python
output = pipe.run_pmids(
    ["30243656", "PMC6821189"],
    "results.xml",
    text_source="fulltext",
    batch_size=32,
    passage_chunk_size=128,
)
```

## Command line

```bash
celldn predict-text \
  --text "Macrophages accumulated near the tumor." \
  --output predictions.json
```

```bash
celldn run-bioc input.xml output.xml \
  --task end-to-end \
  --passage-chunk-size 128
```

```bash
celldn run-files first.pdf second.pdf \
  --results-dir outputs/pdfs/ \
  --batch-size 32 \
  --passage-chunk-size 128
```

```bash
celldn run-files input_directory/ \
  --results-dir result_directory/ \
  --batch-size 32 \
  --passage-chunk-size 128
```

```bash
celldn predict-pmid \
  --ids-file examples/PMID_list.txt \
  --text-source fulltext \
  --output results.xml \
  --batch-size 100 \
  --passage-chunk-size 128
```


