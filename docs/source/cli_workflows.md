# CLI workflows

This page collects CellExLink workflows from the command-line interface. If you want the cross-interface overview first, see {doc}`workflows`. If you want the Python version of the same ideas, see {doc}`python_api_workflows`.

## What the CLI covers

The CLI supports:

- `NER only`
- `NEN only`
- `End-to-end`
- PMID and PMCID retrieval workflows
- plain text input
- BioC XML input
- BioC JSON input
- generic JSON input
- directory and multi-file chunk processing
- PMID lists from a text file

## Before running examples

The examples below assume local model folders in this layout:

```text
models/
├── CellExLink-bioformer16L/
├── CellExLink-Sapbert/
└── models.json
```

Create them with:

```bash
cellexlink download-models --output-dir models
```

Create an output directory:

```bash
mkdir -p outputs
```

## CLI demonstrations

### 1. Plain-text NER only

Use this to show mention detection without ontology normalization.

```bash
cellexlink predict-text \
  --task ner \
  --text "CD8+ T cells infiltrated the tumor and macrophages accumulated nearby." \
  --output outputs/text_ner.json \
  --ner-model models/CellExLink-bioformer16L
```

### 2. Plain-text end-to-end

Use this to show the shortest complete CellExLink workflow.

```bash
cellexlink predict-text \
  --text "CD8+ T cells infiltrated the tumor and macrophages accumulated nearby." \
  --output outputs/text_e2e.json \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 3. BioC XML NER only

Use this when the audience works with corpus-style biomedical documents.

```bash
cellexlink run-bioc \
  examples/sample_input.xml \
  outputs/sample.ner.xml \
  --task ner \
  --chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L
```

### 4. BioC XML normalization only

Use this to show how CellExLink can normalize mentions produced by an upstream recognizer.

```bash
cellexlink run-bioc \
  outputs/sample.ner.xml \
  outputs/sample.normalized.xml \
  --task nen \
  --chunk-size 128 \
  --nen-model models/CellExLink-Sapbert
```

### 5. BioC XML end-to-end

Use this to show the standard structured-document workflow.

```bash
cellexlink run-bioc \
  examples/sample_input.xml \
  outputs/sample.end_to_end.xml \
  --ner-output-xml outputs/sample.intermediate_ner.xml \
  --chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 6. BioC JSON input

Use `run-bioc` when the input is BioC JSON rather than BioC XML.

```bash
cellexlink run-bioc \
  inputs/sample_bioc.json \
  outputs/sample_bioc.normalized.json \
  --task end-to-end \
  --input-format bioc-json \
  --output-format bioc-json \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 7. PubTator3 BioC JSON integration

Use this when you want to start from PubTator3 annotations and add CellExLink cell-type predictions to the same BioC JSON document.

```bash
curl -L "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=30243656" \
  -o outputs/pubtator3_annotations.bioc.json

cellexlink run-bioc \
  outputs/pubtator3_annotations.bioc.json \
  outputs/cellexlink_added.bioc.json \
  --input-format bioc-json \
  --output-format bioc-json \
  --task end-to-end \
  --preserve-existing-annotations \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

Notes:

- PubTator3 BioC JSON can be passed directly to `run-bioc`.
- `--preserve-existing-annotations` keeps the original PubTator3 annotations while adding CellExLink cell-type results.
- With `pmids=...`, this PubTator3 export is typically title plus abstract content.

### 8. Generic JSON input from another tool

Use this when integrating CellExLink with upstream tools that emit JSON rather than BioC.

```bash
cellexlink run-bioc \
  inputs/upstream_annotations.json \
  outputs/upstream_annotations.normalized.json \
  --task nen \
  --input-format json \
  --output-format bioc-json \
  --nen-model models/CellExLink-Sapbert
```

### 9. PMID abstract retrieval plus prediction

Use this to show retrieval followed by CellExLink inference.

```bash
cellexlink predict-pmid \
  --pmid 31261512 \
  --text-source abstract \
  --output outputs/pmid_31261512.predicted.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 10. PMID full-text selection

Use this to show full-text mode when a PMC OA or Europe PMC full text is available.

```bash
cellexlink predict-pmid \
  --pmid PMC6821189 \
  --text-source fulltext \
  --output outputs/pmc6821189.fulltext.predicted.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

Full-text retrieval depends on source availability. If the requested record is not available as full text, use `--text-source abstract` instead.

### 11. PMID list from a text file

Use this to show batch processing.

If `pmids.txt` contains one or more PMIDs or PMCIDs:

```text
31261512
PMC6821189
```

run:

```bash
cellexlink predict-pmid \
  --ids-file pmids.txt \
  --text-source abstract \
  --output outputs/pmid_batch.predicted.xml \
  --chunk-size 100 \
  --bioc-chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 13. Many files through one process

Use `run-files` for a directory or a list of files. One pipeline is created for
the command, and its NER model, NEN model, and static ontology embeddings are
reused across all chunks.

```bash
cellexlink run-files corpus/ \
  --results-dir outputs/corpus_results/ \
  --chunk-size 32 \
  --bioc-chunk-size 128 \
  --task end-to-end \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

Plain-text files produce compact JSON. Structured files produce BioC XML or
BioC JSON. Add `--no-recursive` to process only the direct contents of an input
directory.


