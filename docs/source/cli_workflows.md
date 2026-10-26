# CLI workflows

This page collects CellDN workflows from the command-line interface. If you want the cross-interface overview first, see {doc}`workflows`. If you want the Python version of the same ideas, see {doc}`python_api_workflows`.

## What the CLI covers

The CLI supports:

- `NER only`
- `NEN only`
- `End-to-end`
- PMID and PMCID retrieval workflows
- plain text input
- individual PDFs, PDF lists, and PDF directories with readable text layers
- BioC XML input
- BioC JSON input
- generic JSON input
- directory and multi-file chunk processing
- PMID lists from a text file
- a separate download of the latest official Cell Ontology release, converted to JSONL

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
celldn download-models --output-dir models
```

Create an output directory:

```bash
mkdir -p outputs
```

For `run-files` and `predict-pmid`, `--batch-size` controls the number of files or
publication identifiers per batch, matching Python's `batch_size` argument.
`run-bioc`, `run-files`, and `predict-pmid` all use `--passage-chunk-size` to limit
the passages processed in each chunk, matching Python's `passage_chunk_size`
argument. The separate `--inference-batch-size` option controls model inference
batches on prediction commands.

## CLI demonstrations

### 1. Plain-text NER only

Use this to show mention detection without ontology normalization.

```bash
celldn predict-text \
  --task ner \
  --text "CD8+ T cells infiltrated the tumor and macrophages accumulated nearby." \
  --output outputs/text_ner.json \
  --ner-model models/CellExLink-bioformer16L
```

### 2. Plain-text end-to-end

Use this to show the shortest complete CellDN workflow.

```bash
celldn predict-text \
  --text "CD8+ T cells infiltrated the tumor and macrophages accumulated nearby." \
  --output outputs/text_e2e.json \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 3. BioC XML NER only

Use this when the audience works with corpus-style biomedical documents.

```bash
celldn run-bioc \
  examples/sample_input.xml \
  outputs/sample.ner.xml \
  --task ner \
  --passage-chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L
```

### 4. BioC XML normalization only

Use this to show how CellDN can normalize mentions produced by an upstream recognizer.

```bash
celldn run-bioc \
  outputs/sample.ner.xml \
  outputs/sample.normalized.xml \
  --task nen \
  --passage-chunk-size 128 \
  --nen-model models/CellExLink-Sapbert
```

### 5. BioC XML end-to-end

Use this to show the standard structured-document workflow.

```bash
celldn run-bioc \
  examples/sample_input.xml \
  outputs/sample.end_to_end.xml \
  --ner-output-xml outputs/sample.intermediate_ner.xml \
  --passage-chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 6. BioC JSON input

Use `run-bioc` when the input is BioC JSON rather than BioC XML.

```bash
celldn run-bioc \
  inputs/sample_bioc.json \
  outputs/sample_bioc.normalized.json \
  --task end-to-end \
  --input-format bioc-json \
  --output-format bioc-json \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 7. PubTator3 BioC JSON integration

Use this when you want to start from PubTator3 annotations and add CellDN cell-type predictions to the same BioC JSON document.

```bash
curl -L "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=30243656" \
  -o outputs/pubtator3_annotations.bioc.json

celldn run-bioc \
  outputs/pubtator3_annotations.bioc.json \
  outputs/pubtator3.celldn.bioc.json \
  --input-format bioc-json \
  --output-format bioc-json \
  --task end-to-end \
  --preserve-existing-annotations \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

Notes:

- PubTator3 BioC JSON can be passed directly to `run-bioc`.
- `--preserve-existing-annotations` keeps the original PubTator3 annotations on retained passages while adding CellDN cell-type results. Excluded passages remain excluded.
- The flag is optional and also works with `run-files` and `predict-pmid`. Without it, `ner` and `end-to-end` replace existing annotations with CellDN predictions. NEN-only processing continues to normalize supplied cell mentions without adding duplicate originals.
- With `pmids=...`, this PubTator3 export is typically title plus abstract content.

### 8. Generic JSON input from another tool

Use this when integrating CellDN with upstream tools that emit JSON rather than BioC.

```bash
celldn run-bioc \
  inputs/upstream_annotations.json \
  outputs/upstream_annotations.normalized.json \
  --task nen \
  --input-format json \
  --output-format bioc-json \
  --nen-model models/CellExLink-Sapbert
```

### 9. PMID abstract retrieval plus prediction

Use this to show retrieval followed by CellDN inference.

```bash
celldn predict-pmid \
  --pmid 31261512 \
  --text-source abstract \
  --output outputs/pmid_31261512.predicted.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 10. PMID or PMCID full text with abstract fallback

Use this to show full-text mode when a PMC OA or Europe PMC full text is available.

```bash
celldn predict-pmid \
  --pmid PMC6821189 \
  --text-source fulltext \
  --output outputs/pmc6821189.fulltext.predicted.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

`--text-source` defaults to `abstract`, including for PMCID inputs. Full-text
retrieval requires the explicit `--text-source fulltext` shown above. If fulltext is not available then  it fallbacks to  abstract.

### 11. PMID list from a text file

Use this to show batch processing.

If `pmids.txt` contains one or more PMIDs or PMCIDs:

```text
31261512
PMC6821189
```

run:

```bash
celldn predict-pmid \
  --ids-file pmids.txt \
  --text-source fulltext \
  --output outputs/pmid_batch.predicted.xml \
  --batch-size 100 \
  --passage-chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### 12. Download current Cell Ontology resource

It reads the official public
CL JSON release.

```bash
celldn download-ontology --output-dir resources
```

The command saves a versioned JSONL file and a companion `.metadata.json` file
in `resources/` relative to the current working directory. Select that exact path with `--ontology-path` for subsequent prediction.
For example, if the reported filename is `cell_ontology_v2026-06-08.jsonl`:

```bash
celldn predict-text \
  --text "Macrophages accumulated near the tumor." \
  --ontology-path resources/cell_ontology_v2026-06-08.jsonl \
  --output outputs/current_ontology.predicted.json
```
The bundled
`2025-12-17` snapshot remains the default for prediction. 


### 13. Many files through one process

Use `run-files` for a directory or a list of files. One pipeline is created for
the command, and its NER model, NEN model, and static ontology embeddings are
reused across all chunks.

```bash
celldn run-files first.pdf \
  --results-dir outputs/ \
  --task end-to-end \
  --passage-chunk-size 128 \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

```bash
celldn run-files papers/ --results-dir outputs/papers/ \
  --batch-size 32 \
  --passage-chunk-size 128 \
  --task end-to-end
```

```bash
celldn run-files first.pdf second.pdf --results-dir outputs/pdfs/
```

Each file receives its own output. 
