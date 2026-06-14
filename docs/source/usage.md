# Usage

This page shows the main ways to run CellExLink after installation and model setup. For detailed file formats and output fields, see {doc}`input_output`.

CellExLink supports three user-facing workflows:

1. **NER only**: detect cell-type mention spans.
2. **NEN only**: normalize existing cell-type mentions to Cell Ontology identifiers.
3. **End-to-end extraction**: detect cell-type mentions and normalize them in one workflow.

The Python API exposes all three workflows. The command-line interface exposes end-to-end prediction for plain text and BioC XML, plus normalization of existing BioC annotations.

If you specifically need NER-only inference from the command line, use the Python API or the benchmark helper scripts. The installed `cellexlink` CLI does not currently provide a dedicated NER-only subcommand.

---

## 1. Before running examples

The examples below assume that CellExLink is installed. For the most predictable workflow, first download the released checkpoints with:

```bash
cellexlink download-models --output-dir models
```

Most examples below use local model paths in this layout:

```text
models/
├── CellExLink-bioformer16L/
└── CellExLink-Sapbert/
```

CellExLink can also load the default Hugging Face model IDs directly, but using local paths is often more convenient for reproducible runs and offline reuse.

Create an output directory for the examples:

```bash
mkdir -p outputs
```

---

## 2. Python API

The high-level Python interface is `CellExLinkPipeline`. Use it for notebooks, scripts, or integration into larger biomedical text-mining workflows.

### Create a pipeline

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)
```

If you prefer to load the default Hugging Face model IDs directly, you can omit the local paths:

```python
pipe = CellExLinkPipeline.from_pretrained()
```

This uses the default CellExLink model IDs documented in {doc}`model_checkpoints`.

The package includes the default Cell Ontology and abbreviation resources. You normally do not need to pass resource paths manually. Pass `ontology_path` or `abbreviations_path` only when using custom resources.

```python
pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
    ontology_path="resources/custom_cell_ontology.jsonl",
    abbreviations_path="resources/custom_abbreviations.tsv",
)
```

### End-to-end extraction from text

Use `extract_text()` when you want CellExLink to detect cell-type spans and link them to Cell Ontology identifiers.

```python
from cellexlink import CellExLinkPipeline

text = (
    "The stromal vascular fraction contained mesothelial cells, "
    "smooth muscle cells (SMCs), endothelial cells (ECs), and macrophages. "
    "Mesotheliocytes and SMC clusters formed a separate population."
)

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

results = pipe.extract_text(text, document_id="example-1")

for result in results:
    print(result.to_dict())
```

Each result contains the detected mention text, absolute character offsets, and the predicted Cell Ontology link when available.

### NER only

Use `recognize_text()` when you only need detected cell-type spans.

```python
ner_results = pipe.recognize_text(text, document_id="example-1")

for result in ner_results:
    print(result.to_dict())
```

This is useful when you want to inspect span detection separately from normalization.

### NEN only for known mentions

Use `normalize_mentions()` when the mention spans are already known. This is useful for manually curated mentions, gold-span evaluation in which the mention spans are fixed and the quality of Cell Ontology linking is measured independently of named entity recognition errors, or spans produced by another recognizer.

```python
mentions = [
    {"text": "mesothelial cells", "start": text.index("mesothelial cells")},
    {"text": "SMCs", "start": text.index("SMCs")},
    {"text": "macrophages", "start": text.index("macrophages")},
]

nen_results = pipe.normalize_mentions(
    mentions,
    document_text=text,
    document_id="example-1",
)

for result in nen_results:
    print(result.to_dict())
```

The `mentions` input can contain strings, dictionaries, `RecognizedMention` objects, or `ExtractionResult` objects.

### Write text predictions to JSONL

Use `extract_text_file()` for a plain-text file and JSON Lines output.

```python
pipe.extract_text_file(
    input_txt="examples/sample_input.txt",
    output_jsonl="outputs/sample_predictions.jsonl",
    document_id="sample-text",
)
```

Each line in the JSONL output is one predicted mention. 

---

## 3. BioC XML workflows with the Python API

Use the BioC methods when processing biomedical corpora, benchmark files, or article collections stored as BioC XML.

### NER-only BioC prediction

```python
pipe.recognize_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/sample.ner.xml",
)
```

The output BioC file contains predicted cell-type mention annotations.

### NEN-only BioC normalization

Use `normalize_bioc()` when the input BioC file already contains cell-type mention annotations and you only want to add Cell Ontology links.

```python
pipe.normalize_bioc(
    input_xml="outputs/sample.ner.xml",
    output_xml="outputs/sample.normalized.xml",
)
```

This mode is appropriate for gold-span normalization experiments and for normalizing annotations produced by another NER system. It can also be used on NER output generated by `recognize_bioc()`.

### End-to-end BioC extraction

Use `extract_bioc()` to run NER followed by Cell Ontology normalization.

```python
pipe.extract_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/sample.end_to_end.xml",
    ner_output_xml="outputs/sample.ner.xml",
)
```

The optional `ner_output_xml` argument saves the intermediate NER-only BioC file.

---

## 4. Command-line interface

The `cellexlink` command is installed with the Python package. Use `--help` to inspect available commands and options:

```bash
cellexlink --help
cellexlink predict-text --help
cellexlink predict-bioc --help
cellexlink normalize-bioc --help
```

### End-to-end prediction from a text string

```bash
cellexlink predict-text \
  --text "The mesothelial cell and SMC clusters formed the third population." \
  --output outputs/text_predictions.jsonl \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### End-to-end prediction from a text file

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/sample_text_predictions.jsonl \
  --document-id sample-text \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

The plain-text CLI writes JSON Lines output.

### End-to-end prediction from BioC XML

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/sample.end_to_end.xml \
  --ner-output outputs/sample.ner.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

The `--ner-output` argument is optional. Use it when you want to keep the intermediate NER-only BioC file.

### Normalize existing BioC annotations

```bash
cellexlink normalize-bioc \
  --input outputs/sample.ner.xml \
  --output outputs/sample.normalized.xml \
  --nen-model models/CellExLink-Sapbert
```

This command expects the input BioC XML to already contain cell-type mention annotations.

### Large-corpus BioC example

For larger corpus-style runs, you can use the repository CellLink BioC XML file directly:

```bash
cellexlink predict-bioc \
  --input benchmarks/data/Celllink/test.xml \
  --output outputs/celllink.end_to_end.xml \
  --ner-output outputs/celllink.ner.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --batch-size 8
```

The repository CellLink evaluation file contains 501 unique documents in BioC XML format. Its passage metadata includes PubMed identifiers for all documents and PMC identifiers for most documents, reflecting full-text biomedical literature content prepared for corpus-style processing.

---

## 5. Example scripts

The repository includes small example scripts that can be run from the repository root:

```bash
python examples/quickstart_api.py
python examples/quickstart_text.py
python examples/quickstart_bioc.py
```

These examples are intended as quick checks and templates for custom scripts.
