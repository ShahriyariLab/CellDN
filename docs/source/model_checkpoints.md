# Model checkpoints and resources

CellDN uses two model checkpoints and a small set of packaged resources. The checkpoints are downloaded from Hugging Face or loaded from local folders; the resource files are distributed with the Python package.

---

## Default checkpoints

| Component | Default model ID | Purpose |
|---|---|---|
| NER | [`almire/CellExLink-bioformer16L`](https://huggingface.co/almire/CellExLink-bioformer16L) | Detects cell-type mention spans in text. |
| NEN | [`almire/CellExLink-Sapbert`](https://huggingface.co/almire/CellExLink-Sapbert) | Links recognized or user-provided mentions to Cell Ontology identifiers. |

> **Naming note:** The Hugging Face repository IDs retain the `CellExLink-*` names under which the checkpoints were published with the companion methodological study. These are external checkpoint identifiers; the installable package, Python namespace, and command-line interface are named CellDN, `celldn`, and `celldn`, respectively.

---

## Recommended download method

After installing CellDN, use the built-in command-line interface:

```bash
celldn download-models --output-dir models
```

This command is implemented directly in the installed `celldn` CLI. It downloads both default checkpoints from Hugging Face and writes a small `models.json` manifest describing the local model paths.

This creates the local model directory used in the examples:

```text
models/
├── CellExLink-bioformer16L/
├── CellExLink-Sapbert/
└── models.json
```

If `huggingface_hub` is not installed in your environment, install it first:

```bash
python -m pip install huggingface-hub
```

After the download finishes, you can use the local folders in the CLI and Python API examples from {doc}`usage`.


## Manual download from Hugging Face

The checkpoints can also be downloaded directly from Hugging Face.

Using the current Hugging Face CLI:

```bash
hf download almire/CellExLink-bioformer16L \
  --local-dir models/CellExLink-bioformer16L

hf download almire/CellExLink-Sapbert \
  --local-dir models/CellExLink-Sapbert
```

If your environment uses the older Hugging Face CLI command name, the equivalent commands are:

```bash
huggingface-cli download almire/CellExLink-bioformer16L \
  --local-dir models/CellExLink-bioformer16L

huggingface-cli download almire/CellExLink-Sapbert \
  --local-dir models/CellExLink-Sapbert
```

---

## Packaged resources

CellDN also uses smaller resource files for normalization and abbreviation handling:

```text
src/celldn/resources/
├── abbreviations.tsv
└── cell_ontology_v2025-12-17.jsonl
```

These files are included with the Python package, so users normally do not need to download them separately.

To use custom resources, pass their paths explicitly:

```bash
--ontology-path resources/custom_cell_ontology.jsonl \
--abbreviations-path resources/custom_abbreviations.tsv
```


