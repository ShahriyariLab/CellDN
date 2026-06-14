# Training

This folder contains developer-facing training scripts for the two main CellExLink model components:

- `recognition/train_ner.py` for cell-type named entity recognition
- `normalization.py/train_nen.py` for Cell Ontology normalization

Training is not required for normal package use. Most users should download the released checkpoints with `cellexlink download-models` and follow the main documentation under `docs/source/`.

Use this folder when you want to:

- fine-tune CellExLink on new datasets
- modify the developer training pipeline

The CellLink training data can be downloaded from the original Zenodo record: <https://doi.org/10.5281/zenodo.18090009>. The released CellExLink training workflow used the training split from that dataset release.

## Install training dependencies

From the repository root:

```bash
python -m pip install -e ".[training]"
```

The core package already includes the main model dependencies. The `training` extra adds training-oriented packages.


## NER training

`training/recognition/train_ner.py` fine-tunes a token-classification model for CellExLink cell-type recognition.

The script accepts either:

- BioC XML input with `--train-xml`,
- JSON, JSONL, or CSV input with `--train-file`.

For JSON-like inputs, each record should contain:

```text
text
entities
```

where `entities` is a list of span dictionaries such as:

```json
{"start": 4, "end": 20, "label": "cell_type"}
```

These offsets are local to the record text.

Example NER training command:

```bash
python training/recognition/train_ner.py \
  --model-path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --train-xml path/to/train.xml \
  --validation-xml path/to/validation.xml \
  --output-dir models/CellExLink-bioformer16L \
  --num-train-epochs 10 \
  --per-device-train-batch-size 8 
```

The script writes converted inputs, checkpoints, and training outputs under the requested `--output-dir`.

## NEN training

`training/normalization.py/train_nen.py` fine-tunes a sentence-transformers/SapBERT-style encoder for Cell Ontology normalization.

Example NEN training command:

```bash
python training/normalization.py/train_nen.py \
  --model-name-or-path cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --ontology-jsonl src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --output-dir models/CellExLink-Sapbert \
  --epochs 10 \
  --batch-size 32
```

The NEN script writes the trained model to the requested output directory.

## Reusing trained checkpoints

After training, use the resulting model directories exactly as you would use downloaded checkpoints:

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)
```

You can also point the CLI to trained checkpoints with:

```bash
cellexlink predict-text \
  --text "The mesothelial cell was detected." \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --output outputs/predictions.jsonl
```
