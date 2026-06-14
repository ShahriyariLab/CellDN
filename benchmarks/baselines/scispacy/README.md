# ScispaCy baseline for CellExLink benchmarks

This folder contains a minimal ScispaCy baseline for the CellLink benchmark.
It is provided only as benchmark/reproducibility code, not as part of the main
`cellexlink` package.

The baseline is intentionally simple: `run_scispacy.py` runs prediction and then
calls the shared benchmark evaluators.

## What is included

- `run_scispacy.py`: runs ScispaCy NER, gold-span NEN, and strict end-to-end NER+NEN on CellLink.
- `environment.yml`: optional Conda environment for the ScispaCy baseline.


ScispaCy, spaCy models, and PyOBO may have dependency constraints that are
different from the main CellExLink PyTorch/Transformers environment. Keep this
baseline in its own environment.

## Install baseline environment

From the repository root:

```bash
conda env create -f benchmarks/baselines/scispacy/environment.yml
conda activate cellexlink-scispacy
```

Install ScispaCy and the NER model used by this baseline:

```bash
python -m pip install scispacy
python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.6.0/en_ner_craft_md-0.6.0.tar.gz
```

## Run the CellLink benchmark

From the repository root:

```bash
python benchmarks/baselines/scispacy/run_scispacy.py
```

By default, this uses:

```text
benchmarks/data/Celllink/test.xml
```

and writes outputs under:

```text
benchmarks/benchmark_outputs/scispacy/
```

## What the runner does

`run_scispacy.py` runs three benchmark modes:

```text
NER:
  writes ner/Celllink_test.scispacy.ner.xml
  calls benchmarks/evaluation/evaluate_ner.py

Gold-span NEN:
  links the gold CellLink annotations
  writes nen/Celllink_test.scispacy.normalized.xml


Strict end-to-end:
  runs ScispaCy NER
  links the predicted spans
  writes end_to_end/Celllink_test.scispacy.normalized.xml
  calls benchmarks/evaluation/evaluate_end_to_end.py
```

The evaluation CSV files are written to:

```text
benchmarks/benchmark_outputs/scispacy/scispacy_celllink_ner.csv
benchmarks/benchmark_outputs/scispacy/scispacy_celllink_nen.csv
benchmarks/benchmark_outputs/scispacy/scispacy_celllink_end_to_end.csv
```

## Run one benchmark mode

```bash
python benchmarks/baselines/scispacy/run_scispacy.py --mode ner
python benchmarks/baselines/scispacy/run_scispacy.py --mode nen
python benchmarks/baselines/scispacy/run_scispacy.py --mode e2e
```


## Notes

- This is a baseline comparison, not a production pipeline.
- The script keeps the evaluation centralized by calling the existing shared
  evaluators: `benchmarks/evaluation/evaluate_ner.py`, `benchmarks/evaluation/evaluate_nen.py`, and `benchmarks/evaluation/evaluate_end_to_end.py`.
