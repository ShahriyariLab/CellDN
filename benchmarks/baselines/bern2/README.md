# BERN2 baseline

This folder contains the BERN2 baseline for the CellExLink benchmark.

BERN2 is treated as an external baseline, similar to VANER2 and scispaCy. Install BERN2 separately using the external-baseline instructions here:

https://github.com/dmis-lab/BERN2

## Expected layout

Place the BERN2 code and model under this baseline folder:

```text
benchmarks/baselines/bern2/
  run_bern2.py
  BERN2/
  models/
    dmis-lab/
      bern2-ner/
  model_outputs/
```

Benchmark input files are read from:

```text
benchmarks/data/<DATASET>/test.xml
```

Outputs are written to:

```text
benchmarks/baselines/bern2/model_outputs/
```

## Run BERN2

From the repository root:

```bash
python benchmarks/baselines/bern2/run_bern2.py
```

By default this runs CRAFT test data. To run another dataset:

```bash
DATASET=Celllink python benchmarks/baselines/bern2/run_bern2.py
```

## Reproduce benchmark evaluation

BERN2 output files are saved under `benchmarks/baselines/bern2/model_outputs/`. Run evaluation from the CellExLink repository root.


NER evaluation:

```bash
python benchmarks/evaluation/evaluate_ner.py \
  --gold JNLPBA=benchmarks/data/JNLPBA/test.xml \
  --pred JNLPBA=benchmarks/baselines/bern2/model_outputs/JNLPBA_test.bern2.ner.xml \
  --system JNLPBA \
  --output-csv benchmarks/baselines/bern2/model_outputs/bern2_ner_results.csv
```

Gold-span NEN evaluation:

```bash
python benchmarks/evaluation/evaluate_nen.py \
  --gold CRAFT=benchmarks/data/CRAFT/test.xml \
  --pred Celllink=benchmarks/baselines/bern2/model_outputs/CRAFT_test.bern2.gold.normalized.xml \
  --model-names BERN2 \
  --output-csv benchmarks/baselines/bern2/model_outputs/bern2_nen_results.csv
```

Strict end-to-end evaluation:

```bash
python benchmarks/evaluation/evaluate_end_to_end.py \
  --gold CRAFT=benchmarks/data/CRAFT/test.xml \
  --pred CRAFT=benchmarks/baselines/bern2/model_outputs/CRAFT_test.bern2.normalized.xml \
  --model-names BERN2 \
  --output-csv benchmarks/baselines/bern2/model_outputs/bern2_end_to_end_results.csv
```
