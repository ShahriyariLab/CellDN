# Benchmarking and evaluation

This README is the benchmark evaluation workflow. It focuses on the evaluation commands for CellExLink itself. Dataset details remain in `benchmarks/data/README.md`. Baseline-specific setup remains in the README files under  `benchmarks/baselines/`. 

CellExLink is evaluated at three levels:

1. **NER**: detect cell-type mention spans.
2. **Gold-span NEN**: normalize gold cell-type mentions to Cell Ontology identifiers.
3. **Strict end-to-end extraction**: predict both the mention span and the Cell Ontology identifier correctly.

These tasks correspond to the two-stage CellExLink pipeline: recognition first, then normalization.

## Benchmark inputs

The benchmark scripts expect BioC XML input from the repository evaluation set under `benchmarks/data/`.

For CellLink specifically, the repository evaluation workflow uses labeled validation data for evaluation because the official test split does not provide gold labels. 

The original CellLink resource distinguishes `cell phenotype`, `heterogeneous cell population`, and `vague cell population` annotations. Following the CellLink evaluation setup described in the [CellLink paper](https://www.biorxiv.org/content/10.64898/2026.02.11.705457v1.full), vague cell population annotations are excluded because their identities cannot be determined precisely from context and they are not linked to Cell Ontology identifiers. The evaluation therefore focuses on identifiable cell phenotypes and heterogeneous cell populations. 

For normalization evaluation, the current evaluator supports two gold-identifier interpretation modes:

- `exactIDsOnly_iterator`: default strict mode
- `allLabels_iterator`: optional broader all-label mode

The default benchmark behavior uses `exactIDsOnly_iterator` with `cell_type` annotations across the benchmark corpora.

## Evaluation

Use the current benchmark entrypoints in this repository for  evaluation. 

## Run end-to-end

```bash
python benchmarks/run_cellexlink.py \
  --mode full \
  --input benchmarks/data/Celllink/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/end_to_end \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --manifest benchmarks/benchmark_outputs/cellexlink/end_to_end/run_manifest.csv
```

This produces normalized BioC XML, the intermediate NER output, and a run manifest in:

```text
benchmarks/benchmark_outputs/cellexlink/end_to_end/
```

## Run NER

```bash
python benchmarks/run_cellexlink.py \
  --mode ner \
  --input benchmarks/data/CRAFT/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/ner/
```

## Run NEN

```bash
python benchmarks/run_cellexlink.py \
  --mode normalize \
  --input benchmarks/data/CRAFT/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/nen
```

## NER evaluation

```bash
python benchmarks/evaluation/evaluate_ner.py \
  --gold CRAFT=benchmarks/data/CRAFT/test.xml \
  --pred CRAFT=benchmarks/benchmark_outputs/cellexlink/ner/CRAFT_test.ner.xml \
  --system CellExLink \
  --output-csv benchmarks/benchmark_outputs/table_ner_results.csv
```

This evaluator reports exact-span and relaxed-span precision, recall, and F1 for cell-type mention detection.

## Gold-span NEN evaluation

```bash
python benchmarks/evaluation/evaluate_nen.py \
  --gold CRAFT=benchmarks/data/CRAFT/test.xml \
  --pred CRAFT=benchmarks/benchmark_outputs/cellexlink/nen/CRAFT_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmarks/benchmark_outputs/table5_gold_span_normalization_results.csv
```

To use the broader all-label gold interpretation instead of the default strict mode, add:

```bash
--iterator allLabels_iterator
```

## Strict end-to-end

```bash
python benchmarks/evaluation/evaluate_end_to_end.py \
  --gold BioID=benchmarks/data/BioID/test.xml \
  --pred BioID=benchmarks/benchmark_outputs/cellexlink/end_to_end/BioID_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmarks/benchmark_outputs/table7_end_to_end_results.csv
```

## Baselines and repository README files

This README focuses on CellExLink evaluation. For comparison systems details, see:

- `benchmarks/README.md`
- `benchmarks/data/README.md`
- `benchmarks/baselines/README.md`
- `benchmarks/baselines/scispacy/README.md`
- `benchmarks/baselines/bern2/README.md`
- `benchmarks/baselines/vaner2/README.md`
