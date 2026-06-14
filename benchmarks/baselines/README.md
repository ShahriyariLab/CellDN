# Baseline comparison materials

This folder contains optional baseline materials used for CellExLink benchmark comparisons. These baselines are kept separate from the main `cellexlink` package because they can require different dependencies, different environments, or external upstream repositories.

For the CellExLink-only manuscript workflow, start with:

- `benchmarks/README.md`
- `benchmarks/evaluation/README.md`

Use this folder when you want to reproduce or inspect comparison systems.

## Included baselines

- `scispacy/`: runnable ScispaCy baseline code is included in this repository.
- `bern2/`: wrapper notes for running and evaluating the external BERN2 system.
- `vaner2/`: wrapper notes for running and evaluating the external VANER2 system.

Each baseline keeps its own README because setup and runtime requirements differ.

## Recommended use

There are two common baseline steps:

1. Run a baseline in its own environment.
2. Score the generated XML outputs with the shared CellExLink evaluators.

## Output locations

Baseline prediction files are stored under the corresponding baseline folder, for example:

```text
benchmarks/baselines/bern2/model_outputs/
benchmarks/baselines/vaner2/model_outputs/
```

ScispaCy writes benchmark outputs under:

```text
benchmarks/benchmark_outputs/scispacy/
```

These directories store the generated output files for baseline runs.

## Shared evaluation scripts

Use the shared CellExLink benchmark evaluators for scoring baseline predictions:

```text
benchmarks/evaluation/evaluate_ner.py
benchmarks/evaluation/evaluate_nen.py
benchmarks/evaluation/evaluate_end_to_end.py
```

Baseline-specific README files describe which evaluator is appropriate for each system:

- some baselines are NER only
- some support gold-span NEN
- some support strict end-to-end comparison

## Baseline-specific notes

- `scispacy/README.md`: local runnable baseline in a separate environment.
- `bern2/README.md`: external BERN2 setup plus evaluation guidance.
- `vaner2/README.md`: external VANER2 setup plus NER-only evaluation guidance.

Keep the main documentation focused on CellExLink itself, and use these baseline notes only for comparison-system reproduction.
