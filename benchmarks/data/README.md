# Benchmark datasets

This README summarizes the evaluation datasets used in the benchmark experiments. For the benchmark workflow and evaluation commands, see `benchmarks/evaluation/README.md`.

## Dataset location

The repository stores the evaluation BioC XML files under:

```text
benchmarks/data/
├── AnatEM/test.xml
├── BioID/test.xml
├── CRAFT/test.xml
├── Celllink/test.xml
└── JNLPBA/test.xml
```

All files are BioC XML inputs used by the benchmark runner and evaluation scripts.

The datasets can also be downloaded from publicly available third-party corpora. The benchmark data are available from the original Zenodo record: <https://doi.org/10.5281/zenodo.18090009>. After downloading, place each dataset under `benchmarks/data/` using the dataset folder names shown above.

All datasets are used for named entity recognition evaluation. However, only `BioID`, `CRAFT`, and `CellLink` are used for named entity normalization and end-to-end evaluation, because the remaining datasets do not provide Cell Ontology identifier ground-truth labels.

## Important note on the CellLink dataset

For CellLink data, the original test split does not provide ground-truth normalization labels. Therefore, this repository uses the validation split for evaluation. The CellLink data provided in this repository for evaluation correspond to the validation split from the original source.

Following the CellLink evaluation setup described in the [CellLink paper](https://www.biorxiv.org/content/10.64898/2026.02.11.705457v1.full), annotations labeled as vague cell populations are excluded because their identities cannot be determined precisely from context and they are not linked to Cell Ontology identifiers. Therefore, the CellLink evaluation data provided in this repository already exclude vague cell population annotations.

For CellLink normalization, following the CellLink evaluation setup described in the [CellLink paper](https://www.biorxiv.org/content/10.64898/2026.02.11.705457v1.full), `exactIDsOnly_iterator` is the default because `(skos:related)` does not define a single exact Cell Ontology target. It marks broader, narrower, or otherwise associated CL terms and may correspond to multiple biologically acceptable related identifiers. The primary NEN and strict end-to-end evaluations therefore use only exact Cell Ontology links. `allLabels_iterator` retains the `(skos:related)` IDs for supplementary analysis.

To enable the optional broader evaluation mode, use:

```bash
--iterator allLabels_iterator
```
