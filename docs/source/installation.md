# Installation

This page describes how to install CellExLink for normal use.

---

## Requirements

Recommended environment:

- Python >= 3.10
- Linux, macOS, or Windows

CellExLink uses task-focused biomedical encoder models for cell-type recognition and Cell Ontology normalization.

Neural networks model checkpoints are not stored in the repository. After installing the package, see {doc}`model_checkpoints` for model download and path configuration.

---

## 1. Create a Python environment

Using `venv` on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using `conda`:

```bash
conda create -n cellexlink python=3.12 -y
conda activate cellexlink
python -m pip install --upgrade pip
```

---

## 2. Install from PyPI

Install the published package from PyPI:

```bash
python -m pip install cellexlink
```

---

## 3. Install from GitHub

Install it directly from GitHub:

```bash
python -m pip install "git+https://github.com/ShahriyariLab/CellExLink.git"
```

---

## 4. Install from a local clone


```bash
git clone https://github.com/ShahriyariLab/CellExLink.git
cd CellExLink
python -m pip install .
```

These installation methods install the Python package from `src/cellexlink`
and its packaged resource files. They do not install repository folders such
as `docs/`, `examples/`, or `tests/`.

---

## 5. Check the Python installation

Check that the package imports correctly:

```bash
python - <<'PY'
import cellexlink
from cellexlink import CellExLinkPipeline

print("CellExLink version:", cellexlink.__version__)
print("Pipeline class:", CellExLinkPipeline.__name__)
PY
```

This check does not require downloaded model checkpoints.

---

## 6. Check the command-line interface

The command-line interface is installed automatically with the Python package. No separate CLI package is needed.

Check that the `cellexlink` command is available:

```bash
cellexlink --help
cellexlink --version
```

The main CLI commands are:

```text
cellexlink download-models
cellexlink predict-text
cellexlink run-bioc
cellexlink run-files
cellexlink predict-pmid
```

You can inspect each command without downloading models:

```bash
cellexlink download-models --help
cellexlink predict-text --help
cellexlink run-bioc --help
cellexlink run-files --help
```

If the command is not found, confirm that the environment where CellExLink was installed is activated, then try:

```bash
python -m cellexlink.cli --help
```

---

## 7. Next steps

After installation:

- See {doc}`model_checkpoints` to download or configure the NER and NEN model checkpoints.
- See {doc}`usage` for Python API and command-line examples.
- See {doc}`input_output` for plain-text, JSON, and BioC XML formats.
