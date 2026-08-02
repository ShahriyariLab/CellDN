# CellExLink: An open-source Python package for Cell Ontology-linked cell-type extraction

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue)](LICENSE.txt)
[![CI](https://github.com/ShahriyariLab/CellExLink/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ShahriyariLab/CellExLink/actions/workflows/ci.yml)
![Language: Python](https://img.shields.io/badge/Language-Python-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShahriyariLab/CellExLink/blob/main/examples/demo.ipynb)

CellExLink is a Python package for recognizing cell-type mentions in biomedical text and linking them to Cell Ontology concepts.

Full installation and usage instructions are available in the
[CellExLink documentation](https://shahriyarilab.github.io/CellExLink/).

## Features

- Cell-type mention recognition in biomedical text
- Cell Ontology normalization for recognized or user-provided mention spans
- Both recognition and  normalization end-to-end processing
- Support for plain text, BioC XML, BioC JSON, and compatible JSON documents
- PMID and PMCID retrieval from NCBI or Europe PMC, using abstracts or available full text
- Compatibility with PubTator3 BioC JSON, with preservation of existing annotations
- Processing of individual files, directories, and multi-document collections
- Configurable batching for identifier lists and local files, with passage-level chunking for structured documents
- Automatic reuse of loaded models, ontology embeddings, and static abbreviation resources
- Visual inspection of predictions in notebook environments
- Python and command-line interfaces
- Documentation, executable examples, automated tests, and continuous integration

## Requirement

The installation of `CellExLink` requires python version 3.10 and above.

## Installation
```bash
python -m pip install cellexlink
python -m pip install "git+https://github.com/ShahriyariLab/CellExLink.git"
```

## Quick Start and usage

Launch the demo notebook in Google Colab, running in Google Colab does not require local compute resources.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShahriyariLab/CellExLink/blob/main/examples/demo.ipynb)

## Citation

Citation
metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

CellExLink is released under the GPL-3.0 license. See [`LICENSE.txt`](LICENSE.txt).

## Contact and contributions

Questions and contributions are welcome through the
[GitHub issue](https://github.com/ShahriyariLab/CellExLink/issues).
