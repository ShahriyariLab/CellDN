# CellExLink: cell-type extraction from biomedical text

![License: GPL-3.0](https://img.shields.io/github/license/ShahriyariLab/CellExLink)
[![CI](https://github.com/ShahriyariLab/CellExLink/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ShahriyariLab/CellExLink/actions/workflows/ci.yml)
![Language](https://img.shields.io/github/languages/top/ShahriyariLab/CellExLink)

CellExLink is a Python package for cell-type named entity recognition (NER),
Cell Ontology normalization (NEN), from biomedical text.

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

## Quick Start

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
