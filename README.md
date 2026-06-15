# CellExLink:  cell-type extraction tool
![License: Apache 2.0](https://img.shields.io/github/license/ssciwr/mailcom)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/mailcom/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/mailcom)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_mailcom&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/mailcom)


CellExLink is a biomedical text-mining package for cell-type named entity
recognition, Cell Ontology normalization, and end-to-end cell-type extraction
from biomedical text. It supports both plain-text and BioC XML workflows.

Software package documentation which includes installation and usage is available here:[CellExLink documentation](https://almihan.github.io/CellExLink/build/html/index.html)


## Installation

Create a Python virtual environment, ie. conda. Install `CellExLink` into the environment using  
`python -m pip install cellexlink`

CellExLink's NER and NEN components' large model checkpoints  are **not** stored in this repository and  are hosted on Hugging Face.
Download model checkpoints separately and place them under models/, or pass
their paths explicitly through the Python API or command-line interface. See the [documentation](https://almihan.github.io/CellExLink/build/html/index.html) for details.

## Quick example

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

results = pipe.extract_text(
    "The stromal vascular fraction contained mesothelial cells, smooth muscle cells (SMCs), endothelial cells (ECs), and macrophages. Mesotheliocytes and SMC clusters formed a separate population."
)

for result in results:
    print(result.to_dict())

```

## Additioanl guides:

Additional repository README files are available for development  workflows:

- Benchmark evaluation: [benchmarks/README.md](https://github.com/ShahriyariLab/CellExLink/blob/main/benchmarks/README.md)

- Test suite: [tests/README.md](https://github.com/ShahriyariLab/CellExLink/blob/main/tests/README.md)

- Training workflows: [training/README.md](https://github.com/ShahriyariLab/CellExLink/blob/main/training/README.md)

## Benchmark data

Benchmark test datasets are provided in the repository under `benchmarks/data/`. See `benchmarks/data/README.md` for details.

Use of these datasets is subject to the license and terms specified by the original source. Please refer to the Zenodo record [link](https://zenodo.org/records/18090009) for citation and licensing information.

## Citation

If you use CellExLink in a manuscript, please include a formal software citation in your reference list for the exact released version you used, for example `CellExLink v0.1.0`. This software citation is separate from citing any companion article about the method.

The repository includes a standard [CITATION.cff](CITATION.cff) file that GitHub and citation managers can use to generate a versioned software reference.


## Getting in touch
Do not hesitate to [open an issue](https://github.com/ShahriyariLab/CellExLink/issues) to get in touch with us with requests or questions. Any community contributions are encouraged! 
