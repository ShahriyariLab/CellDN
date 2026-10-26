# Features

This page presents CellDN as a capability set.

## Core workflows

- `NER only`: detect cell mentions
- `NEN only`: normalize supplied mention spans to Cell Ontology
- `End-to-end`: run NER and NEN together
- `PMID-driven`: fetch abstract or full text first, then run prediction
- `PDF`: extract local article text, then recognize and normalize cell mentions

## Interfaces

- `CLI`
- `Python API`

## Inputs

- plain text
- PDF
- BioC XML
- BioC JSON
- PMID / PMCID
- a file list or directory 
- PMID list from a text file

## Content for selection

- abstract only
- full text when available

## Demo matrix

- Cell-type mention recognition in biomedical text
- Cell Ontology normalization for recognized or user-provided mention spans
- End-to-end cell-type recognition and normalization
- Support for plain text, PDF, BioC XML, BioC JSON, and compatible JSON documents
- PMID and PMCID retrieval from NCBI or Europe PMC, with abstract fallback for unavailable full text 
- A separate download command that converts the latest official CL resource
- Compatibility with PubTator3 output, with optional preservation of existing annotations
- Processing of individual files, directories, and multi-document collections
- Configurable batching for identifier lists and local files, with passage-level chunking for structured documents
- Automatic reuse of loaded models, ontology embeddings, and static abbreviation resources
- Visual inspection of predictions in notebook environments
- Python and command-line interfaces
- Documentation, executable examples, automated tests, and continuous integration

For full workflow examples, continue to:

- {doc}`cli_workflows`
- {doc}`python_api_workflows`
- {doc}`workflows`
