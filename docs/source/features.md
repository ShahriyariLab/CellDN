# Features

This page presents CellDN as a capability set rather than as a command reference. It is useful for documentation planning, onboarding, presentations, and choosing the right demo structure.

## Core workflows

- `NER only`: detect cell mentions
- `NEN only`: normalize existing mentions to Cell Ontology
- `End-to-end`: run NER and NEN together
- `PMID-driven`: fetch abstract or full text first, then run prediction

## Interfaces

- `CLI`
- `Python API`

## Inputs

- plain text
- BioC XML
- BioC JSON
- PMID / PMCID
- a file list or directory processed in chunks
- PMID list from a text file

## Content selection

- abstract only
- full text when available

## Demo matrix

- CLI end-to-end on plain text
  This shows the easiest user entry point. Input is plain text and output is JSON.
- Python API end-to-end on plain text
  This shows the same workflow programmatically and is good for notebook and pipeline users.
- CLI NER-only on BioC XML
  This shows structured corpus processing and is useful for users who only want mention detection.
- CLI NEN-only on existing BioC annotations
  This shows integration with external NER tools and is an important interoperability case.
- CLI end-to-end on BioC XML or BioC JSON
  This shows format flexibility and is useful for corpus-scale processing.
- PMID input with abstract selection
  This shows literature retrieval and prediction in one workflow.
- PMID input with full-text selection
  This shows the extended retrieval workflow beyond abstract-only mode.
- PMID list from a `.txt` file
  This shows practical batch processing for real use cases.
- A directory processed with `run-files`
  This shows bounded-memory corpus processing with automatic model reuse.

For full workflow examples, continue to:

- {doc}`cli_workflows`
- {doc}`python_api_workflows`
- {doc}`workflows`
