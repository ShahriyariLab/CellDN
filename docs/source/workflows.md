# Workflow overview

CellDN uses one pipeline class for four public workflows.

| Workflow | Python API | CLI | Input |
|---|---|---|---|
| Plain-text NER | `run_text(task="ner")` | `predict-text --task ner` | text string or CLI text file |
| Plain-text end-to-end | `run_text(task="end-to-end")` | `predict-text` | text string or CLI text file |
| Structured NER, NEN, or end-to-end | `run_bioc(task=...)` | `run-bioc --task ...` | BioC XML, BioC JSON, compatible JSON |
| Multiple local files | `run_files(task=...)` | `run-files --task ...` | files or directories |
| PMID/PMCID retrieval plus prediction | `run_pmids(task=...)` | `predict-pmid --task ...` | one identifier or an identifier list |

## Automatic component reuse

The NER predictor and NEN linker are loaded only when their task is first used.
The pipeline then keeps the loaded model components, Cell Ontology aliases,
ontology embeddings, and static abbreviation resources in memory for later
calls. 

Document-specific abbreviation mappings are rebuilt for every collection or
processing chunk. Context from one document is therefore not carried into an
unrelated document.

## Chunking

`run_bioc()` processes a large collection in passage chunks. It still provides
the complete text of each document to abbreviation resolution.

`run_files()` combines a bounded number of input files and saves one separate
result file for each input. A second limit controls the number of BioC passages
processed together.

`run_pmids()` retrieves identifiers in bounded groups, processes each group,
and merges the results into the single requested output file in identifier order.


