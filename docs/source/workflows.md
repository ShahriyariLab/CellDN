# Workflow overview

CellDN uses one pipeline class for the public workflows below.

| Workflow | Python API | CLI | Input |
|---|---|---|---|
| Plain-text NER | `run_text(task="ner")` | `predict-text --task ner` | text string or CLI text file |
| Plain-text end-to-end | `run_text(task="end-to-end")` | `predict-text` | text string or CLI text file |
| Structured NER, NEN, or end-to-end | `run_bioc(task=...)` | `run-bioc --task ...` | BioC XML, BioC JSON, compatible JSON |
| PDF NER or end-to-end | `run_files(task=...)` | `run-files --task ...` | PDF, PDF list, or PDF directory |
| Separate results for local files |
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

`run_bioc()`, `run_files()`, and `run_pmids()` all use `passage_chunk_size`
(`--passage-chunk-size` in the CLI) to limit the number of passages processed
in one inference chunk. The default is `128`.

`run_bioc()` processes a large collection in passage chunks. It still provides
the retained text of each document to abbreviation resolution.

`run_files()` combines a bounded number of input files and saves one separate
result file for each processable input. 

`run_pmids()` retrieves identifiers in bounded groups, processes each group,
and merges the results into the single requested output file in identifier order.

