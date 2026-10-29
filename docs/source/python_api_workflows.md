# Python API workflows

## Create a pipeline

```python
from celldn import CellDNPipeline

pipe = CellDNPipeline.from_pretrained()
```

The required model components load automatically on first use and remain
available for later calls on the same pipeline.

## Plain text

```python
text = "CD8+ T cells infiltrated the tumor."

ner_results = pipe.run_text(text, task="ner")
end_to_end_results = pipe.run_text(
    text,
    task="end-to-end",
)
```

Plain text supports `ner` and `end-to-end`. NEN-only processing requires
supplied mention spans and therefore uses structured input.

Save compact JSON when needed:

```python
from celldn import write_predictions_json

write_predictions_json(end_to_end_results, "text_results.json")
```

## BioC XML, BioC JSON, or compatible JSON

```python
pipe.run_bioc(
    "input.bioc.json",
    "output.bioc.json",
    task="end-to-end",
    passage_chunk_size=128,
)
```

Valid tasks are `ner`, `nen`, and `end-to-end`. `passage_chunk_size` is the
maximum number of passages processed in one inference unit.

Set `preserve_existing_annotations=True` to keep original annotations, such as
PubTator3 genes and diseases, alongside CellDN predictions for `ner` or
`end-to-end`. This option is available on `run_bioc()`, `run_files()`, and
`run_pmids()` and defaults to `False`. Without it, `ner` and `end-to-end`
replace existing annotations.
Original annotations are merged after prediction on retained passages;
excluded passages remain excluded. NEN-only processing continues to normalize
supplied cell mentions without adding duplicate originals. An intermediate
`ner_output_xml` file contains only CellDN-generated annotations.

NEN-only example:

```python
pipe.run_bioc(
    "cell_mentions.xml",
    "normalized.xml",
    task="nen",
)
```

## PDF input

The following example processes a
list of PDFs and saves a separate result file for each successfully processed input. Sample
files are provided in examples/pdfs directory in github repo. Replace the example filenames with paths to
local PDF files. The examples below assume these files are in your current working directory.

```python
outputs = pipe.run_files(
    "first.pdf",
    "outputs/",
    task="end-to-end",
    passage_chunk_size=128,
)
```

## Multiple local files

```python
pdf_inputs = ["first.pdf", "second.pdf", "third.pdf"]
# Alternatively, use a folder containing PDFs:
# pdf_inputs = "pdfs/"

outputs = pipe.run_files(
    pdf_inputs,
    "outputs/",
    task="end-to-end",
)
```
Every input receives a separate output file.

## PMID and PMCID processing

```python
output = pipe.run_pmids(
    ["30243656", "PMC6821189"],
    "publication_results.xml",
    task="end-to-end",
    text_source="fulltext",
    batch_size=100,
    passage_chunk_size=128,
)
```

The method retrieves and processes large identifier lists in bounded chunks,
then merges them into the requested output file. 

## Reuse one pipeline

```python
pipe.run_text("T cells were detected.", task="ner")
pipe.run_bioc("corpus.xml", "corpus_results.xml")
pipe.run_files("first.pdf", "article_results/")
pipe.run_pmids(["30243656"], "pmid_results.xml")
```

The same NER model, NEN encoder, and static ontology embeddings are reused
across these calls. Document-specific abbreviation context is rebuilt for each
input collection.
