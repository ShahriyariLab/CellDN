# Python API workflows

## Create a pipeline

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)
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
existing spans and therefore uses structured input.

Save compact JSON when needed:

```python
from cellexlink import write_predictions_json

write_predictions_json(end_to_end_results, "text_results.json")
```

## BioC XML, BioC JSON, or compatible JSON

```python
pipe.run_bioc(
    "input.bioc.json",
    "output.bioc.json",
    task="end-to-end",
    preserve_existing_annotations=True,
    passage_chunk_size=128,
)
```

Valid tasks are `ner`, `nen`, and `end-to-end`. `passage_chunk_size` is the
maximum number of passages processed in one inference unit.

NEN-only example:

```python
pipe.run_bioc(
    "cell_mentions.xml",
    "normalized.xml",
    task="nen",
)
```

## Multiple local files

```python
outputs = pipe.run_files(
    ["first.txt", "second.xml", "third.bioc.json"],
    "results/",
    task="end-to-end",
    batch_size=32,
    passage_chunk_size=128,
)
```

Every input receives a separate output file. Plain-text files produce compact
JSON; structured files produce BioC XML or BioC JSON.

## PMID and PMCID processing

```python
pipe.run_pmids(
    ["30243656", "PMC1234567"],
    "publication_results.xml",
    task="end-to-end",
    text_source="abstract",
    batch_size=100,
    passage_chunk_size=128,
)
```

The method retrieves and processes large identifier lists in bounded chunks,
then merges them into the requested output file. Use
`keep_intermediate_bioc="retrieved.xml"` to keep the retrieved text as a
separate BioC XML file.

## Reuse one pipeline

```python
pipe.run_text("T cells were detected.", task="ner")
pipe.run_bioc("corpus.xml", "corpus_results.xml")
pipe.run_pmids(["30243656"], "pmid_results.xml")
```

The same NER model, NEN encoder, and static ontology embeddings are reused
across these calls. Document-specific abbreviation context is rebuilt for each
input collection.
