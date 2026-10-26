# PMID, PMCID, and PubTator3 workflows

The examples below assume that a `CellDNPipeline` object named `pipe` has already been created.


## Add CellDN annotations to PubTator3 output

Download a PubTator3 BioC JSON record.  A request made with the `pmids` parameter normally returns the title and abstract. Full-text PubTator3 exports require an available PubMed Central record and the `pmcids` parameter.

```bash
curl -L "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=30243656" \
  -o inputs/pubtator3_annotations.bioc.json
```

Then preserve the existing PubTator3 annotations and add CellDN cell-type annotations:

```python
pipe.run_bioc(
    "inputs/pubtator3_annotations.bioc.json",
    "outputs/pubtator3_plus_celldn.bioc.json",
    task="end-to-end",
    preserve_existing_annotations=True,
)
```

The output remains a BioC JSON document containing both annotation layers on
retained passages; excluded passages remain excluded. A PubTator3 file may
contain one publication or several publications.

`preserve_existing_annotations` is optional and defaults to `False`. Set it to
`True` for `ner` or `end-to-end` to merge original annotations after CellDN
prediction. Without it, those tasks replace existing annotations. The option
also works with `run_files()` and `run_pmids()`; NEN-only processing continues
to use the supplied annotations without adding duplicate originals.

## Process many separate local files

Use `run_files()` when a directory contains many BioC, PubTator3  files and each input should have its own output:

```python
output_paths = pipe.run_files(
    "inputs/",
    "outputs/",
    task="end-to-end",
    preserve_existing_annotations=True,
)
```

CellDN processes the files with the same loaded models and saves one result file for each input file.
