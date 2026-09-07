# Input and output formats

This page describes the formats that the current CellDN CLI and Python API actually read and write. The main public formats are:

- plain text input for end-to-end extraction
- BioC XML input and output for corpus-style processing
- JSON output for plain-text extraction results


## Plain-text input

The `celldn predict-text` command accepts either a text string or a plain-text file:

```bash
celldn predict-text \
  --text "The mesothelial cell and SMC clusters formed the third population." \
  --output outputs/text_predictions.json
```

```bash
celldn predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.json
```

Internally, CellDN converts the text to a one-document, one-passage BioC XML file and then runs the same end-to-end pipeline used for BioC input.

## JSON output from plain text

`predict-text` writes one JSON object per document. Each object contains an `annotations` array with all extracted mentions for that document.

Example:

```json
{"annotations":[{"mention":"mesothelial cell","entity_type":"cell_type","span":{"begin":4,"end":20},"passage_index":0,"identifier":"CL:0000077","label":"mesothelial cell"},{"mention":"SMC","entity_type":"cell_type","span":{"begin":25,"end":28},"passage_index":0,"identifier":"CL:0000192","label":"smooth muscle cell"}]}
```

Common fields are:

| Field | Description |
|---|---|
| `document_id` | Optional document identifier for the input text. Present only when supplied. |
| `annotations` | Array of extracted mention records for the document. |

Each annotation may include:

| Field | Description |
|---|---|
| `mention` | Extracted mention text. |
| `entity_type` | Predicted entity type, typically `cell_type`. |
| `span.begin` | Absolute start offset. |
| `span.end` | Absolute end offset. |
| `passage_index` | Passage index in the generated BioC document. For plain text, this is usually `0`. |
| `identifier` | Top predicted Cell Ontology identifier. |
| `label` | Top predicted ontology label. |

## Minimal BioC XML input

The BioC workflows expect an input collection with at least one document and one passage:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>CellDN examples</source>
  <date></date>
  <key>cell-type-extraction</key>
  <document>
    <id>sample-doc-001</id>
    <passage>
      <infon key="type">abstract</infon>
      <offset>0</offset>
      <text>The mesothelial cell and SMC clusters formed the third population.</text>
    </passage>
  </document>
</collection>
```

Required elements for normal BioC use are:

```text
collection/document/id
collection/document/passage/offset
collection/document/passage/text
```

Useful optional elements are:

```text
collection/source
collection/date
collection/key
passage/infon[@key="type"]
```

Use:

- `celldn run-bioc --task end-to-end` for extraction plus normalization
- `celldn run-bioc --task nen` when the BioC file already contains cell-type annotations

## BioC offsets

CellDN treats BioC annotation locations as absolute document offsets.

Example:

```xml
<passage>
  <offset>100</offset>
  <text>CD8+ T cells were detected.</text>
</passage>
```

If the mention `CD8+ T cells` begins at the start of that passage, the written annotation location is:

```xml
<location offset="100" length="12" />
```

## BioC input for normalization-only workflows

For `run-bioc --task nen`, the input BioC XML must already contain mention annotations.

Example:

```xml
<annotation id="0">
  <infon key="type">cell_type</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

## NER BioC output

`run-bioc --task ner` and the NER stage of the pipeline write BioC annotations for detected mentions.

Example:

```xml
<annotation id="T0">
  <infon key="type">cell_type</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

Normal NER output annotation contains:

```text
infon key="type"
location offset
location length
text
```
## Normalized BioC output

The normalization stage adds Cell Ontology predictions to each annotation as BioC infons.

Example:

```xml
<annotation id="T0">
  <infon key="type">cell_type</infon>
  <infon key="identifier">CL:0000077</infon>
  <infon key="label">mesothelial cell</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```


| Infon key | Meaning |
|---|---|
| `identifier` | Top predicted Cell Ontology identifier. |
| `label` | Top predicted ontology label or matched name. |

## Cell Ontology JSONL resource

```json
{
  "norm_concept_id": "CL:0000077",
  "norm_preferred_label": "mesothelial cell",
  "synonyms": ["mesotheliocyte"],
  "namespace": "CL"
}
```

The required field names are:

```text
norm_concept_id
norm_preferred_label
synonyms
namespace
```

Use the packaged ontology file or dowload from other resources.

```text
src/celldn/resources/cell_ontology_v2025-12-17.jsonl
```
