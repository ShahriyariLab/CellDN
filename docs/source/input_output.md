# Input and output formats

This page describes the formats that the current CellExLink CLI and Python API actually read and write. The main public formats are:

- plain text input for end-to-end extraction
- BioC XML input and output for corpus-style processing
- JSON Lines output for plain-text extraction results

BioC XML is the recommended format for document-level biomedical text mining because it preserves document identifiers, passage offsets, and annotation locations.

## Plain-text input

The `cellexlink predict-text` command accepts either a text string or a plain-text file:

```bash
cellexlink predict-text \
  --text "The mesothelial cell and SMC clusters formed the third population." \
  --output outputs/text_predictions.jsonl
```

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.jsonl
```

Internally, CellExLink converts the text to a one-document, one-passage BioC XML file and then runs the same end-to-end pipeline used for BioC input.

For plain text, the default document identifier is `doc0`. You can override it with `--document-id` in the CLI or `document_id=...` in the Python API.

## JSONL output from plain text

`predict-text` writes one JSON object per extracted mention. These rows correspond to the public `ExtractionResult` API object.

Example:

```json
{"document_id":"doc0","passage_index":0,"mention":"mesothelial cell","start":4,"end":20,"entity_type":"cell_type","cl_id":"CL:0000077","cl_label":"mesothelial cell","score":0.99,"source":"dense_retrieval"}
```

Common fields are:

| Field | Description |
|---|---|
| `document_id` | Document identifier used for the input text. |
| `passage_index` | Passage index in the generated BioC document. For plain text, this is usually `0`. |
| `mention` | Extracted mention text. |
| `start` | Absolute start offset. |
| `end` | Absolute end offset. |
| `entity_type` | Predicted entity type, typically `cell_type`. |
| `cl_id` | Top predicted Cell Ontology identifier. |
| `cl_label` | Top predicted ontology label. |
| `score` | Top normalization score. |
| `source` | Matching source reported by the normalizer. |

The JSONL writer omits fields whose values are `None`, `{}`, or `[]`, so some rows may be shorter than others.

## Minimal BioC XML input

The BioC workflows expect an input collection with at least one document and one passage:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>CellExLink examples</source>
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

- `cellexlink predict-bioc` for end-to-end extraction on BioC XML
- `cellexlink normalize-bioc` when the BioC file already contains cell-type annotations

## BioC offsets

CellExLink treats BioC annotation locations as absolute document offsets.

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

The same rule applies throughout the package:

- BioC `location/@offset` is absolute
- `PredictedEntity.start` and `PredictedEntity.end` in `cellexlink.io` are absolute
- plain-text extraction results also report absolute offsets within the generated document

## BioC input for normalization-only workflows

For `normalize-bioc` or gold-span evaluation, the input BioC XML must already contain mention annotations.

Example:

```xml
<annotation id="0">
  <infon key="type">cell_type</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

If gold `identifier` infons are present, CellExLink preserves them during normalization-only workflows. This is important for benchmark evaluation.

## NER BioC output

`predict-bioc` and the NER stage of the pipeline write BioC annotations for detected mentions.

Example:

```xml
<annotation id="T0">
  <infon key="type">cell_type</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

The exact annotation ID is not part of the public contract, but a normal NER output annotation contains:

```text
infon key="type"
location offset
location length
text
```

Additional infons such as `score` or model-specific metadata may also be present.

## Normalized BioC output

The normalization stage adds Cell Ontology predictions to each annotation as BioC infons.

Example:

```xml
<annotation id="T0">
  <infon key="type">cell_type</infon>
  <infon key="CellExLink-Sapbert_id_0">CL:0000077</infon>
  <infon key="CellExLink-Sapbert_identifier_name_0">mesothelial cell</infon>
  <infon key="CellExLink-Sapbert_identifier_score_0">0.99</infon>
  <infon key="CellExLink-Sapbert_match_source">dense_retrieval</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

The current public reader logic looks for normalization keys by suffix, especially:

| Infon suffix | Meaning |
|---|---|
| `_id_0` | Top predicted Cell Ontology identifier. |
| `_identifier_name_0` | Top predicted ontology label or matched name. |
| `_identifier_score_0` | Top prediction score. |
| `_match_source` | Matching route used by the normalizer. |

Additional fields such as `_preferred_label_0`, `_embedding_score_0`, or abbreviation-related infons may also be present.

## Python NEN-only mention input

The Python method `normalize_mentions()` accepts:

- mention strings
- `RecognizedMention` objects
- `ExtractionResult` objects
- dictionaries with a `text`, `mention`, or `mention_text` field

Optional dictionary fields include:

```text
document_id
start
end
length
entity_type
```

Example:

```python
mentions = [
    {"text": "mesothelial cells", "start": 4},
    {"mention": "SMCs"},
    "macrophages",
]
```

If `length` is provided without `end`, CellExLink computes `end = start + length`.

## Internal passage-level JSONL utilities

The `cellexlink.io` module also provides a passage-level JSONL format for utilities such as `convert_bioc_to_jsonl()`. This is different from the public `predict-text` output.

Example row:

```json
{
  "record_id": 0,
  "document_id": "sample-doc-001",
  "passage_id": 0,
  "passage_offset": 0,
  "text": "The mesothelial cell and SMC clusters formed the third population.",
  "entities": [
    {
      "start": 4,
      "end": 20,
      "label": "cell_type",
      "text": "mesothelial cell",
      "score": null,
      "infons": {}
    }
  ]
}
```

Important differences from the plain-text prediction JSONL:

- it is passage-oriented rather than mention-oriented
- it uses `record_id`, not `id`
- entity `start` and `end` values are passage-relative in this JSONL representation
- `passage_offset` is stored separately and can be used to recover absolute BioC offsets

## Shared I/O utilities

The file `src/cellexlink/io.py` is the shared lightweight I/O layer used by the package. It does not run the machine-learning models itself. Instead, it handles the format conversions that connect plain text, BioC XML, and JSONL.

In particular, it is responsible for:

- wrapping plain text as minimal BioC XML with `write_text_as_bioc_xml()`
- reading BioC passages and annotations into Python records
- writing predicted entities back into BioC XML
- converting BioC collections into passage-level JSONL for utilities and testing

For most users, these details matter mainly because they define the offset rules and field layouts described on this page. If you are using the Python API directly, the `cellexlink.io` helpers can also be useful for preparing BioC inputs or inspecting outputs outside the full prediction pipeline.

## Cell Ontology JSONL resource

The current normalization loader is intentionally strict for reproducibility. It expects the same JSONL schema used by the packaged ontology resource:

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

For reproducibility, use the packaged ontology file:

```text
src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl
```
