# Testing

This folder contains developer-facing tests for validating CellExLink package behavior.

## Install test dependencies

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

Current coverage includes:

- package import and public API smoke checks
- CLI availability
- BioC input/output behavior
- ontology loader behavior

Run the full test suite from the repository root with:

```bash
pytest
```

Run a smaller subset during development with commands such as:

```bash
pytest tests/test_import.py
pytest tests/test_bioc_io.py
```

Tests are most important after changes to:

- public APIs
- command-line commands
- BioC or JSONL input/output handling
- packaged resources
- release or manuscript reproduction workflows
