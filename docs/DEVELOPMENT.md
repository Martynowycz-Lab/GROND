# Development

Run the public checks before committing:

```bash
python -m pip install -e ".[dev]"
ruff check src/grond tests
pytest
GROND validate
GROND doctor --probe-bloch
```

Keep public modules in the domain folders. Historical experiments should not be added to the package tree unless they are callable, documented, tested, and free of dataset-specific assumptions.
