# Contributing

## Development setup

Use a Python 3.9 or newer virtual environment, then install runtime and
development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Checks before opening a pull request

Run the same focused checks used by CI:

```bash
python -m ruff check app.py data src tests
python -m compileall -q app.py data src tests
python -m pytest -q
python -m pip_audit
```

If training behavior changes, also run `python -m src.train` and the documented
JSON inference example. Do not commit generated model or dataset changes unless
the change is intentional, reviewed, and documented.

## Change guidelines

- Preserve the CLI JSON schema and existing classifier behavior unless a change
  is explicitly requested.
- Add or update tests for behavior changes.
- Keep email content, credentials, tokens, and proprietary data out of commits.
- Prefer focused, reproducible changes over automated or meaningless activity.
