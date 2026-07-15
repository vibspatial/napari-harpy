## Python environment
Canonical environment: `.venv`

Run Python, tests, lint, and tooling by calling the environment's binaries directly via
their `.venv/bin/` path.

```bash
.venv/bin/pytest
.venv/bin/python -m pytest
.venv/bin/pre-commit run ruff --all-files
```

## Test scope

Run only the focused unit tests directly affected by a change.

Do not run the full test suite by default. Run it only when:
- the user explicitly requests it; or
- the change is sufficiently broad that focused tests cannot provide reasonable coverage, in which case ask the user first.

Prefer focused commands such as:

```bash
.venv/bin/pytest -q path/to/test_module.py
.venv/bin/pytest -q path/to/test_module.py::test_specific_behavior
```

Run linting only on the changed or directly affected files where possible.

## Codex config
Repository-local Codex settings for this project live in `.codex/config.toml`.

When checking or updating Codex sandbox, cache, approval, or environment settings for this repo,
use `.codex/config.toml` first rather than `~/.codex/config.toml`.
