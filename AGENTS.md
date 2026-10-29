# Project Conventions for mini-pi

## Core Philosophy
1. Keep the core minimal and fast.
2. Rely on Pydantic models for explicit schema validation.
3. Keep providers behind unified abstractions.
4. Tools must be deterministic and return exact, structured feedback.
5. Never edit without reading the file first.
6. Safeguard workspace boundary against directory traversal.

## Environment
Always run Python code, tests, and agent invocations inside the conda env `aisystem`:

```bash
conda run -n aisystem python -m pytest
conda run -n aisystem python -m mini_pi.cli "..."
```

Do not use the base env or bare `python`.

