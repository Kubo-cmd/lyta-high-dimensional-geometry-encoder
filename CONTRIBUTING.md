# Contributing

## Setup

Use Python 3.9 or newer in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]" build
```

## Required verification

```bash
python -m pytest -q
python -m build
lyta-geometry-encoder --codes 8 --dim 7 --noise 0.02 --trials 100 --seed 42
```

The wheel must contain `lyta_geometry_encoder.py`. Install the exact wheel into
a clean environment and verify the import from outside the repository root.

## Mathematical changes

- State assumptions and domains explicitly.
- Add tests for construction properties, invalid domains, and numerical
  tolerances.
- Compare optimization claims against the regular-simplex baseline whenever
  `code_count <= dim + 1`.
- Label empirical simulation results separately from proofs.
- Do not add machine-checked-proof claims unless the proof compiles without
  admitted goals.

## Pull requests

Keep changes focused, explain compatibility impact, update public claims, and do
not commit environments, caches, generated distributions, credentials, or
unreproducible benchmark output.

Contributions are licensed under the repository's MIT License.
