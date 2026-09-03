# LYTA High-Dimensional Geometry Encoder

[![CI](https://github.com/Kubo-cmd/lyta-high-dimensional-geometry-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/Kubo-cmd/lyta-high-dimensional-geometry-encoder/actions/workflows/ci.yml)

A small, deterministic Python library for spherical-code experiments, regular
simplex construction, seeded random projection, and empirical noisy-recall
measurement.

## What is implemented

- `encode_high_dim_geometry`: normalize a vector or map it through a seeded
  Gaussian projection into a requested dimension.
- `regular_simplex`: construct `n` unit vectors in dimension `d` when
  `2 <= n <= d + 1`.
- `verify_packing_bound`: measure minimum pairwise distance and maximum pairwise
  cosine, then check whether the regular-simplex bound is achieved.
- `evaluate_noisy_recall`: run bounded, seeded nearest-cosine recovery trials
  under Gaussian noise.

## Mathematical boundary

For `n <= d + 1`, a regular simplex has pairwise inner product
`-1 / (n - 1)` and pairwise Euclidean distance
`sqrt(2n / (n - 1))`. The implementation constructs this baseline numerically
and the tests verify those identities within floating-point tolerance.

This repository does not claim a machine-checked proof, global optimality for
arbitrary code sizes, zero drift, cryptographic security, privacy guarantees,
or production memory-system integration. The noisy-recall score is an empirical
measurement for the supplied codebook and parameters, not a theorem.

## Install

```bash
git clone https://github.com/Kubo-cmd/lyta-high-dimensional-geometry-encoder.git
cd lyta-high-dimensional-geometry-encoder
python -m pip install .
```

## Python API

```python
from lyta_geometry_encoder import (
    encode_high_dim_geometry,
    evaluate_noisy_recall,
    regular_simplex,
    verify_packing_bound,
)

encoded = encode_high_dim_geometry([1.0, 2.0, 3.0], dim=64, seed=7)
codes = regular_simplex(code_count=16, dim=15)
report = verify_packing_bound(codes)
recall = evaluate_noisy_recall(
    codes,
    noise_std=0.05,
    trials_per_code=500,
    seed=42,
)

print(report)
print(recall)
```

## Command line

```bash
lyta-geometry-encoder --codes 16 --dim 15 --noise 0.05 --trials 500 --seed 42
```

The command prints one JSON result. Work is rejected when the requested noisy
trials exceed 1,000,000 or an intermediate matrix exceeds 8,000,000 elements.

## Development and verification

```bash
python -m pip install -e ".[test]" build
python -m pytest -q
python -m build
```

CI tests Python 3.9, 3.11, and 3.13, builds the wheel, installs that exact wheel
outside the source tree, imports the public API, and runs a command-line smoke
test. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Release status

No package-index or GitHub release has been published. Version `0.1.0` is a
source and wheel candidate only; publication requires a separate release review.

## License

MIT

PATTERN PERSISTS.
