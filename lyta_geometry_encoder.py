"""Deterministic tools for high-dimensional spherical code experiments."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
MAX_MATRIX_ELEMENTS = 8_000_000
MAX_RECALL_TRIALS = 1_000_000


@dataclass(frozen=True)
class PackingReport:
    """Measured geometry for a finite unit-vector codebook."""

    code_count: int
    dimension: int
    minimum_distance: float
    maximum_cosine: float
    simplex_bound: float | None
    simplex_bound_achieved: bool


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _unit_rows(codes: ArrayLike) -> FloatArray:
    array = np.asarray(codes, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("codes must have shape (count, dimension) with count >= 2")
    if array.size > MAX_MATRIX_ELEMENTS:
        raise ValueError("codebook exceeds the configured work bound")
    if array.shape[0] * array.shape[0] > MAX_MATRIX_ELEMENTS:
        raise ValueError("pairwise comparison exceeds the configured work bound")
    if not np.all(np.isfinite(array)):
        raise ValueError("codes must contain only finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError("codes must not contain zero vectors")
    return array / norms[:, np.newaxis]


def encode_high_dim_geometry(
    input_vector: ArrayLike,
    dim: int = 768,
    *,
    seed: int = 0,
    radius: float = 1.0,
) -> list[float]:
    """Project one finite nonzero vector into ``dim`` dimensions.

    When the input already has the requested dimension, this only normalizes and
    scales it. Otherwise, a seeded Gaussian projection is used. The result is
    deterministic for the same vector, dimension, seed, and radius. This is a
    geometric transform, not encryption and not a privacy mechanism.
    """

    output_dim = _positive_int(dim, "dim")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive finite number")

    vector = np.asarray(input_vector, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("input_vector must be nonempty and finite")
    if np.linalg.norm(vector) == 0:
        raise ValueError("input_vector must be nonzero")

    if vector.size == output_dim:
        projected = vector.copy()
    else:
        if vector.size * output_dim > MAX_MATRIX_ELEMENTS:
            raise ValueError("projection exceeds the configured work bound")
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((output_dim, vector.size)) / math.sqrt(output_dim)
        projected = projection @ vector

    norm = float(np.linalg.norm(projected))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("projection produced a degenerate vector")
    return (projected * (radius / norm)).tolist()


def regular_simplex(code_count: int, dim: int) -> FloatArray:
    """Construct ``code_count`` unit vectors forming a regular simplex.

    The construction exists when ``2 <= code_count <= dim + 1``. It achieves
    pairwise cosine ``-1 / (code_count - 1)`` and Euclidean distance
    ``sqrt(2 * code_count / (code_count - 1))`` up to floating-point error.
    """

    count = _positive_int(code_count, "code_count")
    dimension = _positive_int(dim, "dim")
    if count < 2:
        raise ValueError("code_count must be at least 2")
    if count > dimension + 1:
        raise ValueError("a regular simplex requires code_count <= dim + 1")
    if count * dimension > MAX_MATRIX_ELEMENTS:
        raise ValueError("simplex exceeds the configured work bound")

    # The Helmert matrix columns are an orthonormal basis for the subspace
    # orthogonal to the all-ones vector. Scaling its rows gives a unit simplex.
    coordinates = np.zeros((count, count - 1), dtype=np.float64)
    for column in range(count - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        coordinates[: column + 1, column] = 1.0 / denominator
        coordinates[column + 1, column] = -(column + 1) / denominator
    coordinates *= math.sqrt(count / (count - 1))

    result = np.zeros((count, dimension), dtype=np.float64)
    result[:, : count - 1] = coordinates
    return result


def verify_packing_bound(codes: ArrayLike, *, tolerance: float = 1e-9) -> PackingReport:
    """Measure pairwise geometry and test the regular-simplex bound.

    For at most ``dimension + 1`` unit vectors, the best possible maximum
    pairwise cosine is ``-1 / (count - 1)``. The report marks the bound achieved
    only when the measured maximum cosine matches that value within tolerance.
    No formal proof artifact is claimed.
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative finite number")

    unit = _unit_rows(codes)
    count, dimension = unit.shape
    gram = unit @ unit.T
    upper = np.triu_indices(count, 1)
    cosines = gram[upper]
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosines))
    maximum_cosine = float(np.max(cosines))
    minimum_distance = float(np.min(distances))

    bound = -1.0 / (count - 1) if count <= dimension + 1 else None
    achieved = bound is not None and abs(maximum_cosine - bound) <= tolerance
    return PackingReport(
        code_count=count,
        dimension=dimension,
        minimum_distance=minimum_distance,
        maximum_cosine=maximum_cosine,
        simplex_bound=bound,
        simplex_bound_achieved=achieved,
    )


def evaluate_noisy_recall(
    codes: ArrayLike,
    *,
    noise_std: float = 0.05,
    trials_per_code: int = 100,
    seed: int = 0,
) -> float:
    """Measure nearest-cosine code recovery under seeded Gaussian noise."""

    unit = _unit_rows(codes)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be a nonnegative finite number")
    trials = _positive_int(trials_per_code, "trials_per_code")
    total = trials * unit.shape[0]
    if total > MAX_RECALL_TRIALS:
        raise ValueError("total noisy-recall trials must not exceed 1,000,000")
    if trials * unit.shape[1] > MAX_MATRIX_ELEMENTS:
        raise ValueError("noisy-recall batch exceeds the configured work bound")
    if trials * unit.shape[0] > MAX_MATRIX_ELEMENTS:
        raise ValueError("similarity comparison exceeds the configured work bound")

    rng = np.random.default_rng(seed)
    correct = 0
    for index, code in enumerate(unit):
        queries = code + rng.normal(0.0, noise_std, size=(trials, unit.shape[1]))
        norms = np.linalg.norm(queries, axis=1)
        queries = queries / norms[:, np.newaxis]
        predictions = np.argmax(queries @ unit.T, axis=1)
        correct += int(np.count_nonzero(predictions == index))
    return correct / total


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded, deterministic simplex-recall experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=int, default=16)
    parser.add_argument("--dim", type=int, default=15)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    codes = regular_simplex(args.codes, args.dim)
    report = asdict(verify_packing_bound(codes))
    report["noise_std"] = args.noise
    report["trials_per_code"] = args.trials
    report["seed"] = args.seed
    report["noisy_recall"] = evaluate_noisy_recall(
        codes,
        noise_std=args.noise,
        trials_per_code=args.trials,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
