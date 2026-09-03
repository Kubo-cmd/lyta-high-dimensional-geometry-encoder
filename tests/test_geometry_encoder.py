"""Tests for the high-dimensional geometry encoder."""

import math

import numpy as np
import pytest

from lyta_geometry_encoder import (
    encode_high_dim_geometry,
    evaluate_noisy_recall,
    regular_simplex,
    verify_packing_bound,
)


def test_encoder_returns_requested_radius_and_dimension():
    encoded = np.asarray(encode_high_dim_geometry([1.0, 2.0, 3.0], dim=32, radius=2.5))

    assert encoded.shape == (32,)
    assert np.linalg.norm(encoded) == pytest.approx(2.5)


def test_encoder_is_seed_deterministic():
    first = encode_high_dim_geometry([1.0, 2.0, 3.0], dim=16, seed=7)
    second = encode_high_dim_geometry([1.0, 2.0, 3.0], dim=16, seed=7)
    different = encode_high_dim_geometry([1.0, 2.0, 3.0], dim=16, seed=8)

    assert first == second
    assert first != different


def test_encoder_same_dimension_only_normalizes():
    encoded = np.asarray(encode_high_dim_geometry([3.0, 4.0], dim=2, seed=999))

    assert encoded == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    "vector",
    [[], [0.0, 0.0], [1.0, float("nan")], [1.0, float("inf")]],
)
def test_encoder_rejects_degenerate_input(vector):
    with pytest.raises(ValueError):
        encode_high_dim_geometry(vector)


def test_regular_simplex_achieves_bound():
    count = 8
    codes = regular_simplex(count, dim=7)
    report = verify_packing_bound(codes)

    assert np.linalg.norm(codes, axis=1) == pytest.approx(np.ones(count))
    assert report.maximum_cosine == pytest.approx(-1.0 / (count - 1), abs=1e-12)
    assert report.minimum_distance == pytest.approx(
        math.sqrt(2.0 * count / (count - 1)), abs=1e-12
    )
    assert report.simplex_bound_achieved


def test_regular_simplex_embeds_in_larger_dimension():
    codes = regular_simplex(4, dim=10)

    assert codes.shape == (4, 10)
    assert np.count_nonzero(codes[:, 3:]) == 0


def test_regular_simplex_rejects_impossible_size():
    with pytest.raises(ValueError, match=r"code_count <= dim \+ 1"):
        regular_simplex(5, dim=3)


def test_duplicate_codes_do_not_achieve_simplex_bound():
    report = verify_packing_bound([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    assert report.maximum_cosine == pytest.approx(1.0)
    assert not report.simplex_bound_achieved


def test_packing_verifier_rejects_invalid_codes():
    with pytest.raises(ValueError, match="zero vectors"):
        verify_packing_bound([[1.0, 0.0], [0.0, 0.0]])


def test_noisy_recall_is_deterministic_and_high_for_small_noise():
    codes = regular_simplex(16, dim=15)

    first = evaluate_noisy_recall(codes, noise_std=0.02, trials_per_code=200, seed=42)
    second = evaluate_noisy_recall(codes, noise_std=0.02, trials_per_code=200, seed=42)

    assert first == second
    assert first >= 0.99


def test_noisy_recall_rejects_unbounded_work():
    codes = regular_simplex(2, dim=1)

    with pytest.raises(ValueError, match="must not exceed"):
        evaluate_noisy_recall(codes, trials_per_code=500_001)


def test_projection_rejects_unbounded_matrix_before_allocation():
    with pytest.raises(ValueError, match="projection exceeds"):
        encode_high_dim_geometry(np.ones(3_001), dim=3_000)
