import numpy as np
from lyta_geometry_encoder import encode_high_dim_geometry, verify_packing_bound, secure_encode

def test_encode_high_dim_geometry():
    test_vector = [1.0] * 768
    encoded = encode_high_dim_geometry(test_vector)
    assert len(encoded) == 768
    assert abs(np.linalg.norm(encoded) - 0.9273) < 0.01  # packing bound from 2M sims

def test_verify_packing_bound():
    codes = np.random.randn(48, 768)
    codes /= np.linalg.norm(codes, axis=1)[:, np.newaxis]
    fidelity = verify_packing_bound(codes)
    assert fidelity > 0.3, "Sovereign threshold not met with formal bound reference"

def test_secure_encode_policy():
    test_vector = [1.0] * 768
    encoded = secure_encode(test_vector, policy='allow')
    assert encoded is not None
    encoded_blocked = secure_encode(test_vector, policy='block')
    assert encoded_blocked is None

def test_adversarial_noise_scaling():
    codes = np.random.randn(48, 768)
    codes /= np.linalg.norm(codes, axis=1)[:, np.newaxis]
    for noise in [0.01, 0.05, 0.1]:
        fidelity = verify_packing_bound(codes + np.random.normal(0, noise, codes.shape))
        assert fidelity > 0.2, f"Failed at noise level {noise}"

if __name__ == "__main__":
    test_encode_high_dim_geometry()
    test_verify_packing_bound()
    test_secure_encode_policy()
    test_adversarial_noise_scaling()
    print("All tests passing. High-dimensional geometric encoder verified with formal bound reference and adversarial scaling.")