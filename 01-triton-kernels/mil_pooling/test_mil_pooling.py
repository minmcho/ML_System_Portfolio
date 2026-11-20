"""Correctness ladder (spec section 5). Reference tests run on CPU anywhere;
Triton tests skip unless CUDA + Triton are present (5070 Ti / WSL gate)."""
import pytest
import torch

from reference import mil_pool_ref, abmil_logits_ref

N_LADDER = [1, 127, 128, 1024, 4096, 10000, 27000]
D_LIST = [512, 768, 1024, 1280]


def _bag(N, D, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(N, D, generator=g, dtype=dtype)
    logits = torch.randn(N, generator=g)
    return H, logits


def test_reference_uniform_attention():
    H, _ = _bag(128, 768)
    Z = mil_pool_ref(H, torch.zeros(128))
    assert torch.allclose(Z, H.float().mean(dim=0), atol=1e-5)


def test_reference_dominant_logit():
    # +30 dominant logit: without max-subtraction exp() overflows in fp32 paths
    H, _ = _bag(1024, 512)
    logits = torch.full((1024,), -5.0)
    logits[7] = 30.0
    Z = mil_pool_ref(H, logits)
    assert torch.allclose(Z, H[7].float(), atol=1e-3)


@pytest.mark.parametrize("N", N_LADDER)
@pytest.mark.parametrize("D", D_LIST)
def test_reference_shapes_and_normalisation(N, D):
    H, logits = _bag(N, D, seed=N + D)
    Z = mil_pool_ref(H, logits)
    assert Z.shape == (D,) and Z.dtype == torch.float32
    A = torch.softmax(logits, dim=0)
    assert torch.allclose(Z, A @ H.float(), atol=1e-5)


def test_abmil_logits_shape():
    H, _ = _bag(256, 768)
    Wv = torch.randn(768, 256) * 0.05
    Wu = torch.randn(768, 256) * 0.05
    w = torch.randn(256) * 0.05
    assert abmil_logits_ref(H, Wv, Wu, w).shape == (256,)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton gate runs on the 5070 Ti in WSL"
)
def test_triton_matches_reference():
    from triton_kernel import mil_pool_triton

    for N in N_LADDER:
        for D in D_LIST:
            H, logits = _bag(N, D, seed=1)
            Z_ref = mil_pool_ref(H, logits)
            Z_tri = mil_pool_triton(H.cuda(), logits.cuda()).cpu()
            assert torch.allclose(Z_tri, Z_ref, rtol=1e-5, atol=1e-5), (N, D)
