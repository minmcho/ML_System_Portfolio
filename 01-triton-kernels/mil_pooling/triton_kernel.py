"""Triton MIL pooling kernel -- SKELETON (spec section 3, v1 single bag).

Status: signature, grid and tile structure are fixed by MIL_POOLING_SPEC.md;
the kernel body is marked TODO and is NOT a working kernel yet. Nothing in
this repo claims a benchmark until section 5 correctness passes on the
RTX 5070 Ti in WSL (Python 3.12 venv -- Triton is Linux/CUDA only).

Design recap: one program per bag; Pass A finds the logits max; Pass B loops
over N in BLOCK_N tiles, computes exp(logit - max) once, accumulates the
denominator and Z_partial over BLOCK_D column tiles, all in FP32. H is read
once in Pass B -- that single read is the whole point of the fusion.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU / non-CUDA machines: tests import-guard on this
    triton = None
    tl = None

BLOCK_N = 1024
BLOCK_D = 128


if triton is not None:

    @triton.jit
    def _mil_pool_kernel(
        H_ptr, logits_ptr, Z_ptr,
        N, D,
        stride_hn,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # TODO(Week 3): Pass A -- running max over logits tiles.
        # TODO(Week 3): Pass B -- exp(logit - max), denom, Z_partial[d_block].
        # TODO(Week 3): Z = Z_partial / denom, store FP32.
        pass


def mil_pool_triton(H: torch.Tensor, A_logits: torch.Tensor) -> torch.Tensor:
    """Fused softmax + weighted pooling. Same contract as reference.mil_pool_ref."""
    if triton is None or not H.is_cuda:
        raise RuntimeError(
            "mil_pool_triton needs Triton + CUDA (5070 Ti in WSL). "
            "Use reference.mil_pool_ref on CPU."
        )
    raise NotImplementedError(
        "Skeleton only -- kernel body lands in Week 3 per MIL_POOLING_SPEC.md s.8"
    )
