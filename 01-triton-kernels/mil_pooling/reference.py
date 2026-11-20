"""PyTorch ground truth for MIL attention pooling (MIL_POOLING_SPEC.md section 4).

Pure PyTorch, FP32, max-subtracted softmax. This file is the contract the
Triton kernel is graded against -- it is never optimised, only trusted.
CPU-runnable so the correctness ladder can be checked without a GPU.
"""
import torch


def mil_pool_ref(H: torch.Tensor, A_logits: torch.Tensor) -> torch.Tensor:
    """Z = softmax(A_logits) @ H for a single bag.

    H: [N, D] patch embeddings (any float dtype). A_logits: [N].
    Returns Z: [D] in FP32.
    """
    assert H.dim() == 2, f"H must be [N, D], got {tuple(H.shape)}"
    assert A_logits.dim() == 1 and A_logits.shape[0] == H.shape[0], (
        f"A_logits must be [N={H.shape[0]}], got {tuple(A_logits.shape)}"
    )
    H32 = H.float()
    logits = A_logits.float()
    A = torch.softmax(logits - logits.max(), dim=0)  # max-subtraction is the spec
    return A @ H32


def abmil_logits_ref(
    H: torch.Tensor, Wv: torch.Tensor, Wu: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Gated ABMIL attention logits (Ilse et al. 2018): [N].

    H: [N, D]; Wv, Wu: [D, L]; w: [L].
    """
    V = torch.tanh(H.float() @ Wv.float())
    U = torch.sigmoid(H.float() @ Wu.float())
    return (V * U) @ w.float()
