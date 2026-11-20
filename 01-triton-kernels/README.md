# 01 — Triton Kernels (Weeks 2–3, Project P2)
Status: planned. Done when: fused softmax + tiled matmul (3 tuned configs) + one MIL attention-pooling kernel are committed with a benchmark harness (warmup, median, GB/s) and a README table vs cuBLAS / eager PyTorch, reproduced on the RTX 5070 Ti (12 GB). Rubrics: R02, R03 (see ../rubrics/).

## MIL pooling kernel (P2a) — spec drafted 2026-11-20
- **Spec:** [MIL_POOLING_SPEC.md](MIL_POOLING_SPEC.md) — fused softmax + weighted pooling for the ABMIL/CLAM head of the digital pathology benchmark (40 CAMELYON17 slides, 730 runs). GEMMs stay on cuBLAS in v1; the kernel fuses the memory-bound pooling stage. Backward out of scope for v1.
- **Code skeleton:** `mil_pooling/` — `reference.py` (PyTorch ground truth, CPU-tested), `triton_kernel.py` (signature/grid fixed, body TODO Week 3), `test_mil_pooling.py` (§5 correctness ladder), `bench_mil_pooling.py` (§6 harness, baselines eager / torch.compile / Triton).
- **Open items before the coding gate:** O1 confirm ABMIL hidden size `L` (assumed 256, gated form) from the pathology repo config; O2 confirm `D` per backbone + real per-slide `N` distribution from the CAMELYON17 feature files; O3 v1-vs-v2 scope ruling on Week 4 profile evidence.
- No benchmark numbers exist yet — none will be quoted until the §5 correctness gate passes on the 5070 Ti.
