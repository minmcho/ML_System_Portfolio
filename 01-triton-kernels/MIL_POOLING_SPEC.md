# MIL Attention Pooling — Triton Kernel Spec (Project P2a)

Status: **SPEC DRAFTED 2026-11-20** — no kernel code claimed working yet.
Owner project: `01-triton-kernels/` (Weeks 2–3) · Portfolio Map: P2a
Source research: Digital pathology benchmark — `github.com/minmcho/multi-agent-digital-pathology`
40 CAMELYON17 slides × 3 completed backbones, ABMIL + CLAM × 10 seeds, 730 audited runs:
ViT-B AUC 0.990 / r 0.98, ResNet-18 0.953 / 0.97, ViT-Tiny 0.681 / 0.01.
6-backbone upgrade in progress (UNI, Virchow, CONCH + LC25000) — this spec is parametric
in embedding dim `D` so it survives that upgrade without a rewrite.

---

## 1. The problem, exactly

For one slide (one *bag*):

- `H` — patch embeddings, shape `[N, D]`, one row per patch. `N` varies per slide.
- Attention network (gated ABMIL, Ilse et al. 2018 — the form used by ABMIL/CLAM):

```
V = tanh(H @ Wv)          # [N, L]
U = sigmoid(H @ Wu)       # [N, L]
A_logits = (V * U) @ w    # [N]
A = softmax(A_logits)     # [N], sums to 1 over the bag
Z = A @ H                 # [D]  slide embedding
logits = classifier(Z)
```

- `Wv, Wu`: `[D, L]`, `w`: `[L]`. `L = 256` is the usual ABMIL hidden size — **verify
  against the actual pathology repo config before the coding gate (Open Item O1).**

### What this kernel fuses — and what it deliberately does not

| Stage | Owner | Why |
|---|---|---|
| `H @ Wv`, `H @ Wu` (two `[N,D]×[D,L]` matmuls) | cuBLAS / PyTorch (v1) | Dense GEMM. cuBLAS is hard to beat and it is not the learning goal. |
| **softmax(A_logits) + `A @ H` (pooling)** | **This Triton kernel (v1)** | Memory-bound, variable-`N`, and done as ≥3 separate passes in eager PyTorch: max, exp/sum, weighted sum — each re-reading `H` (`N×D`). One fused pass reads `H` once. |
| Full gated attention (`tanh/sigmoid` + GEMMs + pooling) in one kernel | Stretch v2, only if Week 4 profiling shows the GEMMs are launch/memory bound at real `N`, not compute bound | Do not pre-build it. Profiling decides. |

Honest expectation, stated in advance: at large `N` the two GEMMs dominate FLOPs, so
v1's win is on the pooling stage alone (target: pooling stage ≥2× vs eager, measured
in §6) and on end-to-end latency at small-to-mid `N` where kernel-launch and memory
traffic dominate. If the benchmark does not show that, the README says so — a measured
negative result is a valid portfolio artifact, same rule as ProNeuro-AD.

## 2. Shapes, dtypes, and the verification gate

`D` by backbone — standard published embedding dims, **to be confirmed from the actual
feature files before any benchmark is published (Open Item O2)**:

| Backbone | D (expected) | Status in paper |
|---|---|---|
| ResNet-18 | 512 | Run (0.953 / 0.97) |
| ViT-Tiny (1aurent/phikon-distil — community distillation, NOT official Phikon) | 768 | Run (0.681 / 0.01) |
| ViT-B | 768 | Run (0.990 / 0.98) |
| UNI | 1024 | Upgrade, not yet run |
| CONCH | 512 | Upgrade, not yet run |
| Virchow | 1280 | Upgrade, not yet run |

- `N` test ladder: `1, 127, 128, 1,024, 4,096, 10,000, 27,000` — spans empty-ish bags,
  non-power-of-2, and large WSI bags. Real per-slide `N` distribution from CAMELYON17
  features replaces the top end once O2 is done.
- Inputs: `H` in FP16 or BF16 (as stored by the extractor), `A_logits` in FP32.
- Accumulation: **FP32 everywhere inside the kernel** (softmax sums and the `Z`
  accumulation). Output `Z` FP32; caller casts if needed.
- Layout: `H` row-major, contiguous. Non-contiguous input = caller error, asserted.

## 3. Kernel design — v1 (single bag)

One Triton program per bag. Two-pass online-softmax structure in a single launch:

1. **Pass A — logits reduction:** loop over `N` in `BLOCK_N=1024` tiles, load
   `A_logits` tile, track running max `m`.
2. **Pass B — fused exp + weighted sum:** loop over `N` again; per tile load
   `A_logits` tile, compute `e = exp(logit - m)`, accumulate `denom += sum(e)`, and
   accumulate `Z_partial[d_block] += sum_n e[n] * H[n, d_block]` over `BLOCK_D=128`
   column tiles held in registers/SRAM. Final `Z = Z_partial / denom`.
   - Re-reading `A_logits` (`N` floats) is cheap; re-reading `H` is what we avoid
     doing twice — eager PyTorch reads `H` once for softmax-weighted sum *plus*
     materialises `A` and the broadcast multiply as separate kernels.
   - Numerical rule: max-subtraction is mandatory, not optional. A kernel without it
     fails the large-logit correctness test in §5 and Rubric R03 must say so.

Grid / batching (v1.1, after single-bag is correct): one program per bag in a batch,
bags described by a CSR-style `bag_offsets` int32 tensor `[B+1]` — no padding, no
wasted compute on pad patches. Batched form is what the pathology training loop
actually needs; single-bag is the correctness stepping stone.

Backward pass: **out of scope for v1.** This is an inference / evaluation-time
optimisation first. Training use requires a backward kernel (grad through softmax
pooling) — named as v3, only if the profiling project (P1, Week 4) shows MIL head
training time matters next to feature extraction. Expected verdict: it does not —
extraction dominates — and v3 will likely be cut. Say so in the README if cut.

## 4. Reference implementation (ground truth)

`reference.py` in this folder is the contract:

- `mil_pool_ref(H, A_logits) -> Z` — pure PyTorch, FP32, max-subtracted softmax.
- `abmil_logits_ref(H, Wv, Wu, w) -> A_logits` — gated attention, pure PyTorch.
- The Triton kernel is correct iff it matches `mil_pool_ref` under §5 tolerances.
  The reference never gets optimised; it gets *trusted*.

## 5. Correctness gate (all must pass on the 5070 Ti before any timing is quoted)

- Sizes: every `N` in the §2 ladder × `D ∈ {512, 768, 1024, 1280}`.
- Tolerances vs reference: `Z` — rtol 1e-3, atol 1e-3 (FP16 input) / rtol 1e-5,
  atol 1e-5 (FP32 input test mode).
- Adversarial cases: all-equal logits (uniform attention), one dominant logit
  (+30 vs rest — tests max-subtraction), `N=1`, `N` not divisible by `BLOCK_N`,
  logits containing large negative values.
- Determinism check: same input twice → bitwise-identical `Z` in v1 (single program,
  no atomics). If a future split-K variant uses atomics, this test is re-scoped and
  the README must flag the change.

## 6. Benchmark harness (same discipline as every project in this repo)

- Hardware: **RTX 5070 Ti (12 GB GDDR7) only** for headline numbers. Never the Spark
  (unified memory distorts bandwidth) and never Jetson (Triton is not the Jetson path —
  TensorRT is, Week 9 / P4).
- Method: 25 warmup + 100 timed iterations, median reported, CUDA events, inputs
  resident on GPU before timing. Report: ms/bag, effective GB/s
  (`bytes = N*D*sizeof(H) + N*4`, read-once ideal), peak memory.
- Baselines, all three, same inputs: (a) eager PyTorch pooling, (b) `torch.compile`
  pooling, (c) this kernel. A result table with only (c) is a failed deliverable.
- End-to-end check: pooling-stage win must also be reported as % of full ABMIL head
  time (GEMMs + pooling) so the claim is proportionate — Amdahl, stated plainly.

## 7. File layout (created with this spec)

```
01-triton-kernels/
  MIL_POOLING_SPEC.md     this file
  mil_pooling/
    reference.py          PyTorch ground truth (§4) — CPU-runnable, tested
    triton_kernel.py      Triton skeleton: signature, grid, tile loops marked TODO
    test_mil_pooling.py   §5 correctness ladder (Triton tests skip without CUDA)
    bench_mil_pooling.py  §6 harness skeleton (median/GB/s table, baselines a–c)
```

## 8. Done when

1. §5 gate green on the 5070 Ti in WSL (uv-managed Python 3.12 venv — Triton does not
   run on native Windows or on system Python 3.14).
2. §6 table in `01-triton-kernels/README.md` with all three baselines.
3. Rubric R03 gains a worked example: one flawed MIL kernel (no max-subtraction, or
   FP16 accumulation) graded against this spec.
4. One paragraph in the README: what the win was, where it vanishes (large-`N` GEMM
   dominance), and the v2/v3 decision with its evidence.

## Open items (Min / source verification, in order)

- **O1:** Confirm `L` and the exact gated-attention form from the pathology repo's
  ABMIL/CLAM config — spec assumes Ilse gated form, `L=256`.
- **O2:** Confirm `D` per backbone and the real per-slide `N` distribution from the
  actual CAMELYON17 feature files (they live with the pathology runs, not in this repo).
- **O3:** Ruling on scope if Week 4 profiling lands first: v1 pooling-only (this spec)
  vs jumping straight to v2 — default is v1, change only on profile evidence.
