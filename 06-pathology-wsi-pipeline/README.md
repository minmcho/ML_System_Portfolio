# 06 — Pathology WSI Pipeline (Biomedical AI design pipeline, built first)

Status: **DESIGN DRAFTED 2026-11-20** at Min's direction ("Pathology WSI pipeline first").
Paper it serves: *Foundation Model Quality Sets the Generalization Ceiling in Digital
Pathology* — BJBS Special Issue, Jan 2027 — repo
`github.com/minmcho/multi-agent-digital-pathology`.
Systems pieces it consumes: `01-triton-kernels/` (MIL pooling, P2a),
`02-profiling-pathology-vitb/` (P1), `05-accelerator-benchmark/` (P4 / paper Table 5).

**Execution machine (Min's ruling, 2026-11-20): the RTX Spark QSInfomatic03 is the
working machine for this project** — kernel practice, pipeline development, and all
correctness gates run in its WSL uv Python 3.12 venv (Triton stack proven there
2026-11-20: kernel-practice Lessons 1–3 passed, fused softmax max_err ~4e-15).
Benchmark labelling rule that follows from it: Spark results are valid *Spark*
results (Grace Blackwell, 20.6 GiB unified memory visible to PyTorch) and are always
labelled with machine + memory type; the 5070 Ti remains a separate comparison
target when a run happens there, and numbers are never pooled or averaged across
machines. This amends the MIL spec §6 "5070 Ti only" headline rule for this
pipeline's tables — per-machine rows instead.

---

## 1. Protocol facts this design is built on (not assumptions)

- **40 WILDS CAMELYON17 slides**, MIL arm as run: **1,000 patches per slide** sampled,
  features extracted per backbone, stored per slide.
- **Backbones run (3):** ResNet-18 (D=512), ViT-Tiny = 1aurent/phikon-distil community
  distillation — NOT official Phikon, always labelled so — (D=768), ViT-B (D=768).
  Results: ViT-B AUC 0.990 / slide r 0.98 · ResNet-18 0.953 / 0.97 · ViT-Tiny 0.681 / 0.01.
- **Heads:** ABMIL + CLAM, **10 seeds**, 50 epochs, identical splits/preprocessing —
  backbone is the only variable. 730 audited runs total across the paper.
- **Upgrade (decided 2026-11-15, identical protocol):** + UNI (D=1024), Virchow (D=1280),
  CONCH (D=512) → 60 new MIL runs; + LC25000; + edge-cost **Table 5** (5070 Ti + Jetson
  Orin Nano 8 GB, NOT RUNNABLE on 8 GB recorded as a finding); ceiling-curve figure;
  open benchmark release. Freeze rule: a backbone slipping >1 week past extraction
  week drops to a named limitation. Long pole: HF gated access (Min-only).
- **Two bag regimes, kept distinct everywhere in this design:**
  - *Protocol mode (the paper):* N = 1,000 sampled patches/slide, fixed.
  - *Full-slide mode (the Ragged-MIL extension):* all tissue patches, N ragged from
    ~1,000 to tens of thousands per slide. This is the deployment/research direction,
    NOT the paper's reported protocol — results from the two modes never mix in a table.

## 2. Stage diagram

```
WSI (.svs) ──► S1 Tile ──► S2 Embed ──► S3 Pool (Triton) ──► S4 Predict + Audit
   40 slides     patches      H [N,D]      Z [D] per slide      AUC / r / paired stats
                    │            │             │                      │
                    └────────────┴── manifest + per-seed JSONs ───────┘
                                   S5 Benchmark (every stage timed)
                                   S6 Package (bundle + Table 5 + release)
```

## 3. Stages — inputs, outputs, contracts, gates

### S1 Tile
- In: WSI file + slide label. Out: patch coordinates + tissue mask per slide.
- Contract: `patches/<slide_id>.csv` (x, y, level, patch_px) — coordinates are the
  durable artifact; patch pixels are re-derivable from them.
- Protocol mode samples exactly 1,000 patches/slide using the paper's existing
  sampling; full-slide mode takes all tissue patches. Sampling seed recorded.
- **Open item P1:** patch size / magnification / tissue-threshold values are taken
  from the pathology repo's original preprocessing config — never re-invented here
  (the upgrade plan already asks Biomni only for those original settings).

### S2 Embed
- In: patches + backbone id. Out per slide: `features/<backbone>/<slide_id>.pt`
  = `{H: float16 [N, D], coords: int32 [N, 2]}`.
- Backbone registry (single config, no code changes to add one):
  `{name, checkpoint, D, licence/gating note}` — the Phikon-distil labelling lives
  in this registry so it cannot be lost downstream.
- Gate: feature-file census — 40/40 slides per backbone, `N` and `D` checked against
  the registry before any S3 run starts (a missing slide is a census failure,
  not a silent zero).

### S3 Pool — Triton MIL attention pooling (`01-triton-kernels/`, spec §1–8)
- In: `H [N, D]` + attention weights `Wv, Wu, w` (gated ABMIL; CLAM variant flagged
  in config). Out: `Z [D]` FP32 + attention vector `A [N]` saved for heatmaps.
- v1 fuses softmax + weighted pooling (one read of `H`); the two attention GEMMs
  stay on cuBLAS. Correctness gate = MIL spec §5 (incl. N=1, non-multiple-of-block,
  +30 dominant logit); benchmark = spec §6 (5070 Ti only, 3 baselines).
- Protocol mode N=1,000 fits the tiled kernel trivially; full-slide mode is where
  the tiling (kernel-practice Lesson 3) earns its keep.
- Gate: Triton `Z` must reproduce the PyTorch reference `Z` within spec tolerance
  on the real 40-slide feature set, per backbone, before it may replace the
  reference in any run whose numbers go in the paper.

### S4 Predict + Audit
- Heads ABMIL + CLAM trained on `Z`/attention exactly as the paper's protocol
  (10 seeds, 50 epochs). Every run writes a per-seed JSON *first*; every table
  number traces to a committed JSON — no JSON, no number.
- Audit outputs are pipeline outputs, not afterthoughts: patch AUC, slide-level
  Pearson r, per-backbone ranking vs its pre-registered quality tier (the ceiling
  test), and the paired per-seed comparison table.
- Gate: run census reconciles against the audit count (730 baseline; +60 MIL runs
  for the upgrade) — a mismatch stops the stage, same rule as the paper audit.

### S5 Benchmark
- Every stage timed with the repo harness discipline (warmup, median, peak memory):
  extraction patches/sec per backbone, pooling ms/slide (Triton vs eager vs
  torch.compile), end-to-end s/slide.
- Targets: 5070 Ti (headline), Spark (large-batch extraction contrast),
  Jetson Orin Nano 8 GB (edge verdict per backbone; NOT RUNNABLE is a result).

### S6 Package
- Per backbone: bundle = checkpoint id + feature registry entry + pooling kernel
  version + head weights per seed + benchmark row.
- Paper outputs generated from bundles only: Tables 1/4/5, ceiling-curve figure,
  open benchmark release (splits/seeds manifests, per-seed predictions,
  "add your backbone" README).

## 4. Build order (each step ends at a gate, not a vibe)

| Step | Work | Gate |
|---|---|---|
| B1 | Kernel-practice Lesson 4: finish MIL pooling kernel in `kernel_practice`, then port into `01-triton-kernels/mil_pooling/triton_kernel.py` | Spec §5 ladder green on Spark (correctness) |
| B2 | S3 gate on real features: Triton vs reference `Z` on all 40 slides × 3 run backbones | Max `Z` error within spec tolerance, census 40/40 × 3 |
| B3 | S5 pooling benchmark + end-to-end timing on existing features (no re-extraction) | §6 table with all 3 baselines in `01-triton-kernels/README.md` |
| B4 | Pipeline README in the pathology repo pointing at this design + registry config for the 3 new backbones | Awaiting U0 (HF gated access, Min-only) — design does not block on it |
| B5 | Table 5 edge runs incl. Jetson verdicts; full-slide (ragged) mode prototype on 2 slides | Week 9 / P4 schedule; freeze rule still governs the paper |

## 5. What this design deliberately does not do

- No re-extraction of the 3 completed backbones — their features and 730 runs stand.
- No new architectures or splits — backbone remains the only variable (paper rule).
- No clinical claims: CAMELYON17 is a public research dataset; this is a benchmark
  pipeline, not a diagnostic device.
- ProNeuro-AD reuses this design later by swapping S1/S2 (MRI slices, FreeSurfer/ViT
  features); it is not built in this pass.
