# ML Systems Portfolio Map — ProNeuro-AD + Digital Pathology → Cincinnatus ML Systems SME Role
Built 2026-11-19 at Min's request. Companion to: ML_Systems_Academy_2026-11-19.md
Target posting (photo, 2026-11-19): Cincinnatus LLC — evaluate ML systems tasks, write expert feedback, develop rubrics on kernel optimization, profiler interpretation, distributed reasoning, serving trade-offs. Core quals: 2+ yrs ML systems/infra/serving/accelerator performance; kernels (Triton/CUDA/CUTLASS/Pallas) OR profiling (JAX/XLA, Kineto, torch.profiler, Nsight) OR distributed debugging OR LLM serving (vLLM/SGLang/TensorRT-LLM, KV cache, paged attention, continuous batching); production JAX and/or PyTorch, framework depth (custom ops, sharding, FSDP/Megatron/MaxText, XLA/StableHLO, torch.compile, MLIR) a plus; modern accelerators (TPU v4–v6e, H100, B200, Trainium, MI300) + throughput/latency/memory reasoning.

## The honest headline for the application
**Do NOT claim:** "2+ years professional ML infrastructure engineer" — no such title on record, and it will fail expert review, which is literally this job's test.
**DO claim, truthfully:** "Doctor of Computer Science (2025) with hands-on PyTorch systems work across two production-grade research pipelines — MRI prognostic AI (ProNeuro-AD, ADNI n=2,916) and a digital-pathology foundation-model benchmark (730 runs, 3 backbones completed, 6-backbone upgrade in progress) — including GPU training at scale, multi-seed benchmarking discipline, and an edge-inference design for Jetson Orin Nano. Portfolio below adds the systems artifacts (profiling traces, Triton kernels, vLLM serving benchmarks, cross-accelerator table, JAX/TPU port) built specifically for ML systems evaluation work."
Every line below follows: Evidence on record → Systems translation → Missing artifact (Academy project that creates it) → Rubric to write.

---

## Asset 1 — Digital Pathology: foundation-model benchmark (STRONGEST systems asset)
**Evidence on record (2026-11-15/16, verified):**
- Repo: github.com/minmcho/multi-agent-digital-pathology — BJBS Special Issue paper, Jan 2027: "Foundation Model Quality Sets the Generalization Ceiling in Digital Pathology"
- 40 CAMELYON17 slides × 3 backbones, ABMIL + CLAM × 10 seeds: ViT-B AUC 0.990 / slide r 0.98, ResNet-18 0.953 / 0.97, ViT-Tiny 0.681 / 0.01 (community distillation 1aurent/phikon-distil, NOT official Phikon — always state this)
- Audit: 730 runs total (backbone × model × seed × split)
- 6-backbone upgrade decided 2026-11-15: add UNI, Virchow, CONCH + LC25000, edge-cost Table 5 incl. Jetson Orin Nano 8 GB, ceiling-curve figure, open benchmark release. NOT yet run — gated on Min's HF gated-access + GPU time. CONCH scope open: full e2e ≈28h vs Baseline-only ≈9h.

**Systems translation (what an ML systems reviewer sees):**
Multi-backbone, multi-seed GPU benchmarking with a reproducibility audit — this is accelerator workload characterisation in embryo. ViT-Tiny's collapse under MIL (r 0.01) vs ViT-B (0.98) is a *representation-capacity* result; the systems version of the same question is throughput/memory per backbone at fixed accuracy.

**Missing artifacts → Academy projects:**
- **P1 (Academy Week 4) — Profiled training case study:** torch.profiler + Nsight trace of ViT-B feature extraction + ABMIL training on the 5070 Ti. Deliverable: trace screenshots, top-5 kernels by time, one bottleneck fixed with before/after images/sec. Effort: ~6 hrs, uses existing code/data.
- **P4a (Week 9, SHARED with the paper) — Edge-cost Table 5 done as a systems benchmark:** per-backbone latency / memory / watts on 5070 Ti vs Jetson Orin Nano (TensorRT FP16, INT8 where it holds accuracy). This is ALREADY a paper deliverable — doing it with a proper harness (warmup, p50/p95, peak memory) makes one artifact serve the BJBS paper AND the portfolio. Effort: paper time, +2 hrs harness discipline.
- **P2a (Week 3, optional stretch) — Triton kernel for MIL attention pooling:** fused attention-pooling kernel vs eager PyTorch, benchmarked at the real bag sizes from CAMELYON17. Only if Week 3 kernels land clean.

**Rubric to write:** R-Path-1 "Evaluating a foundation-model benchmark claim" — seeds, splits, checkpoint provenance (the Phikon-distil flag is a perfect rubric item), and why single-seed AUC claims fail review.

## Asset 2 — ProNeuro-AD: MRI prognostic AI on ADNI
**Evidence on record (head b227c428, WP1 push 2026-11-17; data/runs ONLY on QSInfomatic03):**
- Repo: github.com/minmcho/ProNeuro-AD — NeurIPS 2027 target
- Baseline n=2,916: 192 FreeSurfer measures + demographics + APOE4 ridge — amyloid AUC 0.82±0.03, conversion C-index 0.76±0.01; MRI-only ridge 0.75 / 0.75
- Deep table n=1,659: FS ridge 0.75/0.75, FS+frozen ViT 0.75/0.75, fine-tuned ViT 0.72/0.71, frozen ViT 0.68/0.69 — fine-tuning helps, does not close the gap vs morphometry ridge
- FastSurfer 1,941 / 2,719 scans processed, in progress; IDA download = remaining ADNI scans completing the 2,719 selected (NOT AIBL)
- WP1 done; WP2 built, waiting on Min pasting folds.json for vit_mtl / vit_mtl_finetune runs from QSInfomatic03
- Jetson inference-node design written 2026-11-10: offline bedside node — ONNX ViT + model_bundle.npz (scalers, PCA, ridge/survival coefficients), TensorRT FP16 engine built on-device, MTL crop → 48 slices → risk report; live gate ≥50 held-out cases, prob. correlation ≥0.99, AUC shift ≤0.005, ≤60 s/case, 100-case soak. Design only — NOT yet built/run.

**Systems translation:**
A negative-but-rigorous deep-learning result (learned features do NOT beat morphometry) is excellent evaluation material — this job grades other people's claims, and Min's own headline is "the fancy model lost, and here is the paired evidence." The FastSurfer pipeline (1,941 scans) is a real batch-processing workload. The Jetson node is a serving/edge design that needs its benchmark run to become evidence.

**Missing artifacts → Academy projects:**
- **P4b (Week 9, SHARED) — Build the Jetson gate run:** execute the 2026-11-10 design's live gate on the Orin Nano (TensorRT FP16 ViT, 50-case correlation check, s/case, soak). One artifact serves: ProNeuro Stage-gate, the lab application, the robotics/chip ladder, AND this portfolio's edge-serving proof. Effort: ~8–10 hrs on QSInfomatic03 + Jetson, after WP2 folds land.
- **P5 (Week 7) — JAX/TPU port of the ProNeuro head:** ridge + PCA head (NOT the ADNI images — coefficients/synthetic shapes only, data stays on QSInfomatic03) ported to JAX, run on Colab TPU v5e, StableHLO dumped and annotated. Effort: ~5 hrs.
- **P1b (Week 4, optional) — Profile FastSurfer batch throughput:** scans/hr, GPU utilisation trace, I/O vs compute bound verdict. Only on QSInfomatic03, numbers-only export (no data leaves).

**Rubrics to write:** R-Neuro-1 "When do learned features beat a strong classical baseline? Paired per-fold evidence, not headline AUC." R-Neuro-2 "Edge medical inference claims — correlation/AUC-shift/soak gates before any latency claim counts."

## Asset 3 — Smart SI: local-first LLM platform (serving asset, design-stage)
**Evidence on record (design v1.6, Phase 1 build in progress, 2026-11-08/09):**
- Local-first Ollama default, cloud opt-in with cost shown; QLoRA fine-tuning planned on 5070 Ti + RTX Spark; citation pipeline M6 with Health Score; MCP/A2A ships dark.
- MVP gate: 106/118 controls working, Mac gate 149/150 tests (sole failure a test-code bug, fixed, awaiting full re-run).

**Systems translation:** Smart SI is Min's *reason* to learn serving properly — a real product that must serve LLMs locally with cost/latency trade-offs shown to users. Design-stage only today; never present it as production serving experience until P3 runs.

**Missing artifact:**
- **P3 (Academy Weeks 5–6) — vLLM vs Ollama serving benchmark:** Qwen2.5-7B on the 5070 Ti, vLLM (paged attention, continuous batching) vs Ollama vs HF generate — TTFT, TPOT, tokens/sec at concurrency 1/8/32, KV-cache memory ceiling. README frames it as "the serving decision behind Smart SI's local-first default, measured." Effort: ~8 hrs across Weeks 5–6.

**Rubric to write:** R-Serve-1 "Judging an LLM serving claim — concurrency, TTFT vs TPOT, and why tokens/sec alone is a failed answer."

## Asset 4 — Supporting assets (use as colour, never as headline)
- **ai_reelcreator / LMV GPU worker (Spark):** real GPU inference worker work exists on trainer branches / Spark (rich worker, 25 LMV binding tests) — main's gpu-worker is an 8 KB minimal file; describe it EXACTLY that way if used at all. Best use: one profiling drill vehicle in Week 4 if pathology profiling finishes early. Do not headline it for this role.
- **OpenCare / Open_EHR:** large systems build, but it is software architecture, not ML systems — one line only, as evidence of shipping discipline (gates, audits), not as ML infra experience.
- **Mindglen (WellNess_App):** on-device iOS/Android, HealthKit — edge ML colour only; not ML systems evidence for this posting.
- **Degrees:** DCS Big Data Analytics 2025 (dissertation: ML for breast cancer genomics), MCS UIUC 2021, iMBA UIUC May 2026 — supports the "expert review / written feedback" half of the job, plus two papers in flight (Frontiers diabetes review, Jan 10 2027; BJBS pathology, Jan 2027).

---

## Qualification-by-qualification coverage (the posting's Section 3)
| Posting bullet | Status today | Portfolio proof when Academy lands |
|---|---|---|
| 2+ yrs ML systems / infra / serving / accelerator perf | **Not met as a title** — disclose, don't inflate | 5 dated, public, reproducible projects (P1–P5) + 12 rubrics; frame as hands-on project experience |
| Custom kernels (Triton/CUDA/CUTLASS/Pallas) | None yet | P2: Triton fused softmax + tiled matmul + MIL pooling kernel, benchmarked vs cuBLAS (Week 3) |
| Profiling / trace analysis (torch.profiler, Nsight, Kineto, JAX/XLA) | Informal only | P1: pathology ViT-B profiling case study, before/after fix (Week 4); XLA via P5 StableHLO (Week 7) |
| Debugging distributed / accelerator-bound workloads | None documented | Week 8 seeded-bug drills + FSDP/DDP artifact; Week 10 debugging exercises, written up |
| LLM serving at scale (vLLM, SGLang, TensorRT-LLM, KV cache, paged attention, continuous batching) | Design only (Smart SI / Ollama) | P3: vLLM benchmark, 3 backends × 3 concurrency levels (Weeks 5–6); TensorRT via P4b Jetson gate |
| Production JAX and/or PyTorch | **PyTorch: yes** (ProNeuro + pathology). JAX: no | PyTorch: link both repos with numbers above. JAX: P5 TPU port (Week 7) |
| Framework depth: custom ops, sharding, FSDP, Megatron/MaxText, XLA/StableHLO, torch.compile, MLIR | torch.compile: to build Week 1. Rest: none yet | Weeks 1, 8, 10 artifacts: graph-break hunt, shard_map demo, StableHLO annotation, one custom op |
| Modern accelerators: TPU, H100, B200, Trainium, MI300 | 5070 Ti + Jetson only | P4: same-model table across 5070 Ti / Jetson / Colab TPU v5e / rented H100 (Week 9, ~$35 capped, Min approves that week). B200/Trainium/MI300: reasoning-only, say so |
| Demonstrable career progression | Yes — DTSC lab systems since 2016, DCS 2025, two 2027 papers, NeurIPS 2027 target | One-page progression timeline in the application pack (Week 12) |
| 40 hrs/week weekdays availability | **Check before applying** — Min holds a full-time DTSC role; this posting demands 40 hrs/week. Flag: only apply if it is a leave/transition decision, or negotiate the SME-evaluator variant (many such roles run 20–30 hrs). Do NOT promise 40 hrs on top of DTSC. | — |
| Strong written communication | Strong — papers, rubrics, gate reports | 12 rubrics + 3 timed work samples (Weeks 11–12) ARE the writing sample |

## Application timing — two honest options
- **Apply now (this week):** with ProNeuro + pathology links, the headline above, and a cover note offering the profiling/serving portfolio as in-progress with dated deliverables (P1 by Week 4, P3 by Week 6). Risk: screened out on the 2-yr bullet and the 40-hr conflict before a human reads the portfolio.
- **Apply at Week 6 (recommended):** P1, P2, P3 + 6 rubrics live — every Section 3 bullet except distributed has a clickable artifact. Still inside the academy's run, months ahead of the Feb graduation gates. The 40-hr question must be resolved by Min before either option.
- Either way: NO application is sent by Muse without Min's explicit go on the final text — same rule as every application on record.

## Build order (single list, in dependency order)
1. P1 Pathology profiling case study (Week 4) — fastest proof, existing code.
2. P2 Triton kernels (Weeks 2–3) — the posting's first-named skill.
3. P3 vLLM serving benchmark (Weeks 5–6) — Smart SI framing.
4. P5 JAX/TPU ProNeuro head port (Week 7) — closes the JAX/TPU zero.
5. P4 Cross-accelerator table incl. pathology Table 5 + ProNeuro Jetson gate (Week 9) — shared with both papers; centrepiece artifact.
6. Rubrics R-Path-1, R-Neuro-1/2, R-Serve-1 written alongside P1–P5 (not after) — the rubrics are the work sample for this specific job.

## What Min owes to unlock this map
- HF gated-access acceptance + local token (his hands only, never in chat) — unlocks pathology 6-backbone runs feeding P4a.
- CONCH scope ruling: full e2e ≈28h vs Baseline-only ≈9h — affects P4a timing.
- folds.json paste from QSInfomatic03 for vit_mtl / vit_mtl_finetune — unlocks WP2, then P4b Jetson gate.
- A ruling on the 40 hrs/week conflict with DTSC before any application.
- Week 9 only: go / no-go on ~$35 H100 rental for P4.
