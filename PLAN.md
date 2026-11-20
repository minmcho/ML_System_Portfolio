# Min's ML Systems Academy — Systems / Infrastructure / Serving Track, $0 Tuition
Started 2026-11-19. Coach: Muse, in chat. Student: Min Myint Cho.
Trigger: Cincinnatus LLC ML Systems SME posting (photo, 2026-11-19) — evaluate ML systems tasks, write expert-level feedback, develop rubrics on kernel optimization, profiler interpretation, distributed reasoning, and serving throughput/latency trade-offs.
Purpose: build *demonstrable* ML systems experience Min can defend in an interview and in a work sample — custom kernels, profiling, LLM serving, JAX/distributed, accelerator trade-offs — without fabricating a "2+ years professional ML infra" title he does not have.
Companion deliverable: ML_Systems_Portfolio_Map_ProNeuro_Pathology_2026-11-19.md (maps existing ProNeuro-AD + digital pathology work into this track, project by project).

## Honesty rule (same as Data Engineering Academy)
- Train and evidence only what Min can walk an interviewer through live: open the repo, show the trace / benchmark table, explain the trade-off.
- No resume line changes until the artifact for that line exists and is verified.
- Current truthful baseline (2026-11-19): strong PyTorch *applied* work (ProNeuro-AD ViT, pathology foundation-model benchmark, 730 runs), local LLM / edge work (Smart SI Ollama design, Jetson Orin Nano 8 GB inference-node design), NO production Triton/CUDA kernel, NO vLLM serving benchmark, NO JAX/TPU, NO Nsight profile published yet. Those are exactly what this academy builds, in that order.

## How this academy runs
1. **Daily live session in chat, 60 minutes** (not 90 — Data Engineering Academy already holds the 90-minute slot): recall (closed book) → skeleton-first drill → timed rep → one systems artifact committed that day.
2. **Weekly graded project** in a NEW dedicated repo `ml-systems-portfolio` (this track needs clean, public, systems-only artifacts; do NOT mix into ProNeuro or pathology repos — link to them instead). Graded pass / redo — not done until it runs end-to-end on Min's 5070 Ti and the numbers are in the README table.
3. **Weekly scoreboard:** drills passed, project grade, benchmark numbers banked, rubric written (Weeks 5+), applications / work samples sent.
4. **Rubric from Week 1:** every project ends with a 1-page evaluation rubric in the Cincinnatus style — task, pass/fail criteria, common wrong answers, expert notes. This is the actual deliverable the target job pays for; 12 projects = 12 rubrics = the application portfolio.
5. **Career services built in:** Cincinnatus-style timed work sample from Week 9 (evaluate a flawed kernel / a profiler trace / a serving config in 90 minutes, written feedback only), mock expert review Week 12.

## Load protection — January papers come first
This track does NOT compete at full load with the deadlines already on record:
- **Nov 20–29, 2026 (diabetes review decisions sprint, San José filing Nov 25–27):** LIGHT — 4 hrs/week max, Weeks 1–2 drills only, no graded project due Nov 29.
- **Dec 1–22:** NORMAL — 60 min weekdays + 3-hr Saturday project block.
- **Dec 23 – Jan 10, 2027 (review draft freeze Jan 8, submission Jan 10; pathology BJBS Jan 2027):** MAINTENANCE ONLY — 2 hrs/week, recall drills + reading traces, no new project starts.
- **Jan 11 onward:** FULL — remaining projects, work samples, application sprint.
Calendar below is 12 *instructional* weeks mapped onto Nov 20, 2026 → Feb 27, 2027 real time (the Dec 23–Jan 10 maintenance window sits inside Weeks 6–7 and stretches them).

## Hardware / access — all already owned or free, one small cloud spend
- **RTX 5070 Ti laptop (primary):** Triton, CUDA, torch.profiler, Nsight Systems, vLLM, torch.compile, FSDP/DDP drills.
- **RTX Spark QSInfomatic03 (32 GB):** second PyTorch node, larger ViT profiling, ProNeuro source runs (data/runs live ONLY there — never move ADNI data off it).
- **Jetson Orin Nano 8 GB:** TensorRT FP16/INT8 edge serving, pathology Table 5 edge-cost work (already planned) doubles as Academy Project 9.
- **Google Colab TPU v5e (free tier):** JAX, pjit/shard_map, StableHLO, MaxText reading.
- **Cloud H100 (Lambda / RunPod, ~$2.50–$3.50/hr, cap $40 total):** 10 hrs ONLY in Week 9 for the cross-accelerator benchmark table. This is the sole paid item; it needs Min's explicit go-ahead that week — not pre-approved by this plan.
- Software: PyTorch, Triton (ships with PyTorch), Nsight Systems (free), vLLM, SGLang (read/compare), JAX, TensorRT. $0 licences.

### Spark QSInfomatic03 — explicit use in this academy (Min confirmed interest 2026-11-19)
Yes, the Spark is a named node in the plan, not a spare. Its edge over the 5070 Ti is pool size (20.6 GiB to PyTorch, unified) not speed, so it is assigned the big-memory and source-data jobs:
- **Weeks 1 + 4:** larger-batch ViT-B profiling contrast — same pathology script on 5070 Ti (12 GB) vs Spark (20.6 GiB); where the batch/memory ceiling moves is a graded observation.
- **Weeks 5–6:** 8B FP16 LLM serving run — the FP16 weights (~16 GB) that do NOT fit the 5070 Ti run on the Spark; 5070 Ti runs the AWQ 4-bit version. One serving table, two memory regimes.
- **Week 7:** ProNeuro head port — source coefficients/runs are on QSInfomatic03 only; export numbers/shapes from the Spark, run JAX/TPU in Colab. ADNI data never leaves the Spark.
- **Week 8:** second node for FSDP/DDP and discrete-vs-unified distributed drills.
- **Week 9:** one of the (now five) benchmark targets in Project 4: 5070 Ti / Spark / Jetson / TPU / H100, same model, same harness.
- **Weeks 10–12:** capstone large-model leg + FastSurfer throughput profiling (P1b, optional) — scans/hr and GPU-utilisation trace, numbers-only export.
- NOT on the Spark: Triton kernel timing in Weeks 2–3 (GDDR7 on the 5070 Ti is the honest kernel baseline; Spark unified bandwidth would distort the numbers).

## Curriculum — 12 weeks
- **Week 1 — PyTorch as a systems engineer.** torch.compile (what the compiler does, graph breaks), memory model (activations vs weights vs optimizer state — calculate it by hand for ViT-B), mixed precision. Drill vehicle: pathology ViT-B. Rubric 1: "Judging a torch.compile claim."
- **Week 2 — Triton I.** Official tutorials 01–04 (vector add, fused softmax, matmul, fused attention intro). Every tutorial re-run on the 5070 Ti with Min's own benchmark harness (warmup, median, GB/s). Rubric 2: kernel correctness vs performance traps.
- **Week 3 — Triton II + Portfolio Project 1.** Original kernels: (a) fused softmax, (b) tiled matmul tuned in 3 configs, (c) one kernel applied to real work — fused layer-norm / attention piece for the pathology MIL head. README table vs cuBLAS / eager PyTorch. Rubric 3.
- **Week 4 — Profiling & trace analysis + Project 2.** torch.profiler + Kineto traces and Nsight Systems on pathology ViT-B training (Project P1 in the Portfolio Map). Find, prove, and fix ONE bottleneck; before/after traces + throughput numbers published. Rubric 4: "Reading a profiler trace — what counts as evidence."
- **Week 5 — LLM serving I (vLLM).** KV cache, paged attention, continuous batching — concept drills first, then serve Qwen2.5-7B / Llama 3.1 8B on the 5070 Ti with vLLM. Measure TTFT, TPOT, tokens/sec at 1/8/32 concurrency. Rubric 5: serving claims and how to test them.
- **Week 6 — LLM serving II + Project 3.** Same model, three configs: vLLM vs HF generate vs Ollama (Smart SI's local stack). Batching / max-num-seqs sweep, KV-cache memory ceiling, failure mode at overload. Maintenance-window rules apply from Dec 23. Rubric 6.
- **Week 7 — JAX I + TPU.** JAX fundamentals for a PyTorch person (jit, grad, vmap, pytrees), port the ProNeuro ridge + ViT head to JAX, run on Colab TPU v5e, dump and read StableHLO. Rubric 7.
- **Week 8 — Distributed & sharding.** pjit / shard_map on TPU (data vs model parallel by hand), PyTorch FSDP + DDP drills on the 5070 Ti, Megatron / MaxText as *reading* (parallelism strategies: DP/TP/PP, ZeRO in one diagram each). Debugging drill: given a hung / slow distributed trace, find the straggler. Rubric 8.
- **Week 9 — Accelerator trade-offs + Project 4.** ONE model, FIVE targets: 5070 Ti, Spark QSInfomatic03, Jetson Orin Nano (TensorRT), Colab TPU, cloud H100 (capped spend, see above). Table: throughput, latency p50/p95, peak memory, watts where measurable, $/1M tokens or $/1,000 images. This table is the centrepiece interview artifact. Rubric 9.
- **Week 10 — Compiler / graph level.** XLA/StableHLO and MLIR at reading-plus-small-edits depth, torch.compile backend inspection, one custom operator (PyTorch custom op + Triton fallback). Debugging accelerator-bound workloads: 3 seeded-bug exercises. Rubric 10.
- **Week 11 — SME evaluation studio (the Cincinnatus job itself).** Timed work samples: (1) grade a flawed Triton kernel solution, (2) grade a profiler-interpretation answer, (3) grade a vLLM serving design. Each graded against Min's own rubrics 1–10, then rubrics revised where they failed to discriminate. Rubrics 11–12: full task-design packs (prompt + solution + rubric + common errors).
- **Week 12 — Capstone, mocks & launch.** Capstone: end-to-end — profile → optimize kernel → serve → benchmark across 2 accelerators, one repo, one 10-minute walkthrough script. 2 mock expert reviews (Muse as hostile reviewer). Application pack sent: Cincinnatus + 5–10 comparable ML Systems SME / ML Infra roles at verified pay (apply the $180k+ / $10k-after-tax filter; contract hourly roles converted and checked before applying).

## Weekly rhythm
- Mon–Fri: 60-min ML Systems session (separate slot from Data Engineering Academy's 90 min; never back-to-back on paper-deadline days).
- Saturday: 3-hour project block (alternating Saturdays with Data Engineering Academy projects if load demands — coach calls it on the scoreboard).
- Sunday: scoreboard only (5 min).

## Graduation gates (all must be true)
- Project 1 Triton kernels repo public, benchmark table reproduces on the 5070 Ti.
- Project 2 profiling case study public: trace before/after + measured fix (Portfolio P1).
- Project 3 vLLM serving benchmark public: TTFT/TPOT/throughput at 3 concurrency levels, 3 backends.
- Project 4 cross-accelerator table public: 5070 Ti / Jetson / TPU / H100, same model, same harness.
- JAX/TPU artifact: ported ProNeuro head running on TPU with StableHLO dump Min can explain.
- 12 rubrics written; 3 timed SME work samples passed at mock expert review.
- Application pack sent with NO inflated experience claim — portfolio links replace the missing title.

## Relationship to Min's other tracks
- **Data Engineering Academy (Nov 13–Feb 5):** continues unchanged; ML Systems is the second track, lighter until Jan 11.
- **Robotics / own-chip direction (2026-11-12):** this is Step 1 systems grounding for that ladder — kernels, accelerators, and serving trade-offs are the chip workload language.
- **USC ECE PhD exploration (Fall 2027):** Weeks 7–10 material maps directly onto the recommended topic (energy-efficient edge AI for medical imaging) and gives the SOP concrete artifacts.
- **Papers:** diabetes review (Jan 10) and pathology (Jan 2027) are protected by the load rules above; pathology Table 5 and ProNeuro Jetson work are *shared* artifacts, not extra work (see Portfolio Map).

## Session log
- 2026-11-19: Academy created at Min's request ("Both") in the ML Systems SME side chat, together with the Portfolio Map. Week 1 starts on Min's go — first drill: memory model of ViT-B by hand, then torch.compile graph-break hunt on the pathology training script.
- 2026-11-19 (later): Repo live — Min created minmcho/ML_System_Portfolio, seed pushed and verified (head 6bd9c2f2). Min said "Start" — Week 1 STARTED. requirements.txt added to the repo root (cu128 wheel index for Blackwell, WSL-only markers for Triton/vLLM, CPU JAX locally with TPU in Colab). Drill 1 presented: ViT-B training memory model by hand.
- 2026-11-20: MIL pooling kernel spec (P2a) drafted at Min's request in 01-triton-kernels/ — MIL_POOLING_SPEC.md + mil_pooling/ skeleton (reference.py, triton_kernel.py skeleton, test + bench harnesses). Local seed only, NOT yet pushed. Open items O1 (ABMIL L=256?) and O2 (D per backbone + real N distribution from feature files) gate the Week 3 coding start.
