# ML Systems Portfolio — Min Myint Cho

Hands-on ML systems work built in a 12-week ML Systems Academy (started 2026-11-19): custom kernels, profiling and trace analysis, LLM serving, JAX/TPU, and cross-accelerator trade-off studies — each project paired with an expert evaluation rubric of the kind used to grade ML systems work professionally.

**Plan:** [PLAN.md](PLAN.md) · **How existing research maps into this portfolio:** [PORTFOLIO_MAP.md](PORTFOLIO_MAP.md)

## Hardware (all owned, no cloud dependence except one capped benchmark)
| Node | Memory seen by PyTorch | Role in this portfolio |
|---|---|---|
| RTX 5070 Ti laptop | 12 GB GDDR7 (discrete) | Triton kernels, torch.profiler / Nsight, AWQ serving — the honest kernel-timing baseline |
| RTX Spark QSInfomatic03 (HP, N1X) | 20.6 GiB (32 GB unified, WSL 24 GB) | Large-batch profiling, 8B FP16 serving, ProNeuro source runs (data never leaves this node) |
| Jetson Orin Nano 8 GB | 8 GB unified | TensorRT FP16/INT8 edge deployment target |
| Google Colab TPU v5e | — | JAX, pjit/shard_map, StableHLO |
| Cloud H100 (rented, capped) | 80 GB | Week 9 comparison target only |

## Projects
| # | Project | Folder | Status |
|---|---|---|---|
| P2 | Triton kernels: fused softmax, tiled matmul, MIL attention pooling — benchmarked vs cuBLAS / eager PyTorch | `01-triton-kernels/` | Weeks 2–3 |
| P1 | Profiling case study: pathology ViT-B training (torch.profiler + Nsight), one bottleneck found, proven, fixed | `02-profiling-pathology-vitb/` | Week 4 |
| P3 | LLM serving benchmark: vLLM vs Ollama vs HF generate — TTFT, TPOT, tokens/sec at concurrency 1/8/32; AWQ on 5070 Ti, FP16 on Spark | `03-vllm-serving/` | Weeks 5–6 |
| P5 | JAX/TPU port of the ProNeuro-AD prediction head, with annotated StableHLO | `04-jax-tpu-proneuro-head/` | Week 7 |
| P4 | Cross-accelerator benchmark: one model, five targets (5070 Ti / Spark / Jetson / TPU / H100), same harness | `05-accelerator-benchmark/` | Week 9 |
| R01–R12 | Evaluation rubrics, one per academy week | `rubrics/` | Weeks 1–12 |
| P6 | Pathology WSI pipeline: biomedical AI design pipeline (tile → embed → Triton MIL pool → predict+audit → benchmark → package) for the BJBS paper — 40 CAMELYON17 slides × 1,000 patches, 3→6 backbones | `06-pathology-wsi-pipeline/` | Design drafted 2026-11-20 |

## Research this portfolio builds on (separate repos, linked not copied)
- **ProNeuro-AD** — MRI prognostic AI for Alzheimer's on ADNI: github.com/minmcho/ProNeuro-AD — baseline n=2,916, FreeSurfer ridge amyloid AUC 0.82±0.03 / conversion C-index 0.76±0.01; deep table n=1,659 (learned features do not beat morphometry — a paired-evidence result).
- **Digital pathology benchmark** — github.com/minmcho/multi-agent-digital-pathology — 40 CAMELYON17 slides × 3 backbones, ABMIL+CLAM × 10 seeds, 730 audited runs: ViT-B AUC 0.990 / r 0.98, ResNet-18 0.953 / 0.97, ViT-Tiny 0.681 / 0.01.

## Honesty note
This portfolio evidences hands-on project experience. It does not claim a professional ML-infrastructure job title the author has not held — every number here is reproducible from the code and harnesses in this repo, on the hardware in the table above.
