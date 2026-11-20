"""Benchmark harness skeleton (spec section 6).

5070 Ti only for headline numbers (never Spark, never Jetson). 25 warmup +
100 timed iters, CUDA events, median. Baselines: (a) eager PyTorch,
(b) torch.compile, (c) Triton -- a table with only (c) is a failed deliverable.
Run: python bench_mil_pooling.py   (WSL, CUDA required)
"""
import statistics
import torch

from reference import mil_pool_ref

WARMUP, ITERS = 25, 100


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(ITERS):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)  # ms


def gbs(N, D, ms, dtype_bytes=2):
    total_bytes = N * D * dtype_bytes + N * 4  # H read once + logits
    return total_bytes / (ms * 1e-3) / 1e9


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required -- run on the 5070 Ti in WSL.")
    compiled = torch.compile(mil_pool_ref)
    print(f"{'N':>7} {'D':>5} {'eager ms':>9} {'compile ms':>10} {'triton ms':>10} {'GB/s':>7}")
    for N in [1024, 4096, 10000, 27000]:
        for D in [512, 768, 1024, 1280]:
            H = torch.randn(N, D, device="cuda", dtype=torch.float16)
            logits = torch.randn(N, device="cuda")
            ms_a = bench(mil_pool_ref, H, logits)
            ms_b = bench(compiled, H, logits)
            # TODO(Week 3): ms_c = bench(mil_pool_triton, H, logits) once s.5 passes
            print(f"{N:>7} {D:>5} {ms_a:>9.3f} {ms_b:>10.3f} {'--':>10} {'--':>7}")


if __name__ == "__main__":
    main()
