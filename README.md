# GPU & Distributed Systems Portfolio

Three hands-on systems programming projects — each with real, measured performance data, profiler evidence, and an honest account of what worked and what didn't. Built to demonstrate practical GPU and distributed computing skills beyond certificates: real code, real hardware, real numbers.

## Projects

### [CUDA Matrix Multiplication: Naive vs. Tiled](./cuda-matmul-tiling)
**1.56x speedup** (9.04ms → 5.78ms) from shared-memory tiling, verified with Nsight Compute profiling on a Tesla T4. Diagnoses the exact bottleneck (L1 cache pipe congestion from redundant global memory requests, not DRAM bandwidth) rather than just reporting a faster number.

### [MPI Scaling: Single-Node and True Multi-Node](./mpi-scaling)
**98.3% parallel efficiency** (3.93x speedup on 4 processes) on a dedicated 4-core cloud VM, plus genuine **multi-node scaling across two independent physical machines** connected over a real network — including the infrastructure work that requires (security groups, firewall configuration, SSH trust between nodes). Also includes an honest negative-scaling result on a shared 2-core environment, included deliberately to show the difference between a correctness problem and a hardware capacity problem.

### [NCCL Multi-GPU Collective Operations Benchmark](./nccl-benchmark)
Benchmarks NVIDIA's NCCL library across all-reduce, all-gather, and broadcast on a 2x GPU system, using NVIDIA's official `nccl-tests` suite. Includes an honest note on a hardware constraint encountered mid-project (no NVLink on the available consumer GPUs) and how the benchmark was adapted to still produce a legitimate, explainable comparison.

## What ties these together

Each project follows the same approach: build something real, measure it with the actual profiling/benchmarking tools used in production (Nsight Compute, MPI's own timing, NCCL's official test suite), and explain *why* the numbers look the way they do — not just report a headline figure. Where results were messy, non-ideal, or constrained by available hardware, that's documented rather than hidden, because that's a more honest signal of hands-on understanding than a uniformly clean result would be.
