# MoE EP And Scientific Application Notes

NCCL-SAI includes expert AlltoAll research, but SAI application performance is
a full-stack problem. Public guidance should distinguish direct NCCL users,
MPI applications with GPU buffers, and applications whose dominant
communication is halo exchange, reductions, or solver-specific patterns.

## MoE Expert Parallelism

Large MoE training systems usually avoid one global all-to-all across every
rank. Common strategies include limiting expert-parallel groups, using
hierarchical dispatch, overlapping token exchange with compute, batching small
messages, reusing communicators, and relying on runtime libraries that manage
connection pressure. NCCL-SAI AlltoAll research is relevant to this class of
traffic, but it is not a transparent release capability. Public claims should
state the communicator size, topology class, and explicit expert configuration
instead of implying that one microbenchmark covers all MoE runtimes.

## Scientific Applications

Many scientific applications do not call `ncclAlltoAll()` directly:

- VASP is a closed-source application for this purpose; optimize through the
  MPI, CUDA, NCCL, and scheduler/runtime environment without requiring source
  changes.
- ABACUS, DeePMD-kit, NEP/GPUMD, and related open ecosystems can be optimized
  collaboratively through runtime defaults, optional profiling, and upstreamable
  communication choices.
- CP2K, GROMACS, and similar codes often depend heavily on MPI collectives,
  halo exchange, PME/FFT decomposition, or solver reductions. NCCL-SAI should
  not claim application speedups from alltoall alone.

## Transparent Optimization Priorities

Prefer optimizations that are invisible to application source code:

- packaged NCCL-SAI runtime with conservative defaults;
- site MPI/UCX/UCC settings that preserve CUDA-aware behavior;
- topology-aware rank ordering and scheduler placement;
- separately documented environment overrides for controlled expert research;
- optional PMPI/NCCL observation tools that can be used without recompiling
  closed-source applications.

When source-level work is possible with friendly upstreams, keep it optional
and compatible with the transparent runtime path.

## Collective Coverage

Track at least these communication families:

- alltoall and alltoallv-style exchange;
- allreduce and reduce-scatter;
- allgather and broadcast;
- neighbor or halo exchange;
- point-to-point send/recv loops;
- GPU-buffer MPI collectives.

This prevents overfitting the release to one benchmark while missing the
communication pattern that dominates a real application.
