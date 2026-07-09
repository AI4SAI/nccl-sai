# NCCL-SAI Public Guide

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL for SAI
UltraPOD and SlimPOD GPU fabrics. The current public branch focuses on
application-transparent communication behavior: users should be able to load a
site-provided NCCL-SAI runtime without modifying application source code.

The first optimized API is `ncclAlltoAll()`. Other collectives, including
allreduce, reduce-scatter, allgather, broadcast, and point-to-point operations,
must remain correct and pass non-regression gates before any broad default
deployment.

## Activation Model

Public source must not hard-code a private cluster, hostname pattern, scheduler
partition, or filesystem path. NCCL-SAI uses generic opt-in controls:

- `NCCL_SAI_FABRIC_PROFILE=<name>` enables default SAI behavior for a site
  profile. Recommended public names are product-family names such as
  `ultrapod` or `slimpod`.
- Disabled profile values are `0`, `false`, `off`, `none`, `native`, and
  `upstream`.
- `NCCL_SAI_A2A_ENABLE=1` explicitly enables the alltoall SAI path.
- `NCCL_SAI_A2A_ENABLE=0` explicitly disables the alltoall SAI path.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1` explicitly enables the local PCIe
  path-relaxation guard used for selected single-node layouts.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0` disables that local path-relaxation guard.

Site-specific modulefiles, prologs, or container entrypoints may set these
variables privately. Public code and docs should only describe the generic
profile mechanism.

## Current Tunables

- `NCCL_SAI_A2A_LANE_ENABLE`: enable or disable the large-message lane path.
- `NCCL_SAI_A2A_LANES`: number of logical lanes used by the lane path.
- `NCCL_SAI_A2A_GROUP_NODES`: node count in one local aggregation domain.
- `NCCL_SAI_A2A_MIN_PEER_BYTES`: lower per-peer size bound for the lane path.
- `NCCL_SAI_A2A_ISLAND_ENABLE`: enable or disable the small-message island path.
- `NCCL_SAI_A2A_ISLAND_SIZE`: rank count in one local island.
- `NCCL_SAI_A2A_ISLAND_MAX_PEER_BYTES`: upper per-peer size bound for the
  island path.

These defaults are implementation policy, not a promise that one setting is
optimal for every SAI system. Site packages may ship conservative defaults and
leave expert overrides available.

## Public Validation Standard

Before a source branch or binary package is presented as a public NCCL-SAI
candidate, validate at least:

- build reproducibility from a clean source tree;
- CUDA architecture coverage required by the target package;
- single-node alltoall small and large messages;
- same-domain multi-node alltoall small and large messages;
- cross-domain alltoall small and large messages;
- multi-domain alltoall at sustained runtime, not only a short smoke test;
- allreduce, reduce-scatter, allgather, broadcast, and common P2P paths for
  correctness and non-regression;
- fallback behavior when `NCCL_SAI_FABRIC_PROFILE` is unset or disabled.

Public performance summaries should report only sanitized scale classes,
message sizes, NCCL/CUDA versions, and topology classes. Keep private job IDs,
node names, switch labels, raw logs, internal paths, and operational incident
notes outside the public repository.

## Supported Fabric Families

NCCL-SAI is intended for SAI GPU fabric families rather than a single private
deployment. Public documentation should describe behavior across UltraPOD and
SlimPOD style systems, including rail-aware and topology-aware deployments.
Site-specific topology labels and scheduler policies belong in private
deployment configuration.

## License And Notice

NCCL-SAI is derived from NVIDIA NCCL. Keep upstream copyright and license
notices intact, include `LICENSE.txt` with source or binary redistribution, and
include `docs/sai/NOTICE.md` or equivalent release-note text that makes the
AI4SAI modification boundary clear. Do not imply endorsement by NVIDIA or other
upstream contributors.
