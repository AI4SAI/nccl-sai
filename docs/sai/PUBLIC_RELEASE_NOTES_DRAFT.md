# NCCL-SAI Public Release Notes Draft

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL. The current
source candidate is based on NCCL `v2.28.9-1` and targets SAI UltraPOD and
SlimPOD GPU fabric families.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is not endorsed by NVIDIA. Original NVIDIA NCCL copyright and license notices
are retained in `LICENSE.txt`; see `docs/sai/NOTICE.md`.

## Optimized API

- `ncclAlltoAll()` on eligible SAI fabric profiles.

Eligible small-message calls use a two-stage GPU-island aggregation path.
Eligible large out-of-place calls use the phased planner. Unsupported layouts
and sizes use upstream scheduling. The island path supports distinct and
exactly aliased buffers; partially overlapping buffers are not claimed.
Automatic grouped use requires complete rank-consistent fabric metadata. An
equal total topology-node count alone does not prove that one group is complete.

NCCL-SAI also includes a narrowly guarded local P2P transport-selection path
for an eligible single-host 8-rank, two-island communicator. That path can
affect any operation using local P2P transport, while the allreduce, allgather,
reduce-scatter, broadcast, and reduce collective algorithms remain unchanged.

## Transparent Runtime Model

SAI sites can enable default NCCL-SAI behavior by setting:

```bash
export NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh
```

The recognized public families are `ultrapod` and `slimpod`; a nonempty
`-<variant>` suffix is accepted for site packaging. The current island and
phased defaults are specific to the `ultrapod-fullmesh` profile; recognizing a
broader family name does not enable those topology-specific paths. Empty,
unknown, or disabled-style values such as `0`, `false`, `off`, `none`,
`native`, and `upstream` fail closed to upstream behavior.

Explicit overrides:

- `NCCL_SAI_A2A_ENABLE=1`: enable the alltoall SAI path.
- `NCCL_SAI_A2A_ENABLE=0`: disable the alltoall SAI path.
- `NCCL_SAI_A2A_ISLAND_ENABLE=0`: disable the small-message island path while
  retaining other eligible AlltoAll behavior.
- `NCCL_SAI_A2A_ISLAND_SIZE=<N>`: expert override for the number of contiguous
  ranks in one regular GPU island.
- `NCCL_SAI_A2A_ISLAND_MIN_RANKS=<N>`: expert override for the automatic
  island-path scale guard; the profile default is 576 ranks.
- `NCCL_SAI_FABRIC_GROUP_ID=<N>`: provide rank-local numeric metadata used to
  validate complete multi-group layouts. This explicit value takes priority
  over automatic scheduler metadata.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1`: enable the selected local P2P path guard.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0`: disable that local path guard.
- Standard `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings take precedence
  over the profile-driven local path relaxation.
- `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE=1`: enable an advanced P2P schedule for
  locally grouped fabric domains. It uses `NCCL_SAI_A2A_GROUP_NODES`, requires
  complete multi-group metadata, and never activates for a single group.
  Automatic mode enables it only for the validated full-mesh profile; other
  profiles remain upstream by default.

The transparent profile does not set a communicator-wide minimum channel
count. Standard upstream `NCCL_MIN_NCHANNELS` controls remain explicit expert
tuning because a global floor can change unrelated collective and P2P paths.

When the full-mesh profile is active and no explicit fabric-group ID is set,
NCCL-SAI can derive the group from standard rank-local Slurm topology address
and pattern variables. Parsing and communicator-wide completeness checks fail
closed; no scheduler command, file lookup, or fabric-management query occurs in
the library.

## Preliminary Performance Boundary

In the published RC1, a sustained same-domain 64-GPU `alltoall_perf` run using
a 1 GiB per-rank test size measured 6.16 GB/s bus bandwidth on the eligible
out-of-place phased path. Disabling the NCCL-SAI AlltoAll policy in the same
build measured 4.80 GB/s. The exactly aliased benchmark path remained upstream
at 4.76 GB/s, producing a combined average of 5.46 GB/s.

This historical RC1 result demonstrates recovery of large out-of-place
AlltoAll performance on the validated topology class. It does not qualify the
new island path or demonstrate a universal application-level speedup. Public
island performance claims remain gated on a clean exact-binary scale run.

## Release Scope

This draft describes a source-level public candidate. Public binary packages
should be published only after clean rebuild, architecture coverage, and
sanitized validation evidence are available. Build and packaging expectations
are documented in `docs/sai/BUILD_AND_PACKAGING.md`.

Minimum public gates:

- clean source build;
- CUDA SASS/PTX coverage and portable host ISA target evidence for the advertised
  package;
- single-node, same-domain, cross-domain, and multi-domain alltoall;
- KB, MB, and GB message-size coverage;
- CUDA Graph capture and replay through the enqueue-based paths;
- allreduce, reduce-scatter, allgather, broadcast, and P2P non-regression;
- fallback behavior when `NCCL_SAI_FABRIC_PROFILE` is unset or unrecognized.

For the full-mesh layout class with 16 equal-bandwidth endpoints per group and
one equal-bandwidth direct edge per group pair, final headline performance
qualification must use 16-18 complete, uniformly occupied groups. Smaller
layouts remain correctness or topology-normalized efficiency evidence.

## Not Claimed

This release draft does not claim:

- universal speedup for all NCCL or MPI collectives;
- application-level speedup for programs that do not use the optimized path;
- performance guarantees for every topology, scheduler allocation, or
  concurrent production workload;
- optimized performance for partially occupied fabric groups;
- upstream vendor endorsement.

## Redistribution

Publish under the `AI4SAI` organization as a public NCCL-SAI repository derived
from NVIDIA NCCL. Keep upstream license files and copyright notices intact.
Keep upstream provenance explicit through documentation, tags, and an upstream
remote, but do not imply NVIDIA endorsement. Binary packages or tarballs must
include `LICENSE.txt`, `docs/sai/NOTICE.md`, and release notes that reproduce
the required upstream notice/disclaimer boundary.
