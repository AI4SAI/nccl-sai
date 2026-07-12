# NCCL-SAI v2.28.9-1-sai.1 Release Notes

NCCL-SAI is an AI4SAI optimization release derived from NVIDIA NCCL. This
release is based on NCCL `v2.28.9-1` and targets SAI UltraPOD and SlimPOD GPU
fabric families.

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

The validated dual-rail mode requires
`NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh`, `NCCL_IB_MERGE_NICS=0`, and
`NCCL_CROSS_NIC=0`. A GPU with exactly two topology-local NET candidates then
selects them by channel parity. Ordinary ring/tree graph endpoints use the same
channel-aligned local NET and recompute the network device and proxy rank;
CollNet and NVLS graphs retain their existing endpoint selection. Broader
family-only profiles, merged-NIC mode, nonzero cross-NIC mode, and any other
local-NET count retain the corresponding upstream path.

## Transparent Runtime Model

SAI sites can enable the validated AlltoAll defaults and dual-rail mode with:

```bash
export NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh
export NCCL_IB_MERGE_NICS=0
export NCCL_CROSS_NIC=0
```

The recognized public families are `ultrapod` and `slimpod`; a nonempty
`-<variant>` suffix is accepted for site packaging. The profile selects the
current island and phased defaults. Dual-rail local-NET selection additionally
requires NIC merging to be explicitly disabled, and ordinary ring/tree graph
endpoint override additionally requires cross-NIC mode to be explicitly
disabled. Recognizing a broader family name does not enable those
topology-specific paths. Empty, unknown, or disabled-style profile values such
as `0`, `false`, `off`, `none`, `native`, and `upstream` fail closed to upstream
behavior.

Explicit overrides:

- `NCCL_IB_MERGE_NICS=0`: keep the two validated physical NET candidates
  separate so channel-parity selection can activate.
- `NCCL_CROSS_NIC=0`: allow the ordinary ring/tree graph endpoint override;
  other values retain the graph-selected endpoint.
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

## Historical RC1 Performance Boundary

In the published RC1, a sustained same-domain 64-GPU `alltoall_perf` run using
a 1 GiB per-rank test size measured 6.09 GB/s bus bandwidth on the eligible
out-of-place phased path. Disabling the NCCL-SAI AlltoAll policy in the same
build measured 4.78 GB/s. The exactly aliased benchmark path remained upstream
at 4.71 GB/s, producing a combined average of 5.40 GB/s.

This historical RC1 result demonstrates recovery of large out-of-place
AlltoAll performance on the validated topology class. It does not qualify the
new island path or demonstrate a universal application-level speedup. Public
island performance claims remain gated on a clean exact-binary scale run.

## Release Qualification

The binary package distributed with this release passed the release gates
recorded in its sanitized `RUNTIME_EVIDENCE.json`. Its `BUILD_INFO.txt` binds
the exact source commit, source tree, source archive SHA256, shared-library
SHA256, static-library SHA256, CUDA architecture coverage, and portable host
ISA. Matching checksums and contents manifests are published with the source
and binary assets.

The build audit records the clean source identity, advertised SASS/PTX
architecture coverage, portable host ISA target, package contents, and library
digests. Communication, topology, message-size, CUDA Graph, fallback, and
cross-collective claims are limited to the exact gate entries present in the
published sanitized evidence summary. A gate or scale class not listed there
is not claimed by this release.

Application acceptance is a separate private gate bound to the exact final
shared-library hash. Private workloads, scheduler records, and raw application
logs are not release assets. This release does not use application acceptance
as island-path evidence and does not claim a VASP application speedup.

For the full-mesh layout class with 16 equal-bandwidth endpoints per group and
one equal-bandwidth direct edge per group pair, any new headline performance
qualification requires 16-18 complete, uniformly occupied groups. Smaller
layouts remain correctness or topology-normalized efficiency evidence.

## Not Claimed

This release does not claim:

- universal speedup for all NCCL or MPI collectives;
- application-level speedup for programs that do not use the optimized path;
- performance guarantees for every topology, scheduler allocation, or
  concurrent production workload;
- optimized performance for partially occupied fabric groups;
- upstream vendor endorsement.

## Redistribution

This release is published under the `AI4SAI` organization in a public NCCL-SAI
repository derived from NVIDIA NCCL. Upstream license files and copyright
notices remain intact, and the repository keeps upstream provenance explicit
without implying NVIDIA endorsement. Binary packages and tarballs include
`LICENSE.txt`, `docs/sai/NOTICE.md`, and release notes that reproduce the
required upstream notice/disclaimer boundary.
