# NCCL-SAI v2.28.9-1-sai.2 Release Notes

NCCL-SAI is an AI4SAI optimization release derived from NVIDIA NCCL. This
release is based on NCCL `v2.28.9-1` and targets SAI UltraPOD and SlimPOD GPU
fabric families.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is not endorsed by NVIDIA. Original NVIDIA NCCL copyright and license notices
are retained in `LICENSE.txt`; see `docs/sai/NOTICE.md`.

## Automatic Release Capabilities

- Dual-rail-by-channel transport selection on the exact supported SM70,
  four-GPU-NVLink-clique and dual-port physical topology.
- Local P2P transport selection on the exact single-host eight-rank, two-clique
  SM70 topology with distinct CPU locality domains.

Both capabilities require their complete runtime hardware predicate and
communicator-wide agreement. Neither identifies the site from Slurm, hostnames,
partitions, device-name strings, configuration paths, or environment-variable
strings. Missing or ambiguous evidence fails closed to upstream behavior.

## Expert-Only AlltoAll Implementation

The source contains `ncclAlltoAll()` paths for explicit expert experiments on
a hardware-validated topology with rank-consistent fabric-group metadata.

Eligible small-message calls use a two-stage GPU-island aggregation path.
Eligible large out-of-place calls use the phased planner. Unsupported layouts
and sizes use upstream scheduling. The island path supports distinct and
exactly aliased buffers; partially overlapping buffers are not claimed.
Normal zero-variable runs keep grouped AlltoAll on the upstream path because
NCCL 2.28 does not expose a stable cross-node hardware fabric-group identity.
Expert use requires complete rank-consistent fabric metadata. An equal total
topology-node count alone does not prove that one group is complete.

NCCL-SAI also includes a narrowly guarded local P2P transport-selection path
for an eligible single-host 8-rank, two-island communicator. That path can
affect any operation using local P2P transport, while the allreduce, allgather,
reduce-scatter, broadcast, and reduce collective algorithms remain unchanged.

The automatic dual-rail path requires the actual topology visible to NCCL to
contain one to four complete, equal-bandwidth four-GPU NVLink cliques of SM70
GPUs, each on a distinct CPU locality domain, and exactly eight unmerged NET
endpoints arranged as four capability-symmetric, GDR-capable dual-port
adapters. Each clique must resolve one distinct physical port pair, and all
GPU-to-port paths must have the same type and bandwidth. Slurm metadata, HCA
strings, and merge-variable strings are not activation tests.
The internal IB transport must report two distinct selected GID subnet
prefixes, with port 1 and port 2 each communicator-wide consistent; missing or
inconsistent hardware identity retains upstream selection.
Effective `NCCL_CROSS_NIC=0` is a rail-behavior gate. Physical port 1 serves
even channels and physical port 2 serves odd channels; NET enumeration order
has no rail meaning. Ordinary ring/tree endpoints use the same channel-aligned
port and recompute the network device and proxy rank. CollNet/NVLS, merged-NIC
topologies, nonzero cross-NIC mode, and rank-inconsistent layouts retain the
corresponding upstream path. An explicit non-`AUTO` `NCCL_NETDEVS_POLICY`
likewise keeps upstream device selection.

## Transparent Runtime Model

Normal users, applications, and shared MPI stacks set no `NCCL_SAI_*`
variables. NCCL-SAI does not identify a site from Slurm, hostnames, partitions,
device-name strings, or configuration paths. Rail and local-P2P activation use
strict runtime topology plus capability and safety checks, followed by
communicator-wide agreement.
Grouped AlltoAll/P2P schedules are not enabled automatically: scheduler
metadata is neither hardware truth nor a site identity. Upstream controls such as
`NCCL_IB_HCA`, `NCCL_IB_MERGE_NICS`, and `NCCL_CROSS_NIC` retain their upstream
meaning and may be supplied by normal site policy; NCCL-SAI evaluates the
topology they actually produce rather than matching their text values.

`NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh` remains a compatibility/expert
request for grouped AlltoAll/P2P, not a normal prerequisite and not a hardware
bypass. The structural topology predicate must still pass. Profile values do
not enable or suppress automatic rail or local-P2P capability detection.

Rollback and expert controls:

- `NCCL_SAI_DISABLE=1`: globally disable SAI-specific behavior.
- Upstream `NCCL_IB_MERGE_NICS=0` normally leaves physical endpoints visible;
  the automatic predicate still decides from the resulting topology.
- Upstream `NCCL_IB_MERGE_NICS=1` requests merged virtual devices. This mode
  can run; a successful four-vdev result naturally does not match the strict
  eight-endpoint automatic class.
- Upstream `NCCL_CROSS_NIC=0` allows the ordinary ring/tree rail override;
  other effective values retain the graph-selected endpoint.
- `NCCL_SAI_A2A_ENABLE=1`: request expert evaluation of the AlltoAll SAI path;
  it does not bypass the hardware predicate or fabric-group validation.
- `NCCL_SAI_A2A_ENABLE=0`: disable the alltoall SAI path.
- `NCCL_SAI_A2A_ISLAND_ENABLE=0`: disable the small-message island path while
  retaining other eligible AlltoAll behavior.
- `NCCL_SAI_A2A_ISLAND_SIZE=<N>`: expert override for the number of contiguous
  ranks in one regular GPU island.
- `NCCL_SAI_A2A_ISLAND_MIN_RANKS=<N>`: expert override for the island-path
  scale guard; the profile default is 576 ranks.
- `NCCL_SAI_FABRIC_GROUP_ID=<N>`: provide rank-local numeric metadata used to
  validate complete multi-group layouts in controlled expert tests. It is not
  set by the default module or normal application runtime.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE` unset: use strict automatic local-P2P
  hardware detection.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0`: disable that local path as a rollback.
  Positive values still cannot force an unsupported topology through the
  hardware predicate.
- Standard `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings retain upstream
  policy meaning and are not site-identity signals. `NCCL_P2P_DISABLE=1` and
  explicit `LOC`/`PIX`/`PXB`/`PHB` limits retain precedence. An unset level or
  `NVL` may admit the strictly hardware-qualified two-island SYS exception
  after communicator-wide agreement, with no `NCCL_SAI_*` activation variable.
- Rank-local P2P policy mismatch is a deterministic initialization error.
  Hardware eligibility disagreement disables only the automatic exception and
  falls back to the common upstream policy.
- `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE=1`: enable an advanced P2P schedule for
  locally grouped fabric domains. It is a subpolicy of the enabled,
  hardware-validated grouped-A2A profile, uses `NCCL_SAI_A2A_GROUP_NODES`,
  requires complete multi-group metadata, and never activates for a single
  group. A standalone positive setting cannot activate it; this is an
  expert-only path in this release. This control does not activate the separate
  ragged AlltoAll planner, whose metadata, occupancy, and build-consensus gates
  remain independent.

The automatic policy does not set a communicator-wide minimum channel
count. Standard upstream `NCCL_MIN_NCHANNELS` controls remain explicit expert
tuning because a global floor can change unrelated collective and P2P paths.

The release deliberately does not derive fabric groups from Slurm variables,
rank order, hostnames, HCA names, LIDs, or guessed GUID ranges. None of those is
a stable cross-node hardware grouping proof. Until a hardware-backed provider
exists, normal zero-SAI-variable grouped paths remain upstream. Explicit
numeric group IDs are retained only for controlled expert validation and never
determine rail or local-P2P eligibility.

## Release Qualification

Release assets are publishable only after the sanitized runtime evidence binds
the exact source commit, source tree, source archive SHA256, shared-library
SHA256, static-library SHA256, CUDA architecture coverage, and portable host
ISA. Matching checksums and contents manifests accompany the source and binary
assets.

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
claim for the ideal balanced fabric requires 16-18 complete, uniformly occupied
groups. That scale is not a general release gate. Smaller layouts remain valid
for correctness, fallback, and topology-normalized non-regression evidence.

## Not Claimed

This release does not claim:

- universal speedup for all NCCL or MPI collectives;
- application-level speedup for programs that do not use the optimized path;
- performance guarantees for every topology, scheduler allocation, or
  concurrent production workload;
- optimized performance for partially occupied fabric groups;
- a performance ordering between `NCCL_IB_MERGE_NICS=0` and `1` until a
  sustained same-candidate A/B is recorded;
- that a `GPU Direct RDMA Enabled` capability message proves the actual
  transport selected by `NCCL INFO Channel` lines;
- upstream vendor endorsement.

## Redistribution

This release is published under the `AI4SAI` organization in a public NCCL-SAI
repository derived from NVIDIA NCCL. Upstream license files and copyright
notices remain intact, and the repository keeps upstream provenance explicit
without implying NVIDIA endorsement. Binary packages and tarballs include
`LICENSE.txt`, `docs/sai/NOTICE.md`, and release notes that reproduce the
required upstream notice/disclaimer boundary.
