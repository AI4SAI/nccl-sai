# NCCL-SAI Public Guide

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL for SAI
UltraPOD and SlimPOD GPU fabrics. The current public branch focuses on
application-transparent communication behavior: users should be able to load a
site-provided NCCL-SAI runtime without modifying application source code.

The first optimized API is `ncclAlltoAll()`. NCCL-SAI also has a narrowly
guarded local P2P transport-selection path for one eligible single-host layout;
that transport choice can affect allreduce and other operations even though
their collective algorithms are unchanged. All communication families must
remain correct and pass non-regression gates before any broad default
deployment.

## Activation Model

Public source must not hard-code a private cluster, hostname pattern, scheduler
partition, or filesystem path. NCCL-SAI uses generic opt-in controls:

- `NCCL_SAI_A2A_ENABLE=1` explicitly enables the alltoall SAI path.
- `NCCL_SAI_A2A_ENABLE=0` explicitly disables the alltoall SAI path.
- When `NCCL_SAI_A2A_ENABLE` is unset, the alltoall SAI path follows
  `NCCL_SAI_FABRIC_PROFILE`.
- `NCCL_SAI_FABRIC_PROFILE=<family>[-<variant>]` enables default SAI behavior
  only for recognized product families. The current public families are
  `ultrapod` and `slimpod`; matching is case-insensitive and the optional
  variant suffix must be nonempty.
- Empty, unknown, or disabled-style values such as `0`, `false`, `off`,
  `none`, `native`, and `upstream` fail closed to upstream behavior.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1` explicitly enables the local PCIe
  path-relaxation guard used for selected single-node layouts.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0` disables that local path-relaxation guard.
- When `NCCL_SAI_LOCAL_P2P_SYS_ENABLE` is unset, the local path-relaxation
  guard follows `NCCL_SAI_FABRIC_PROFILE`.
- Explicit `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings take precedence
  over the profile-driven local path relaxation.

Site-specific modulefiles, prologs, or container entrypoints may set these
variables privately. Public code and docs should only describe the generic
profile mechanism. When both explicit enable variables are unset and
`NCCL_SAI_FABRIC_PROFILE` is unset or unrecognized, NCCL-SAI falls back to
upstream NCCL behavior.

## Current Tunables

- `NCCL_SAI_A2A_LANE_ENABLE`: enable or disable the large-message lane path.
- `NCCL_SAI_A2A_LANES`: number of logical lanes used by the lane path.
- `NCCL_SAI_A2A_GROUP_NODES`: NCCL topology-node count in one lane-local
  aggregation unit. A topology node is not necessarily a physical host; sites
  that split one host into multiple locality domains must set this from the
  communicator view reported by NCCL.
- `NCCL_SAI_A2A_MIN_PEER_BYTES`: lower per-peer size bound for the lane path.
- `NCCL_SAI_A2A_MIN_RANKS`: minimum communicator size for the lane path.
- `NCCL_SAI_A2A_LANE_TRACE`: emit rank-zero eligibility diagnostics for the
  alltoall SAI paths.
- `NCCL_SAI_A2A_ISLAND_ENABLE`: enable or disable the small-message island path.
- `NCCL_SAI_A2A_ISLAND_SIZE`: rank count in one local island.
- `NCCL_SAI_A2A_ISLAND_MAX_PEER_BYTES`: upper per-peer size bound for the
  island path.
- `NCCL_SAI_A2A_ISLAND_MIN_RANKS`: minimum communicator size for the island
  path.
- `NCCL_SAI_A2A_ISLAND_BULK_LOCAL_ENABLE`: enable or disable the island-local
  bulk copy fast path when ranks are globally contiguous by island. Disabling
  it, or failing its rank-layout guard, falls back to upstream AlltoAll.
- `NCCL_SAI_LOCAL_P2P_SYS_TRACE`: emit initialization diagnostics when the local
  path-relaxation guard accepts a candidate path.
- `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE`: enable the advanced P2P schedule that
  groups peer steps by local fabric-domain size. This is disabled by default and
  should be treated as an expert-only tuning knob until validated for a site.
- `NCCL_SAI_P2P_FABRIC_NODES`: topology-node count in one local fabric-domain
  group for `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE`. This is expressed in the
  communicator's NCCL topology nodes, not scheduler host count.

These defaults are implementation policy, not a promise that one setting is
optimal for every SAI system. Site packages may ship conservative defaults and
leave expert overrides available.

The optimized paths currently require distinct send and receive buffers, a
blocking communicator, and a top-level `ncclAlltoAll()` call. Aliased buffers
fall back to upstream NCCL; NCCL does not document an in-place AlltoAll
contract. Calls made inside an outer `ncclGroupStart()` / `ncclGroupEnd()`
region, nonblocking communicators, nonuniform rank layouts, and layouts that
fail the configured topology guards also fall back upstream. Lane and fabric
grouping assume that topology-node numbering is contiguous inside each
configured unit; site validation must confirm that communicator view before
enabling a profile by default.

The local P2P path-relaxation guard is intentionally narrow. It is eligible
only for a single-host 8-rank communicator that forms exactly two local
4-rank GPU islands, and only for cross-island paths whose topology distance is
outside the normal NVB class but still within SYS. It is not a general override
for multi-node P2P behavior or arbitrary single-node layouts.

The island-bulk path uses stream-ordered temporary device memory. For `N`
islands of size `I`, its scratch bound is
`peerBytes * N * I + 2 * peerBytes * N * (I - 1)`. Site validation must leave
that headroom on every rank. A scratch allocation failure is returned as a CUDA
error; NCCL-SAI does not let one rank silently switch to the upstream schedule
after other ranks selected island-bulk.

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
- fallback behavior when `NCCL_SAI_FABRIC_PROFILE` is unset or unrecognized.

See `docs/sai/BUILD_AND_PACKAGING.md` for the public binary-package build
matrix, including CUDA architecture coverage and portable x86-64-v3/v4 host ISA
targets.

Public performance summaries should report only sanitized scale classes,
message sizes, NCCL/CUDA versions, and topology classes. Keep private job IDs,
node names, switch labels, raw logs, internal paths, and operational incident
notes outside the public repository.

## Offline Algorithm Checks

`tools/sai/verify_a2a_algorithms.py` mirrors the SAI schedule formulas without
requiring CUDA devices. It verifies:

- every P2P schedule rank has complete, duplicate-free send and receive peer
  coverage, with matching peers in every round;
- the lane formula assigns both directions of a rank pair to the same phase and
  balances each modeled fabric edge across phases;
- island-bulk pack, exchange, and unpack indexing reproduces exact alltoall
  placement for both distinct and defensively modeled aliased buffers. The
  alias model is not a public in-place support claim; dispatch sends aliased
  calls to upstream NCCL;
- large power-of-two and non-power-of-two fabric-group schedule shapes remain
  structurally valid without encoding a private deployment size.

Run the full model before publishing source changes that touch these paths:

```bash
python3 tools/sai/verify_a2a_algorithms.py --full-scale
```

The CI workflow also compiles and runs
`tools/sai/test_profile_activation.cc` to keep profile recognition fail-closed.

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
