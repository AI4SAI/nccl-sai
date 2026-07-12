# NCCL-SAI Public Guide

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL for SAI
UltraPOD and SlimPOD GPU fabric families. It is designed as a drop-in runtime:
applications continue to use the NCCL API without source changes.

The first optimized API is `ncclAlltoAll()`. Eligible small-message calls use
a two-stage island planner that aggregates network traffic across regular GPU
islands before a local redistribution. Eligible large out-of-place calls use a
phased P2P planner that limits the peer rounds admitted to each kernel plan.
Unsupported sizes and layouts fail closed to upstream AlltoAll scheduling.

NCCL-SAI also has a narrowly guarded local P2P transport-selection path for one
eligible single-host layout. That transport choice can affect operations that
use local P2P, although their collective algorithms are unchanged.

## Activation Model

Public source does not detect private hostnames, partitions, or filesystem
paths. Sites enable NCCL-SAI through generic controls:

- `NCCL_SAI_FABRIC_PROFILE=<family>[-<variant>]` enables profile defaults for
  recognized product families. Current public families are `ultrapod` and
  `slimpod`; matching is case-insensitive.
- `NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh` selects the currently validated
  island and phased AlltoAll defaults. A family name alone does not assert that
  one topology-specific planner is valid for every product in that family.
- Empty, unknown, or disabled-style values such as `0`, `false`, `off`,
  `none`, `native`, and `upstream` fail closed to upstream behavior.
- `NCCL_SAI_A2A_ENABLE=1` explicitly enables the AlltoAll policy.
- `NCCL_SAI_A2A_ENABLE=0` explicitly disables the AlltoAll policy.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1` explicitly enables the guarded local P2P
  path relaxation; `0` disables it.
- Standard `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings take precedence
  over profile-driven local P2P behavior.

Site modulefiles, prologs, SPANK plugins, or container entrypoints may set these
variables privately. When no recognized profile or explicit enable is present,
NCCL-SAI follows upstream NCCL behavior.

## AlltoAll Paths

Size thresholds are per-peer bytes, not the total bytes passed to
`ncclAlltoAll()`.

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_A2A_ENABLE` | `-1` | Follow a recognized profile; `0` disables and positive values enable explicitly. |
| `NCCL_SAI_A2A_PLANNER_ENABLE` | `-1` | Enable the phased planner for the validated full-mesh profile; `0` disables it. |
| `NCCL_SAI_A2A_PLANNER_ROUNDS` | `-1` | Choose a profile-aware phase window automatically; positive values are expert overrides. |
| `NCCL_SAI_A2A_GROUP_NODES` | `-1` | Follow the profile's NCCL topology-node count per aggregation group; the current validated full-mesh class resolves to 16. |
| `NCCL_SAI_A2A_MULTIGROUP_ENABLE` | `-1` | In automatic mode, require complete rank-consistent fabric-group metadata; positive values are expert overrides. |
| `NCCL_SAI_A2A_MIN_PEER_BYTES` | `131072` | Enter the phased planner at 128 KiB per peer. |
| `NCCL_SAI_A2A_MIN_RANKS` | `32` | Require at least 32 ranks. |
| `NCCL_SAI_A2A_ISLAND_ENABLE` | `-1` | Enable the small-message island path for the validated full-mesh profile; `0` disables it. |
| `NCCL_SAI_A2A_ISLAND_SIZE` | `4` | Number of contiguous ranks in one regular GPU island. |
| `NCCL_SAI_A2A_ISLAND_MAX_PEER_BYTES` | `4096` | Maximum per-peer payload admitted to the island path. |
| `NCCL_SAI_A2A_ISLAND_MIN_RANKS` | `576` | Minimum communicator size for automatic island aggregation. |
| `NCCL_SAI_A2A_ISLAND_SCRATCH_CAP_BYTES` | `67108864` | Maximum per-rank scratch reservation for island aggregation. |
| `NCCL_SAI_A2A_PLAN_TRACE` | `0` | Emit rank-zero planner eligibility and phase diagnostics. |

### Small-Message Island Path

The island path treats a regular group of contiguous ranks as one external
fabric endpoint. In stage one, ranks with the same island-local index exchange
one contiguous destination-island block. In stage two, ranks redistribute the
staged data within each island. This reduces network peer fanout by the island
size and increases the network message size by the same factor without changing
the public NCCL API.

Automatic eligibility requires a rank-consistent configuration, uniform local
rank counts, complete islands, a global rank order that is contiguous by
island, rank-consistent scratch allocation, and complete fabric-group metadata.
The path supports distinct buffers and exactly aliased in-place buffers.
Unsupported layouts, allocation failure, invalid sizes, missing group metadata,
and expert limits fall back to upstream scheduling.

The default 576-rank scale guard is deliberate. Sustained same-allocation A/B
showed that upstream scheduling remains faster at a 64-rank complete group,
while the island path first demonstrated a material gain at 576 ranks. Smaller
communicators therefore remain upstream by default; experts can lower
`NCCL_SAI_A2A_ISLAND_MIN_RANKS` for controlled validation.

The default scratch reservation is the smaller of
`ISLAND_MAX_PEER_BYTES * nranks` and `ISLAND_SCRATCH_CAP_BYTES` per rank. It is
allocated during communicator initialization only when the communicator meets
the configured rank threshold, so CUDA Graph capture and nonblocking operation
do not allocate memory from the AlltoAll call path and lower-scale fallback
communicators reserve no island scratch.

### Large-Message Phased Path

The phased path requires distinct send and receive buffers. Exactly aliased
buffers and calls below the phased per-peer threshold use another eligible path
or upstream scheduling. The planner admits bounded peer-round windows to each
kernel plan while preserving the ordinary P2P ordering established before the
AlltoAll call.

Schedule-affecting configuration is compared during communicator
initialization. A mismatch fails initialization before ranks can select
different communication schedules.

The transparent profile does not change communicator-wide
`NCCL_MIN_NCHANNELS` or its legacy alias. Standard upstream channel controls
remain available as explicit expert tuning, but AlltoAll defaults must not
silently alter unrelated collectives or ordinary P2P traffic.

## Fabric Metadata

Automatic grouped planning requires every topology node to resolve a fabric
group. An explicit numeric `NCCL_SAI_FABRIC_GROUP_ID` has highest priority. For
the full-mesh profile, NCCL-SAI can otherwise parse standard
`SLURM_TOPOLOGY_ADDR` and `SLURM_TOPOLOGY_ADDR_PATTERN` rank-local metadata and
derive an ID from the path through the deepest `switch` component.

The Slurm parser requires matching nonempty components, only `switch` components
followed by one final `node`, and at least one switch. Values are gathered during
communicator initialization and are accepted only when every rank provides
valid metadata and each group has the configured number of topology nodes. An
invalid explicit ID is never replaced by an automatic source. Missing,
malformed, conflicting, or incomplete metadata leaves automatic grouped
AlltoAll on the upstream path.

The total topology-node count is not evidence that an allocation contains one
complete group: the same count can be assembled from several fragmented groups.
Automatic mode therefore requires communicator-wide completeness metadata even
when the total equals the configured group size. A positive
`NCCL_SAI_A2A_MULTIGROUP_ENABLE` value can bypass this check for controlled
expert validation; the operator then owns layout correctness and performance.
The current transparent paths do not implement occupancy-weighted scheduling
for fragmented groups.

Metadata identifies logical fabric groups, not private device names. Sites
without Slurm topology metadata may inject the explicit numeric ID from another
trusted rank-local runtime mechanism.

## P2P Controls

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_LOCAL_P2P_SYS_ENABLE` | `-1` | Follow a recognized profile; `0` disables and positive values enable explicitly. |
| `NCCL_SAI_LOCAL_P2P_SYS_TRACE` | `0` | Emit local path-relaxation diagnostics. |
| `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` | `-1` | Follow the validated full-mesh profile; `0` disables the expert schedule. Automatic and explicit modes still require complete multi-group metadata. |

The local P2P relaxation is intentionally narrow. It applies only to a
single-host eight-rank communicator with two four-rank local GPU islands, and
only to cross-island paths that remain within the same host and shared-memory
domain. It is not a general multi-node P2P override.

For the validated 16-endpoint full-mesh class, NCCL 2.28 exposes each regular
four-rank GPU island as one topology node. One complete group therefore contains
16 NCCL topology nodes. Planner phase sizing and metadata completeness use this
NCCL-visible topology count; physical host count must not be substituted for it.

`NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` changes the peer order used by P2P-based
operations. It uses `NCCL_SAI_A2A_GROUP_NODES` as the single group-size source,
never activates for a single fabric group, and fails closed when complete group
metadata is unavailable. Treat explicit settings as expert controls and
validate them for the target fabric before site-wide deployment.

## Public Validation Standard

Before publishing a source branch or binary package, validate at least:

- reproducible build from a clean source commit;
- advertised CUDA SASS and PTX architecture coverage;
- portable host ISA settings;
- small-message island eligibility/fallback and large-message phased AlltoAll;
- same-domain, cross-domain, and multi-domain sustained runs;
- CUDA Graph, nonblocking communicator, repeated-call, and mixed P2P ordering;
- allreduce, reduce-scatter, allgather, broadcast, and common P2P
  non-regression;
- fail-closed behavior for missing, invalid, or inconsistent profile metadata.

See `docs/sai/BUILD_AND_PACKAGING.md` for build and packaging requirements.
Public performance summaries should contain only sanitized topology classes,
scale classes, versions, and message sizes. Keep private job IDs, hostnames,
switch labels, paths, and raw operational logs outside the public repository.

For a full-mesh topology class with 16 equal-bandwidth endpoints per group and
one equal-bandwidth direct edge per group pair, 17 complete groups are the
injection/edge balance point. Headline qualification for that layout must use
16-18 complete, uniformly occupied groups. Smaller layouts are correctness or
topology-normalized efficiency evidence, not the ideal full-fabric result.

The published RC1 has a sustained same-domain 64-GPU result for a
1 GiB per-rank `alltoall_perf` case: the eligible out-of-place path measured
6.16 GB/s bus bandwidth versus 4.80 GB/s through the upstream scheduler in the
same NCCL-SAI build. In that RC1, the exactly aliased case remained on the
upstream path, so the combined benchmark average was 5.46 GB/s. This historical
result does not qualify the newer island path; it is a scoped AlltoAll result,
not a universal application or collective speedup claim.

## Offline Checks

`tools/sai/verify_a2a_algorithms.py` verifies P2P schedule coverage, phased
planner boundaries, island scratch sizing and two-stage data mapping,
fabric-edge balance, queue ordering models, and metadata ordering for
power-of-two and non-power-of-two scale classes:

```bash
python3 tools/sai/verify_a2a_algorithms.py --full-scale
```

CI also compiles and runs `tools/sai/test_profile_activation.cc` to keep
profile recognition and metadata parsing fail-closed.

## License And Notice

NCCL-SAI is derived from NVIDIA NCCL. Keep upstream copyright and license
notices intact, include `LICENSE.txt` with source or binary redistribution, and
include `docs/sai/NOTICE.md` or equivalent release-note text. Do not imply
endorsement by NVIDIA or other upstream contributors.
