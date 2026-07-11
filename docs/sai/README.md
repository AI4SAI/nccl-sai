# NCCL-SAI Public Guide

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL for SAI
UltraPOD and SlimPOD GPU fabric families. It is designed as a drop-in runtime:
applications continue to use the NCCL API without source changes.

The first optimized API is `ncclAlltoAll()`. Eligible large out-of-place calls
use a phased P2P planner that limits the peer rounds admitted to each kernel
plan. Small messages, aliased buffers, incomplete topology metadata, and
unsupported layouts use upstream AlltoAll scheduling.

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
  phased AlltoAll defaults. A family name alone does not assert that one
  topology-specific planner is valid for every product in that family.
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

## AlltoAll Planner

Size thresholds are per-peer bytes, not the total bytes passed to
`ncclAlltoAll()`.

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_A2A_ENABLE` | `-1` | Follow a recognized profile; `0` disables and positive values enable explicitly. |
| `NCCL_SAI_A2A_PLANNER_ENABLE` | `-1` | Enable the phased planner for the validated full-mesh profile; `0` disables it. |
| `NCCL_SAI_A2A_PLANNER_ROUNDS` | `-1` | Choose a profile-aware phase window automatically; positive values are expert overrides. |
| `NCCL_SAI_A2A_GROUP_NODES` | `-1` | Follow the profile's NCCL topology-node count per aggregation group. |
| `NCCL_SAI_A2A_MULTIGROUP_ENABLE` | `-1` | In automatic mode, require complete rank-consistent fabric-group metadata. |
| `NCCL_SAI_A2A_MIN_PEER_BYTES` | `131072` | Enter the phased planner at 128 KiB per peer. |
| `NCCL_SAI_A2A_MIN_RANKS` | `32` | Require at least 32 ranks. |
| `NCCL_SAI_A2A_PLAN_TRACE` | `0` | Emit rank-zero planner eligibility and phase diagnostics. |

The phased path requires distinct send and receive buffers. Exactly aliased
buffers use upstream scheduling; this release does not claim an in-place or
partially overlapping AlltoAll optimization. Calls below the per-peer threshold
also use upstream scheduling because the release candidate does not claim a
validated small-message speedup.

Schedule-affecting configuration is compared during communicator
initialization. A mismatch fails initialization before ranks can select
different communication schedules.

The validated full-mesh profile also raises the communicator's implicit
minimum channel count to at most eight channels when the user has not set
`NCCL_MIN_NCHANNELS` or its legacy alias. This preserves the measured mixed-size
profile, but it is communicator-wide rather than AlltoAll-only. Explicit
upstream channel settings take precedence, and release validation therefore
includes non-AlltoAll collectives.

## Fabric Metadata

Multi-group automatic planning requires every topology node to provide a
numeric `NCCL_SAI_FABRIC_GROUP_ID`. The values are gathered during communicator
initialization and are accepted only when every rank provides valid metadata
and each group has the configured number of topology nodes.

Metadata identifies logical fabric groups, not private device names. Sites may
inject it from scheduler topology, a launcher, or another rank-local runtime
mechanism. Incomplete or invalid metadata leaves multi-group AlltoAll on the
upstream path unless an administrator explicitly enables an expert override.

## P2P Controls

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_LOCAL_P2P_SYS_ENABLE` | `-1` | Follow a recognized profile; `0` disables and positive values enable explicitly. |
| `NCCL_SAI_LOCAL_P2P_SYS_TRACE` | `0` | Emit local path-relaxation diagnostics. |
| `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` | `-1` | Follow the validated full-mesh profile; `0` disables the expert schedule. |
| `NCCL_SAI_P2P_FABRIC_NODES` | `-1` | Follow the profile's topology-node count per fabric group. |

The local P2P relaxation is intentionally narrow. It applies only to a
single-host eight-rank communicator with two four-rank local GPU islands, and
only to cross-island paths that remain within the same host and shared-memory
domain. It is not a general multi-node P2P override.

`NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` changes the peer order used by P2P-based
operations. Treat explicit settings as expert controls and validate them for
the target fabric before site-wide deployment.

## Public Validation Standard

Before publishing a source branch or binary package, validate at least:

- reproducible build from a clean source commit;
- advertised CUDA SASS and PTX architecture coverage;
- portable host ISA settings;
- small-message upstream fallback and large-message phased AlltoAll;
- same-domain, cross-domain, and multi-domain sustained runs;
- CUDA Graph, nonblocking communicator, repeated-call, and mixed P2P ordering;
- allreduce, reduce-scatter, allgather, broadcast, and common P2P
  non-regression;
- fail-closed behavior for missing, invalid, or inconsistent profile metadata.

See `docs/sai/BUILD_AND_PACKAGING.md` for build and packaging requirements.
Public performance summaries should contain only sanitized topology classes,
scale classes, versions, and message sizes. Keep private job IDs, hostnames,
switch labels, paths, and raw operational logs outside the public repository.

The current release candidate has a sustained same-domain 64-GPU result for a
1 GiB per-rank `alltoall_perf` case: the eligible out-of-place path measured
6.16 GB/s bus bandwidth versus 4.80 GB/s through the upstream scheduler in the
same NCCL-SAI build. The exactly aliased case remained on the upstream path, so
the combined benchmark average was 5.46 GB/s. This is a scoped AlltoAll result,
not a universal application or collective speedup claim.

## Offline Checks

`tools/sai/verify_a2a_algorithms.py` verifies P2P schedule coverage, phased
planner boundaries, fabric-edge balance, queue ordering models, and metadata
ordering for power-of-two and non-power-of-two scale classes:

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
