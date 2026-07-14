# NCCL-SAI Public Guide

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL for SAI
UltraPOD and SlimPOD GPU fabric families. Applications continue to use the
NCCL API without source changes. This loading and API compatibility is not a
cross-version performance guarantee or a site-wide default-promotion claim.

The automatic behaviors are a narrowly guarded local P2P transport-selection
path for one eligible single-host layout and a strict dual-rail-by-channel path.
These transport choices do not change collective channel, algorithm, or
protocol selection.

The source also contains expert-only `ncclAlltoAll()` implementations.
Eligible small-message calls can use a two-stage island planner that aggregates
network traffic across regular GPU islands before a local redistribution.
Eligible large out-of-place calls can use a phased P2P planner that limits the
peer rounds admitted to each kernel plan. They are not transparent release
capabilities because NCCL 2.28 cannot derive stable cross-node fabric groups
from hardware. Normal runs keep upstream AlltoAll scheduling.
Normal upstream NCCL 2.28 grouped `ncclSend`/`ncclRecv` scheduling is not
claimed to match NCCL 2.18.x small-message latency.

## Activation Model

Normal users, applications, and shared MPI stacks set no `NCCL_SAI_*`
environment variables. Public source does not match private hostnames,
partitions, scheduler strings, device names, or filesystem paths. There is no
site-identity test. Each automatic optimization evaluates the runtime
GPU/PCIe/NIC graph and operation shape needed for that optimization: SM70 GPUs
arranged as complete, equal-bandwidth four-GPU NVLink cliques, one distinct CPU
locality domain per clique, the supported physical dual-port ASIC shape,
communicator-wide agreement, and the documented communicator and message-size
guards.

Upstream NCCL controls retain upstream semantics. `NCCL_IB_HCA` and
`NCCL_IB_MERGE_NICS` influence the topology that NCCL actually discovers, but
their text values are not activation signals. A successful merged-NIC result has
a different runtime topology and therefore fails the physical dual-port
predicate naturally. Effective `NCCL_CROSS_NIC=0` is a behavior gate for the
automatic rail policy, not an identity test.

The internal IB plugin adds a fail-closed preflight only when
`NCCL_IB_MERGE_NICS`, `NCCL_NET_MERGE_LEVEL`, and `NCCL_NET_FORCE_MERGE` are
all unset, effective `NCCL_CROSS_NIC=0`, and the NET-device policy is automatic.
An exact local shape of eight equal-capability IB endpoints, four normalized
PCI adapter pairs exposing NCCL logical ports `1` and `2`, and two distinct
port-consistent subnets may start a physical-endpoint probe only after every
rank reports the same upstream policy and eligible shape. The complete
GPU/CPU/GDR/path and subnet vote then runs
before graph construction. Any rejection releases the probe topology and
rebuilds every rank with upstream default fusion. External network plugins are
unaffected.

If upstream policy itself naturally leaves eight physical endpoints, the same
full classifier may qualify that resulting topology without the automatic
probe. The preflight is an exposure mechanism, not an activation identity.

Scheduler metadata is not used for site identity or automatic fabric-group
classification. NCCL 2.28 does not expose enough cross-node switch identity to
prove the physical group from hardware, so normal zero-variable runs keep
grouped AlltoAll/P2P schedules on the upstream path.

The single-host eight-rank, two-island P2P relaxation is selected from exactly
two SM70 four-GPU NVLink cliques with equal link bandwidth and distinct CPU
locality domains. It does not depend on NIC or scheduler metadata. Standard
`NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings remain upstream policy gates,
not site-identity signals. An unset level or `NCCL_P2P_LEVEL=NVL` may admit the
strictly hardware-qualified cross-island exception; other explicit levels and
`NCCL_P2P_DISABLE=1` retain precedence. No `NCCL_SAI_*` activation variable is
required.

Compatibility and rollback controls remain available:

- `NCCL_SAI_DISABLE=1` disables all SAI-specific behavior.
- `NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh` is a legacy compatibility and
  expert-validation request for grouped AlltoAll/P2P, not a normal runtime
  requirement or a hardware bypass. Profile values do not enable or suppress
  automatic rail or local-P2P capability detection.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0` is a local-P2P rollback. Leaving it unset
  uses strict automatic hardware detection; a positive value cannot force an
  unsupported topology to pass.
- AlltoAll and grouped-P2P `NCCL_SAI_*_ENABLE` values are expert controls for
  those expert-only paths. They do not enable or suppress automatic rail or
  local-P2P capability detection, and they do not
  bypass structural layout, metadata-consistency, or buffer-safety checks.

### Dual-Rail Local-NET Selection

The automatic dual-rail class contains one to four complete, equal-bandwidth
four-GPU NVLink cliques of SM70 GPUs. Each clique must belong to a distinct CPU
locality domain. The node exposes four capability-symmetric dual-port network
adapters; each adapter must provide two GDR-capable NET endpoints on one ASIC
with physical ports `{1,2}`. The final topology must contain eight unmerged NET
endpoints, every clique must resolve one distinct port pair, all GPU-to-port
pairs must be type/bandwidth symmetric per GPU and no farther than PXB,
all accepted GPU-to-port bandwidths must match, and all ranks must report the
same normalized capability class. A clique may contain both PIX-local and
PXB-local GPUs. The internal IB transport must also expose two
distinct selected GID subnet prefixes: every physical port 1 must resolve the
same prefix across all ranks, every physical port 2 must resolve the other
prefix, and the two prefixes must differ. External network plugins or missing
hardware identity fail closed to upstream selection.

The selection matrix is:

| Behavior | Required conditions |
| --- | --- |
| Topology-local NET selection by channel parity | The communicator-wide automatic hardware predicate passed and the calling GPU has the validated physical port pair. |
| Ordinary ring/tree graph endpoint override | All local-NET conditions above, plus effective `NCCL_CROSS_NIC=0`; CollNet and NVLS graphs retain their existing endpoint selection. |

With those gates satisfied, even channels select physical port 1 and odd
channels select physical port 2. NET enumeration order is ignored. For ordinary
ring and tree graphs, a graph endpoint is replaced by that channel-aligned port
and its network device and proxy rank are recomputed. This keeps both endpoints
of a channel aligned to the same rail, including when the original graph
endpoint is not topology-local to the calling rank.

With `NCCL_IB_MERGE_NICS` unset, the internal IB preflight above may suppress
virtual-device fusion temporarily so the strict final predicate can inspect
all eight physical endpoints. The probe is retained only after an all-rank
full-topology vote; otherwise all ranks rebuild with upstream fusion. Explicit
`NCCL_IB_MERGE_NICS=0` keeps the upstream
no-fusion behavior. Explicit `NCCL_IB_MERGE_NICS=1` is not an error and does not
mean the job cannot run: upstream NCCL creates merged virtual devices, the
strict eight-NET predicate no longer matches, and the upstream merged-NIC path
runs. Explicit `NCCL_NET_MERGE_LEVEL` and `NCCL_NET_FORCE_MERGE` likewise keep
their upstream semantics. With merge controls unset, `NCCL_SAI_DISABLE=1`
restores the upstream default fusion behavior.
No same-candidate sustained unset/explicit-`0`/explicit-`1` performance result
is claimed until that comparison is measured. Nonzero cross-NIC mode, CollNet/NVLS
graphs, or any other ineligible layout likewise retains upstream selection.
An explicit non-`AUTO` `NCCL_NETDEVS_POLICY` also retains upstream selection;
the automatic rail path does not reinterpret an upstream device-count policy.

### Collective Channel Policy

NCCL-SAI does not add collective connection channels or override collective
algorithm and protocol selection. Standard upstream minimum/maximum channel,
algorithm, protocol, and tuner controls retain their normal semantics.

## AlltoAll Paths

Size thresholds are per-peer bytes, not the total bytes passed to
`ncclAlltoAll()`.

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_A2A_ENABLE` | `-1` | Keep the grouped SAI path off in normal runtime; `0` disables and positive values request expert evaluation. |
| `NCCL_SAI_A2A_PLANNER_ENABLE` | `-1` | Enable the phased planner only for an explicitly requested, hardware-validated full-mesh class; `0` disables it. |
| `NCCL_SAI_A2A_PLANNER_ROUNDS` | `-1` | Derive a phase window from the expert topology settings; positive values override it. |
| `NCCL_SAI_A2A_GROUP_NODES` | `-1` | Use the configured expert default of 16 topology nodes per aggregation group. |
| `NCCL_SAI_A2A_MULTIGROUP_ENABLE` | `-1` | Require complete rank-consistent metadata for grouped layouts; positive values request expert multigroup evaluation but cannot override completeness. |
| `NCCL_SAI_A2A_MIN_PEER_BYTES` | `131072` | Enter the phased planner at 128 KiB per peer. |
| `NCCL_SAI_A2A_MIN_RANKS` | `32` | Require at least 32 ranks. |
| `NCCL_SAI_A2A_ISLAND_ENABLE` | `-1` | Enable the small-message island path only within an eligible expert configuration; `0` disables it. |
| `NCCL_SAI_A2A_ISLAND_SIZE` | `4` | Number of contiguous ranks in one regular GPU island. |
| `NCCL_SAI_A2A_ISLAND_MAX_PEER_BYTES` | `4096` | Maximum per-peer payload admitted to the island path. |
| `NCCL_SAI_A2A_ISLAND_MIN_RANKS` | `576` | Minimum communicator size for the expert island path. |
| `NCCL_SAI_A2A_ISLAND_SCRATCH_CAP_BYTES` | `67108864` | Maximum per-rank scratch reservation for island aggregation. |
| `NCCL_SAI_A2A_PLAN_TRACE` | `0` | Emit rank-zero planner eligibility and phase diagnostics. |

### Small-Message Island Path

The island path treats a regular group of contiguous ranks as one external
fabric endpoint. In stage one, ranks with the same island-local index exchange
one contiguous destination-island block. In stage two, ranks redistribute the
staged data within each island. This reduces network peer fanout by the island
size and increases the network message size by the same factor without changing
the public NCCL API.

Expert-path eligibility requires a rank-consistent configuration, uniform local
rank counts, complete islands, a global rank order that is contiguous by
island, rank-consistent scratch allocation, and complete fabric-group metadata.
The path supports distinct buffers and exactly aliased in-place buffers.
Unsupported layouts, allocation failure, invalid sizes, missing group metadata,
and expert limits fall back to upstream scheduling.

The default 576-rank threshold is a conservative expert-path guard, not a
general performance-crossover claim. Smaller communicators remain upstream
unless an expert lowers `NCCL_SAI_A2A_ISLAND_MIN_RANKS` for controlled
validation.

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

The automatic policy does not set communicator-wide `NCCL_MIN_NCHANNELS` or
its legacy alias. Standard upstream channel controls remain available as
explicit expert tuning.

## Fabric Metadata

NCCL 2.28 exposes local GPU/PCI/NIC topology but no stable cross-node switch or
aggregation-group identity. HCA GUIDs identify local adapters; subnet prefixes
identify rails or subnets; LIDs are assigned by the subnet manager. None alone
proves membership in the validated physical group. Slurm topology strings,
hostnames, rank order, and device names are also not hardware evidence.

Therefore normal zero-variable grouped AlltoAll/P2P remains upstream. An
explicit numeric `NCCL_SAI_FABRIC_GROUP_ID` is retained only for controlled
expert tests. Values are accepted as metadata only when every rank provides a
valid ID and all ranks in one topology node agree. The layout is marked complete
only when every group has the configured number of topology nodes. An invalid
explicit ID is never replaced by another source.

The total topology-node count is not evidence that an allocation contains one
complete group: the same count can be assembled from several fragmented groups.
The expert path therefore requires communicator-wide completeness metadata
even when the total equals the configured group size. A positive
`NCCL_SAI_A2A_MULTIGROUP_ENABLE` value can request multigroup evaluation, but
it cannot turn missing or incomplete metadata into a complete grouped layout.
Fragmented groups enter only the separately validated ragged schedule when all
ranks provide consistent metadata and every occupancy bound passes; otherwise
the operation stays on the upstream path.

Metadata identifies logical fabric groups, not private device names. Local rail
and single-host P2P recognition still work without it. An explicit profile and
numeric ID can be used for expert validation of grouped paths.

## P2P Controls

| Variable | Default | Behavior |
| --- | ---: | --- |
| `NCCL_SAI_LOCAL_P2P_SYS_ENABLE` | `-1` | Follow strict automatic two-island GPU topology detection; `0` is a rollback, while positive values still cannot bypass the hardware predicate. |
| `NCCL_SAI_LOCAL_P2P_SYS_TRACE` | `0` | Emit local path-relaxation diagnostics. |
| `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` | `-1` | Keep the grouped schedule off in normal runtime; positive values request expert evaluation only under an enabled, hardware-validated grouped-A2A policy with complete metadata. |

The local P2P relaxation is intentionally narrow. It applies only to a
single-host eight-rank communicator with two four-rank local GPU islands, and
only to cross-island paths that remain within the same host and shared-memory
domain, pass bidirectional NVML read/write capability checks, and agree across
all ranks. It is not a general multi-node P2P override. Upstream P2P controls
remain policy inputs, not activation inputs. `NCCL_P2P_DISABLE=1` disables the
path. `NCCL_P2P_LEVEL=NVL` is treated as a compatible site baseline only after
the complete hardware and communicator-wide predicate passes; explicit
`LOC`, `PIX`, `PXB`, or `PHB` limits are not overridden.
Rank-local P2P policy signatures must match. A mismatch is rejected during
communicator initialization instead of allowing different ranks to construct
different transports; topology or NVML eligibility disagreement only disables
the automatic exception and retains the common upstream policy.

For the validated 16-endpoint full-mesh class, NCCL 2.28 exposes each regular
four-rank GPU island as one topology node. One complete group therefore contains
16 NCCL topology nodes. Planner phase sizing and metadata completeness use this
NCCL-visible topology count; physical host count must not be substituted for it.

`NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE` changes the peer order used by P2P-based
operations. It uses `NCCL_SAI_A2A_GROUP_NODES` as the single group-size source,
is a subpolicy of the enabled hardware-validated grouped-A2A profile, never
activates for a single fabric group, and fails closed when complete group
metadata is unavailable. A standalone positive setting cannot activate it.
Treat explicit settings as expert controls and validate them for the target
fabric before site-wide deployment. This control governs only the global P2P
peer order; it does not activate the separately validated ragged AlltoAll round
map, which remains under the expert AlltoAll planner and multigroup gates.

## Transport Evidence

`GPU Direct RDMA Enabled` reports capability, not the transport ultimately
selected for a connection. Validate actual behavior from `NCCL INFO Channel`
lines. In the single-host eight-rank two-island gate, cross-island local edges
must use `via P2P/CUMEM` (or the corresponding local P2P transport), not
`via NET/IB/.../GDRDMA`. In multi-node runs, NET/GDR on inter-node edges is
expected and must not be mixed into the local cross-island decision.

## Public Validation Standard

Validation must match the behavior changed and the claims being published. The
minimum release gate for the automatic capabilities in this branch is:

- one reproducible CUDA build from the final clean source commit, with the
  advertised SASS/PTX coverage, portable host ISA, package contents, and
  checksums recorded;
- offline activation and fail-closed checks for the changed predicates, with
  normal qualification using an empty `NCCL_SAI_*` environment;
- one adjacent stable/candidate comparison on a supported single-host GPU
  topology, reporting small-message latency and large-message bandwidth;
- one two-host dual-rail initialization and data-path regression; and
- one adjacent stable/candidate multi-host comparison beyond the historical
  scale-failure boundary, plus a short initialization trace confirming
  automatic rail consensus and the upstream-selected collective/P2P channel
  layout.

Additional CUDA Graph, nonblocking, collective, merge-mode, topology, or scale
matrices are required only when the source change or published claim touches
those behaviors. Functional AlltoAll descriptions require correctness and
fail-closed coverage; performance-specific AlltoAll matrices may be omitted
when no AlltoAll performance claim is made.

See `docs/sai/BUILD_AND_PACKAGING.md` for build and packaging requirements.
Public performance summaries should contain only sanitized topology classes,
scale classes, versions, and message sizes. Keep private job IDs, hostnames,
switch labels, paths, and raw operational logs outside the public repository.

## Offline Checks

`tools/sai/verify_a2a_algorithms.py` verifies P2P schedule coverage, phased
planner boundaries, island scratch sizing and two-stage data mapping,
fabric-edge balance, queue ordering models, and metadata ordering for
power-of-two and non-power-of-two scale classes:

```bash
python3 tools/sai/verify_a2a_algorithms.py --full-scale
```

CI also compiles and runs `tools/sai/test_profile_activation.cc` to keep the
zero-SAI-variable contract, compatibility profiles, topology-versus-metadata
separation, physical-port rail ordering, automatic-policy provenance, rollback
controls, and expert metadata parsing fail-closed.

## License And Notice

NCCL-SAI is derived from NVIDIA NCCL. Keep upstream copyright and license
notices intact, include `LICENSE.txt` with source or binary redistribution, and
include `docs/sai/NOTICE.md` or equivalent release-note text. Do not imply
endorsement by NVIDIA or other upstream contributors.
