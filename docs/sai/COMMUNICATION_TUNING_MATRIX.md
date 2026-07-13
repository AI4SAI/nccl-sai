# NCCL-SAI Communication Tuning Matrix

This matrix defines the public validation shape for NCCL-SAI. It is not a
dump of private benchmark evidence. Use it to decide which communication
families must pass before a branch becomes a default runtime.

## Topology Classes

Use portable topology classes in public reports:

| Class | Meaning | Purpose |
| --- | --- | --- |
| Local island | Ranks sharing the tightest GPU locality domain | Small-message and local-copy behavior |
| Single node | All ranks are on one host | P2P, shared memory, and PCIe/NVLink interaction |
| Same aggregation domain | Multiple hosts under one local fabric domain | Fast-path smoke and sustained same-domain bandwidth |
| Cross aggregation domain | Hosts span two fabric domains | Fallback and rail correctness |
| Multi-domain | Hosts span several domains | Scaling, rank-order sensitivity, and congestion risk |
| Full-system class | The intended production scale class | Release qualification, not early smoke testing |

Do not publish private domain labels, node lists, switch names, job IDs, or
private artifact paths.

## Expert Full-Mesh Research Boundary

For a direct full-mesh fabric with `E` equal-bandwidth endpoints per group and
one endpoint-bandwidth edge per group pair, the injection/edge balance point is
`E + 1` complete groups. A 16-endpoint group is therefore balanced at 17
groups, with 16 and 18 groups forming the practical qualification bracket.

Runs below that bracket can still prove correctness and report efficiency
against their layout-specific pair-edge bound. They must not be presented as
the ideal full-fabric bandwidth result. Fragmented allocations and concurrent
traffic are separate isolation gates rather than substitutes for a balanced
complete-group run.

Any expert complete-group algorithm must fail closed when occupancy metadata
is missing or any group is partial, even if the total endpoint count equals an
integer number of groups. Fragmented-group performance requires a separate
occupancy-aware schedule and must not be inferred from group-ID discovery alone.

## Collective Gates

| Collective/API | Why It Matters | Required Public Gate |
| --- | --- | --- |
| `ncclAlltoAll()` | Expert-only implementation in this release; important for exchange-heavy research but not an automatic runtime capability. | When an AlltoAll claim is made, test small and large messages across local, same-domain, cross-domain, and multi-domain classes. |
| `ncclAllReduce()` | Dominant data-parallel and solver collective; ring order can expose the slowest fabric segment. | Latency and bandwidth samples across same-domain and multi-domain classes, plus fragmented or concurrent-layout stress when available. |
| `ncclReduceScatter()` / `ncclAllGather()` | Common in modern distributed training and optimizer pipelines. | Correctness and bandwidth non-regression against upstream behavior. |
| `ncclBroadcast()` / `ncclReduce()` | Common control and solver phases. | Correctness and latency non-regression. |
| P2P send/recv | Used directly by some runtimes and internally by collective implementations. | Correctness, fallback, and connection-resource stability at scale. |
| MPI collectives with GPU buffers | Many scientific applications use MPI rather than NCCL APIs directly. | Verify the site MPI stack separately; do not infer MPI behavior from NCCL-only tests. |

## Transparent Activation Gates

Qualification of the default runtime must additionally prove:

- the normal run has an empty `NCCL_SAI_*` set;
- a supported physical topology activates rail and local-P2P capability without
  Slurm identity, partition, hostname, or HCA-string matching;
- the runtime implementation does not read `SLURM_*` variables for SAI policy;
- normal zero-variable grouped AlltoAll/P2P remains upstream until a stable
  hardware-backed fabric-group provider exists;
- fabricated scheduler metadata cannot activate any SAI path;
- `NCCL_IB_MERGE_NICS=1` remains runnable through upstream behavior, while
  rail activation is decided from the resulting NET topology;
- actual transport is established from `NCCL INFO Channel ... via ...` lines,
  not from a GDR capability message.

## Message-Size Gates

At each relevant topology class, include:

- KB-scale small messages for latency and dispatch overhead;
- MB-scale mid-sized messages for algorithm transition behavior;
- GB-scale large messages for bandwidth and rail utilization;
- sustained runs long enough for monitoring and fabric counters to become
  meaningful.

Topology-specific small-message aggregation must also have a measured scale
guard. Below the first validated beneficial scale, transparent mode should keep
the upstream path instead of assuming that reduced network fanout always wins.

Short smoke tests are useful for correctness, but they are not sufficient for
release performance claims.

## Public Reporting

Public reports should include:

- NCCL-SAI commit or tag;
- upstream NCCL base version;
- CUDA version and compiled GPU architecture list;
- benchmark name and message-size range;
- sanitized topology class and rank count;
- confirmation that the normal qualification set no `NCCL_SAI_*`, plus any
  upstream NCCL controls and expert overrides used by separate comparison runs;
- explicit rollback knobs.

Public reports should not include:

- private hostnames, partitions, scheduler job IDs, switch labels, raw logs, or
  file paths;
- credentials, private links, GUID maps, or field-operation worklists;
- performance claims that depend on unsanitized production diagnostics.

## Release Decision

AlltoAll optimization is not a default-release gate while stable hardware
fabric-group identity is unavailable. A publishable runtime must prove the
automatic rail and local-P2P paths, upstream fallback, and correctness of common
collectives. Any expert AlltoAll performance claim is a separate qualification
and must not be presented as transparent default behavior.

`NCCL_IB_MERGE_NICS=0` versus `1` performance ordering may be reported only
from a sustained same-candidate, same-layout A/B. A correctness pass in either
mode is not a performance comparison.
