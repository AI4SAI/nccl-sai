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

## Collective Gates

| Collective/API | Why It Matters | Required Public Gate |
| --- | --- | --- |
| `ncclAlltoAll()` | Primary NCCL-SAI optimized path; important for exchange-heavy workloads. | Small and large messages across local, same-domain, cross-domain, and multi-domain classes. |
| `ncclAllReduce()` | Dominant data-parallel and solver collective; ring order can expose the slowest fabric segment. | Latency and bandwidth samples across same-domain and multi-domain classes, plus fragmented or concurrent-layout stress when available. |
| `ncclReduceScatter()` / `ncclAllGather()` | Common in modern distributed training and optimizer pipelines. | Correctness and bandwidth non-regression against upstream behavior. |
| `ncclBroadcast()` / `ncclReduce()` | Common control and solver phases. | Correctness and latency non-regression. |
| P2P send/recv | Used directly by some runtimes and internally by collective implementations. | Correctness, fallback, and connection-resource stability at scale. |
| MPI collectives with GPU buffers | Many scientific applications use MPI rather than NCCL APIs directly. | Verify the site MPI stack separately; do not infer MPI behavior from NCCL-only tests. |

## Message-Size Gates

At each relevant topology class, include:

- KB-scale small messages for latency and dispatch overhead;
- MB-scale mid-sized messages for algorithm transition behavior;
- GB-scale large messages for bandwidth and rail utilization;
- sustained runs long enough for monitoring and fabric counters to become
  meaningful.

Short smoke tests are useful for correctness, but they are not sufficient for
release performance claims.

## Public Reporting

Public reports should include:

- NCCL-SAI commit or tag;
- upstream NCCL base version;
- CUDA version and compiled GPU architecture list;
- benchmark name and message-size range;
- sanitized topology class and rank count;
- environment variables that affect NCCL-SAI behavior;
- explicit rollback knobs.

Public reports should not include:

- private hostnames, partitions, scheduler job IDs, switch labels, raw logs, or
  file paths;
- credentials, private links, GUID maps, or field-operation worklists;
- performance claims that depend on unsanitized production diagnostics.

## Release Decision

Treat alltoall optimization as necessary but not sufficient. A publishable
runtime must also show that common non-alltoall collectives remain correct and
that transparent defaults do not surprise applications that never call
`ncclAlltoAll()`.
