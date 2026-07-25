# NCCL-SAI 2.18.5 communication qualification matrix

This matrix describes the public validation shape. It intentionally excludes
private cluster names, scheduler records, node lists, and raw benchmark logs.

## Required communication families

| Path | Required checks |
| --- | --- |
| Dense grouped Send/Recv | Small-message fallback, large-message activation, correctness, single-host performance, and representative multi-host scale |
| P2P preconnection | Small-message connector cardinality, later larger-message expansion, correctness, and adjacent latency/bandwidth non-regression |
| Sparse or asymmetric Send/Recv | Correctness and proof that dense phases remain inactive |
| AllReduce | Small-message latency and large-message bandwidth non-regression |
| Graph and graphless NET rail policy | Enabled and disabled behavior, ring/tree endpoint parity, graphless endpoint parity, and networkless no-op |
| Communicator rail agreement | Policy mismatch, external plugin, CROSS_NIC mismatch, port-shape mismatch, subnet mismatch, and mixed networkless/networked fail-closed behavior |
| Initialization wire | Homogeneous runtime success, sai.2-side mixed-runtime detection, and an explicit prohibition on inferring communicator-wide graceful failure |
| Proxy listener | Partial-magic progress, invalid-client rejection, established-client progress, delayed first use, and scale |
| Unsupported topology | Upstream fallback for automatic non-rail predicates; explicit initialization failure when active rail policy is ambiguous |

## Message sizes

Use at least one KiB-scale latency point and one sustained large-message point.
Dense-exchange qualification must include a message below the 128 KiB per-peer
threshold and a message above it. Correctness checks must report zero invalid
values for every measured row.

## Runtime policy boundary

Applications and shared MPI stacks do not set an `NCCL_SAI_*` variable bundle.
A site integration may manage the optional rail-by-channel policy. Qualification
must prove that source and runtime behavior do not depend on scheduler
variables, hostnames, partitions, device-name strings, or deployment paths.

Active rail-policy qualification must prove one canonical port-1/port-2 subnet
pair across all ranks before connection creation. A fully networkless
communicator must preserve upstream behavior as an agreed no-op. Policy-disabled
qualification must prove that graph and graphless NET selection remain
upstream, while the internal wire check and agreement AllGather still complete.

Upstream channel, algorithm, protocol, and P2P controls retain their upstream
meaning. Active policy on a networked communicator requires CROSS_NIC=0.
Performance claims require the same payload, rank layout, test binary, runtime
policy, and topology class.

## Reporting boundary

Publish only sanitized topology classes, rank counts, message sizes, benchmark
names, checksums, and scope-matched results. Application stability and site
default promotion are deployment gates and are not inferred from a
microbenchmark alone.
