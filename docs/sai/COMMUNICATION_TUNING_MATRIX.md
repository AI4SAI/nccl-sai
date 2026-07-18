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
| Graphless NET rail policy | Enabled and disabled behavior, exact two-endpoint activation, endpoint parity, and unsupported-shape fallback |
| Proxy listener | Partial-magic progress, invalid-client rejection, established-client progress, delayed first use, and scale |
| Unsupported topology | Upstream fallback when an automatic predicate does not match |

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

Upstream channel, algorithm, protocol, P2P, cross-NIC, and NIC-merge controls
retain their upstream meaning. Performance claims require the same payload,
rank layout, test binary, runtime policy, and topology class.

## Reporting boundary

Publish only sanitized topology classes, rank counts, message sizes, benchmark
names, checksums, and scope-matched results. Application stability and site
default promotion are deployment gates and are not inferred from a
microbenchmark alone.
