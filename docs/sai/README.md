# NCCL-SAI 2.18.5 public guide

NCCL-SAI 2.18.5 is derived from NVIDIA NCCL `v2.18.5-1`. It preserves the
upstream public API and SONAME while adding a narrow dual-rail policy, exact
dense peer scheduling, size-aware P2P preconnection, and defensive
proxy-listener progress.

Applications do not need an `NCCL_SAI_*` variable bundle. A site may enable the
rail policy through managed runtime configuration. The source does not identify
a deployment through scheduler metadata, hostnames, partitions, device-name
strings, or filesystem paths.

## Dense grouped Send/Recv scheduling

An exact dense grouped Send/Recv exchange may use host-local phases in the
upstream NCCL 2.18 P2P executor. Eligibility requires all of the following:

- at least 16 communicator ranks;
- a uniform physical-rank count on every host, with at least eight ranks per
  host and a count divisible by eight;
- exactly one send and one receive task for every peer;
- equal send and receive sizes for every peer and one common size across the
  communicator;
- at least 128 KiB per peer; and
- physical-host and host-local-rank identity derived from NCCL peer identity.

Eligible peers are admitted in eight host-local phases. Sparse, asymmetric,
mixed-size, small, partially queued, or otherwise inexact traffic keeps the
upstream schedule. This implementation does not import a later NCCL work-batch
executor or an experimental AlltoAll planner.

## Size-aware P2P preconnection

NCCL 2.18 schedules each P2P operation over a message-dependent number of
channel offsets. NCCL-SAI applies the same calculation when marking connectors
for preconnection. Small messages therefore do not open channel offsets that
the existing scheduler will not use, while large messages retain the upstream
multi-channel selection.

This behavior is automatic. It does not lower the communicator channel count,
change channel-to-rail mapping, replace the P2P executor, or add a tuning
variable. If a later operation for the same peer needs more channel offsets,
the additional connectors are marked at that time.

## Graphless dual-rail-by-channel selection

A site may set `NCCL_SAI_RAIL_BY_CHANNEL=1` as managed runtime policy. When a
GPU has exactly two topology-local NET endpoints, graphless local NET selection
uses channel parity to alternate between those endpoints. Endpoint ordering is
therefore part of the site policy and must represent the intended two rails.

When the policy is unset, or when the local shape does not expose exactly two
NET endpoints, the upstream local-NET selection remains active. This change
does not rewrite ring/tree graph endpoints, CollNet or NVLS selection, NIC
merge policy, PXN policy, or collective channel counts.

## Proxy listener progress

Proxy listeners accept connections asynchronously. Partial handshake bytes are
retained across event-loop iterations; invalid magic is closed and reset; and
an incomplete new client does not block requests from already connected proxy
clients. The behavior is automatic and adds no tuning variable.

This is protocol hardening, not permission to expose NCCL listeners to
untrusted networks. Deployments should still restrict monitoring and service
discovery to explicit endpoints.

## Deliberate non-features

This release does not add or automatically enable:

- a native AlltoAll API or native-API-specific algorithm;
- a later NCCL work-batch executor;
- topology-group or GPU-island planners;
- local-P2P or PXN policy changes;
- P2P channel-count or executor changes;
- collective graph rail rewriting;
- collective channel expansion; or
- collective algorithm or protocol overrides.

Upstream NCCL controls retain their normal meaning. Operational rollback uses
the previous versioned runtime; sites using the optional rail policy may also
unset it to restore upstream graphless local-NET selection.
