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

## Communicator-agreed dual-rail selection

A site may set `NCCL_SAI_RAIL_BY_CHANNEL=1` as managed runtime policy. Each
networked rank must then expose exactly two topology-local physical NET
endpoints on one nonzero ASIC, with topology port numbers 1 and 2. Port order,
not provider enumeration or a device-name string, defines channel parity:
even channels use port 1 and odd channels use port 2.

Before graph search or transport connection, every rank exchanges one fixed
rail record. Active policy requires all of the following:

- every rank requests the policy;
- every rank in a networked communicator uses NCCL's internal IB transport and
  sets `NCCL_CROSS_NIC=0`;
- the port-1 and port-2 endpoints have distinct flat NCCL 2.18 IB device
  indices and nonzero, distinct GID subnet prefixes; and
- every rank reports the same ordered port-1/port-2 subnet pair.

The agreed mapping applies to ring/tree graph endpoints and graphless local NET
selection. CollNet and NVLS remain outside the override. A communicator whose
trimmed topology is networkless on every rank accepts the policy as a no-op;
this preserves NCCL 2.18 single-host and node-local split behavior. Mixing
networkless and networked ranks fails communicator initialization. A fully
networkless communicator does not inspect transport, CROSS_NIC, endpoint, or
subnet fields. For a networked communicator, using an external NET plugin or
reporting an incomplete or inconsistent rail shape fails before graph or QP
creation.

When the policy is unset consistently, upstream local-NET and graph selection
remain active. The homogeneous sai.2 runtime still performs its internal wire
revision check and one small agreement AllGather. A mixed sai.1/sai.2
communicator is unsupported: a sai.2 rank detects the mismatch, closes its
bootstrap ring, and returns an initialization error. Because sai.1 has no
communicator-wide wire check, this does not guarantee that every old rank exits
cleanly. Deployments must drain active jobs and switch the shared runtime
atomically; the code-level check is not a rolling-upgrade protocol.

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
- collective channel expansion; or
- collective algorithm or protocol overrides.

Upstream NCCL controls retain their normal meaning. Operational rollback uses
the previous versioned runtime; sites using the optional rail policy may unset
it to restore upstream graph and graphless local-NET selection.
