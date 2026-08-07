# NCCL-SAI v2.29.3-1-sai.2 Release Notes

This release derives from NVIDIA NCCL `v2.29.3-1`. It carries forward the
communicator-wide, channel-aligned dual-rail contract from sai.1 and hardens
incoming proxy and RAS control connections against incomplete or unrelated
socket handshakes.

## Changes From sai.1

- Accepted sockets now expose a distinct bad-magic state. Callers that manage
  their own event loop can reject that connection instead of immediately
  retrying another accept operation on the same socket object.
- Proxy listening sockets use asynchronous accept. An incomplete handshake is
  retained as per-peer state and progressed only when its descriptor becomes
  readable, so it cannot prevent established proxy clients from making
  progress.
- Proxy shutdown stops admitting new peers and closes pending handshakes through
  the existing service-loop cleanup path.
- RAS listening sockets use the same explicit rejection behavior and remove
  invalid descriptors from their poll set.

The bad-magic and RAS event-loop handling follows upstream NCCL fixes
[`8fdd98c`](https://github.com/NVIDIA/nccl/commit/8fdd98c9b7157a3c64c1be4b92e68eb973f3d8c4)
and
[`7310803`](https://github.com/NVIDIA/nccl/commit/7310803b6670d4136cbdbc9e0a0ca1396de7a8d4).
The proxy-listener wiring is adapted to make the same nonblocking accept
behavior active in the NCCL 2.29.3 service loop.

These changes add no user setting and do not identify a deployment from
hostnames, scheduler metadata, network addresses, or filesystem paths.

## Inherited Rail Contract

- Operator-managed `NCCL_SAI_RAIL_BY_CHANNEL=1` requests the policy.
- Every rank must use NCCL's internal IB transport and effective
  `NCCL_CROSS_NIC=0`.
- Every participating GPU must expose exactly one topology-local dual-port NET
  pair whose physical ports are identified as 1 and 2. Each endpoint must map
  to one physical device; merged virtual NICs are rejected.
- Port 1 and port 2 must expose distinct GID subnet prefixes, and every rank
  must report the same ordered pair. This is the communicator-wide logical
  rail identity check.
- Port 1 serves even channels and port 2 serves odd channels for collective
  ring/tree graph endpoints and graphless P2P traffic. NVLS and CollNet retain
  upstream endpoint selection.
- The selected graph endpoint, network device, and proxy rank are recomputed
  together.
- Policy, transport, and topology eligibility are agreed across the entire
  communicator before graph construction. A mismatch is an initialization
  error rather than a rank-local fallback.
- An initialization wire identifier rejects a communicator that mixes this
  fork with an ABI-compatible upstream NCCL runtime before the SAI-specific
  agreement exchange.
- When the operator policy is unset, ring/tree and graphless P2P endpoint
  selection retain upstream behavior. The wire check and SAI agreement
  exchange still run during communicator initialization.

External network plugins are not supported while rail-by-channel policy is
active. The generic plugin interface does not expose sufficient rail identity
for this implementation, so the communicator fails fast instead of risking
cross-rail endpoint selection.

## Compatibility Boundary

This release preserves the NCCL 2.29.3 public API, version code, and
`libnccl.so.2` SONAME. It does not change collective algorithms, protocols,
channel counts, graph endpoint policy, or the sai.1 initialization wire
contract. The accept-state changes are internal control-path behavior.

## Qualification Boundary

The sai.1 source tree was qualified on a dual-independent-rail, SM70 GPU fabric
at 32- and 128-GPU communicator sizes. Those results establish the inherited
rail baseline; they are not qualification for the sai.2 binary.

Promotion of sai.2 requires checks bound to the exact final source and binary
for valid proxy clients, partial handshakes, rejected bad-magic connections,
RAS accept cleanup, communicator initialization, real network transport, and
collective and grouped Send/Recv correctness. Public release artifacts must not
contain raw cluster logs, scheduler identifiers, hostnames, or private topology
mappings.

This release makes no universal performance-improvement, external-network-
plugin, or application-level speedup claim. Other GPU architectures require
topology-specific runtime qualification before a site makes a production-
default claim.

## Provenance

NCCL-SAI modifications are maintained by AI4SAI contributors. This project is
not endorsed by NVIDIA. Original NVIDIA copyright and license notices remain
in `LICENSE.txt`; see `docs/sai/NOTICE.md`.
