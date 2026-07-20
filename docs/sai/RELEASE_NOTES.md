# NCCL-SAI v2.29.3-1-sai.1 Release Notes

This release derives from NVIDIA NCCL `v2.29.3-1`. It adds communicator-wide,
channel-aligned endpoint selection for a topology class with two independent
network rails exposed as the two physical ports of each local adapter.

## Behavior

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

## Validation Boundary

The release source tree was qualified on a dual-independent-rail, SM70 GPU
fabric at 32- and 128-GPU communicator sizes. The exact release-equivalent tree
completed real-IB/GDR AllReduce and grouped Send/Recv AlltoAll correctness
checks without Socket fallback. Public release artifacts do not contain raw
cluster logs, scheduler identifiers, hostnames, or private topology mappings.

This qualification does not claim universal performance improvement, external
network-plugin support, or application-level speedup. Other GPU architectures
are build-covered but require topology-specific runtime qualification before a
site makes a production-default claim.

## Provenance

NCCL-SAI modifications are maintained by AI4SAI contributors. This project is
not endorsed by NVIDIA. Original NVIDIA copyright and license notices remain
in `LICENSE.txt`; see `docs/sai/NOTICE.md`.
