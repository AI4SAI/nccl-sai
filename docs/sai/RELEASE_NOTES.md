# NCCL-SAI v2.18.5-1-sai.2 release notes

This release is derived from NVIDIA NCCL `v2.18.5-1` and is maintained by
AI4SAI/SAI contributors. It is not endorsed by NVIDIA. Upstream copyright and
license notices are retained.

## Changes from sai.1

- Ring/tree graph endpoints and graphless local NET selection now use one
  communicator-agreed, port-ordered channel-to-rail mapping.
- Active rail policy on a networked communicator validates internal IB
  transport, `NCCL_CROSS_NIC=0`, two distinct flat NCCL 2.18 IB endpoints, and
  one consistent pair of distinct nonzero subnet prefixes across every rank
  before graph or QP creation.
- Fully networkless communicators accept active rail policy as a no-op, while a
  mixed networkless/networked communicator fails initialization. The no-op does
  not inspect transport, CROSS_NIC, endpoint, or subnet fields.
- The internal peer bootstrap record uses its former x86-64 tail padding for a
  wire revision without changing that record's 64-byte size. A sai.2 rank that
  detects a mixed runtime closes its bootstrap ring and returns an
  initialization error.

The rail agreement AllGather and internal wire check also run when the policy
is disabled. In that case, sai.2 preserves upstream graph and local-NET
selection. External NET plugins remain usable with the policy disabled and are
rejected by active policy on a networked communicator.

## Inherited sai.1 behavior

- Strict physical-host-based phase scheduling for eligible exact dense grouped
  Send/Recv exchanges in the NCCL 2.18 executor.
- Message-size-aware P2P preconnection using the existing NCCL 2.18 scheduler's
  channel calculation.
- Asynchronous proxy-listener handshake progress which preserves partial state,
  rejects invalid clients, and continues servicing established clients.

These mechanisms remain automatic and require no application-managed
`NCCL_SAI_*` bundle. The optional rail policy is supplied by managed runtime
configuration. The source contains no scheduler, hostname, partition,
device-name, or deployment-path identity test.

## Compatibility boundary

The release preserves the NCCL 2.18 public API, version code `21805`, and
`libnccl.so.2` SONAME. The internal initialization wire is intentionally not
compatible with sai.1, so a deployment must use one homogeneous runtime per
communicator and switch shared stacks only after draining active jobs. Because
sai.1 has no communicator-wide wire revision check, sai.2 local detection is
not a global abort protocol and does not make rolling mixed-runtime upgrades
safe.

This release does not import a later NCCL executor, native AlltoAll scheduling,
GPU-island planners, local-P2P policy, PXN policy, channel-count changes, or
collective algorithm/protocol overrides. CollNet and NVLS graph endpoints are
not rewritten.

## Qualification boundary

Release promotion is tied to the exact final library checksum and requires:

- focused rail-contract checks covering graph and graphless channel parity,
  ordered subnet agreement, the fully networkless no-op, representative
  policy/transport/CROSS_NIC/topology/subnet fail-closed cases, and only the
  documented sai.2-side detection boundary for a mixed runtime;
- an adjacent 16-rank comparison covering KiB-scale latency and sustained
  AllReduce and dense AlltoAll-equivalent bandwidth;
- a fixed 24-node application run which completes its required SCF iterations;
- one greater-than-24-node application run; and
- one topology-equivalent large-scale dense AlltoAll check with zero validation
  errors.

Results from sai.1 are historical baselines, not proof for the sai.2 binary.
Public artifacts contain sanitized summaries only. Private application inputs,
scheduler records, hostnames, raw logs, and deployment paths are not release
assets.

## Release boundary

A successful package and benchmark result is not a universal default
recommendation. Each deployment must validate its endpoint ordering, topology,
application behavior, monitoring policy, homogeneous-runtime switch, and
rollback path. The runtime should remain versioned so operators can restore the
previous library without changing applications.
