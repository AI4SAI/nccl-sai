# NCCL-SAI v2.18.5-1-sai.1 release notes

This release is derived from NVIDIA NCCL `v2.18.5-1` and is maintained by
AI4SAI/SAI contributors. It is not endorsed by NVIDIA. Upstream copyright and
license notices are retained.

## Changes from upstream

- Optional channel-parity selection across exactly two topology-local NET
  endpoints for graphless communication.
- Strict physical-host-based phase scheduling for exact dense grouped
  Send/Recv exchanges in the NCCL 2.18 executor.
- Message-size-aware P2P preconnection using the existing NCCL 2.18 scheduler's
  channel calculation, avoiding connectors that an operation cannot use.
- Asynchronous proxy-listener handshake progress which preserves partial state,
  rejects invalid clients, and continues servicing established clients.

The source contains no scheduler, hostname, partition, or deployment-path
identity test. Dense scheduling, size-aware preconnection, and proxy hardening
are automatic. The rail policy is explicitly enabled by managed runtime
configuration.

## Compatibility boundary

The release preserves the NCCL 2.18 public API and `libnccl.so.2` SONAME. It
does not import a later NCCL executor, native AlltoAll scheduling, GPU-island
planners, local-P2P policy, PXN policy, collective graph rewriting, or channel
count changes. P2P execution uses the upstream message-size calculation for
both scheduling and preconnection. Collective algorithm, protocol, and tuner
selection remain upstream.

## Qualification summary

The release candidate was validated with grouped Send/Recv and collective
tests on SM70 GPUs using CUDA 12.4:

- A 16-rank, 1 GiB dense AlltoAll-equivalent exchange reached about
  `7.34 GB/s` bus bandwidth; a same-run upstream-schedule control reached about
  `6.19 GB/s`.
- A 16-rank, 256 MiB AllReduce comparison measured `24.07 GB/s` for the
  candidate and `24.17 GB/s` for the control. KiB-scale AllReduce latency was
  unchanged within the measurement resolution.
- A 768-rank, 1 GiB dense exchange completed three samples with zero validation
  errors at about `1.27 GB/s`. This release does not claim that the single-host
  gain reaches the historical approximately `2 GB/s` scale target.
- An adjacent 16-rank preconnection comparison measured `6.25 GB/s` for the
  final candidate and `6.01 GB/s` for its immediate control at 1 GiB. A reverse
  adjacent 4 KiB comparison measured `50.87 us` and `52.02 us`, respectively.
- A deterministic 384-rank delayed-first-use proxy stress completed with zero
  validation errors after the fix.
- A 768-rank application qualification completed ten SCF iterations and
  reached normal timing output with the final source candidate.

Public artifacts contain sanitized summaries only. Private application inputs,
scheduler records, hostnames, raw logs, and deployment paths are not release
assets.

## Release boundary

A successful package and benchmark result is not a universal default
recommendation. Each deployment must validate its endpoint ordering, topology,
application behavior, monitoring policy, and rollback path. The runtime should
remain versioned so operators can restore the previous library without changing
applications.
