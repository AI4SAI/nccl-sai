# NCCL-SAI Public Release Notes Draft

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL. The current
source candidate is based on NCCL `v2.28.9-1` and targets SAI UltraPOD and
SlimPOD GPU fabric families.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is not endorsed by NVIDIA. Original NVIDIA NCCL copyright and license notices
are retained in `LICENSE.txt`; see `docs/sai/NOTICE.md`.

## Optimized API

- `ncclAlltoAll()` on eligible SAI fabric profiles.

Eligible large out-of-place AlltoAll calls use the phased planner. Small
messages and exactly aliased buffers use upstream scheduling. This release does
not claim an in-place, partially overlapping, or universal small-message
AlltoAll speedup. Release performance claims use out-of-place buffers.

NCCL-SAI also includes a narrowly guarded local P2P transport-selection path
for an eligible single-host 8-rank, two-island communicator. That path can
affect any operation using local P2P transport, while the allreduce, allgather,
reduce-scatter, broadcast, and reduce collective algorithms remain unchanged.

## Transparent Runtime Model

SAI sites can enable default NCCL-SAI behavior by setting:

```bash
export NCCL_SAI_FABRIC_PROFILE=ultrapod-fullmesh
```

The recognized public families are `ultrapod` and `slimpod`; a nonempty
`-<variant>` suffix is accepted for site packaging. The current phased planner
defaults are specific to the `ultrapod-fullmesh` profile; recognizing a broader
family name does not enable that topology-specific planner. Empty, unknown, or
disabled-style values such as `0`, `false`, `off`, `none`, `native`, and
`upstream` fail closed to upstream behavior.

Explicit overrides:

- `NCCL_SAI_A2A_ENABLE=1`: enable the alltoall SAI path.
- `NCCL_SAI_A2A_ENABLE=0`: disable the alltoall SAI path.
- `NCCL_SAI_FABRIC_GROUP_ID=<N>`: provide rank-local numeric metadata used to
  validate complete multi-group layouts.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1`: enable the selected local P2P path guard.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0`: disable that local path guard.
- Standard `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL` settings take precedence
  over the profile-driven local path relaxation.
- `NCCL_SAI_P2P_FABRIC_GROUP_SCHEDULE=1`: enable an advanced P2P schedule for
  locally grouped fabric domains. It is disabled by default and should not be
  enabled in public packages without site validation.
- `NCCL_SAI_P2P_FABRIC_NODES=<N>`: local fabric-domain node count for that
  advanced schedule, expressed as NCCL topology nodes rather than scheduler
  host count.

When the validated full-mesh profile is active and the user has not explicitly
set `NCCL_MIN_NCHANNELS` or its legacy alias, NCCL-SAI applies an implicit
channel floor of at most eight channels. This setting is communicator-wide and
is included in cross-collective release testing; explicit upstream channel
settings take precedence.

## Preliminary Performance Boundary

In a sustained same-domain 64-GPU `alltoall_perf` run using a 1 GiB per-rank
test size, the eligible out-of-place phased path measured 6.16 GB/s bus
bandwidth. Disabling the NCCL-SAI AlltoAll policy in the same build measured
4.80 GB/s. The exactly aliased benchmark path remained upstream at 4.76 GB/s,
producing a combined average of 5.46 GB/s.

This result demonstrates recovery of large out-of-place AlltoAll performance
on the validated topology class. It does not demonstrate a universal
small-message speedup, an in-place optimization, or an application-level
speedup.

## Release Scope

This draft describes a source-level public candidate. Public binary packages
should be published only after clean rebuild, architecture coverage, and
sanitized validation evidence are available. Build and packaging expectations
are documented in `docs/sai/BUILD_AND_PACKAGING.md`.

Minimum public gates:

- clean source build;
- CUDA SASS/PTX coverage and portable host ISA target evidence for the advertised
  package;
- single-node, same-domain, cross-domain, and multi-domain alltoall;
- KB, MB, and GB message-size coverage;
- CUDA Graph capture and replay through the enqueue-based fallback;
- allreduce, reduce-scatter, allgather, broadcast, and P2P non-regression;
- fallback behavior when `NCCL_SAI_FABRIC_PROFILE` is unset or unrecognized.

## Not Claimed

This release draft does not claim:

- universal speedup for all NCCL or MPI collectives;
- application-level speedup for programs that do not use the optimized path;
- performance guarantees for every topology, scheduler allocation, or
  concurrent production workload;
- upstream vendor endorsement.

## Redistribution

Publish under the `AI4SAI` organization as a public NCCL-SAI repository derived
from NVIDIA NCCL. Keep upstream license files and copyright notices intact.
Keep upstream provenance explicit through documentation, tags, and an upstream
remote, but do not imply NVIDIA endorsement. Binary packages or tarballs must
include `LICENSE.txt`, `docs/sai/NOTICE.md`, and release notes that reproduce
the required upstream notice/disclaimer boundary.
