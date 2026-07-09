# NCCL-SAI Public Release Notes Draft

NCCL-SAI is an AI4SAI optimization branch derived from NVIDIA NCCL. The current
source candidate is based on NCCL `v2.28.9-1` and targets SAI UltraPOD and
SlimPOD GPU fabric families.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is not endorsed by NVIDIA. Original NVIDIA NCCL copyright and license notices
are retained in `LICENSE.txt`; see `docs/sai/NOTICE.md`.

## Optimized API

- `ncclAlltoAll()` on eligible SAI fabric profiles.

Other NCCL collectives retain upstream behavior unless explicitly modified and
validated by future releases.

## Transparent Runtime Model

SAI sites can enable default NCCL-SAI behavior by setting:

```bash
export NCCL_SAI_FABRIC_PROFILE=ultrapod
```

or another site-selected public profile name. Disabled profile values are `0`,
`false`, `off`, `none`, `native`, and `upstream`.

Explicit overrides:

- `NCCL_SAI_A2A_ENABLE=1`: enable the alltoall SAI path.
- `NCCL_SAI_A2A_ENABLE=0`: disable the alltoall SAI path.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=1`: enable the selected local P2P path guard.
- `NCCL_SAI_LOCAL_P2P_SYS_ENABLE=0`: disable that local path guard.

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
- allreduce, reduce-scatter, allgather, broadcast, and P2P non-regression;
- fallback behavior when `NCCL_SAI_FABRIC_PROFILE` is unset or disabled.

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
