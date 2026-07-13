# NCCL

Optimized primitives for inter-GPU communication.

## NCCL-SAI Branch

This branch carries AI4SAI changes for SAI UltraPOD and SlimPOD GPU fabrics.
Its automatic release capabilities are narrowly guarded local P2P and
dual-rail transport selection. Normal application use requires no
`NCCL_SAI_*` environment variables. The runtime combines strict GPU/NIC
topology checks, physical ASIC/port relationships, and communicator-wide
agreement. Missing, ambiguous, merged, plugin-backed, or rank-inconsistent
layouts retain the corresponding upstream path.

The source also contains experimental `ncclAlltoAll()` implementations: in
controlled expert validation, eligible small messages use a two-stage
GPU-island path and eligible large messages use a phased P2P planner. NCCL 2.28
does not expose a stable cross-node hardware fabric-group identity, so these
grouped AlltoAll/P2P schedules are not automatic release capabilities and
normal zero-variable runs leave them on the upstream path. They remain
available only for explicit expert experiments with rank-consistent group
metadata.

The current automatic class is intentionally exact: SM70 GPUs form complete,
equal-bandwidth four-GPU NVLink cliques, and each clique belongs to a distinct
CPU locality domain. On the validated dual-port topology, physical port 1
serves even channels and physical port 2 serves odd channels. NET enumeration
order has no rail meaning.
The internal IB transport must report two distinct GID subnet prefixes, with
the port-1 prefix and port-2 prefix each identical across the communicator.
Ordinary ring/tree graph endpoints use the same channel-aligned port; CollNet
and NVLS endpoint selection remains unchanged. The single-host eight-rank
two-island P2P path requires NCCL's same-host/shared-memory identity, the exact
GPU topology, bidirectional NVML P2P capability, and communicator-wide
agreement; it is independent from NIC merge/cross-NIC mode. The host-domain
identity and upstream P2P controls are safety/policy gates, not signals used to
identify a site or activate the exception.
An unset level or `NCCL_P2P_LEVEL=NVL` is compatible with the strict automatic
two-island predicate, while other explicit distance limits remain unchanged.

Collective algorithms remain unchanged unless explicitly documented and
validated. See
`docs/sai/README.md` and `docs/sai/COMMUNICATION_TUNING_MATRIX.md` for scope,
rollback knobs, and validation requirements.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is derived from NVIDIA NCCL and is not endorsed by NVIDIA. Original NVIDIA NCCL
copyright and license notices are retained; see `LICENSE.txt` and
`docs/sai/NOTICE.md`.

For SAI users, the intended runtime mode is drop-in replacement: put the
NCCL-SAI build's `lib/` directory before the system NCCL in `LD_LIBRARY_PATH`.
Applications and shared MPI stacks do not set SAI activation variables.
`NCCL_IB_HCA`, `NCCL_IB_MERGE_NICS`, and `NCCL_CROSS_NIC` retain their upstream
semantics and may be supplied by ordinary site policy. NCCL-SAI classifies the
topology those controls actually produce; their strings are not activation
signals. Legacy
`NCCL_SAI_FABRIC_PROFILE` values remain available for compatibility and expert
testing of grouped paths, but are not a production prerequisite and do not
change automatic rail or local-P2P detection. `NCCL_SAI_DISABLE=1` is the global
emergency rollback; feature-specific `*_ENABLE=0` controls remain available
where documented.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

Original NVIDIA NCCL source code and documentation retain their upstream
copyright notices. NCCL-SAI modifications are copyright (c) 2026, AI4SAI
contributors and are redistributed under `LICENSE.txt`; see
`docs/sai/NOTICE.md` for the modification boundary.
