# NCCL

Optimized primitives for inter-GPU communication.

## NCCL-SAI 2.18 Branch

This branch carries a small AI4SAI change set derived from NVIDIA NCCL
`v2.18.5-1`. It keeps the NCCL 2.18 public API and SONAME and focuses on three
bounded behaviors:

- an operator-managed, opt-in rail-by-channel policy for graphless NET
  selection when exactly two topology-local NET endpoints are available;
- automatic host-local phasing for an exact, sufficiently large, dense grouped
  Send/Recv exchange; and
- nonblocking proxy-listener handshake progress so a partial or non-NCCL
  client cannot monopolize the proxy service thread.

Dense scheduling and listener hardening require no application environment
variables. A deployment which uses the dual-rail policy may set
`NCCL_SAI_RAIL_BY_CHANNEL=1` through site-managed runtime configuration.
The implementation does not inspect scheduler metadata, hostnames, partition
names, or deployment paths.

This branch does not add a native AlltoAll API, replace the NCCL 2.18 executor,
change PXN policy, expand collective channels, or override collective
algorithm/protocol selection. See [the NCCL-SAI guide](docs/sai/README.md) and
[release notes](docs/sai/RELEASE_NOTES.md) for exact predicates and validation
boundaries.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is derived from NVIDIA NCCL and is not endorsed by NVIDIA. Original NVIDIA
copyright and license notices are retained; see `LICENSE.txt` and
`docs/sai/NOTICE.md`.

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

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.

NCCL-SAI modifications are copyright (c) 2026, AI4SAI contributors and are
redistributed under `LICENSE.txt`; see `docs/sai/NOTICE.md`.
