# NCCL

Optimized primitives for inter-GPU communication.

## NCCL-SAI Branch

This branch carries AI4SAI changes for SAI UltraPOD and SlimPOD GPU fabrics.
The first optimized API is `ncclAlltoAll()`: eligible small messages use a
two-stage GPU-island aggregation path, while eligible large messages use a
phased P2P planner. NCCL-SAI also has a narrowly guarded local P2P
transport-selection path that can affect any operation on an eligible
single-host communicator. On the `ultrapod-fullmesh` profile, GPUs with exactly
two local network devices use channel parity to keep channel traffic aligned to
the two rails. Other profiles and local-network-device counts retain upstream
selection. Collective algorithms remain unchanged unless explicitly documented
and validated. See
`docs/sai/README.md` and `docs/sai/COMMUNICATION_TUNING_MATRIX.md` for scope,
rollback knobs, and validation requirements.

NCCL-SAI modifications are maintained by AI4SAI/SAI contributors. This project
is derived from NVIDIA NCCL and is not endorsed by NVIDIA. Original NVIDIA NCCL
copyright and license notices are retained; see `LICENSE.txt` and
`docs/sai/NOTICE.md`.

For SAI users, the intended runtime mode is drop-in replacement: put the
NCCL-SAI build's `lib/` directory before the system NCCL in `LD_LIBRARY_PATH`.
SAI site modules or prologs can enable transparent SAI behavior by setting
`NCCL_SAI_FABRIC_PROFILE` to a recognized product-family profile. The current
island and phased AlltoAll defaults are selected by `ultrapod-fullmesh`;
that profile also enables the exactly-two-local-NET channel alignment described
above. Broader family names such as `ultrapod` and `slimpod` are recognized
activation namespaces but do not imply that the same topology-specific behavior
is valid for every layout.
Unknown profile names and unsupported layouts fall back to upstream NCCL
behavior unless an expert explicitly opts in with `NCCL_SAI_A2A_ENABLE=1` and
the related controls.

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
