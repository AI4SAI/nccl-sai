# NCCL

Optimized primitives for inter-GPU communication.

## NCCL-SAI Branch

This branch derives from NVIDIA NCCL `v2.29.3-1` and carries a narrowly
guarded dual-rail endpoint-selection change for SAI GPU fabric products.
When operator-managed site policy requests rail-by-channel selection, every
rank must agree on the policy and use NCCL's internal IB transport with
`NCCL_CROSS_NIC=0`. Each participating GPU must also expose one topology-local
dual-port NET pair backed by two unmerged physical endpoints. The two logical
rails must have distinct GID subnet prefixes, and every rank must report them
in the same port order. Physical port 1 serves even channels and physical port
2 serves odd channels for ring/tree graph endpoints and graphless P2P traffic.
NVLS and CollNet endpoints retain their upstream selection behavior.

The policy fails during communicator initialization when configuration,
transport, physical-endpoint eligibility, or logical rail identity differs
across ranks. The fork also carries an initialization wire identifier so that
mixing NCCL-SAI and an ABI-compatible upstream runtime in one communicator
fails before the SAI-specific agreement step. With the policy unset, ring/tree
and graphless P2P endpoint selection retain upstream behavior; the wire check
and one SAI agreement exchange still run during initialization. Applications
and users do not need to set an NCCL-SAI environment variable; activation and
unmerged-NIC policy are site-operator responsibilities. External network
plugins are intentionally rejected while this policy is active because the
generic plugin interface does not provide the rail identity needed by this
implementation.

NCCL-SAI modifications are maintained by AI4SAI contributors. This project is
derived from NVIDIA NCCL and is not endorsed by NVIDIA. Original NVIDIA
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

Original NVIDIA source code and documentation retain their existing NVIDIA
copyright notices. NCCL-SAI modifications are copyright (c) 2026, AI4SAI
contributors. See `LICENSE.txt` and `docs/sai/NOTICE.md`.
