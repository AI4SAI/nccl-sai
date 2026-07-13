# NCCL-SAI Build And Packaging Guide

This guide defines public build expectations for NCCL-SAI source and binary
packages. It is intentionally generic: site-specific module names, scheduler
policies, filesystem paths, and deployment defaults belong in private site
configuration.

## Source Build

NCCL-SAI should build with the same upstream NCCL build flow unless a release
note states otherwise:

```bash
make -j src.build CUDA_HOME=/path/to/cuda
```

Release candidates should record:

- upstream NCCL base version;
- NCCL-SAI commit or tag;
- CUDA toolkit version;
- compiler family and version;
- `NVCC_GENCODE`;
- whether the build is source-only, tarball, Debian package, RPM package, or a
  site module package.

## CUDA Architecture Coverage

Public SAI binary packages intended to cover mixed GPU generations should include
SASS for these CUDA architectures unless a narrower package is clearly labeled:

```text
sm_70 sm_75 sm_80 sm_86 sm_89 sm_90
```

The broad SAI package should include PTX for every advertised architecture, in
addition to SASS. This keeps each supported generation independently
inspectable and preserves a virtual-architecture image for compatible JIT
paths. A typical release build can use:

```bash
NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_70,code=compute_70 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_75,code=compute_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_80,code=compute_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_86,code=compute_86 \
-gencode=arch=compute_89,code=sm_89 \
-gencode=arch=compute_89,code=compute_89 \
-gencode=arch=compute_90,code=sm_90 \
-gencode=arch=compute_90,code=compute_90"
```

If a package omits an architecture, state that limitation in the package name or
release notes.

## Host CPU ISA Policy

Do not build public NCCL-SAI CPU objects with vendor-specific `-march` targets
such as AMD Zen or Intel server microarchitecture names unless the package is
explicitly labeled for that CPU family. Such flags may enable or tune for
features that are not a portable cross-vendor contract.

Use x86-64 microarchitecture levels for portable binary packages:

- `x86-64-v3`: recommended default package level for broad modern x86-64
  systems. This covers AVX2-class CPUs and avoids requiring AVX-512.
- `x86-64-v4`: optional AVX-512 package level for systems that satisfy the v4
  feature set. This is a cross-vendor ISA level, not a synonym for one AMD or
  Intel CPU generation.

For GCC-compatible compilers, set the target consistently through the build:

```bash
export CFLAGS="-O3 -DNDEBUG -march=x86-64-v3 -mtune=generic"
export CXXFLAGS="$CFLAGS"
export FCFLAGS="$CFLAGS"
export FFLAGS="$CFLAGS"
```

For an AVX-512 package, replace `x86-64-v3` with `x86-64-v4`.

The NCCL-SAI 2.28.9 release build is pinned to GNU GCC because NCCL's default
`PROFAPI` uses GNU alias/weak declarations that NVIDIA HPC SDK 25.7 `nvc++`
does not compile correctly. Loading an NVHPC/OpenMPI consumer module can export
`CXX=nvc++`, so the release command must override it explicitly:

```bash
export CFLAGS="-O3 -DNDEBUG -march=x86-64-v3 -mtune=generic"
export CXXFLAGS="$CFLAGS"
make -j "${BUILD_JOBS}" src.build \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  CUDA_HOME=<cuda-12.4-root> \
  BUILDDIR="$PWD/build-release"
```

Start from a fresh build directory. Record the real paths and versions of
`gcc`, `g++`, and `nvcc`, the source/archive hash, `CUDA_HOME`,
`NVCC_GENCODE`, and the complete make command in `BUILD_INFO.txt`. NVHPC remains
a supported consumer MPI/application stack; it is not a claimed NCCL compiler
for this release. Avoid `native`, `host`, `znver*`, and Intel
microarchitecture targets for public generic packages.

## Dependency Consistency

The host ISA target must apply to every compiled CPU dependency in the package,
not only to NCCL itself. Audit at least:

- NCCL CPU objects and shared libraries;
- CUDA-aware MPI stack components if bundled with the package;
- UCX, UCC, PMIx, PRRTE, libevent, hwloc, and plugin libraries when bundled;
- wrapper compiler metadata such as `mpicc --showme` or equivalent;
- build logs for accidental `native`, `host`, `znver*`, or Intel
  microarchitecture targets.

Directory names such as `avx2` or `avx512` are not proof of ISA coverage.
Record compiler flags, wrapper metadata, and a clean run of the package's
instruction-set audit in release evidence.

## Runtime Selection

Sites may expose multiple packages through environment modules or container
tags. A safe public pattern is:

- load `x86-64-v3` by default;
- load `x86-64-v4` only when the current host satisfies the v4 feature set;
- never silently load a v4 package on a v3-only host;
- require no `NCCL_SAI_*` activation variable in normal use; each optimization
  must enable only after its runtime topology predicate and communicator-wide
  agreement pass, as described in `docs/sai/README.md`.

Runtime selection should be based on feature-level checks, not on CPU vendor
strings or cluster-specific node names.

## Release Evidence

Before publishing a binary package, preserve sanitized evidence for:

- clean source tree and reproducible build command;
- CUDA architecture list in the resulting library;
- host ISA target flags for NCCL and bundled CPU dependencies;
- smoke tests proving the package loads on every advertised CPU ISA level;
- NCCL-SAI transparent fallback and explicit disable knobs;
- communication gates listed in `docs/sai/COMMUNICATION_TUNING_MATRIX.md`.

Do not publish private hostnames, job IDs, raw scheduler logs, internal paths,
switch labels, credentials, GUID maps, or field-operation notes as release
evidence.
