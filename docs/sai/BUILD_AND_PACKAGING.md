# NCCL-SAI build and packaging guide

NCCL-SAI uses the upstream NCCL build flow:

```bash
make -j src.build \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  CUDA_HOME=/path/to/cuda \
  BUILDDIR=/path/to/fresh-build
```

Release builds should start from a clean source archive and a fresh build
directory. Record the source commit and tree, archive checksum, CUDA toolkit,
compiler versions, complete build command, host flags, and `NVCC_GENCODE`.

## Canonical source archive

Create the public source asset directly from the final clean commit. Use the
commit time as `SOURCE_DATE_EPOCH`, suppress gzip timestamps, build twice, and
compare the byte streams:

```bash
commit=$(git rev-parse HEAD)
tree=$(git rev-parse HEAD^{tree})
SOURCE_DATE_EPOCH=$(git show -s --format=%ct "$commit")
export SOURCE_DATE_EPOCH

git archive --format=tar --prefix=nccl-sai/ "$commit" | gzip -n \
  > nccl-sai-2.18.5-1-sai.2-source.tar.gz.first
git archive --format=tar --prefix=nccl-sai/ "$commit" | gzip -n \
  > nccl-sai-2.18.5-1-sai.2-source.tar.gz.second
cmp nccl-sai-2.18.5-1-sai.2-source.tar.gz.first \
  nccl-sai-2.18.5-1-sai.2-source.tar.gz.second
mv nccl-sai-2.18.5-1-sai.2-source.tar.gz.first \
  nccl-sai-2.18.5-1-sai.2-source.tar.gz
rm nccl-sai-2.18.5-1-sai.2-source.tar.gz.second
git get-tar-commit-id \
  < <(gzip -dc nccl-sai-2.18.5-1-sai.2-source.tar.gz)
sha256sum nccl-sai-2.18.5-1-sai.2-source.tar.gz
```

Record `commit`, `tree`, `SOURCE_DATE_EPOCH`, the embedded commit ID, and the
archive checksum together. The embedded commit ID must equal `commit`.

## Portable binary policy

The broad CUDA 12.4 x86-64 package uses portable GNU host compilation with:

```text
-O3 -DNDEBUG -march=x86-64-v3 -mtune=generic
```

The broad package includes SASS and PTX for:

```text
sm_70 sm_75 sm_80 sm_86 sm_89 sm_90
```

Do not use `-march=native`, vendor-specific CPU names, or an unlabeled
single-GPU-generation build for a general release asset.

## Package contents

A binary package must contain one top-level directory and retain:

- `LICENSE.txt` and the NCCL-SAI modification notice;
- release notes and the public guide;
- public NCCL headers;
- versioned shared libraries and correct SONAME symlinks; and
- build identity and checksums.

Static libraries may be distributed in a full development package. A smaller
runtime asset may omit the static library if that limitation is stated.

## Release evidence

Before publication, verify archive integrity, reproducible repacking, shared
library loadability, NCCL version identity, CUDA architecture coverage,
portable host ISA, license contents, and disclosure scans. Public artifacts
must not contain private hostnames, scheduler job identifiers, node lists,
internal paths, credentials, raw operational logs, or private application data.
