# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Receiver image for the Thor launchers (igx_run.sh, gpu_ingest_bench/sweep.sh): Holoscan
# SDK 5.0, NVIDIA DAQIRI (DPDK engine + Python binding) built from source with the Thor
# steering patch, and the app dependencies. Build from the repository root:
#   docker build -f deploy/igx_thor/Dockerfile -t holoscan-finance:hsdk5-daqiri .
# Only the DAQIRI/DPDK build outputs, no image (a build check):
#   docker build -f deploy/igx_thor/Dockerfile --target daqiri-artifacts \
#       --output type=local,dest=out/daqiri-artifacts .
# The checkout is not copied in: the launchers mount it read-only. DAQIRI does not link
# Holoscan; it shares glibc, CPython 3.12 and CUDA 13 with the base image.

ARG HOLOSCAN_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
# DOCA apt repo for rdma-core: 3.3.0 gives 2601.0.7, the userspace validated on Thor;
# 3.2.1 gives 2510.0.11 (Thor's DOCA-OFED 25.10). The base image ships 59.1.
ARG DOCA_VERSION=3.3.0
# Fingerprint of the key that signs the DOCA apt repository (Mellanox Signing Key v3, the same
# file for x86_64 and arm64-sbsa). The build fails if the downloaded key does not carry it.
ARG DOCA_GPG_FINGERPRINT=CAE649B3B3929F8470D4D33BA024F6F0E6D6A281

# ---------------------------------------------------------------------------------------------
FROM ${HOLOSCAN_IMAGE} AS doca-repo
SHELL ["/bin/bash", "-eou", "pipefail", "-c"]
ARG DOCA_VERSION
ARG DOCA_GPG_FINGERPRINT
RUN DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}") \
    && ARCH=$([ "$(uname -m)" = aarch64 ] && echo arm64-sbsa || echo x86_64) \
    && URL="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/${DISTRO}/${ARCH}" \
    && curl -fsSLo /tmp/mellanox.pub "${URL}/GPG-KEY-Mellanox.pub" \
    && { gpg --batch --show-keys --with-colons /tmp/mellanox.pub \
         | awk -F: '$1 == "pub" {want = 1} $1 == "fpr" && want {print $10; want = 0}' \
         | grep -x "${DOCA_GPG_FINGERPRINT}" > /dev/null \
         || { echo "DOCA signing key does not carry fingerprint ${DOCA_GPG_FINGERPRINT}" >&2; exit 1; }; } \
    && gpg --batch --dearmor < /tmp/mellanox.pub > /usr/share/keyrings/mellanox-archive-keyring.gpg \
    && rm /tmp/mellanox.pub \
    && echo "deb [signed-by=/usr/share/keyrings/mellanox-archive-keyring.gpg] ${URL} ./" > /etc/apt/sources.list.d/mellanox.list

# ---------------------------------------------------------------------------------------------
FROM doca-repo AS daqiri-build
ARG DEBIAN_FRONTEND=noninteractive
# The NVIDIA/daqiri commit (2026.7.0, between #233 and #238) whose public headers match the
# DAQIRI validated on Thor; daqiri_thor_steering.patch applies to it, not to the v2026.07.00 tag.
ARG DAQIRI_REPO=https://github.com/NVIDIA/daqiri.git
ARG DAQIRI_REF=7c21e331179bb679d1ef3a6efb72b24e9f75143d
ARG DPDK_VERSION=25.11
# SHA-256 of dpdk-${DPDK_VERSION}.tar.xz. Changing DPDK_VERSION means passing the new
# tarball's hash as well (DPDK publishes none next to it: compute it from a download you trust).
ARG DPDK_SHA256=52e90d2a531ef3ded0283bd91abc94980698f1f6471fa09658a0217cf6609526
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libibverbs-dev librdmacm-dev ibverbs-providers libnuma-dev \
        meson ninja-build pkgconf python3-pyelftools pybind11-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "${DAQIRI_REPO}" /src/daqiri && git -C /src/daqiri checkout --detach "${DAQIRI_REF}"
COPY deploy/igx_thor/daqiri_thor_steering.patch /tmp/
RUN git -C /src/daqiri apply /tmp/daqiri_thor_steering.patch
# Upstream DPDK with DAQIRI's dma-buf/NVIDIA patches and the options of DAQIRI's own Dockerfile.
RUN curl -fsSLo /tmp/dpdk.tar.xz "https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz" \
    && echo "${DPDK_SHA256}  /tmp/dpdk.tar.xz" | sha256sum --check --strict - \
    && tar xJf /tmp/dpdk.tar.xz -C /tmp && rm /tmp/dpdk.tar.xz \
    && cd "/tmp/dpdk-${DPDK_VERSION}" \
    && for p in /src/daqiri/dpdk_patches/*.patch; do \
         if [[ "$(basename "$p")" == dmabuf.patch ]]; then \
           git apply --exclude=.mailmap --exclude=doc/guides/rel_notes/release_26_03.rst "$p"; \
         else git apply "$p"; fi; \
       done \
    && meson setup /tmp/dpdk-build --prefix=/usr/local -Dtests=false -Dplatform=generic -Denable_docs=false \
        -Ddisable_drivers='baseband/*,bus/ifpga/*,common/cpt,common/dpaax,common/iavf,common/octeontx,common/octeontx2,crypto/nitrox,net/ark,net/atlantic,net/avp,net/axgbe,net/bnx2x,net/bnxt,net/cxgbe,net/e1000,net/ena,net/enic,net/fm10k,net/hinic,net/hns3,net/i40e,net/ixgbe,vdpa/ifc,net/igc,net/liquidio,net/mana,net/netvsc,net/nfp,net/qede,net/sfc,net/thunderx,net/vdev_netvsc,net/vmxnet3,regex/octeontx2' \
    && ninja -C /tmp/dpdk-build && meson install -C /tmp/dpdk-build && ldconfig \
    && rm -rf /tmp/dpdk-*
RUN cmake -S /src/daqiri -B /tmp/daqiri-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/daqiri \
        -DCMAKE_CUDA_ARCHITECTURES=all-major -DBUILD_SHARED_LIBS=ON \
        -DDAQIRI_BUILD_PYTHON=ON -DDAQIRI_ENGINE="dpdk ibverbs" \
    && cmake --build /tmp/daqiri-build -j "$(nproc)" \
    && cmake --install /tmp/daqiri-build
# Stage what the runtime needs at its install paths: DPDK's EAL loads every PMD from the
# compiled-in /usr/local/lib/<multiarch>/dpdk directory, so the libraries keep that location.
RUN LIBDIR="/usr/local/lib/$(gcc -print-multiarch)" \
    && rm -f "${LIBDIR}"/librte_*.a \
    && mkdir -p "/staging${LIBDIR}" /staging/opt \
    && cp -a "${LIBDIR}/." "/staging${LIBDIR}/" \
    && cp -a /opt/daqiri /staging/opt/

# Artifacts only (for `--output type=local`): exactly what the runtime stage copies.
FROM scratch AS daqiri-artifacts
COPY --from=daqiri-build /staging/ /

# ---------------------------------------------------------------------------------------------
FROM doca-repo AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 ibverbs-providers librdmacm1 libnuma1 pybind11-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=daqiri-build /staging/ /
# The binding has no RUNPATH: register /opt/daqiri/lib with the loader instead of touching
# LD_LIBRARY_PATH (its CUDA compat entry must stay first). PYTHONPATH is never replaced:
# Holoscan 5 is found only through the image's /usr/lib/python3.12/site-packages, so DAQIRI's
# Python package goes in through a .pth file in a site directory.
RUN echo /opt/daqiri/lib > /etc/ld.so.conf.d/daqiri.conf && ldconfig \
    && echo /opt/daqiri/lib/python > /usr/local/lib/python3.12/dist-packages/daqiri.pth
ENV DAQIRI_ROOT=/opt/daqiri CMAKE_PREFIX_PATH=/opt/daqiri
# App and test dependencies; ruff, pre-commit and build are developer tools. Never pip
# pybind11: the native drain must use the apt pybind11 2.x to share daqiri.BurstParams.
COPY requirements-dev.txt /tmp/
RUN grep -vE '^(ruff|pre-commit|build)\b' /tmp/requirements-dev.txt > /tmp/requirements-runtime.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && python3 -c 'import importlib.util as u; assert all(u.find_spec(n) for n in ("holoscan", "cupy", "daqiri"))'
