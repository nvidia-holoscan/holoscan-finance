# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for Holoscan Finance.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/finance when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holohub run-container
#
# Manual build (without the CLI):
#   docker build -t holoscan-finance .
#
# For a digest-pinned base image (NGC login required to resolve the digest):
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan@sha256:<digest> .

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.3.0-cuda13
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install holoscan-cli developer tools. Override with --build-arg; default is
# a pinned release (no --pre) from NVIDIA PyPI.
ARG HOLOSCAN_CLI_INSTALL_ARGS="--extra-index-url https://pypi.nvidia.com holoscan-cli==4.5.0"
RUN if ! python3 -m holoscan_cli --help 2>/dev/null | grep -qw build; then \
        python3 -c "import shlex, subprocess, sys; \
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + shlex.split('''${HOLOSCAN_CLI_INSTALL_ARGS}'''))" \
    ; fi

# The module itself uses only the Python standard library beyond Holoscan SDK.

# Python tooling: pytest-timeout for tests, build + scikit-build-core for
# `./holohub package --pkg-generator WHEEL` (drives `python -m build`).
RUN pip install --no-cache-dir pytest-timeout==2.4.0 build==1.6.0 scikit-build-core==1.0.3
