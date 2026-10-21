# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development and CI container for holoscan-finance: the Holoscan SDK 5.0 image plus
# holoscan-cli (the image has no `holoscan` command, and the CLI calls it inside the container)
# and the test/tool dependencies. holoscan-cli bind-mounts the source at
# /workspace/holoscan-finance; nothing from the repository is copied in.
#   holoscan run-container                  # build and enter
#   docker build -t holoscan-finance .      # manual
# The DAQIRI/NIC runtime image is deploy/igx_thor/Dockerfile.

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13
FROM ${BASE_IMAGE}

# CUPY_CACHE_DIR is left unset: holoscan-cli points it at /workspace/<name>/.cupy.
COPY requirements-cli.txt requirements-dev.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-cli.txt -r /tmp/requirements-dev.txt \
    && holoscan version
