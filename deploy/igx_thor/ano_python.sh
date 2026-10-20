#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Python in the container with this checkout ahead of the image's packages
# (scripts/sitecustomize.py), keeping the image's DAQIRI paths.
set -euo pipefail
export PYTHONPATH="$HOLOSCAN_FINANCE_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$@"
