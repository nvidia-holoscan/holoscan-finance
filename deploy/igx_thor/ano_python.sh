#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Python in the container against the mounted checkout's holoscan_finance package.
# The checkout is appended: the image's PYTHONPATH (/usr/lib/python3.12/site-packages) is the
# only place Holoscan SDK 5 is found, so it must stay first and is never replaced.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HOLOSCAN_FINANCE_REPO/python"
exec python3 "$@"
