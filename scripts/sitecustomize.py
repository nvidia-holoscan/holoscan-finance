# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Auto-imported at interpreter startup (when this dir is on PYTHONPATH).
# Extends the installed `holoscan` package (container image or .venv) to this repo's source
# tree so `holoscan.equities` / `holoscan.marketdata` resolve without a wheel build.
import os


def _extend():
    repo = os.environ.get("HOLOSCAN_FINANCE_REPO")
    if not repo:
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.dirname(here)
    src = os.path.join(repo, "python", "holoscan")
    if not os.path.isdir(src):
        return
    try:
        import holoscan
    except Exception:
        return
    if src not in holoscan.__path__:
        holoscan.__path__.insert(0, src)


_extend()
