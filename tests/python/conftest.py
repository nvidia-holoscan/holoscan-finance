# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test tiers: skip what this host cannot run, and keep CuPy's kernel cache out of $HOME.

Markers (``pyproject.toml``): ``sdk`` skips unless ``import holoscan`` works, ``gpu`` unless
CuPy sees a CUDA device, ``daqiri`` unless ``import daqiri`` works. ``slow`` and ``timing``
are never skipped here; commands deselect them with ``-m``. A test module that cannot even be
imported because its tier's dependency is missing is skipped as well, so the pure tier can
collect every module. The 4.3 conftest installed a 4.x ``holoscan`` stub and edited
``holoscan.__path__``; in the 5.0 image that hid the real SDK, so neither exists any more.
Shared test doubles live in ``fakes/``, not in fixtures.
"""

from __future__ import annotations

import functools
import importlib
import os
import tempfile
from pathlib import Path

import pytest

# The top-level module each skippable marker needs.
TIER_MODULES = {"sdk": "holoscan", "gpu": "cupy", "daqiri": "daqiri"}


@functools.cache
def missing(marker: str) -> str | None:
    """Why tests marked ``marker`` cannot run here, or None when they can."""
    try:
        module = importlib.import_module(TIER_MODULES[marker])
    except ImportError as exc:
        return f"{marker}: {exc}"
    if marker == "gpu":
        try:
            if module.cuda.runtime.getDeviceCount() < 1:
                return "gpu: no CUDA device"
        except module.cuda.runtime.CUDARuntimeError as exc:
            return f"gpu: CUDA unavailable ({exc})"
    return None


class TierModule(pytest.Module):
    """A test module whose import fails on a missing tier dependency is skipped, not an error."""

    def _getobj(self):
        try:
            return super()._getobj()
        except self.CollectError as exc:
            name = getattr(exc.__cause__, "name", None) or ""
            for marker, module in TIER_MODULES.items():
                # Only a dependency that is really absent here; any other ImportError is a bug.
                if name.partition(".")[0] == module and (reason := missing(marker)):
                    pytest.skip(f"{reason}, imported by {self.nodeid}", allow_module_level=True)
            raise


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Module:
    return TierModule.from_parent(parent, path=module_path)


def pytest_configure(config: pytest.Config) -> None:
    # CuPy reads CUPY_CACHE_DIR once, at import. dev5.sh sets a persistent cache; anywhere
    # else, kernels built by the tests go to a directory removed at exit.
    if "CUPY_CACHE_DIR" not in os.environ:
        cache = tempfile.TemporaryDirectory(prefix="hf-cupy-")
        config.add_cleanup(cache.cleanup)
        os.environ["CUPY_CACHE_DIR"] = cache.name


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in TIER_MODULES:
        if item.get_closest_marker(marker) and (reason := missing(marker)):
            pytest.skip(reason)
