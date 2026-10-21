# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CUDA kernels, their compile flags and the native drain are byte-identical to 4.3.

The digests are of the 43bf4e1 originals, committed here so the check needs no git history
(CI checks out at depth 1, and a squash merge drops the commit). When the commit is
available, the sources are also compared with ``git show 43bf4e1:...`` directly.
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import holoscan_finance
from holoscan_finance.marketdata import gpu_bars, gpu_mitch

BASE = "43bf4e1"
PACKAGE = Path(holoscan_finance.__file__).parent
DRAIN = PACKAGE / "ingest" / "native" / "daqiri_drain.cpp"

# constant -> (module, 4.3 path, SHA-256 of the 4.3 kernel source)
KERNELS = {
    "_SOURCE": (
        gpu_mitch,
        "python/holoscan/marketdata/gpu_mitch.py",
        "dd2cef93496696ddce33ff24a124b8da05d427b511595dc8a3045a4d07f70c1e",
    ),
    "_BARS_SOURCE": (
        gpu_bars,
        "python/holoscan/marketdata/gpu_bars.py",
        "adf2e34b89a75dd502c12c25efc4e1a23fb8ce4ece286358907418c96532afa3",
    ),
}
# The RawModule options: --fmad=false keeps every multiply-add as in the host oracle.
OPTIONS = {
    gpu_mitch: [("-std=c++17", "--device-int128", "--fmad=false")],
    gpu_bars: [("-std=c++17", "--fmad=false")],
}
DRAIN_43 = "python/holoscan/marketdata/native/daqiri_drain.cpp"
DRAIN_SHA256 = "2c9f15b88b7ad7e573a1170beb9d37db9f2cd85947a88a19d9f10c24ed5dfff0"


def sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def constant(source: str, name: str) -> str:
    """The string literal assigned to ``name`` at module level, as written in the file."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return node.value.value
    raise AssertionError(f"{name} not found")


def raw_module_options(source: str) -> list[tuple[str, ...]]:
    return [
        ast.literal_eval(kw.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "RawModule"
        for kw in node.keywords
        if kw.arg == "options"
    ]


def module_source(module) -> str:
    return Path(module.__file__).read_text()


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_kernel_sources_match_the_4_3_digests(name):
    module, _, digest = KERNELS[name]
    source = constant(module_source(module), name)
    assert source == getattr(module, name)
    assert sha256(source) == digest


@pytest.mark.parametrize("module", [gpu_mitch, gpu_bars], ids=lambda m: m.__name__)
def test_raw_module_options_are_unchanged(module):
    assert raw_module_options(module_source(module)) == OPTIONS[module]


def test_native_drain_matches_the_4_3_digest():
    assert sha256(DRAIN.read_bytes()) == DRAIN_SHA256


def git_show(path: str) -> bytes:
    """``git show 43bf4e1:path``; skips when git or the commit is unavailable."""
    repo = PACKAGE.parents[1]
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if subprocess.run(["git", "cat-file", "-e", BASE], cwd=repo, capture_output=True).returncode:
        pytest.skip(f"commit {BASE} is not in this checkout")
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=repo, capture_output=True, check=True
    ).stdout


def test_sources_match_git_history_when_available():
    for name, (module, old_path, _) in KERNELS.items():
        old = git_show(old_path).decode()
        assert constant(old, name) == getattr(module, name)
        assert raw_module_options(old) == OPTIONS[module]
    assert git_show(DRAIN_43) == DRAIN.read_bytes()
