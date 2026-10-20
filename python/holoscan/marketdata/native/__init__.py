# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small native helpers built on first use into a writable cache directory.

``load_daqiri_frames()`` and ``load_daqiri_drain()`` compile ``daqiri_frames.cpp`` and
``daqiri_drain.cpp`` against the installed DAQIRI headers and library (``DAQIRI_ROOT``,
default ``/opt/daqiri``), CUDA and the pybind11 headers, then import them. The source
checkout stays read-only; the shared object goes to ``HOLOSCAN_FINANCE_NATIVE_CACHE``,
else ``$CUPY_CACHE_DIR/native``, else ``~/.cache/holoscan-finance/native``. Each returns
``None`` (callers fall back to Python) if building fails; ``HOLOSCAN_FINANCE_NATIVE=0``
disables both.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import logging
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)
_loaded = {}


def _cache_dir() -> Path:
    explicit = os.environ.get("HOLOSCAN_FINANCE_NATIVE_CACHE")
    if explicit:
        return Path(explicit)
    cupy = os.environ.get("CUPY_CACHE_DIR")
    return Path(cupy) / "native" if cupy else Path.home() / ".cache/holoscan-finance/native"


def _pybind11_include() -> str:
    try:
        import pybind11

        return pybind11.get_include()
    except ImportError:
        return "/usr/include"


def _build(stem, extra_flags=()):
    """Compile ``<stem>.cpp`` into the cache (once per source/compiler/flags) and import it."""
    src = Path(__file__).with_name(f"{stem}.cpp")
    root = Path(os.environ.get("DAQIRI_ROOT", "/opt/daqiri"))
    cuda = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    cxx = os.environ.get("CXX", "c++")
    compiler = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=True).stdout
    key = hashlib.sha256(
        src.read_bytes()
        + sys.version.encode()
        + compiler.encode()
        + str(root).encode()
        + " ".join(extra_flags).encode()
        + b"cuda-inc"
    ).hexdigest()[:16]
    module_name = f"_{stem}"
    target = _cache_dir() / f"{module_name}_{key}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        cmd = [
            cxx, "-O2", "-shared", "-fPIC", "-std=c++17", str(src), "-o", str(tmp),
            f"-I{sysconfig.get_paths()['include']}", f"-I{_pybind11_include()}",
            f"-I{root / 'include'}", f"-I{cuda / 'include'}",
            f"-L{root / 'lib'}", "-ldaqiri", f"-Wl,-rpath,{root / 'lib'}", *extra_flags,
        ]  # fmt: skip
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.replace(tmp, target)
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(target))
    spec = importlib.util.spec_from_file_location(module_name, str(target), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _load(attr, stem, extra_flags=(), switch=None):
    """``attr`` of the built module, or None (logged) if disabled or the build fails."""
    if attr in _loaded:
        return _loaded[attr]
    _loaded[attr] = None
    for var in ("HOLOSCAN_FINANCE_NATIVE", switch):  # ablation / troubleshooting
        if var and os.environ.get(var, "1") == "0":
            logger.info("native %s disabled by %s=0", stem, var)
            return None
    try:
        _loaded[attr] = getattr(_build(stem, extra_flags), attr)
    except (OSError, subprocess.CalledProcessError, ImportError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        logger.warning("native %s unavailable: %s", stem, detail)
    return _loaded[attr]


def load_daqiri_frames():
    """Batch frame-address fetch, or None (callers use per-packet binding calls)."""
    return _load("burst_frames", "daqiri_frames")


def load_daqiri_drain():
    """The GIL-free drain class, or None (callers use the Python drain thread).
    ``HOLOSCAN_FINANCE_NATIVE_DRAIN=0`` disables it alone."""
    cuda = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    stubs = cuda / "lib64" / "stubs"  # link-time only: the driver is loaded by CuPy
    return _load(
        "Drain", "daqiri_drain", (f"-L{stubs}", "-lcuda"), switch="HOLOSCAN_FINANCE_NATIVE_DRAIN"
    )
