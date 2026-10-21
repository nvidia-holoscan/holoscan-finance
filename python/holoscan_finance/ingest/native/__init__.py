# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GIL-free native drain, built on first use into a writable cache directory.

``load_daqiri_drain()`` compiles ``daqiri_drain.cpp`` against the installed DAQIRI (``DAQIRI_ROOT``, default ``/opt/daqiri``), the CUDA driver API
(linked against the stub; CuPy loads the driver) and pybind11, then imports it. The headers
must match ``_daqiri``'s pybind11 internals, so the system headers (apt ``pybind11-dev``)
come before a pip ``pybind11``: ``HOLOSCAN_FINANCE_PYBIND11_INCLUDE``, then ``/usr/include``,
then pip. The shared object goes to ``HOLOSCAN_FINANCE_NATIVE_CACHE``, else
``$CUPY_CACHE_DIR/native``, else ``~/.cache/holoscan-finance/native``; its key covers the
source, Python, compiler, flags, DAQIRI version, pybind11 internals and NumPy major, so an
image change never reuses a stale build. It returns None (logged; the caller uses the
Python drain) when the build fails or ``HOLOSCAN_FINANCE_NATIVE=0`` or
``HOLOSCAN_FINANCE_NATIVE_DRAIN=0``. The sources call it before ``daqiri_init``: the cold
build takes about 5 s, which must not run with the NIC up and unpolled.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import importlib.util
import logging
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("daqiri_drain.cpp")
_SYSTEM_INCLUDE = Path("/usr/include")
_SWITCHES = ("HOLOSCAN_FINANCE_NATIVE", "HOLOSCAN_FINANCE_NATIVE_DRAIN")


def _cache_dir() -> Path:
    explicit = os.environ.get("HOLOSCAN_FINANCE_NATIVE_CACHE")
    if explicit:
        return Path(explicit)
    cupy = os.environ.get("CUPY_CACHE_DIR")
    return Path(cupy) / "native" if cupy else Path.home() / ".cache/holoscan-finance/native"


def _pybind11_include() -> Path:
    """The pybind11 headers to build with: the override, the system ones, else pip's."""
    explicit = os.environ.get("HOLOSCAN_FINANCE_PYBIND11_INCLUDE")
    if explicit:
        return Path(explicit)
    if (_SYSTEM_INCLUDE / "pybind11" / "pybind11.h").exists():
        return _SYSTEM_INCLUDE
    try:
        import pybind11
    except ImportError as exc:
        raise ImportError("pybind11 headers not found (install pybind11-dev)") from exc
    return Path(pybind11.get_include())


def _pybind11_version(include: Path) -> str:
    """``major.minor.patch`` of the headers in ``include`` ("" if unreadable)."""
    try:
        text = (include / "pybind11" / "detail" / "common.h").read_text()
    except OSError:
        return ""
    parts = [
        re.search(rf"#define PYBIND11_VERSION_{part}\s+(\S+)", text)
        for part in ("MAJOR", "MINOR", "PATCH")
    ]
    return ".".join(m.group(1) for m in parts if m)


def _internals_id(daqiri: ModuleType) -> str:
    """The pybind11 internals id compiled into DAQIRI's extension (``__pybind11_internals_v5_…``):
    the drain must be built against the same internals to accept DAQIRI's types."""
    path = getattr(getattr(daqiri, "_daqiri", None), "__file__", None)
    if not path:
        return ""
    found = re.search(rb"__pybind11_internals_v\w+__", Path(path).read_bytes())
    return found.group().decode() if found else ""


def _cache_key(compiler: str, include: Path, flags: list[str]) -> str:
    import daqiri
    import numpy

    parts = (
        _SOURCE.read_bytes(),
        sys.version.encode(),
        compiler.encode(),
        " ".join(flags).encode(),
        getattr(daqiri, "__version__", "").encode(),
        _internals_id(daqiri).encode(),
        f"{include}:{_pybind11_version(include)}".encode(),
        numpy.__version__.split(".")[0].encode(),
    )
    return hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]


def _build() -> type:
    """Compile into the cache (once per key) and import the module's ``Drain``."""
    root = Path(os.environ.get("DAQIRI_ROOT", "/opt/daqiri"))
    cuda = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    cxx = os.environ.get("CXX", "c++")
    include = _pybind11_include()
    flags = [
        f"-I{sysconfig.get_paths()['include']}", f"-I{include}",
        f"-I{root / 'include'}", f"-I{cuda / 'include'}",
        f"-L{root / 'lib'}", "-ldaqiri", f"-Wl,-rpath,{root / 'lib'}",
        f"-L{cuda / 'lib64' / 'stubs'}", "-lcuda",  # link-time only: CuPy loads the driver
    ]  # fmt: skip
    compiler = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=True)
    key = _cache_key(compiler.stdout, include, flags)
    module_name = "_daqiri_drain"
    target = _cache_dir() / f"{module_name}_{key}{sysconfig.get_config_var('EXT_SUFFIX')}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        cmd = [cxx, "-O2", "-shared", "-fPIC", "-std=c++17", str(_SOURCE), "-o", str(tmp), *flags]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.replace(tmp, target)  # atomic: concurrent builders never load a partial file
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(target))
    spec = importlib.util.spec_from_file_location(module_name, str(target), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.Drain


@functools.cache
def load_daqiri_drain() -> type | None:
    """The native ``Drain`` class, or None (logged) if disabled or the build fails.

    Built once per process, cached on disk across processes."""
    for var in _SWITCHES:  # ablation and troubleshooting
        if os.environ.get(var, "1") == "0":
            logger.info("native daqiri_drain disabled by %s=0", var)
            return None
    try:
        return _build()
    except (OSError, subprocess.CalledProcessError, ImportError) as exc:
        logger.warning("native daqiri_drain unavailable: %s", getattr(exc, "stderr", "") or exc)
        return None


__all__ = ["load_daqiri_drain"]
