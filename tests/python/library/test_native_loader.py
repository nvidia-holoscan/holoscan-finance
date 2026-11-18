# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``load_daqiri_drain`` with the compiler mocked: the pybind11 include order (override,
system, pip), the cache key (an image change never reuses a stale build) and the switches.
No DAQIRI and no compiler needed; the real build runs in the DAQIRI image."""

from __future__ import annotations

import logging
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import numpy
import pytest

from holoscan_finance.ingest import native


class Compiler:
    """``subprocess.run`` for the loader: answers ``--version`` and "compiles" a file that is
    not a shared object, so the import that follows fails like a broken build."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if "--version" not in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text("not an ELF file")
        return subprocess.CompletedProcess(cmd, 0, stdout="c++ (fake) 13.3\n", stderr="")

    @property
    def builds(self) -> list[list[str]]:
        return [c for c in self.commands if "--version" not in c]


def headers(root: Path, version: str = "2.11.1") -> Path:
    major, minor, patch = version.split(".")
    (root / "pybind11" / "detail").mkdir(parents=True)
    (root / "pybind11" / "pybind11.h").write_text("")
    (root / "pybind11" / "detail" / "common.h").write_text(
        f"#define PYBIND11_VERSION_MAJOR {major}\n#define PYBIND11_VERSION_MINOR {minor}\n"
        f"#define PYBIND11_VERSION_PATCH {patch}\n"
    )
    return root


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    """A clean loader: no switches, an empty system include, no pip pybind11, a fake
    ``daqiri`` and a private cache directory."""
    for var in native._SWITCHES + ("HOLOSCAN_FINANCE_PYBIND11_INCLUDE",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOLOSCAN_FINANCE_NATIVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(native, "_SYSTEM_INCLUDE", tmp_path / "usr-include")
    monkeypatch.setitem(sys.modules, "pybind11", None)
    monkeypatch.setitem(sys.modules, "daqiri", types.SimpleNamespace(__version__="2026.7.0"))
    fake = Compiler()
    monkeypatch.setattr(native.subprocess, "run", fake)
    native.load_daqiri_drain.cache_clear()
    yield fake
    native.load_daqiri_drain.cache_clear()


def include_of(command: list[str]) -> str:
    """The pybind11 ``-I`` (the second one; the first is Python's)."""
    includes = [c[2:] for c in command if c.startswith("-I")]
    assert includes[0] == sysconfig.get_paths()["include"]
    return includes[1]


def test_system_pybind11_comes_before_pip(compiler, monkeypatch, tmp_path):
    pip = headers(tmp_path / "pip")
    monkeypatch.setitem(
        sys.modules, "pybind11", types.SimpleNamespace(get_include=lambda: str(pip))
    )
    assert native.load_daqiri_drain() is None  # the fake build does not import
    assert include_of(compiler.builds[-1]) == str(pip)

    native.load_daqiri_drain.cache_clear()
    system = headers(tmp_path / "usr-include")
    native.load_daqiri_drain()
    assert include_of(compiler.builds[-1]) == str(system)

    native.load_daqiri_drain.cache_clear()
    monkeypatch.setenv("HOLOSCAN_FINANCE_PYBIND11_INCLUDE", str(tmp_path / "override"))
    native.load_daqiri_drain()
    assert include_of(compiler.builds[-1]) == str(tmp_path / "override")


def test_no_pybind11_headers_is_a_logged_none(compiler, caplog):
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        assert native.load_daqiri_drain() is None
    assert compiler.builds == [] and "pybind11 headers not found" in caplog.text


def test_build_command_target_and_cache_hit(compiler, monkeypatch, tmp_path):
    headers(tmp_path / "usr-include")
    monkeypatch.setenv("DAQIRI_ROOT", "/opt/daqiri")
    monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
    assert native.load_daqiri_drain() is None
    assert native.load_daqiri_drain() is None  # once per process
    (cmd,) = compiler.builds
    assert cmd[cmd.index("-o") - 1] == str(native._SOURCE)
    assert {"-shared", "-fPIC", "-std=c++17", "-ldaqiri", "-lcuda"} <= set(cmd)
    assert "-L/usr/local/cuda/lib64/stubs" in cmd  # link-time only: CuPy loads the driver
    assert "-I/opt/daqiri/include" in cmd and "-Wl,-rpath,/opt/daqiri/lib" in cmd
    (built,) = (tmp_path / "cache").iterdir()
    assert built.name.startswith("_daqiri_drain_")
    assert built.name.endswith(sysconfig.get_config_var("EXT_SUFFIX"))

    native.load_daqiri_drain.cache_clear()
    native.load_daqiri_drain()  # a new process: the cached file is reused
    assert len(compiler.builds) == 1


def test_cache_key_covers_the_image(compiler, monkeypatch, tmp_path):
    include = headers(tmp_path / "usr-include")

    def key(compiler: str = "c++ 13.3", include: Path = include, flags=("-lcuda",)) -> str:
        return native._cache_key(compiler, include, list(flags))

    base = key()
    assert key() == base
    assert key(compiler="c++ 14.1") != base
    assert key(flags=("-lcuda", "-O3")) != base
    newer = headers(tmp_path / "newer", version="3.0.1")
    assert native._pybind11_version(newer) == "3.0.1"
    assert key(include=newer) != base

    monkeypatch.setitem(sys.modules, "daqiri", types.SimpleNamespace(__version__="2026.8.0"))
    assert key() != base
    monkeypatch.setitem(sys.modules, "daqiri", types.SimpleNamespace(__version__="2026.7.0"))
    assert key() == base

    numpy_version = numpy.__version__
    monkeypatch.setattr(numpy, "__version__", "3.0.0")
    assert key() != base
    monkeypatch.setattr(numpy, "__version__", numpy_version)

    # The internals compiled into _daqiri: the drain must share them to take DAQIRI's types.
    extension = tmp_path / "_daqiri.so"
    extension.write_bytes(
        b"\0ELF...\0__pybind11_internals_v11_system_libstdcpp_gxx_abi_1xxx__\0..."
    )
    daqiri = types.SimpleNamespace(
        __version__="2026.7.0", _daqiri=types.SimpleNamespace(__file__=str(extension))
    )
    monkeypatch.setitem(sys.modules, "daqiri", daqiri)
    assert (
        native._internals_id(daqiri) == "__pybind11_internals_v11_system_libstdcpp_gxx_abi_1xxx__"
    )
    assert key() != base


@pytest.mark.parametrize("switch", ["HOLOSCAN_FINANCE_NATIVE", "HOLOSCAN_FINANCE_NATIVE_DRAIN"])
def test_switches_disable_the_build(compiler, monkeypatch, tmp_path, switch):
    headers(tmp_path / "usr-include")
    monkeypatch.setenv(switch, "0")
    assert native.load_daqiri_drain() is None
    assert compiler.commands == []


def test_a_failed_compile_is_a_logged_none(compiler, monkeypatch, tmp_path, caplog):
    headers(tmp_path / "usr-include")

    def broken(cmd, **kwargs):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="c++\n", stderr="")
        raise subprocess.CalledProcessError(1, cmd, stderr="daqiri/common.h: No such file")

    monkeypatch.setattr(native.subprocess, "run", broken)
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        assert native.load_daqiri_drain() is None
    assert "daqiri/common.h: No such file" in caplog.text
    assert not list((tmp_path / "cache").glob("*.so"))
