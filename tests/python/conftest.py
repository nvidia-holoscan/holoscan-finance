# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the pure-Python finance module.

The finance algorithms do not need CUDA.  These tests therefore provide a
small Holoscan API compatibility layer when the SDK is unavailable (notably on
developer macOS hosts).  When the SDK is installed, the real classes are used.
In both cases the source tree and CMake build tree are added to
``holoscan.__path__`` so ``holoscan.finance`` can be tested without installing
the wheel.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_HOLOSCAN = PROJECT_ROOT / "python" / "holoscan"
SOURCE_FINANCE = SOURCE_HOLOSCAN / "finance"
SOURCE_OPERATORS = PROJECT_ROOT / "operators"
BUILD_ROOT = Path(os.environ.get("FINANCE_BUILD_DIR", PROJECT_ROOT / "build")).resolve()
BUILD_HOLOSCAN = BUILD_ROOT / "python" / "lib" / "holoscan"
BUILD_FINANCE = BUILD_HOLOSCAN / "finance"
USE_BUILD_TREE = "FINANCE_BUILD_DIR" in os.environ


def _existing_paths(*paths: Path) -> list[str]:
    return [str(path) for path in paths if path.is_dir()]


def _holoscan_search_paths() -> list[str]:
    if USE_BUILD_TREE:
        return _existing_paths(BUILD_HOLOSCAN, SOURCE_HOLOSCAN)
    return _existing_paths(SOURCE_HOLOSCAN)


def _install_holoscan_stub() -> types.ModuleType:
    """Install the minimal SDK surface needed by the Python operators."""

    holoscan = types.ModuleType("holoscan")
    holoscan.__path__ = _holoscan_search_paths()  # type: ignore[attr-defined]
    holoscan.__package__ = "holoscan"

    core = types.ModuleType("holoscan.core")
    conditions = types.ModuleType("holoscan.conditions")

    class IOSpec:
        def __init__(self, name: str):
            self.name = name

    class OperatorSpec:
        def __init__(self):
            self.inputs: dict[str, IOSpec] = {}
            self.outputs: dict[str, IOSpec] = {}

        def input(self, name: str) -> IOSpec:
            port = IOSpec(name)
            self.inputs[name] = port
            return port

        def output(self, name: str) -> IOSpec:
            port = IOSpec(name)
            self.outputs[name] = port
            return port

    class Fragment:
        pass

    class Application(Fragment):
        def add_flow(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class ExecutionContext:
        pass

    class InputContext:
        pass

    class OutputContext:
        pass

    class Operator:
        def __init__(
            self,
            fragment: Fragment | None = None,
            *args: Any,
            name: str | None = None,
            **kwargs: Any,
        ) -> None:
            del args, kwargs
            self.fragment = fragment
            self.name = name or type(self).__name__
            self.spec = OperatorSpec()
            self.setup(self.spec)

        def setup(self, spec: OperatorSpec) -> None:
            del spec

    class CountCondition:
        def __init__(self, fragment: Fragment | None = None, count: int = 1, **kwargs: Any):
            del kwargs
            self.fragment = fragment
            self.count = count

    core.Application = Application
    core.ExecutionContext = ExecutionContext
    core.Fragment = Fragment
    core.InputContext = InputContext
    core.IOSpec = IOSpec
    core.Operator = Operator
    core.OperatorSpec = OperatorSpec
    core.OutputContext = OutputContext
    conditions.CountCondition = CountCondition

    holoscan.core = core
    holoscan.conditions = conditions
    sys.modules["holoscan"] = holoscan
    sys.modules["holoscan.core"] = core
    sys.modules["holoscan.conditions"] = conditions
    return holoscan


USING_HOLOSCAN_STUB = False
try:
    import holoscan
    import holoscan.core  # noqa: F401
except ImportError:
    holoscan = _install_holoscan_stub()
    USING_HOLOSCAN_STUB = True

existing_holoscan_paths = list(holoscan.__path__)
holoscan.__path__ = list(  # type: ignore[attr-defined]
    dict.fromkeys(_holoscan_search_paths() + existing_holoscan_paths)
)


def _load_finance_from_source_layout() -> None:
    """Load one logical package from shared and per-operator source trees.

    CMake assembles these directories into ``build/python/lib/holoscan/finance``.
    Source-only pytest runs need equivalent package search locations without
    copying or mutating generated files.
    """

    if USE_BUILD_TREE:
        search_locations = _existing_paths(BUILD_FINANCE, SOURCE_FINANCE, SOURCE_OPERATORS)
    else:
        search_locations = _existing_paths(SOURCE_FINANCE, SOURCE_OPERATORS)
    existing = sys.modules.get("holoscan.finance")
    if existing is not None:
        existing.__path__ = list(  # type: ignore[attr-defined]
            dict.fromkeys(search_locations + list(existing.__path__))
        )
        return

    spec = importlib.util.spec_from_file_location(
        "holoscan.finance",
        SOURCE_FINANCE / "__init__.py",
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to construct the holoscan.finance source package")
    finance = importlib.util.module_from_spec(spec)
    sys.modules["holoscan.finance"] = finance
    setattr(holoscan, "finance", finance)
    try:
        spec.loader.exec_module(finance)
    except Exception:
        sys.modules.pop("holoscan.finance", None)
        raise


_load_finance_from_source_layout()


class MockOpInput:
    """Holoscan-like input context containing one value per named port."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = dict(values)
        self.received_ports: list[str] = []

    def receive(self, port: str) -> Any:
        self.received_ports.append(port)
        if port not in self._values:
            raise AssertionError(f"operator received from undeclared test port {port!r}")
        return self._values[port]


class MockOpOutput:
    """Capture all values emitted by an operator compute invocation."""

    def __init__(self) -> None:
        self.emissions: list[tuple[Any, str]] = []

    def emit(self, value: Any, port: str) -> None:
        self.emissions.append((value, port))

    @property
    def emitted(self) -> tuple[Any, str] | None:
        return self.emissions[-1] if self.emissions else None


@pytest.fixture
def fragment():
    from holoscan.core import Fragment

    return Fragment()


@pytest.fixture
def op_input_factory():
    def make(port: str, value: Any) -> MockOpInput:
        return MockOpInput({port: value})

    return make


@pytest.fixture
def op_output_factory():
    return MockOpOutput


def invoke(op: Any, *, port: str | None = None, value: Any = None) -> MockOpOutput:
    """Invoke an operator once and return its captured output context."""

    op_input = MockOpInput({} if port is None else {port: value})
    op_output = MockOpOutput()
    op.compute(op_input, op_output, None)
    return op_output
