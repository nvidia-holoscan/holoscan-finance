# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the equity demo's Python tests.

The CPU paths do not need CUDA.  These tests therefore provide a small Holoscan
API compatibility layer when the SDK is unavailable (notably on developer macOS
hosts).  When the SDK is installed, the real classes are used.  The source tree is
added to ``holoscan.__path__`` so ``holoscan.equities`` and ``holoscan.marketdata``
can be tested without installing the package; under CTest (``FINANCE_BUILD_DIR``)
the staged build tree comes first, so the packaged layout is what gets tested.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

SOURCE_HOLOSCAN = str(Path(__file__).resolve().parents[2] / "python" / "holoscan")
SEARCH_PATHS = [SOURCE_HOLOSCAN]
if "FINANCE_BUILD_DIR" in os.environ:
    build = Path(os.environ["FINANCE_BUILD_DIR"]).resolve() / "python" / "lib" / "holoscan"
    SEARCH_PATHS.insert(0, str(build))


def _install_holoscan_stub() -> types.ModuleType:
    """Install the minimal SDK surface the receive operators need."""

    holoscan = types.ModuleType("holoscan")
    holoscan.__path__ = list(SEARCH_PATHS)  # type: ignore[attr-defined]
    core = types.ModuleType("holoscan.core")
    conditions = types.ModuleType("holoscan.conditions")

    class IOSpec:
        class QueuePolicy:
            POP, REJECT, FAULT = range(3)

        class IOSize:
            def __init__(self, size: int):
                self.size = size

        def condition(self, kind: Any, **kwargs: Any) -> IOSpec:
            del kind, kwargs
            return self

    class ConditionType:
        NONE, MESSAGE_AVAILABLE, DOWNSTREAM_MESSAGE_AFFORDABLE = range(3)

    class OperatorSpec:
        def __init__(self):
            self.inputs: list[str] = []
            self.outputs: list[str] = []

        def input(self, name: str, **kwargs: Any) -> IOSpec:
            del kwargs
            self.inputs.append(name)
            return IOSpec()

        def output(self, name: str) -> IOSpec:
            self.outputs.append(name)
            return IOSpec()

    class Fragment:
        pass

    class Operator:
        def __init__(self, fragment=None, *args: Any, name: str | None = None, **kwargs: Any):
            del args, kwargs
            self.fragment = fragment
            self.name = name or type(self).__name__
            self.spec = OperatorSpec()
            self.setup(self.spec)

        def setup(self, spec: OperatorSpec) -> None:
            del spec

    class BooleanCondition:
        def __init__(self, fragment=None, enable_tick: bool = True, **kwargs: Any):
            del fragment, kwargs
            self.tick_enabled = enable_tick

        def disable_tick(self) -> None:
            self.tick_enabled = False

    class PeriodicCondition:
        def __init__(self, fragment=None, recess_period: int = 0, **kwargs: Any):
            del fragment, kwargs
            self.recess_period = recess_period

    for cls in (ConditionType, Fragment, IOSpec, Operator, OperatorSpec):
        setattr(core, cls.__name__, cls)
    for name in ("ExecutionContext", "InputContext", "OutputContext"):
        setattr(core, name, type(name, (), {}))
    conditions.BooleanCondition = BooleanCondition
    conditions.PeriodicCondition = PeriodicCondition
    holoscan.core, holoscan.conditions = core, conditions
    sys.modules.update(
        {"holoscan": holoscan, "holoscan.core": core, "holoscan.conditions": conditions}
    )
    return holoscan


try:
    import holoscan
    import holoscan.core  # noqa: F401
except ImportError:
    holoscan = _install_holoscan_stub()

holoscan.__path__ = list(  # type: ignore[attr-defined]
    dict.fromkeys([*SEARCH_PATHS, *holoscan.__path__])
)


class MockOpInput:
    """Serve queued values to ``receive(port)``; ``None`` once a port is empty."""

    def __init__(self, **ports: list[Any]) -> None:
        self.ports = {k: list(v) for k, v in ports.items()}
        self.received = 0

    def receive(self, port: str) -> Any:
        queue = self.ports.get(port) or []
        if not queue:
            return None
        self.received += 1
        return queue.pop(0)


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
def op_output_factory():
    return MockOpOutput


@pytest.fixture
def op_input_factory():
    return MockOpInput
