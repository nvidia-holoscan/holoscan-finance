# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live objects shared with operators: ``Ref`` (a Param-safe handle) and ``Knob`` (latest value).

Holoscan deep-copies Param values into every operator instance (at ``graph.op``, at compile
and per run), so a live object such as a stats holder or a gate travels inside a ``Ref``
whose deep copy is itself: every instance then shares one object. ``Knob`` holds the latest
value: writers serialize, data-plane readers take no lock.
No Holoscan import, so library code builds both without the SDK.
"""

from __future__ import annotations

import threading


class Ref:
    """Param-safe live-object handle: deepcopy returns self, so operator instances share obj."""

    __slots__ = ("obj",)

    def __init__(self, obj: object | None = None) -> None:
        self.obj = obj

    def __deepcopy__(self, memo: dict) -> Ref:
        return self

    def __copy__(self) -> Ref:
        return self

    def __bool__(self) -> bool:
        return self.obj is not None

    def __repr__(self) -> str:
        return f"Ref({type(self.obj).__name__ if self.obj is not None else 'None'})"


class Knob[T]:
    """Latest immutable value. Writers serialize on a lock; readers never lock (one attribute read).

    The value and its version are published as one tuple, so a reader that needs both sees
    a consistent pair without locking.
    """

    def __init__(self, value: T | None = None) -> None:
        self._lock = threading.Lock()
        self._state: tuple[int, T | None] = (0, value)

    def get(self) -> T | None:
        return self._state[1]

    def set(self, value: T) -> int:
        """Publish ``value``; returns the new version."""
        with self._lock:
            version = self._state[0] + 1
            self._state = (version, value)
        return version

    @property
    def version(self) -> int:
        return self._state[0]
