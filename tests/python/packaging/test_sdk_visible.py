# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``import holoscan`` in the tests is the installed Holoscan 5 SDK.

This fails if a stub or a checkout copy of ``holoscan`` ever shadows the installed SDK.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.sdk

REPO = Path(__file__).resolve().parents[3]


def test_holoscan_is_the_installed_sdk() -> None:
    import holoscan

    assert hasattr(holoscan, "compile"), f"no holoscan.compile: not the 5.x SDK ({holoscan})"
    assert not Path(holoscan.__file__).resolve().is_relative_to(REPO), holoscan.__file__
