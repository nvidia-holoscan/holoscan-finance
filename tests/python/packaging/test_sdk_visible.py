# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``import holoscan`` in the tests is the installed Holoscan 5 SDK.

The 4.3 conftest installed a 4.x ``holoscan`` stub and put the checkout on
``holoscan.__path__``; in the 5.0 image that hid the SDK from every operator test. This
fails if a stub or a checkout copy of ``holoscan`` ever comes back.
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
