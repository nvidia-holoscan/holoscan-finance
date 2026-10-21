# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything outside the operator layer imports without the Holoscan SDK, CuPy or DAQIRI.

In Holoscan 5, ``import holoscan`` links libcuda, so a module that imports the SDK (or CuPy)
at module level cannot run in the CPU tools, the feed and replay containers (started
without the GPU runtime), spawned analytics workers or the pure CI tier. DAQIRI exists
only in the Thor receiver image, so it is held to the same rule. Each library module is
imported in a fresh interpreter with ``sys.modules`` entries for ``holoscan``, ``cupy`` and
``daqiri`` set to None, so any module-level import of them fails. Only the operator modules
(``operators/<op>.py``, ``_common``, ``edges``) and ``tools/topic_tail`` may import the SDK
at module level; a module whose optional third-party dependency (FastAPI, duckdb, ...) is
missing here is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PYTHON = REPO / "python"
BLOCKED = ("holoscan", "cupy", "daqiri")

CHILD = f"""
import importlib, sys
sys.modules.update(dict.fromkeys({BLOCKED!r}))
try:
    importlib.import_module(sys.argv[1])
except ModuleNotFoundError as exc:
    missing = (exc.name or "").partition(".")[0]
    if missing in {(*BLOCKED, "holoscan_finance")!r}:
        raise
    print(missing)  # an optional third-party dependency that is not installed here
"""


def is_sdk_module(name: str) -> bool:
    """The modules tagged [S] in the design: the operator layer and the holomq subscriber."""
    return (
        name.startswith("holoscan_finance.operators.")
        or name == "holoscan_finance.tools.topic_tail"
    )


def library_modules() -> list[str]:
    names = []
    for path in sorted((PYTHON / "holoscan_finance").rglob("*.py")):
        parts = path.relative_to(PYTHON).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if not is_sdk_module(name):
            names.append(name)
    return names


MODULES = library_modules()


def python(*args: str) -> subprocess.CompletedProcess:
    # -P: never put the working directory on sys.path; the checkout comes from PYTHONPATH.
    path = os.pathsep.join(filter(None, (str(PYTHON), os.environ.get("PYTHONPATH"))))
    return subprocess.run(
        [sys.executable, "-P", *args],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def imports() -> dict[str, subprocess.CompletedProcess]:
    """One fresh interpreter per module, run in parallel."""
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return dict(zip(MODULES, pool.map(lambda m: python("-c", CHILD, m), MODULES)))


@pytest.mark.parametrize("module", MODULES)
def test_library_module_imports_without_holoscan_cupy_or_daqiri(imports, module):
    result = imports[module]
    assert result.returncode == 0, f"{module}:\n{result.stderr}"
    if missing := result.stdout.strip():
        pytest.skip(f"{module} needs {missing}, which is not installed")


def test_the_package_and_its_operator_registry_load_none_of_them():
    result = python(
        "-c",
        "import json, sys, holoscan_finance, holoscan_finance.operators;"
        "dir(holoscan_finance); dir(holoscan_finance.operators);"
        f"print(json.dumps([m for m in {BLOCKED!r} if m in sys.modules]))",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
