# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CI ``gpu`` job runs every test it owns exactly once.

``holoscan test --local`` runs every CTest tier that ``tests/CMakeLists.txt`` registers with
``finance_pytest`` (no label filter), and the job's own pytest step then adds what those tiers
leave out. Running a tier in both steps doubles the GPU runner's time and its exposure to an
intermittent GPU test. ``timing`` tests never run in CI. This reads both files (no import) and
checks the marker expressions against every combination of the markers they name.
"""

from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def selects(expression: str, markers: frozenset[str]) -> bool:
    """Whether ``pytest -m expression`` selects a test that carries exactly ``markers``."""

    def value(node: ast.expr) -> bool:
        match node:
            case ast.Name(id=name):
                return name in markers
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                return not value(operand)
            case ast.BoolOp(op=ast.And(), values=values):
                return all(map(value, values))
            case ast.BoolOp(op=ast.Or(), values=values):
                return any(map(value, values))
        raise ValueError(f"not a marker expression: {expression!r}")

    return value(ast.parse(expression, mode="eval").body)


def ctest_tiers() -> dict[str, str]:
    cmake = (REPO / "tests/CMakeLists.txt").read_text()
    return dict(re.findall(r'^finance_pytest\((\w+) "([^"]+)"\)', cmake, re.M))


def gpu_job() -> str:
    ci = (REPO / ".github/workflows/ci.yml").read_text()
    job = re.search(r"^  gpu:\n(.*?)(?=^  \S|\Z)", ci, re.M | re.S)
    assert job, "ci.yml: no gpu job"
    return job.group(1)


def test_gpu_job_pytest_step_adds_only_what_ctest_leaves_out():
    tiers = ctest_tiers()
    assert tiers, "tests/CMakeLists.txt: no finance_pytest tiers"
    job = gpu_job()
    assert "holoscan test --local" in job  # runs every CTest tier
    (step,) = re.findall(r'pytest[^\n]*?-m "([^"]+)"', job)
    names = sorted(
        {
            node.id
            for expression in [step, *tiers.values()]
            for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Name)
        }
    )
    for flags in itertools.product((False, True), repeat=len(names)):
        markers = frozenset(name for name, on in zip(names, flags, strict=True) if on)
        if "timing" in markers and "slow" not in markers:
            continue  # every timing test is also slow (pyproject.toml markers)
        in_ctest = [tier for tier, expression in tiers.items() if selects(expression, markers)]
        in_step = selects(step, markers)
        assert not (in_ctest and in_step), f"{sorted(markers)}: CTest {in_ctest} and pytest"
        runs = bool(in_ctest) or in_step
        assert runs == ("timing" not in markers), f"{sorted(markers)}: runs={runs}"
