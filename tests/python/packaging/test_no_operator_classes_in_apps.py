# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Applications compose the module's operators; they never define their own.

An application file is a ``compose()`` over ``holoscan_finance.operators`` plus settings and
plan rules, so every operator stays reusable, documented in the catalog and testable on its
own. This scans the application sources (AST, no import) for classes derived from
``hs.Operator`` or from any ``...Op`` class.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def operator_classes(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ClassDef):
            for base in map(ast.unparse, node.bases):
                if base.rpartition(".")[2] == "Operator" or base.endswith("Op"):
                    where = f"{path.relative_to(REPO)}:{node.lineno}"
                    found.append(f"{where} class {node.name}({base})")
    return found


def test_applications_define_no_operator_classes():
    sources = sorted((REPO / "applications").rglob("*.py"))
    assert sources, "no application sources found"
    offenders = [cls for path in sources for cls in operator_classes(path)]
    assert not offenders, "move these into holoscan_finance.operators:\n" + "\n".join(offenders)
