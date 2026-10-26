# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The module's catalog metadata is valid for holoscan-cli 5.0.0 and agrees with the code.

``holoscan list``, ``modes``, ``build`` and ``run`` read the root ``metadata.json`` and the
catalog entries under ``operators/`` and ``examples/``; the wheel and the CMake staging
read ``pyproject.toml`` and ``CMakeLists.txt``. These checks keep the copies of the same facts
(names, versions, the ``holoscan.finance`` entry point) from drifting.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE = json.loads((REPO / "metadata.json").read_text())["module"]

_spec = importlib.util.spec_from_file_location(
    "validate_metadata", REPO / "scripts/validate_metadata.py"
)
validate_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_metadata)


def catalog(kind: str) -> list[str]:
    """Catalog directories (with a metadata.json) under operators/ or examples/."""
    return sorted(path.parent.name for path in (REPO / kind).glob("*/metadata.json"))


def entry(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text())
    (kind,) = [key for key in data if key != "$schema"]
    return kind, data[kind]


def snake_case(class_name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()


def test_every_metadata_file_validates_against_the_cli_schemas_without_warnings():
    if importlib.util.find_spec("holoscan_cli") is None:
        pytest.skip("holoscan-cli is not installed (pip install -r requirements-cli.txt)")
    # The validator's only warnings are FutureWarnings (urn:holohub: labels, binary_packages.pypi).
    result = subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", "scripts/validate_metadata.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout  # the failure first
    lines = result.stdout.splitlines()
    assert lines and all(line.endswith(": ok") for line in lines), result.stdout


def test_every_metadata_file_is_a_holoscan_5_entry_with_the_module_authors():
    # One author list for the module, its operators and its example applications.
    authors = [a["name"] for a in MODULE["authors"]]
    assert authors == ["Bruce Hashemian", "Jake Goldberg"]
    files = sorted(validate_metadata.iter_metadata_files(REPO))
    assert len(files) > 1
    for path in files:
        label = json.loads(path.read_text()).get("$schema", "")
        kind, data = entry(path)
        where = path.relative_to(REPO)
        assert label.startswith("urn:holoscan:"), f"{where}: $schema {label!r}"
        assert data["holoscan_sdk"]["minimum_required_version"] == "5.0.0", where
        assert [a["name"] for a in data["authors"]] == authors, where


def test_subprojects_list_every_catalog_entry():
    assert sorted(MODULE["subprojects"]["operators"]) == catalog("operators")
    # The module schema names the key "applications"; the directory is examples/.
    assert sorted(MODULE["subprojects"]["applications"]) == catalog("examples")


def test_operator_names_match_the_operator_registry_and_the_catalog():
    from holoscan_finance import operators  # lazy: imports no Holoscan

    assert sorted(MODULE["operator_names"]) == sorted(operators.__all__)
    for name in catalog("operators"):
        kind, data = entry(REPO / "operators" / name / "metadata.json")
        assert kind == "operator" and data["name"] in operators.__all__, name
        assert name == snake_case(data["name"]), f"{name} holds {data['name']}"


def test_examples_require_only_catalog_operators():
    operators = set(catalog("operators"))
    for name in catalog("examples"):
        kind, data = entry(REPO / "examples" / name / "metadata.json")
        assert kind == "application" and data["name"] == name
        required = set(data["requirements"].get("operators", []))
        assert required <= operators, f"{name} requires {sorted(required - operators)}"


def test_version_and_entry_point_agree_across_packaging_files():
    import holoscan_finance

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    cmake = re.search(
        r"project\(holoscan_finance\s+VERSION\s+([\d.]+)", (REPO / "CMakeLists.txt").read_text()
    )
    assert cmake, "CMakeLists.txt: no project(holoscan_finance VERSION ...)"
    versions = {MODULE["version"], project["version"], holoscan_finance.__version__, cmake[1]}
    assert versions == {"0.2.0"}
    assert project["entry-points"]["holoscan.modules"] == {"finance": MODULE["namespace"]["python"]}
    assert MODULE["namespace"]["python"] == holoscan_finance.__name__
    assert MODULE["binary_packages"]["wheel"] == project["name"] == MODULE["name"]
