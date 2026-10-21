# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan introspection and rules: what a composition promises, checked before it runs.

``PlanView`` joins a compiled plan's JSON (payload edges -> ports -> operators, partition
placements, clock triggers) into operator-level facts. A ``Rule`` turns those
facts into violation strings, so an application states its invariants next to its graph
(source and bars share a lane, bars -> risk is latest-only, risk polls at 1 ms) and the
runner refuses a plan that breaks them; the plan tests apply the same rules on the CPU
tier. A rule never raises: a key a later SDK drops gives ``plan.json key <k> unavailable``. ``plan_summary`` is what ``--plan-only``, ``GET /plan`` and ``/snapshot.engine`` show.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import holoscan as hs

# plan.json payload_edges[].queue_policy (calibrated in graphs/test_plans.py).
POLICIES = {0: "BLOCKING", 1: "DROP_OLDEST"}
_PYOBJECT = "holoscan.PyObjectPayload"  # ports[].payload_identity.schema.canonical_name
_INPUT, _OUTPUT = 0, 1  # ports[].direction


@dataclass(frozen=True)
class Edge:
    src: str  # "op.port", or "@import.<name>" for an imported topic
    dst: str  # "op.port", or "@export.<name>" for an export topic endpoint
    policy: str  # "BLOCKING" | "DROP_OLDEST"
    depth: int


class PlanView:
    """Operator-level facts of one compiled plan.

    Lane membership comes from the ``plan.partitions`` API, everything else from
    ``plan.json()``. A fact whose keys are missing is left empty and named by
    ``unavailable()``, which the rules report instead of raising.
    """

    lanes: dict[str, list[str]]  # authored name -> sorted operator paths
    automatic: list[list[str]]  # operator paths of each automatic partition
    placements: dict[str, dict]  # lane -> {"cpus": [...], "wake_mode": int}
    edges: list[Edge]
    clocks: dict[str, int]  # operator path -> period_ns
    outputs: dict[str, list[str]]  # operator path -> pyobject output port names
    inputs: dict[str, list[str]]  # operator path -> input port names
    runtime_updates: list[str]  # operator paths with runtime parameter updates enabled

    def __init__(self, plan: hs.Plan) -> None:
        self.lanes, self.automatic = {}, []
        for partition in plan.partitions:
            members = sorted(partition.operator_paths)
            if partition.authored_name is None:
                self.automatic.append(members)
            else:
                self.lanes[partition.authored_name] = members
        doc = json.loads(plan.json())
        self._missing: dict[str, str] = {}
        self.placements = self._fact("placements", lambda: _placements(doc), {})
        self.edges = self._fact("edges", lambda: _edges(doc), [])
        self.clocks = self._fact("clocks", lambda: _clocks(doc), {})
        self.outputs = self._fact("outputs", lambda: _ports(doc, _OUTPUT, pyobject_only=True), {})
        self.inputs = self._fact("inputs", lambda: _ports(doc, _INPUT, pyobject_only=False), {})
        self.runtime_updates = self._fact(
            "runtime_updates",
            lambda: sorted(
                o["operator_path"]
                for o in doc["operator_parameters"]
                if o["runtime_updates_enabled"]
            ),
            [],
        )

    def _fact[T](self, name: str, build: Callable[[], T], empty: T) -> T:
        try:
            return build()
        except (KeyError, TypeError) as exc:
            key = exc.args[0] if isinstance(exc, KeyError) and exc.args else name
            self._missing[name] = str(key)
            return empty

    def unavailable(self, *facts: str) -> list[str]:
        """One violation per plan.json key missing for the named facts (``"edges"``, ...)."""
        keys = dict.fromkeys(self._missing[f] for f in facts if f in self._missing)
        return [f"plan.json key {key} unavailable" for key in keys]

    def lane_of(self, op: str) -> str | None:
        """The authored lane of ``op``; None for an automatic partition or an unknown op."""
        return next((name for name, ops in self.lanes.items() if op in ops), None)


def _operator_paths(doc: dict) -> dict[int, str]:
    return {o["owner"]["id"]: o["operator_path"] for o in doc["operator_parameters"]}


def _port_index(doc: dict) -> dict[int, tuple[str | None, dict]]:
    """Port id -> (owning operator path, or None for a topic endpoint; the port record)."""
    ops = _operator_paths(doc)
    return {
        p["id"]: (ops.get(p["owner"]) if p.get("owner_kind", 0) == 0 else None, p)
        for p in doc["ports"]
    }


def _edges(doc: dict) -> list[Edge]:
    ports = _port_index(doc)

    def name(port_id: int, endpoint: str) -> str:
        op, port = ports[port_id]
        return f"{op}.{port['local_name']}" if op else f"@{endpoint}.{port['local_name']}"

    return [
        Edge(
            name(e["source"], "import"),
            name(e["target"], "export"),
            POLICIES.get(e["queue_policy"], str(e["queue_policy"])),
            e["queue_depth"],
        )
        for e in doc["payload_edges"]
    ]


def _ports(doc: dict, direction: int, *, pyobject_only: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for op, port in _port_index(doc).values():
        if op is None or port["direction"] != direction:
            continue
        schema = port["payload_identity"].get("schema") or {}
        if not pyobject_only or schema.get("canonical_name") == _PYOBJECT:
            out.setdefault(op, []).append(port["local_name"])
    return {op: sorted(names) for op, names in out.items()}


def _clocks(doc: dict) -> dict[str, int]:
    ops = _operator_paths(doc)
    return {
        ops[t["owner"]["id"]]: t["period_ns"]
        for t in doc["clock_triggers"]
        if t["owner"]["id"] in ops
    }


def _placements(doc: dict) -> dict[str, dict]:
    return {
        p["authored_name"]: {
            "cpus": sorted(p["accepted"]["affinity"]),
            "wake_mode": p["accepted"]["wake_mode"],
        }
        for p in doc["partition_placements"]
        if p["authored_name"] is not None
    }


Rule = Callable[[PlanView], list[str]]  # returns violation strings


def same_lane(*ops: str) -> Rule:
    """``ops`` share one authored lane (no thread hop between them)."""

    def rule(view: PlanView) -> list[str]:
        lanes = {op: view.lane_of(op) for op in ops}
        if None not in lanes.values() and len(set(lanes.values())) == 1:
            return []
        where = ", ".join(f"{op} ({lane or 'no authored lane'})" for op, lane in lanes.items())
        return [f"same_lane: {where} are not in one authored lane"]

    return rule


def edge(
    src: str,
    dst: str,
    *,
    policy: str | None = None,
    depth: int | None = None,
    min_depth: int | None = None,
) -> Rule:
    """The edge ``src -> dst`` (``"op.port"``; ``dst="*"`` checks every edge leaving
    ``src``) exists and has the given queue policy and depth."""

    def rule(view: PlanView) -> list[str]:
        if gaps := view.unavailable("edges"):
            return gaps
        found = [e for e in view.edges if e.src == src and dst in ("*", e.dst)]
        if not found:
            return [f"edge {src} -> {dst}: no such edge"]
        out = []
        for e in found:
            if policy is not None and e.policy != policy:
                out.append(f"edge {e.src} -> {e.dst}: policy {e.policy}, expected {policy}")
            if depth is not None and e.depth != depth:
                out.append(f"edge {e.src} -> {e.dst}: depth {e.depth}, expected {depth}")
            if min_depth is not None and e.depth < min_depth:
                out.append(f"edge {e.src} -> {e.dst}: depth {e.depth}, expected >= {min_depth}")
        return out

    return rule


def clock_period_max(op: str, max_us: int) -> Rule:
    """``op`` is clock-triggered with a period of at most ``max_us``."""

    def rule(view: PlanView) -> list[str]:
        if gaps := view.unavailable("clocks"):
            return gaps
        period_ns = view.clocks.get(op)
        if period_ns is None:
            return [f"clock_period_max: {op} has no clock trigger"]
        if period_ns > max_us * 1000:
            return [f"clock_period_max: {op} polls every {period_ns / 1000:g} us > {max_us} us"]
        return []

    return rule


def no_automatic_partitions() -> Rule:
    """Every operator is in an authored, placed lane."""

    def rule(view: PlanView) -> list[str]:
        return [f"automatic partition {members}" for members in view.automatic]

    return rule


def outputs_connected() -> Rule:
    """Every pyobject output port feeds an edge (an emit on an unwired one raises)."""

    def rule(view: PlanView) -> list[str]:
        if gaps := view.unavailable("edges", "outputs"):
            return gaps
        sources = {e.src for e in view.edges}
        return [
            f"output {op}.{port} is not connected"
            for op, ports in view.outputs.items()
            for port in ports
            if f"{op}.{port}" not in sources
        ]

    return rule


def inputs_connected() -> Rule:
    """Every input port is fed (an unwired input fails the run with
    ``RUNTIME_OPERATOR_REPLAY_MISMATCH`` although the plan compiles)."""

    def rule(view: PlanView) -> list[str]:
        if gaps := view.unavailable("edges", "inputs"):
            return gaps
        targets = {e.dst for e in view.edges}
        return [
            f"input {op}.{port} is not connected"
            for op, ports in view.inputs.items()
            for port in ports
            if f"{op}.{port}" not in targets
        ]

    return rule


def lanes_avoid_cpus(cpus: Iterable[int]) -> Rule:
    """No authored lane may run on ``cpus`` (the DAQIRI lcores)."""
    avoid = frozenset(cpus)

    def rule(view: PlanView) -> list[str]:
        if not avoid:
            return []
        if gaps := view.unavailable("placements"):
            return gaps
        out = []
        for lane in view.lanes:
            placement = view.placements.get(lane)
            if placement is None:
                out.append(f"lane {lane} has no CPU placement, so it may run on {sorted(avoid)}")
            elif overlap := sorted(avoid.intersection(placement["cpus"])):
                out.append(f"lane {lane} runs on reserved CPUs {overlap}")
        return out

    return rule


def no_runtime_parameter_updates() -> Rule:
    """No operator has runtime parameter updates (they fail all-Python graphs at start)."""

    def rule(view: PlanView) -> list[str]:
        if gaps := view.unavailable("runtime_updates"):
            return gaps
        return [f"{op} has runtime parameter updates enabled" for op in view.runtime_updates]

    return rule


def check_plan(plan: hs.Plan, rules: Iterable[Rule]) -> list[str]:
    """Every violation of ``rules`` on ``plan``."""
    view = PlanView(plan)
    return [violation for rule in rules for violation in rule(view)]


def plan_summary(plan: hs.Plan) -> dict:
    """Lanes with their CPUs, edges, clock periods and automatic partitions (JSON-ready)."""
    view = PlanView(plan)
    return {
        "lanes": [
            {
                "name": name,
                "operators": ops,
                "cpus": view.placements.get(name, {}).get("cpus", []),
                "wake_mode": _wake_mode_name(view.placements.get(name, {}).get("wake_mode")),
            }
            for name, ops in view.lanes.items()
        ],
        "edges": [
            {"from": e.src, "to": e.dst, "policy": e.policy, "depth": e.depth} for e in view.edges
        ],
        "clocks": [
            {"operator": op, "period_us": period_ns / 1000}
            for op, period_ns in sorted(view.clocks.items())
        ],
        "automatic_partitions": view.automatic,
        "sdk": os.environ.get("HOLOSCAN_SDK_VERSION"),
    }


def _wake_mode_name(value: int | None) -> str | None:
    import holoscan as hs

    try:
        return None if value is None else hs.execution.WakeMode(value).name
    except ValueError:
        return str(value)
