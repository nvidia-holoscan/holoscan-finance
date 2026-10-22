# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``equity_analytics`` flags: canonical ingest names, front-end defaults, rejected combinations,
NIC Params."""

from __future__ import annotations

import argparse

import pytest
from apphelpers import equity_args

from holoscan_finance.app import cli
from holoscan_finance.app.runner import ExitCode
from holoscan_finance.ingest.affinity import CpuSets
from holoscan_finance.settings import INGEST_MODES


def usage_error(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    """The stderr message of a flag set that ``normalize`` rejects with exit status 2."""
    with pytest.raises(SystemExit) as exc:
        equity_args(*argv)
    assert exc.value.code == ExitCode.USAGE == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("mode", INGEST_MODES)
def test_canonical_ingest_names_parse_unchanged(mode: str) -> None:
    assert equity_args("--ingest", mode, "--pcap", "day.pcap", "--tape", "tape").ingest == mode


@pytest.mark.parametrize("name", ["sockets", "gpu-direct", "nic"])
def test_other_ingest_names_are_rejected(capsys: pytest.CaptureFixture[str], name: str) -> None:
    assert "invalid choice" in usage_error(capsys, "--ingest", name)


def test_nic_flags_have_one_spelling(capsys: pytest.CaptureFixture[str]) -> None:
    err = usage_error(capsys, "--ingest", "gpudirect", "--pcie", "0001:02:00.0")
    assert "unrecognized arguments: --pcie" in err


def test_nic_flags_set_the_nic_settings() -> None:
    args = equity_args(
        "--ingest", "gpudirect", "--nic-pcie", "0001:02:00.0", "--nic-config", "nic.yaml",
        "--nic-master-core", "4", "--nic-rx-core", "5", "--nic-num-bufs", "4096",
    )  # fmt: skip
    assert (args.nic_pcie, args.nic_config, args.nic_master_core, args.nic_rx_core) == (
        "0001:02:00.0",
        "nic.yaml",
        4,
        5,
    )
    assert args.nic_num_bufs == 4096


@pytest.mark.parametrize(
    ("ingest", "batch", "ring"),
    [
        ("udp", 16384, 65536),
        ("daqiri-host", 16384, 65536),
        ("gpudirect", 131072, 1 << 20),
        ("replay", 131072, 1 << 20),
    ],
)
def test_batch_and_ring_defaults_follow_the_front_end(ingest: str, batch: int, ring: int) -> None:
    args = equity_args("--ingest", ingest, "--pcap", "day.pcap", "--tape", "tape")
    assert (args.batch_messages, args.ring_size) == (batch, ring)


def test_explicit_batch_and_ring_are_kept() -> None:
    args = equity_args("--ingest", "udp", "--batch-messages", "512", "--ring-size", "1024")
    assert (args.batch_messages, args.ring_size) == (512, 1024)


def test_defaults() -> None:
    args = equity_args()
    assert args.ingest == "udp" and args.analytics_backend == "compare"
    assert (args.host, args.port, args.udp_host, args.udp_port) == (
        "127.0.0.1",
        8080,
        "127.0.0.1",
        26400,
    )
    assert (args.nic_pcie, args.nic_master_core, args.nic_rx_core, args.nic_num_bufs) == (
        "0004:03:00.0",
        3,
        9,
        114688,
    )
    assert (args.bar_seconds, args.window_seconds, args.stale_seconds, args.min_bars) == (
        5,
        300,
        30,
        20,
    )
    assert args.max_spread_bps == 100.0 and args.interval == 0.05
    assert (args.risk_poll_us, args.risk_deadline_us, args.expect_windows) == (250, 0, 0)
    assert args.topics is False and args.topics_catalog == "/tmp/holoscan-finance/endpoints.json"


@pytest.mark.parametrize("ingest", ["gpudirect", "replay"])
def test_gpu_front_ends_refuse_cpu_analytics(
    capsys: pytest.CaptureFixture[str], ingest: str
) -> None:
    err = usage_error(
        capsys, "--ingest", ingest, "--analytics-backend", "cpu", "--pcap", "p", "--tape", "t"
    )
    assert "runs GPU analytics" in err


@pytest.mark.parametrize("missing", ["pcap", "tape"])
def test_replay_needs_a_capture_and_its_tape(
    capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    given = {"pcap": "day.pcap", "tape": "tape"}
    del given[missing]
    flags = [x for name, value in given.items() for x in (f"--{name}", value)]
    assert f"--ingest replay needs --{missing}" in usage_error(capsys, "--ingest", "replay", *flags)


def test_plan_only_replay_gets_placeholder_paths() -> None:
    args = equity_args("--ingest", "replay", "--plan-only")
    assert (args.pcap, args.tape) == ("none", "none")  # the source's setup() rejects ""


def test_exit_when_drained_needs_a_finite_replay(capsys: pytest.CaptureFixture[str]) -> None:
    assert "--exit-when-drained needs --ingest replay" in usage_error(
        capsys, "--ingest", "udp", "--exit-when-drained"
    )
    replay = ("--ingest", "replay", "--pcap", "p", "--tape", "t", "--exit-when-drained")
    assert "--replay-loops 0 never ends" in usage_error(capsys, *replay, "--replay-loops", "0")
    assert equity_args(*replay, "--replay-loops", "2").exit_when_drained


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--risk-poll-us", "50"], "--risk-poll-us"),
        (["--risk-poll-us", "200000"], "--risk-poll-us"),
        (["--interval", "0"], "--interval"),
        (["--bar-seconds", "7"], "invalid bar, window"),  # bars tile the window
        (["--ingest-cpus", "4-x"], "invalid CPU list"),
        (["--ingest", "nic"], "invalid choice"),
    ],
)
def test_invalid_values_are_usage_errors(
    capsys: pytest.CaptureFixture[str], flags: list[str], message: str
) -> None:
    assert message in usage_error(capsys, *flags)


def test_risk_poll_fallback_is_accepted() -> None:
    assert equity_args("--risk-poll-us", "5000").risk_poll_us == 5000


def test_nic_params_never_pass_none_for_a_str_param() -> None:
    args = equity_args("--ingest", "daqiri-host")
    assert args.nic_config == ""
    args.nic_config = None  # a caller-built Namespace
    cpus = CpuSets(*(frozenset({0, 1, 2}),) * 6)
    params = cli.nic_params(args, cpus, gpudirect=False)
    assert params == dict(
        pcie_address="0004:03:00.0", daqiri_config="", master_core=3, rx_core=9, init_cpus=[0, 1, 2]
    )
    assert all(value is not None for value in params.values())


def test_nic_params_add_num_bufs_for_gpudirect_and_widen_init_to_every_cpu() -> None:
    args = equity_args(
        "--ingest", "gpudirect", "--nic-config", "nic.yaml", "--nic-num-bufs", "8192"
    )
    everything, app = frozenset({0, 1, 2, 3}), frozenset({0, 1})
    cpus = CpuSets(
        all=everything, reserved=everything - app, app=app, ingest=app, helper=app, worker=app
    )
    params = cli.nic_params(args, cpus, gpudirect=True)
    assert params["num_bufs"] == 8192 and params["daqiri_config"] == "nic.yaml"
    assert params["init_cpus"] == [0, 1, 2, 3]  # daqiri_init runs on the full mask


def test_bar_settings_come_from_the_bar_flags() -> None:
    args = equity_args("--bar-seconds", "10", "--window-seconds", "600", "--max-spread-bps", "50")
    settings = cli.bar_settings(args)
    assert settings.as_params() == dict(
        bar_seconds=10, window_seconds=600, stale_seconds=30, min_bars=20, max_spread_bps=50.0
    )
    assert settings.window_bars == 60


def test_shared_flag_groups_build_a_parser_of_their_own() -> None:
    p = argparse.ArgumentParser()
    cli.add_bar_flags(p)
    cli.add_placement_flags(p)
    cli.add_plan_flags(p)
    args = p.parse_args(
        ["--ingest-cpus", "4-5", "--ingest-busy-poll", "--plan-only", "--plan-json", "x"]
    )
    assert (args.ingest_cpus, args.ingest_busy_poll, args.plan_only, args.plan_json) == (
        "4-5",
        True,
        True,
        "x",
    )
    assert args.bar_seconds == 5 and args.max_spread_bps == 100.0
