# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment orchestration checks without Docker, hugepages or NIC mutations."""

import importlib.util
import io
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy/igx_thor"
spec = importlib.util.spec_from_file_location("ano_runtime", DEPLOY / "ano_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def dry_run(**overrides):
    result = subprocess.run(
        ["bash", str(DEPLOY / "igx_ano_run.sh"), "--dry-run"],
        env={**os.environ, "LOOPBACK": "0", "START_FEED": "0", **overrides},
        text=True,
        capture_output=True,
        check=True,
    )
    # Keep docker/python3 commands; drops shell-function echoes such as gpu_pm_set.
    return [
        shlex.split(line)
        for line in result.stdout.splitlines()
        if line.startswith(("docker ", "python3 "))
    ]


def test_external_receiver_runs_stream_without_local_sender_or_loopback():
    commands = dry_run(ANO_PCIE="0000:81:00.0", GPU_UUID="GPU-test", WEB_PORT="8091")
    receiver = next(c for c in commands if "--name" in c)
    assert receiver[receiver.index("-m") + 1] == "holoscan.equities.stream"
    assert receiver[receiver.index("--ingest") + 1] == "ano"
    assert receiver[receiver.index("--ano-pcie") + 1] == "0000:81:00.0"
    assert "CUDA_VISIBLE_DEVICES=GPU-test" in receiver
    assert len(commands) == 3  # dependency check, receiver, HTTP readiness gate
    assert commands[-1][commands[-1].index("--url") + 1] == "http://127.0.0.1:8091"


def test_gpu_ingest_receiver_gets_gpudirect_mode_and_persistent_kernel_cache(tmp_path):
    cache = tmp_path / "kernels"
    commands = dry_run(INGEST="gpu", GPU_UUID="GPU-test", KERNEL_CACHE_DIR=str(cache))
    receiver = next(c for c in commands if "--name" in c)
    assert receiver[receiver.index("--ingest") + 1] == "gpu"
    assert f"type=bind,src={cache},dst=/kernel-cache" in receiver
    assert "CUPY_CACHE_DIR=/kernel-cache" in receiver
    assert "ano.ingest=gpu" in receiver


def test_ingest_mode_is_validated():
    result = subprocess.run(
        ["bash", str(DEPLOY / "igx_ano_run.sh"), "--dry-run"],
        env={**os.environ, "LOOPBACK": "0", "START_FEED": "0", "INGEST": "sockets"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "INGEST must be ano or gpu" in result.stderr


def test_local_sender_uses_prepared_quote_preserving_tape_and_quotes_path():
    commands = dry_run(
        START_FEED="1",
        FEED_HOST="10.0.0.2",
        TAPE_DIR="/licensed tape/day one",
        UDP_PORT="27500",
        SPEED="0",
        RATE="12000",
    )
    sender = commands[-1]
    assert "holoscan.equities.replay" in sender
    assert "type=bind,src=/licensed tape/day one,dst=/data/tape,readonly" in sender
    assert sender[sender.index("--data") + 1] == "/data/tape"
    assert sender[sender.index("--port") + 1] == "27500"
    assert sender[sender.index("--loops") + 1] == "0"
    gate = commands[-2]
    assert gate[gate.index("--speed") + 1] == "0"
    assert gate[gate.index("--rate") + 1] == "12000"


def test_loopback_order_is_receiver_then_readiness_then_nic_then_sender():
    result = subprocess.run(
        ["bash", str(DEPLOY / "igx_ano_run.sh"), "--dry-run"],
        env={**os.environ, "LOOPBACK": "1", "START_FEED": "1"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert result.index("holoscan.equities.stream") < result.index("ano_runtime.py wait")
    assert result.index("ano_runtime.py wait") < result.index("--privileged")
    assert result.index("--privileged") < result.index("holoscan.equities.replay")
    assert "--host 192.168.100.2" in result


def response(payload):
    return io.BytesIO(json.dumps(payload).encode())


def test_readiness_waits_for_ano_and_updates_real_feed_controls(monkeypatch):
    states = iter(
        [
            {"graph_status": "running", "network": {"ingest": "async", "ready": True}},
            {"graph_status": "running", "network": {"ingest": "ano", "ready": False}},
            {"graph_status": "running", "network": {"ingest": "ano", "ready": True}},
        ]
    )
    posted = []

    def open_url(request, **kwargs):
        if isinstance(request, str):
            return response(next(states))
        posted.append(json.loads(request.data))
        return response(posted[-1])

    monkeypatch.setattr(runtime, "urlopen", open_url)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    runtime.wait_ready(SimpleNamespace(url="http://test", timeout=1, speed=0, rate=12000))
    assert posted == [{"speed": 0, "rate": 12000}]


@pytest.mark.parametrize(
    "state",
    [
        {"graph_status": "stopped", "error": "init failed"},
        {"graph_status": "running", "network": {"error": "DAQIRI RX failed"}},
    ],
)
def test_readiness_fails_on_graph_or_drain_error(monkeypatch, state):
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: response(state))
    with pytest.raises(RuntimeError, match="failed"):
        runtime.wait_ready(SimpleNamespace(url="http://test", timeout=1, speed=3, rate=5000))


def test_unreachable_receiver_times_out(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(runtime, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="did not become ready"):
        runtime.wait_ready(SimpleNamespace(url="http://test", timeout=0.01, speed=3, rate=5000))


@pytest.fixture
def mock_docker(tmp_path):
    path = tmp_path / "docker"
    path.write_text("""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:2] == ["container", "inspect"]:
    sys.exit(0 if os.environ.get("EXISTING") else 1)
if "check" in args:
    sys.exit(1)
sys.exit(0)
""")
    path.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "DOCKER_CALLS": str(log),
        "LOOPBACK": "0",
        "START_FEED": "0",
    }
    return env, log


@pytest.mark.parametrize("existing", [False, True])
def test_failed_preflight_or_name_collision_never_starts_or_removes_containers(
    mock_docker, existing
):
    env, log = mock_docker
    if existing:
        env["EXISTING"] = "1"
    result = subprocess.run(["bash", str(DEPLOY / "igx_ano_run.sh")], env=env, capture_output=True)
    assert result.returncode != 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any("--name" in call or call[0] in ("rm", "stop") for call in calls)


def test_readiness_failure_rolls_back_only_created_receiver(tmp_path):
    """Exercise the actual shell trap/teardown, with an unavailable HTTP receiver."""
    log = tmp_path / "calls.jsonl"
    state = tmp_path / "container.json"
    docker = tmp_path / "docker"
    docker.write_text("""#!/usr/bin/python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ["CONTAINER_STATE"])
with open(os.environ["DOCKER_CALLS"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[:2] == ["container", "inspect"]:
    sys.exit(0 if state.exists() and args[-1] == "fin-equities-ano" else 1)
if args[0] == "run" and "--name" in args:
    labels = dict(args[i+1].split("=", 1) for i, a in enumerate(args) if a == "--label")
    state.write_text(json.dumps(labels))
if args[0] == "inspect":
    fmt = args[args.index("--format") + 1]
    if fmt == "{{.Config.Image}}":
        print("holoscan-finance:ano")
    else:
        print(json.loads(state.read_text()).get(fmt.split('"')[1], ""))
if args[0] == "rm":
    state.unlink()
sys.exit(0)
""")
    docker.chmod(0o755)
    # READY_TIMEOUT=0: the readiness gate gives up before any HTTP call.
    result = subprocess.run(
        ["bash", str(DEPLOY / "igx_ano_run.sh")],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "DOCKER_CALLS": str(log),
            "CONTAINER_STATE": str(state),
            "LOOPBACK": "0",
            "START_FEED": "0",
            "READY_TIMEOUT": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "did not become ready" in result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c for c in calls if c[0] == "stop"] == [["stop", "--time", "15", "fin-equities-ano"]]
    assert [c for c in calls if c[0] == "rm"] == [["rm", "fin-equities-ano"]]
    assert not state.exists()
    assert not any("--privileged" in c or "holoscan.equities.replay" in " ".join(c) for c in calls)


@pytest.mark.parametrize("owner", ["nyq-equities", "another-application"])
def test_teardown_uses_recorded_nic_settings_and_refuses_unrelated_containers(tmp_path, owner):
    log = tmp_path / "calls.jsonl"
    labels = {
        "com.nvidia.holoscan-finance.demo": owner,
        "ano.loopback": "1",
        "ano.nm-managed": "yes",
        "ano.feed": "saved-feed",
        "ano.pcie": "0000:82:00.0",
        "ano.nic": "saved-nic",
        "ano.cidr": "10.20.0.1/24",
        "ano.peer": "10.20.0.2",
        "ano.mst-image": "saved-mst:image",
        "ano.link-up": "0",
        "ano.replay": "saved-replay",
        "ano.irq-banned": "1,5",
        "ano.gpu-pm": "0000:01:00.0",
        "ano.fwupd-timer": "active",
    }
    docker = tmp_path / "docker"
    docker.write_text("""#!/usr/bin/python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "inspect":
    fmt = args[args.index("--format") + 1]
    if fmt == "{{.Config.Image}}":
        print("saved-app:image")
    else:  # like docker's index: a missing label prints an empty line
        print(json.loads(os.environ["SAVED_LABELS"]).get(fmt.split('"')[1], ""))
sys.exit(0)
""")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(DEPLOY / "igx_ano_down.sh")],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "DOCKER_CALLS": str(log),
            "SAVED_LABELS": json.dumps(labels),
            "ANO_PCIE": "wrong-current-pcie",
            "ANO_NIC": "wrong-current-nic",
            "APP_NAME": "saved-app",
        },
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    if owner != "nyq-equities":
        assert result.returncode != 0
        assert not any(c[0] in ("run", "rm", "stop") for c in calls)
        return
    assert result.returncode == 0, result.stderr
    mutations = [c for c in calls if c[0] in ("run", "rm", "stop")]
    assert mutations[0] == ["rm", "-f", "saved-feed"]
    assert mutations[1] == ["rm", "-f", "saved-replay"]
    assert mutations[2] == ["stop", "--time", "15", "saved-app"]
    cleanup = mutations[3]
    assert "saved-mst:image" in cleanup
    assert cleanup[-8:] == [
        "ano-loopback",
        "down",
        "0000:82:00.0",
        "saved-nic",
        "10.20.0.1/24",
        "10.20.0.2",
        "0",
        "yes",
    ]
    irq, gpu_pm = mutations[4], mutations[5]
    assert "--privileged" in irq and "--pid=host" in irq and "saved-app:image" in irq
    assert irq[-3:] == ["/irq_isolate.py", "set", "1,5"]
    assert "--privileged" in gpu_pm and "saved-app:image" in gpu_pm
    assert gpu_pm[-2:] == ["auto", "0000:01:00.0"]
    fwupd = mutations[6]
    assert "--privileged" in fwupd and "saved-app:image" in fwupd
    assert fwupd[-3:] == ["systemctl", "start", "fwupd-refresh.timer"]
    assert mutations[7] == ["rm", "saved-app"]
