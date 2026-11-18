# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container dependency/hardware preflight and host-side HTTP readiness gate.

``check`` runs in the receiver image (Holoscan SDK 5, DAQIRI, the mounted checkout); ``wait``
runs on the host with the system Python and imports nothing from the checkout.
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def check(args):
    import holoscan as hs

    if not hasattr(hs, "compile"):
        raise RuntimeError(f"{hs!r} is not Holoscan SDK 5 (no holoscan.compile)")
    for name in (
        "cupy",
        "numpy",
        "scipy",
        "threadpoolctl",
        "fastapi",
        "uvicorn",
        "holoscan_finance.operators.gpu_sbe_source",
        "holoscan_finance.dashboard.server",
    ):
        importlib.import_module(name)
    import holoscan_finance
    from holoscan_finance.ingest.daqiri import require_daqiri

    mode = args.ingest
    require_daqiri(gpu=mode == "gpudirect")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy CUDA device is unavailable")
    source = holoscan_finance.__file__
    repo = Path(os.environ["HOLOSCAN_FINANCE_REPO"]).resolve()
    if not Path(source).resolve().is_relative_to(repo):
        raise RuntimeError(f"finance imports must resolve to the mounted branch, got {source}")
    vendor = Path("/sys/bus/pci/devices") / args.pcie / "vendor"
    if not vendor.exists() or vendor.read_text().strip() != "0x15b3":
        raise RuntimeError(f"no Mellanox/NVIDIA network device at PCIe {args.pcie}")
    cores = os.sched_getaffinity(0)
    if args.master_core == args.rx_core or not {args.master_core, args.rx_core} <= cores:
        raise RuntimeError(f"choose distinct DAQIRI master/RX cores from {sorted(cores)}")
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(meminfo["HugePages_Free"].strip()) == 0:
        raise RuntimeError("no free hugepages; provision them using the DAQIRI system setup guide")
    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"Preflight passed ({mode}): {source}; analytics GPU {gpu}")


def wait_ready(args):
    deadline = time.monotonic() + args.timeout
    last_error = "HTTP endpoint unavailable"
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{args.url}/snapshot", timeout=2) as response:
                state = json.load(response)
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = str(exc)
        else:
            if state.get("error") or state.get("graph_status") == "stopped":
                raise RuntimeError(f"receiver graph failed: {state.get('error') or 'stopped'}")
            network = state.get("network", {})
            if network.get("error"):
                raise RuntimeError(f"DAQIRI RX failed: {network['error']}")
            if network.get("ingest") in ("daqiri-host", "gpudirect") and network.get("ready"):
                payload = json.dumps({"speed": args.speed, "rate": args.rate}).encode()
                request = Request(
                    f"{args.url}/feedrate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    json.load(response)
                print("DAQIRI RX ready; replay pacing configured")
                return
            last_error = f"graph={state.get('graph_status')}, network={network.get('status')}"
        time.sleep(0.25)
    raise RuntimeError(f"DAQIRI RX did not become ready in {args.timeout}s: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--pcie", required=True)
    check_parser.add_argument("--master-core", type=int, required=True)
    check_parser.add_argument("--rx-core", type=int, required=True)
    check_parser.add_argument("--ingest", required=True, choices=("daqiri-host", "gpudirect"))
    wait_parser = sub.add_parser("wait")
    wait_parser.add_argument("--url", required=True)
    wait_parser.add_argument("--timeout", type=float, default=90)
    wait_parser.add_argument("--speed", type=float, default=3)
    wait_parser.add_argument("--rate", type=float, default=5000)
    args = parser.parse_args()
    try:
        (check if args.command == "check" else wait_ready)(args)
    except (ImportError, OSError, RuntimeError) as exc:
        parser.exit(1, f"receiver startup failed: {exc}\n")


if __name__ == "__main__":
    main()
