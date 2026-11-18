# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay controller for the dashboard's full-day button: tcpreplay synthetic data at a given rate.

The dashboard POSTs /replay/fullday to the ingest app, which pauses its paced sender and
forwards ``{"pps": N}`` here. Each start registers a new replay generation (the receiver only
accepts a fresh UDP source port per generation), then tcpreplays the capture
rewritten to that port. One rewritten copy is prepared ahead in the background --
after each replay finishes, never during one
(rewriting competes with the receiver for CPU and memory bandwidth).

Runs privileged with host networking (tcpreplay writes raw frames on the NIC):

    python3 replay_control.py --pcap data/synth/day.pcap --base-port <PORT> --packets <PACKETS> \\
        --nic enP4p3s0f0np0 --tcpreplay-dir ~/tools/root/usr/bin --app http://127.0.0.1:8080 \\
        --tape data/synth/tape --work <WORK_DIR>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Controller:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, LD_LIBRARY_PATH=a.tcpreplay_libs)
        self.lock = threading.Lock()
        self.next_port = a.port_start
        self.spare = None  # (port, path) ready to replay
        self.proc = None
        self.run = {}
        self._prepare()

    def _off_receiver_cores(self):
        """Keep the sender off the receiver's DAQIRI master/RX cores (same host)."""
        cores = set(os.sched_getaffinity(0)) - set(self.a.avoid_cores)
        if cores:
            os.sched_setaffinity(0, cores)

    def _rewrite(self):
        port = self.next_port
        self.next_port += 1
        out = self.work / f"run_{port}.pcap"
        subprocess.run(
            [f"{self.a.tcpreplay_dir}/tcprewrite", f"--portmap={self.a.base_port}:{port}", "--fixcsum",
             "-i", self.a.pcap, "-o", str(out)],
            check=True, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        with self.lock:
            self.spare = (port, out)

    def _prepare(self):
        self.preparing = threading.Thread(target=self._rewrite, daemon=True)
        self.preparing.start()

    def stop(self):
        with self.lock:
            proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)

    def start(self, pps):
        if not 1 <= pps <= 250_000:
            raise ValueError("pps must be 1..250000")
        self.stop()
        self.preparing.join()
        with self.lock:
            port, path = self.spare
            self.spare = None
        for old in self.work.glob("run_*.pcap"):  # this controller's own rewritten copies
            if old != path:
                old.unlink(missing_ok=True)
        from holoscan_finance.tools.bench_replay import register

        register(port, self.a.app, self.a.tape)
        cmd = [f"{self.a.tcpreplay_dir}/tcpreplay", "-i", self.a.nic, f"--pps={pps}", str(path)]
        if pps >= 100_000:
            cmd.insert(1, "-K")  # preload: one core cannot read and send >100k pps otherwise
        log = open(self.work / "tcpreplay.log", "w")
        with self.lock:
            self.proc = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                                         preexec_fn=self._off_receiver_cores)  # fmt: skip
            self.run = dict(pps=pps, sender_port=port, started=time.time(), packets=self.a.packets)
        proc = self.proc
        self.preparing = threading.Thread(
            target=lambda: (proc.wait(), self._rewrite()), daemon=True
        )
        self.preparing.start()
        return dict(self.run)

    def status(self):
        with self.lock:
            proc, run = self.proc, dict(self.run)
        run["running"] = bool(proc and proc.poll() is None)
        log = self.work / "tcpreplay.log"
        if log.exists():
            run["tcpreplay"] = log.read_text()[-400:]
        return run


def serve(a):
    ctl = Controller(a)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200, ctl.status())

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 <= size <= 4096:  # bodies are {"pps": N}; never read an unbounded one
                    raise ValueError("request body too large")
                body = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/start":
                    self._reply(200, ctl.start(int(body["pps"])))
                elif self.path == "/stop":
                    ctl.stop()
                    self._reply(200, ctl.status())
                else:
                    self._reply(404, {"error": "unknown path"})
            except Exception as exc:
                self._reply(422, {"error": str(exc)})

        def log_message(self, *args):
            pass

    print(f"replay controller on 127.0.0.1:{a.port}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    finally:
        ctl.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pcap", required=True)
    p.add_argument("--base-port", type=int, required=True, help="UDP source port in the capture")
    p.add_argument("--packets", type=int, help="packets in the capture (reported to the dashboard)")
    p.add_argument(
        "--avoid-cores",
        type=lambda v: [int(c) for c in v.split(",") if c],
        default=[],
        help="comma-separated CPU cores tcpreplay must not use (the receiver's DAQIRI cores)",
    )
    p.add_argument("--nic", default="enP4p3s0f0np0")
    p.add_argument("--tcpreplay-dir", required=True)
    p.add_argument("--tcpreplay-libs", default="")
    p.add_argument("--app", default="http://127.0.0.1:8080")
    p.add_argument("--tape", default="/data/tape")
    p.add_argument("--work", default="/tmp/replay_control")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--port-start", type=int, default=45000)
    serve(p.parse_args())


if __name__ == "__main__":
    main()
