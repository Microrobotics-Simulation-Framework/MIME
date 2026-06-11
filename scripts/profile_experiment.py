#!/usr/bin/env python3
"""Profile a running MIME experiment and save a Perfetto trace.

The experiment runner (``mime.runner.server``) exposes a ``profile`` command
on its ZMQ REP socket (fit-up §8). This connects, requests a profile, and
writes the returned Chrome/Perfetto trace JSON to a file — open it by
drag-and-drop at https://ui.perfetto.dev.

Profiling runs MADDENING's profiler on the live graph; the runner snapshots
and restores graph state around it, so the running simulation is unaffected
(it pauses for the duration of the profile, then continues).

Usage::

    python scripts/profile_experiment.py            # 50 steps -> profile.json
    python scripts/profile_experiment.py --n-steps 200 --out run.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="runner host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5555,
                    help="runner REP port (default: 5555)")
    ap.add_argument("--n-steps", type=int, default=50,
                    help="benchmarked steps (default: 50)")
    ap.add_argument("--n-warmup", type=int, default=3,
                    help="warmup steps (default: 3)")
    ap.add_argument("--out", default="profile.json",
                    help="output trace file (default: profile.json)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds to wait for the reply (default: 180)")
    args = ap.parse_args()

    try:
        import zmq
    except ImportError:
        print("ERROR: pyzmq not installed (pip install pyzmq)", file=sys.stderr)
        return 1

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    sock.connect(f"tcp://{args.host}:{args.port}")

    sock.send_string(json.dumps({
        "command": "profile",
        "n_steps": args.n_steps,
        "n_warmup": args.n_warmup,
    }))
    print(f"Requested a {args.n_steps}-step profile from "
          f"{args.host}:{args.port} — waiting (up to {args.timeout:.0f}s)...")
    try:
        reply = json.loads(sock.recv_string())
    except zmq.Again:
        print(f"ERROR: no reply within {args.timeout:.0f}s — is the runner up?",
              file=sys.stderr)
        return 1
    finally:
        sock.close()
        ctx.term()

    if reply.get("status") != "ok":
        print(f"ERROR: runner returned {reply}", file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        json.dump(reply["report"], f)
    print(f"Perfetto trace written to {args.out} — "
          f"open it at https://ui.perfetto.dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
