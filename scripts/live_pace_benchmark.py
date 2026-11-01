"""Whether A0 served live keeps up with real time on this machine: a measurement, never a CI check (a shared runner
under load falls behind, and that is no fault of the session's).

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/live_pace_benchmark.py <work dir>

It saves A0 (``tests/script_a0.py``) into ``<work dir>``, serves it with ``mime serve`` in a process of its own, a
client in this one, twice: unpaced for 4 s of simulated time, and paced at real time for 2 s. Each is timed from the
first frame after the first chunk (which compiles, a few seconds) to the last: the real-time factor is the simulated
time over the wall time between them. It prints JSON with the machine (its processor, its cores, JAX's backend) and
both factors, every frame's order checked. Run it alone (on the maintainer's machine, under the heavy lock), and record
its numbers with the machine in ``docs/live.md``. Not collected by pytest."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def processor() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine() -> dict[str, Any]:
    """The machine a measurement ran on: its processor, its cores, its platform, JAX's backend and Python."""
    import jax

    return {"processor": processor(), "cores": os.cpu_count(), "platform": platform.platform(),
            "jax_backend": jax.default_backend(), "python": platform.python_version()}


def served(folder: Path, work: Path, name: str, seconds: float, *extra: str) -> dict[str, Any]:
    from mime.live import FrameOrder, LiveClient

    ready = work / f"{name}.ready.json"
    ready.unlink(missing_ok=True)
    process = subprocess.Popen([sys.executable, "-m", "mime.cli", "serve", str(folder), "--commands",
                                "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(ready),
                                "--out", str(work / f"{name}-run"), *extra],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while not ready.exists() or not ready.read_text():
            if process.poll() is not None:
                raise RuntimeError(f"mime serve exited {process.returncode}")
            time.sleep(0.05)
        endpoints = json.loads(ready.read_text())
        live = LiveClient(endpoints["commands"], endpoints["frames"])
        live.request("start")
        frames, arrived = [], []
        while not frames or frames[-1].time < seconds:
            frames.append(live.frame())
            arrived.append(time.monotonic())
        live.request("stop")
        process.wait(timeout=600)
        order = FrameOrder()
        problems = [problem for problem in (order.see(held) for held in frames) if problem]
        simulated, wall = frames[-1].time - frames[1].time, arrived[-1] - arrived[1]
        return {"simulated_s": round(simulated, 4), "wall_s": round(wall, 4),
                "real_time_factor": round(simulated / wall, 3), "frames": len(frames), "order_problems": problems}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def main() -> int:
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "tests"))
    from script_a0 import a0

    folder = work / "script_a0"
    if not (folder / "mime.lock").is_file():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a0().save(folder)
    result = {"machine": machine(), "unpaced": served(folder, work, "unpaced", 4.0, "--unpaced"),
              "paced_at_1": served(folder, work, "paced", 2.0)}
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
