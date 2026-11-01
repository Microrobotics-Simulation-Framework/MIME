"""How long ``mime serve`` takes to answer while A0's first chunk compiles, and what A0's first ``set`` costs: a
measurement, never a CI check (a loaded runner answers later, and that is no fault of the server's).

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/live_responsiveness.py <work dir>

It saves A0 (``tests/script_a0.py``, its controller called back) into ``<work dir>`` and serves it with ``mime serve``
in a process of its own, unpaced. One client sends a ``step`` of one chunk, the session's first, which compiles; a
second sends ``status`` after ``status`` until the step is answered, each timed from its send to its reply. Then a step
of one chunk, a ``set`` of A0's live input, and another step, each timed (before 7n-b the step after the first set
compiled A0's controller again). It prints JSON with the machine and the numbers. Run it alone (on the maintainer's
machine, under the heavy lock), and record its numbers with the machine in ``docs/live.md``. Not collected by pytest."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from live_pace_benchmark import ROOT, machine


def timed(live: Any, command: str, **arguments: Any) -> tuple[dict[str, Any], float]:
    began = time.perf_counter()
    answer = live.request(command, **arguments)
    return answer, time.perf_counter() - began


def measured(folder: Path, work: Path) -> dict[str, Any]:
    from mime.live import LiveClient

    ready = work / "responsiveness.ready.json"
    ready.unlink(missing_ok=True)
    process = subprocess.Popen([sys.executable, "-m", "mime.cli", "serve", str(folder), "--commands",
                                "tcp://127.0.0.1:0", "--frames", "tcp://127.0.0.1:0", "--ready-file", str(ready),
                                "--out", str(work / "responsiveness-run"), "--unpaced"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while not ready.exists() or not ready.read_text():
            if process.poll() is not None:
                raise RuntimeError(f"mime serve exited {process.returncode}")
            time.sleep(0.05)
        endpoints = json.loads(ready.read_text())
        stepper = LiveClient(endpoints["commands"], endpoints["frames"], timeout=600)
        asker = LiveClient(endpoints["commands"], endpoints["frames"], timeout=600)
        answers: list[tuple[dict[str, Any], float]] = []
        thread = threading.Thread(target=lambda: answers.append(timed(stepper, "step", chunks=1)))
        thread.start()
        compiling: list[float] = []
        while not answers:
            answer, took = timed(asker, "status")
            if answer["activity"] == "compiling":
                compiling.append(took)
            time.sleep(0.01)
        thread.join()
        plain = timed(stepper, "step", chunks=1)[1]
        set_took = timed(stepper, "set", values={"offset": 0.002})[1]
        after_set = timed(stepper, "step", chunks=1)[1]
        stepper.request("stop")
        process.wait(timeout=600)
        stepper.close()
        asker.close()
        return {"first_chunk_s": round(answers[0][1], 3), "statuses_while_compiling": len(compiling),
                "status_while_compiling_ms": {"median": round(1e3 * statistics.median(compiling), 2),
                                              "max": round(1e3 * max(compiling), 2)},
                "step_ms": round(1e3 * plain, 2), "set_ms": round(1e3 * set_took, 2),
                "step_after_first_set_ms": round(1e3 * after_set, 2)}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def main() -> int:
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "tests"))
    from script_a0 import a0

    from mime.units import ms

    folder = work / "script_a0_callback"
    if not (folder / "mime.lock").is_file():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a0(mode="callback", rate=ms(10)).save(folder)
    print(json.dumps({"machine": machine(), "a0_callback": measured(folder, work)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
