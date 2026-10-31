"""M3 route 2, stage 2 on Modal: ``m3_stage2.py``'s two programs in one function call on a 32-core CPU container (the
kernel's wall side runs on the host; no GPU): ``--pairs`` (every pair's exact block and the table's read, 32 spawned workers,
one thread each; a deadline saves a partial state to the Volume) and, if it completed, ``--solve`` (the matrices, the
solves, E's statistics, the plants, on all threads). Results go to the Volume ``mime-m3-stage2`` under ``stage2/``
(``started.json`` first, ``done.json`` or ``failed.json`` last, the Volume committed after each write), so they do not
depend on the laptop. ``modal`` is imported only here.

**Cost safety (server-side limits).** App ``mime-m3-stage2``; one function, ``timeout`` 1800 s; no retries; no
``min_containers`` or ``keep_warm``; ``max_containers=1``; 32 cores and 80 GiB (at most about $1.10 for the 1800 s).
Launched by ``m3s2_launch.py`` (``modal run --detach``).

    MIME_M3_WHEEL=<wheel> MIME_M3_REQUIREMENTS=<hashed requirements> MIME_M3_TABLE=<unit_tube_R0.997.npz> \
        modal run --detach scripts/sim2real/m3_stage2_modal.py --out DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE
ENV = {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets"}
WHEEL = os.environ.get("MIME_M3_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_M3_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_M3_TABLE", "")
APP_NAME = "mime-m3-stage2"
VOLUME = "mime-m3-stage2"
PREFIX = "stage2"
WORKERS = 32
PAIRS_DEADLINE_S = 1300

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    image = (image.uv_pip_install(requirements=[REQUIREMENTS], extra_options="--require-hashes --no-deps")
             .add_local_file(WHEEL, f"/w/{Path(WHEEL).name}", copy=True)
             .run_commands(f"pip install --no-deps --no-compile /w/{Path(WHEEL).name}")
             .add_local_file(TABLE, "/assets/unit_tube_R0.997.npz", copy=True)
             .env(ENV)
             .run_commands("mime assets fetch unit_tube_R0.997"))
    return (image.env(ENV)
            .add_local_file(str(ROOT / "tests" / "script_a.py"), "/repo/tests/script_a.py")
            .add_local_file(str(ROOT / "tests" / "differential" / "script_a_steady.py"),
                            "/repo/tests/differential/script_a_steady.py")
            .add_local_dir(str(ROOT / "scripts" / "tube_table"), "/repo/scripts/tube_table", ignore=["__pycache__"])
            .add_local_file(str(ROOT / "tests" / "tube_fourier_bessel.py"), "/repo/tests/tube_fourier_bessel.py")
            .add_local_dir(str(HERE), "/repo/scripts/sim2real", ignore=["__pycache__"]))


image = _image()


@app.function(image=image, cpu=32.0, memory=81920, max_containers=1, timeout=1800, retries=0,
              volumes={"/results": results})
def run_all() -> dict:
    import shutil
    import subprocess
    import sys
    import traceback

    folder = Path("/results", PREFIX)
    folder.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict) -> None:
        (folder / name).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
        results.commit()

    script = "/repo/scripts/sim2real/m3_stage2.py"
    env = dict(os.environ)
    save("started.json", {"started_unix": time.time(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "workers": WORKERS, "pairs_deadline_s": PAIRS_DEADLINE_S, "cpus": os.cpu_count()})
    log = open(work / "run.log", "w")
    try:
        resume = []
        if (folder / "partial.npz").exists():
            shutil.copy(folder / "partial.npz", work / "partial.npz")
            resume = ["--resume"]
        t0 = time.time()
        subprocess.run([sys.executable, "-W", "ignore", script, "--pairs", "--work", str(work), "--workers", str(WORKERS),
                        "--deadline-s", str(PAIRS_DEADLINE_S), *resume], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        pairs = json.loads((work / "pairs_summary.json").read_text())
        save("pairs_summary.json", pairs)
        if not pairs["complete"]:
            shutil.copy(work / "partial.npz", folder / "partial.npz")
            shutil.copy(work / "run.log", folder / "run.log")
            save("failed.json", {"reason": "pairs not complete at the deadline; partial.npz saved, relaunch resumes",
                                 "tasks_done": pairs["tasks_done"], "tasks": pairs["tasks"]})
            return pairs
        t1 = time.time()
        subprocess.run([sys.executable, "-W", "ignore", script, "--solve", "--work", str(work), "--out", str(folder)],
                       check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        shutil.copy(work / "run.log", folder / "run.log")
        out = json.loads((folder / "stage2_solve.json").read_text())
    except BaseException:
        log.flush()
        try:
            shutil.copy(work / "run.log", folder / "run.log")
        except OSError:
            pass
        save("failed.json", {"traceback": traceback.format_exc()})
        raise
    save("done.json", {"done_unix": time.time(), "pairs_seconds": t1 - t0, "solve_seconds": time.time() - t1})
    return out


@app.local_entrypoint()
def main(out: str) -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    found = run_all.remote()
    (target / "stage2_returned.json").write_text(json.dumps(found, indent=1, sort_keys=True, default=str) + "\n")
    print("returned", flush=True)
