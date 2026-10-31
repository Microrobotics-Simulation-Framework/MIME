"""M3 in another tube (register section 6 M3; B4a with review 2 N22 and the light check's F7): M3's two parts at that
tube's floored pose, as they ran at A on the fixed surface (screw 0.2.0, closed, 0.178 mm, blob 0.5, margin 0.1 mm), on
Modal. ``modal`` is imported only here.

- ``rerun`` (one A100-40GB, float64 at the highest matmul precision): ``m3_wall_read``'s ``reference`` (the nominal,
  the floor, the x1.02 control, the pole scan, the construction checks), ``noise`` (model 1) and ``smooth`` (model 2),
  50 draws each at the eight phases, the same seeds as at A. Every solve's reading is appended to the Volume as it is
  found (``solves.jsonl``) and a background thread commits the Volume as often as it can, so each solve is kept within
  a commit's time of its finishing; each stage's JSON is written as it finishes, and a later call skips finished stages.
- ``stage2`` (32 cores, 80 GiB, no GPU): route 2's ``m3_stage2.py --pairs`` (the exact error at every pair at phase 0,
  its truncation's readable radius this tube's) and ``--solve`` (propagated through the solve, the plumbing over every
  entry, the plants). The pairs program writes its partial state every 600 s, which this function copies to the Volume;
  a call (or a preempted container's restart) resumes from it.

**Cost safety.** ``modal run`` (never detached), so the app stops when the local process does; each function's
``timeout`` from ``MIME_M3T_RERUN_TIMEOUT_S`` and ``MIME_M3T_STAGE2_TIMEOUT_S`` (required, set by the caller from the
budget left); no retries; ``max_containers=1``; no ``min_containers`` or ``keep_warm``.

    MIME_A0F_WHEEL=<wheel> MIME_M3T_RERUN_TIMEOUT_S=1500 MIME_M3T_STAGE2_TIMEOUT_S=3600 \
        modal run scripts/sim2real/m3_tubes_modal.py --tube-mm 2.38125 --rho-max 0.9578 --parts rerun,stage2 --out DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE
ENV = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets"}
WHEEL = os.environ.get("MIME_A0F_WHEEL", "")
RERUN_TIMEOUT_S = int(os.environ.get("MIME_M3T_RERUN_TIMEOUT_S", "0"))
STAGE2_TIMEOUT_S = int(os.environ.get("MIME_M3T_STAGE2_TIMEOUT_S", "0"))
APP_NAME = "mime-m3-tubes"
VOLUME = "mime-m3-tubes-results"
WORKERS = 32
#: What the solve program needs after the pairs (A's full solve with its plants took about 15 minutes on 32 cores).
SOLVE_RESERVE_S = 1200
MARGIN_S = 120
#: How often the pairs program writes its partial state (``MIME_M3T_CHECKPOINT_S``, 600 s by default): the 3/8" tube's
#: CPU container was preempted three times within ten minutes, each before its first 600 s checkpoint.
CHECKPOINT_S = int(os.environ.get("MIME_M3T_CHECKPOINT_S", "600"))
RERUN_STAGES = ("reference", "noise", "smooth")

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _image(variant: str) -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    requirements = ROOT / "src" / "mime" / "data" / "env" / f"requirements-{variant}.txt"
    env = {**ENV, **({"JAX_PLATFORMS": "cpu"} if variant == "cpu" else {})}
    return (image.uv_pip_install(requirements=[str(requirements)], extra_options="--require-hashes --no-deps")
            .add_local_file(WHEEL, f"/w/{Path(WHEEL).name}", copy=True)
            .run_commands(f"pip install --no-deps --no-compile /w/{Path(WHEEL).name}")
            .env(env)
            .run_commands("mime assets fetch unit_tube_R0.997")
            .add_local_file(str(ROOT / "tests" / "script_a.py"), "/repo/tests/script_a.py")
            .add_local_file(str(ROOT / "tests" / "differential" / "script_a_steady.py"),
                            "/repo/tests/differential/script_a_steady.py")
            .add_local_file(str(ROOT / "tests" / "tube_fourier_bessel.py"), "/repo/tests/tube_fourier_bessel.py")
            .add_local_dir(str(ROOT / "scripts" / "tube_table"), "/repo/scripts/tube_table", ignore=["__pycache__"])
            .add_local_dir(str(HERE), "/repo/scripts/sim2real", ignore=["__pycache__", "*.pyc"]))


gpu_image = _image("cuda12")
cpu_image = _image("cpu")


def _plain(payload: object) -> object:
    return json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))


def _verify_table() -> str:
    """The image's snapshot changes the table's modification time, which MIME's asset record holds: hash it again in
    full, offline, and rewrite the record (as ``a0f_modal.py`` does)."""
    import subprocess

    return subprocess.run(["mime", "assets", "fetch", "unit_tube_R0.997"], capture_output=True, text=True,
                          env={**os.environ, "MIME_OFFLINE": "1"}, check=True).stdout.strip()


def _folder(tube_mm: float) -> Path:
    folder = Path("/results", f"tube_{tube_mm:.5f}mm")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_plain(payload), indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


@app.function(image=gpu_image, gpu="A100-40GB", cpu=4.0, memory=16384, max_containers=1,
              timeout=max(RERUN_TIMEOUT_S, 60), retries=0, volumes={"/results": results})
def rerun(tube_mm: float) -> dict:
    import sys
    import threading
    import traceback

    verified = _verify_table()
    sys.path.insert(0, "/repo/scripts/sim2real")
    import m3_wall_read as m3
    import operating_point as op

    results.reload()
    folder = _folder(tube_mm) / "rerun"
    folder.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def committer() -> None:
        while not stop.is_set():
            try:
                results.commit()
            except Exception:  # a failed commit is retried on the next turn
                pass
            stop.wait(0.2)

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    _write(folder / f"started_{int(time.time())}.json", {"tube_mm": tube_mm, "verified": verified,
                                                          "environment": op.environment()})
    log = (folder / "solves.jsonl").open("a")
    found: dict = {}
    try:
        for stage in RERUN_STAGES:
            target = folder / f"m3_{stage}.json"
            if target.exists():
                found[stage] = json.loads(target.read_text())
                continue

            def on_solve(row: dict, stage: str = stage) -> None:
                log.write(json.dumps({"stage": stage, "unix": time.time(), **row}) + "\n")
                log.flush()

            found[stage] = m3.run_stage(stage, tube_radius_mm=tube_mm, on_solve=on_solve)
            found[stage]["tube_radius_mm"] = tube_mm
            _write(target, found[stage])
    except BaseException:
        _write(folder / "failed.json", {"traceback": traceback.format_exc(), "unix": time.time()})
        raise
    finally:
        log.close()
        stop.set()
        thread.join()
        results.commit()
    _write(folder / "done.json", {"unix": time.time(), "stages": sorted(found)})
    results.commit()
    return _plain(found)


@app.function(image=cpu_image, cpu=float(WORKERS), memory=81920, max_containers=1,
              timeout=max(STAGE2_TIMEOUT_S, 60), retries=0, volumes={"/results": results})
def stage2(tube_mm: float, rho_max: float, timeout_s: int) -> dict:
    import shutil
    import subprocess
    import sys
    import traceback

    began = time.time()
    verified = _verify_table()
    results.reload()
    folder = _folder(tube_mm) / "stage2"
    folder.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    script = "/repo/scripts/sim2real/m3_stage2.py"
    env = {**os.environ, "M3_TUBE_RADIUS_MM": repr(tube_mm), "M3_RHO_MAX": repr(rho_max),
           "M3_CHECKPOINT_S": str(CHECKPOINT_S)}
    _write(folder / f"started_{int(began)}.json", {"tube_mm": tube_mm, "rho_max": rho_max, "workers": WORKERS,
                                                   "cpus": os.cpu_count(), "verified": verified})
    results.commit()
    log = open(work / "run.log", "a")
    try:
        if not (folder / "pairs_summary.json").exists() or not json.loads(
                (folder / "pairs_summary.json").read_text()).get("complete"):
            resume = []
            if (folder / "checkpoint.npz").exists():
                shutil.copy(folder / "checkpoint.npz", work / "partial.npz")
                resume = ["--resume"]
            deadline = max(60.0, timeout_s - SOLVE_RESERVE_S - MARGIN_S - (time.time() - began))
            process = subprocess.Popen([sys.executable, "-W", "ignore", script, "--pairs", "--work", str(work),
                                        "--workers", str(WORKERS), "--deadline-s", str(deadline), *resume],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            seen = None
            while process.poll() is None:
                time.sleep(15)
                checkpoint = work / "checkpoint.npz"
                if checkpoint.exists() and checkpoint.stat().st_mtime != seen:
                    seen = checkpoint.stat().st_mtime
                    shutil.copy(checkpoint, folder / "checkpoint.tmp.npz")
                    os.replace(folder / "checkpoint.tmp.npz", folder / "checkpoint.npz")
                    results.commit()
            if process.returncode != 0:
                raise RuntimeError(f"the pairs program exited with {process.returncode}")
            pairs = json.loads((work / "pairs_summary.json").read_text())
            pairs["tube_radius_mm"] = tube_mm
            if pairs["complete"]:  # the arrays the solve reads, kept for a later call
                for name in ("exact.npy", "table.npy"):
                    shutil.copy(work / name, folder / f"{name}.tmp")
                    os.replace(folder / f"{name}.tmp", folder / name)
            _write(folder / "pairs_summary.json", pairs)
            if not pairs["complete"]:
                shutil.copy(work / "partial.npz", folder / "checkpoint.npz")
                shutil.copy(work / "run.log", folder / "run.log")
                _write(folder / "failed.json", {"reason": "pairs not complete at the deadline; the checkpoint is kept",
                                                "tasks_done": pairs["tasks_done"], "tasks": pairs["tasks"]})
                results.commit()
                return _plain(pairs)
            results.commit()
        else:
            for name in ("exact.npy", "table.npy"):
                shutil.copy(folder / name, work / name)
        subprocess.run([sys.executable, "-W", "ignore", script, "--solve", "--work", str(work), "--out", str(folder)],
                       check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        shutil.copy(work / "run.log", folder / "run.log")
        out = json.loads((folder / "stage2_solve.json").read_text())
    except BaseException:
        log.flush()
        try:
            shutil.copy(work / "run.log", folder / "run.log")
        except OSError:
            pass
        _write(folder / "failed.json", {"traceback": traceback.format_exc(), "unix": time.time()})
        results.commit()
        raise
    for stale in ("checkpoint.npz", "failed.json", "exact.npy", "table.npy"):
        (folder / stale).unlink(missing_ok=True)
    _write(folder / "done.json", {"unix": time.time(), "seconds": time.time() - began})
    results.commit()
    return _plain({"pairs": json.loads((folder / "pairs_summary.json").read_text()), "solve": out})


@app.local_entrypoint()
def main(tube_mm: float, rho_max: float, parts: str, out: str) -> None:
    chosen = [p for p in parts.split(",") if p]
    if "rerun" in chosen and RERUN_TIMEOUT_S <= 0 or "stage2" in chosen and STAGE2_TIMEOUT_S <= 0 or not WHEEL:
        raise SystemExit("set MIME_A0F_WHEEL and each chosen part's timeout from the budget left")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    calls = []
    for part in chosen:
        began = time.time()
        try:
            found = rerun.remote(tube_mm) if part == "rerun" else stage2.remote(tube_mm, rho_max, STAGE2_TIMEOUT_S)
            (target / f"{part}_returned.json").write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
            status = "returned"
        except Exception as error:  # the Volume holds what was done; the caller reads it
            status = f"error: {error!r}"[:500]
        calls.append({"part": part, "local_start_unix": began, "local_seconds": time.time() - began, "status": status})
        (target / "local_calls.json").write_text(json.dumps(calls, indent=1) + "\n")
        print(part, status, flush=True)
