"""``bem_budget_grid.a_level`` on Modal: A's grid of the BEM error budget (item 2, with the regularised near-wall image
arm of item 4), float64 at the highest matmul precision on one A100-80GB. ``modal`` is imported only here and is no
dependency of MIME. The image is ``a0f_modal.py``'s.

**Cost safety.** Launched with ``modal run`` (never detached), so the app stops when this process does. One function;
``timeout`` from ``MIME_BB_TIMEOUT_S``, which the caller sets from the budget left at the measured rate (required); no
retries; ``max_containers=1``; no ``min_containers`` or ``keep_warm``. The function also stops itself, saving what it
has, at a deadline a margin inside its timeout.

**Checkpoints.** Every solve's result is written to the Volume ``mime-bem-budget-results`` under ``<tag>/`` and the
Volume committed before the next solve; a later call with the same tag resumes from what the Volume holds.

    MIME_BB_WHEEL=<wheel> MIME_BB_TIMEOUT_S=2400 modal run scripts/sim2real/bem_budget_modal.py \
        --tag grid --spacings 0.2,0.16,0.128,0.1024 --out DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets",
             # as in a0f_modal.py: the finest spacing needs 78.3 GB of the device, beyond JAX's default 75% share
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
WHEEL = os.environ.get("MIME_BB_WHEEL", "")
TIMEOUT_S = int(os.environ.get("MIME_BB_TIMEOUT_S", "0"))
APP_NAME = "mime-bem-budget"
VOLUME = "mime-bem-budget-results"
MARGIN_S = 120

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _image(variant: str = "cuda12") -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    requirements = ROOT / "src" / "mime" / "data" / "env" / f"requirements-{variant}.txt"
    env = {**PRECISION, **({"JAX_PLATFORMS": "cpu"} if variant == "cpu" else {})}
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


image = _image()
cpu_image = _image("cpu")
#: The CPU container: item 1 (the wall's pairs and solves, phases 0 then 6) and item 3's controls, in the brief's order.
CPU_CORES = 32
CPU_STEPS = (("wall", ["bem_budget_wall.py", "--pairs", "--phase", "0", "--workers", "30"], "pairs_phase0.json"),
             ("wall", ["bem_budget_wall.py", "--solve", "--phase", "0"], "solve_phase0.json"),
             ("controls", ["bem_budget_grid.py", "--spheres"], "sphere_0.1024.json"),
             ("controls", ["bem_budget_grid.py", "--fine"], "sphere_fine.json"),
             ("wall", ["bem_budget_wall.py", "--pairs", "--phase", "6", "--workers", "30"], "pairs_phase6.json"),
             ("wall", ["bem_budget_wall.py", "--solve", "--phase", "6"], "solve_phase6.json"))


def _plain(payload: object) -> object:
    return json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))


@app.function(image=image, gpu="A100-80GB", cpu=4.0, memory=16384, max_containers=1, timeout=max(TIMEOUT_S, 60),
              retries=0, volumes={"/results": results})
def run(tag: str, spacings: list[float], timeout_s: int, regimage: bool) -> dict:
    import subprocess
    import sys
    import traceback

    # a0f_modal.py's finding 4: the image's snapshot changes the table's mtime, which MIME's record holds; re-hashed
    # offline here
    verified = subprocess.run(["mime", "assets", "fetch", "unit_tube_R0.997"], capture_output=True, text=True,
                              env={**os.environ, "MIME_OFFLINE": "1"}, check=True).stdout
    sys.path.insert(0, "/repo/scripts/sim2real")
    import bem_budget_grid
    import operating_point as op

    began = time.time()
    deadline = began + timeout_s - MARGIN_S
    folder = Path("/results", tag)
    folder.mkdir(parents=True, exist_ok=True)
    results.reload()

    def held(name: str) -> dict | None:
        path = folder / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save(name: str, payload: dict) -> None:
        (folder / f"{name}.json").write_text(json.dumps(_plain(payload), indent=1, sort_keys=True) + "\n")
        results.commit()

    starts = (held("started") or {}).get("starts", [])
    save("started", {"starts": [*starts, {"unix": began, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
                     "spacings": spacings, "timeout_s": timeout_s, "regimage": regimage,
                     "environment": op.environment(), "table_verified": verified.strip()})
    found: dict = {}
    try:
        for spacing in spacings:
            name = f"a_level_{spacing:.4f}"
            previous = held(name)
            if previous is not None and (previous.get("complete") or previous.get("stopped", "deadline")
                                         != "deadline"):
                found[name] = previous
                continue
            found[name] = bem_budget_grid.a_level(spacing, lambda p, name=name: save(name, p), previous, deadline,
                                                  regimage)
            if found[name].get("stopped") == "deadline":
                break
    except BaseException:
        save("failed", {"traceback": traceback.format_exc(), "unix": time.time()})
        raise
    save("done", {"unix": time.time(), "seconds": time.time() - began, "names": sorted(found)})
    return _plain(found)


@app.function(image=cpu_image, cpu=float(CPU_CORES), memory=65536, max_containers=1, timeout=max(TIMEOUT_S, 60),
              retries=0, volumes={"/results": results})
def cpu_chain(tag: str, timeout_s: int) -> dict:
    """``CPU_STEPS`` in order, each a subprocess writing into the Volume, committed every 20 s while it runs and at
    its end; a step whose record says it is complete is skipped (a restart repeats only the step it was in)."""
    import subprocess
    import sys
    import threading

    subprocess.run(["mime", "assets", "fetch", "unit_tube_R0.997"], capture_output=True, text=True,
                   env={**os.environ, "MIME_OFFLINE": "1"}, check=True)
    began = time.time()
    results.reload()
    root = Path("/results", tag)
    log_path = root / "cpu_chain.log"
    root.mkdir(parents=True, exist_ok=True)
    done: dict = {}
    for folder, command, marker in CPU_STEPS:
        where = root / folder
        where.mkdir(parents=True, exist_ok=True)
        record = where / marker
        if record.exists():
            held = json.loads(record.read_text())
            if held.get("complete", True) and (marker != "sphere_0.1024.json" or len(held.get("tube", {}).get(
                    "solves", {})) == 5):
                done[" ".join(command)] = "skipped (complete)"
                continue
        if time.time() > began + timeout_s - MARGIN_S:
            done[" ".join(command)] = "not started (deadline)"
            break
        stop = threading.Event()

        def committer() -> None:
            while not stop.wait(20.0):
                results.commit()

        thread = threading.Thread(target=committer, daemon=True)
        thread.start()
        with open(log_path, "a") as log:
            log.write(f"\n=== {time.strftime('%H:%M:%SZ', time.gmtime())} {' '.join(command)}\n")
            log.flush()
            code = subprocess.run([sys.executable, "-W", "ignore", f"/repo/scripts/sim2real/{command[0]}",
                                   *command[1:], "--out", str(where)], stdout=log, stderr=subprocess.STDOUT,
                                  cwd="/repo/scripts/sim2real").returncode
        stop.set()
        thread.join()
        results.commit()
        done[" ".join(command)] = code
        if code != 0:
            break
    results.commit()
    return {"steps": done, "seconds": time.time() - began}


@app.local_entrypoint()
def main(tag: str, out: str, spacings: str = "", regimage: bool = True, stage: str = "grid") -> None:
    if TIMEOUT_S <= 0:
        raise SystemExit("set MIME_BB_TIMEOUT_S from the budget left (the function's hard timeout)")
    if not WHEEL:
        raise SystemExit("set MIME_BB_WHEEL")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    chosen = [float(s) for s in spacings.split(",") if s]
    began = time.time()
    (target / "local_call.json").write_text(json.dumps({"tag": tag, "spacings": chosen, "timeout_s": TIMEOUT_S,
                                                        "local_start_unix": began}) + "\n")
    if stage == "cpu":
        try:
            found = cpu_chain.remote(tag, TIMEOUT_S)
            (target / "cpu_chain_return.json").write_text(json.dumps(found, indent=1) + "\n")
        finally:
            (target / "local_call_cpu.json").write_text(json.dumps({"tag": tag, "timeout_s": TIMEOUT_S,
                                                                    "local_start_unix": began,
                                                                    "local_end_unix": time.time()}) + "\n")
        return
    try:
        found = run.remote(tag, chosen, TIMEOUT_S, regimage)
        for name, payload in found.items():
            (target / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            print(f"wrote {name}.json", flush=True)
    finally:
        ended = time.time()
        (target / "local_call.json").write_text(json.dumps({"tag": tag, "spacings": chosen, "timeout_s": TIMEOUT_S,
                                                            "local_start_unix": began, "local_end_unix": ended,
                                                            "local_seconds": ended - began}) + "\n")
