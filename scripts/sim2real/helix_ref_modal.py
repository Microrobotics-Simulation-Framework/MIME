"""``helix_ref_run`` on Modal: the helix reference's stages, float64 at the highest matmul precision on one A100-80GB.
``modal`` is imported only here and is no dependency of MIME. The image is ``bem_budget_modal.py``'s.

**Cost safety.** Launched with ``modal run`` (never detached), so the app stops when this process does. One function;
``timeout`` from ``MIME_HR_TIMEOUT_S``, which the caller sets from the budget left at the measured rate (required); no
retries; ``max_containers=1``; no ``min_containers`` or ``keep_warm``. The run also stops itself, saving what it has,
at a deadline a margin inside its timeout.

**Checkpoints.** Every solve's record is written to the Volume ``mime-helix-ref-results`` under ``<tag>/`` as it is
found, and the Volume committed after it (a solve under a second is committed with the next, at most 10 s later); a
later call with the same tag resumes from what the Volume holds.

    MIME_HR_WHEEL=<wheel> MIME_HR_TIMEOUT_S=2400 modal run scripts/sim2real/helix_ref_modal.py --tag ref1 --out DIR
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
             # route (a)'s finest level holds one matrix of up to about 74 GB
             "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
WHEEL = os.environ.get("MIME_HR_WHEEL", "")
TIMEOUT_S = int(os.environ.get("MIME_HR_TIMEOUT_S", "0"))
APP_NAME = "mime-helix-ref"
VOLUME = "mime-helix-ref-results"
MARGIN_S = 120
COMMIT_EVERY_S = 10.0

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    requirements = ROOT / "src" / "mime" / "data" / "env" / "requirements-cuda12.txt"
    return (image.uv_pip_install(requirements=[str(requirements)], extra_options="--require-hashes --no-deps")
            .add_local_file(WHEEL, f"/w/{Path(WHEEL).name}", copy=True)
            .run_commands(f"pip install --no-deps --no-compile /w/{Path(WHEEL).name}")
            .env(PRECISION)
            .run_commands("mime assets fetch unit_tube_R0.997")
            .add_local_file(str(ROOT / "tests" / "script_a.py"), "/repo/tests/script_a.py")
            .add_local_file(str(ROOT / "tests" / "differential" / "script_a_steady.py"),
                            "/repo/tests/differential/script_a_steady.py")
            .add_local_file(str(ROOT / "tests" / "tube_fourier_bessel.py"), "/repo/tests/tube_fourier_bessel.py")
            .add_local_dir(str(ROOT / "scripts" / "tube_table"), "/repo/scripts/tube_table", ignore=["__pycache__"])
            .add_local_dir(str(HERE), "/repo/scripts/sim2real", ignore=["__pycache__", "*.pyc"]))


image = _image()


def _plain(payload: object) -> object:
    return json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))


@app.function(image=image, gpu="A100-80GB", cpu=4.0, memory=32768, max_containers=1, timeout=max(TIMEOUT_S, 60),
              retries=0, volumes={"/results": results})
def run(tag: str, stages: list[str], timeout_s: int) -> dict:
    import subprocess
    import sys
    import traceback

    verified = subprocess.run(["mime", "assets", "fetch", "unit_tube_R0.997"], capture_output=True, text=True,
                              env={**os.environ, "MIME_OFFLINE": "1"}, check=True).stdout
    sys.path.insert(0, "/repo/scripts/sim2real")
    import helix_ref_run
    import operating_point as op

    began = time.time()
    deadline = began + timeout_s - MARGIN_S
    folder = Path("/results", tag)
    folder.mkdir(parents=True, exist_ok=True)
    results.reload()
    last = {"commit": 0.0, "save": time.time(), "commits": 0, "commit_seconds": 0.0}

    def commit() -> None:
        start = time.time()
        results.commit()
        last.update(commit=time.time(), commits=last["commits"] + 1,
                    commit_seconds=last["commit_seconds"] + time.time() - start)

    def held(name: str) -> dict | None:
        path = folder / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save(name: str, payload: dict) -> None:
        (folder / f"{name}.json").write_text(json.dumps(_plain(payload), indent=1, sort_keys=True) + "\n")
        now = time.time()
        slow, last["save"] = now - last["save"] >= 1.0, now
        if slow or now - last["commit"] >= COMMIT_EVERY_S:
            commit()

    starts = (held("started") or {}).get("starts", [])
    save("started", {"starts": [*starts, {"unix": began, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
                     "stages": stages, "timeout_s": timeout_s, "environment": op.environment(),
                     "table_verified": verified.strip()})
    commit()
    try:
        done = helix_ref_run.run_stages(tuple(stages), save, held, deadline, helix_ref_run.FULL)
    except BaseException:
        save("failed", {"traceback": traceback.format_exc(), "unix": time.time()})
        commit()
        raise
    save("done", {"unix": time.time(), "seconds": time.time() - began, "stages": done,
                  "commits": last["commits"], "commit_seconds": last["commit_seconds"]})
    commit()
    return {"done": done, "seconds": time.time() - began}


@app.local_entrypoint()
def main(tag: str, out: str, stages: str = "sphere,nn_frozen,nn_sets,a") -> None:
    if TIMEOUT_S <= 0:
        raise SystemExit("set MIME_HR_TIMEOUT_S from the budget left (the function's hard timeout)")
    if not WHEEL:
        raise SystemExit("set MIME_HR_WHEEL")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    began = time.time()
    chosen = [s for s in stages.split(",") if s]
    (target / "local_call.json").write_text(json.dumps({"tag": tag, "stages": chosen, "timeout_s": TIMEOUT_S,
                                                        "local_start_unix": began}) + "\n")
    try:
        found = run.remote(tag, chosen, TIMEOUT_S)
        (target / "run_return.json").write_text(json.dumps(found, indent=1) + "\n")
    finally:
        ended = time.time()
        (target / "local_call.json").write_text(json.dumps({"tag": tag, "stages": chosen, "timeout_s": TIMEOUT_S,
                                                            "local_start_unix": began, "local_end_unix": ended,
                                                            "local_seconds": ended - began}) + "\n")


@app.local_entrypoint()
def pull(tag: str, out: str) -> None:
    """The Volume's records under ``tag`` into ``out``, read through the client (no container starts)."""
    target = Path(out)
    count = 0
    for entry in results.listdir(f"/{tag}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        local = target / Path(entry.path).relative_to(tag)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"".join(results.read_file(entry.path)))
        count += 1
    print(f"pulled {count} files into {target}", flush=True)
