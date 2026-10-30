"""M3's remaining stages on Modal, built to survive a dropped connection: one function call runs ``smooth`` (model 2, as
preregistered) and then ``scan_fine`` (the correction's scale x1.000 to x1.040 in steps of 0.0025, every phase) in one
container, writing each stage's JSON to the Volume ``mime-7ja-m3-results`` as soon as it finishes (``started.json``
first, ``done.json`` last, the Volume committed after each write), so the results do not depend on the laptop.

**Cost safety (the server-side cap, since a detached run cannot be stopped by the laptop's tracker while it is offline).**
App ``mime-7ja-m3b``; one function, ``timeout`` 1800 s; no retries; no ``min_containers`` or ``keep_warm``;
``max_containers=1``; A100-40GB. Launched by ``m3b_launch.py`` (``modal run --detach``), which never starts a second run
while one is running or done.

    MIME_7JA_WHEEL=... MIME_7JA_REQUIREMENTS=... MIME_7JA_TABLE=... modal run --detach scripts/sim2real/m3b_modal.py --out DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets"}
WHEEL = os.environ.get("MIME_7JA_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_7JA_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_7JA_TABLE", "")
APP_NAME = "mime-7ja-m3b"
VOLUME = "mime-7ja-m3-results"
STAGES = ("smooth", "scan_fine")

app = modal.App(APP_NAME)
results = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    image = (image.uv_pip_install(requirements=[REQUIREMENTS], extra_options="--require-hashes --no-deps")
             .uv_pip_install("jax[cuda12]==0.12.0.1")
             .add_local_file(WHEEL, f"/w/{Path(WHEEL).name}", copy=True)
             .run_commands(f"pip install --no-deps --no-compile /w/{Path(WHEEL).name}")
             .add_local_file(TABLE, "/assets/unit_tube_R0.997.npz", copy=True)
             .env(PRECISION)
             .run_commands("mime assets fetch unit_tube_R0.997"))
    return (image.env(PRECISION)
            .add_local_file(str(ROOT / "tests" / "script_a.py"), "/repo/tests/script_a.py")
            .add_local_file(str(ROOT / "tests" / "differential" / "script_a_steady.py"),
                            "/repo/tests/differential/script_a_steady.py")
            .add_local_dir(str(HERE), "/repo/scripts/sim2real", ignore=["__pycache__"]))


image = _image()


@app.function(image=image, gpu="A100-40GB", cpu=4.0, memory=16384, max_containers=1, timeout=1800,
              volumes={"/results": results})
def run_all(draws: int = 50) -> dict:
    import sys
    import traceback

    sys.path.insert(0, "/repo/scripts/sim2real")
    import m3_wall_read

    def save(name: str, payload: dict) -> None:
        Path("/results", name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        results.commit()

    save("started.json", {"started_unix": time.time(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "stages": list(STAGES), "draws": draws})
    found: dict = {}
    try:
        for stage in STAGES:
            payload = m3_wall_read.run_stage(stage, draws)
            payload = json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))
            found[stage] = payload
            save(f"m3_{stage}.json", payload)
    except BaseException:
        save("failed.json", {"traceback": traceback.format_exc(), "finished_stages": sorted(found)})
        raise
    save("done.json", {"done_unix": time.time(), "stages": sorted(found)})
    return found


@app.local_entrypoint()
def main(out: str, draws: int = 50) -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    found = run_all.remote(draws)
    for stage, payload in found.items():
        (target / f"m3_{stage}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"{stage}: wrote m3_{stage}.json ({payload.get('solves')} solves)", flush=True)
