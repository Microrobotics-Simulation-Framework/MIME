"""M3's rerun on the fixed surface (screw 0.2.0, PR #66) on Modal: ``m3_wall_read.py``'s stages ``reference`` (nominal, floor,
x1.02 control, pole scan, construction checks), ``noise`` (model 1) and ``smooth`` (model 2), 50 draws each, the same seeds,
exactly as preregistered (``smooth_local`` is left out and ``scan_fine`` is not repeated: the cap fix reran it), in one
function call on one A100-40GB, float64 at the highest matmul precision. Each stage's JSON goes to the Volume
``mime-m3-stage2`` under ``rerun/`` as soon as it finishes (``started.json`` first, ``done.json`` last, the Volume
committed after each write), so the results do not depend on the laptop. ``modal`` is imported only here.

**Cost safety (server-side limits).** App ``mime-m3-rerun``; one function, ``timeout`` 1800 s; no retries; no
``min_containers`` or ``keep_warm``; ``max_containers=1``; A100-40GB. Launched by ``m3s2_launch.py`` (``modal run
--detach``), which never starts a second run while one is running or has started.

    MIME_M3_WHEEL=<wheel> MIME_M3_REQUIREMENTS=<hashed requirements> MIME_M3_TABLE=<unit_tube_R0.997.npz> \
        modal run --detach scripts/sim2real/m3_rerun_modal.py --out DIR
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
WHEEL = os.environ.get("MIME_M3_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_M3_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_M3_TABLE", "")
APP_NAME = "mime-m3-rerun"
VOLUME = "mime-m3-stage2"
PREFIX = "rerun"
STAGES = ("reference", "noise", "smooth")

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


@app.function(image=image, gpu="A100-40GB", cpu=4.0, memory=16384, max_containers=1, timeout=1800, retries=0,
              volumes={"/results": results})
def run_all(draws: int = 50) -> dict:
    import sys
    import traceback

    sys.path.insert(0, "/repo/scripts/sim2real")
    import m3_wall_read

    Path("/results", PREFIX).mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict) -> None:
        Path("/results", PREFIX, name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
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
