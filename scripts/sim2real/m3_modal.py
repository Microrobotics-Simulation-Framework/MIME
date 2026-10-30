"""M3's runs on Modal: ``m3_wall_read.py``'s stages on one A100-40GB, float64 at the highest matmul precision, the device
served recorded with each result. ``modal`` is imported only here and is no dependency of MIME.

**Cost safety (preconditions, not options).** One app, ``mime-7ja-m3``; one function; ``timeout`` 2400 s (enforced by
Modal, the backstop that does not depend on the laptop's connection); no retries; no ``min_containers`` or
``keep_warm``; ``max_containers=1``. Launch without ``--detach``, so a lost connection stops the app.

    MIME_7JA_WHEEL=<mime wheel> MIME_7JA_REQUIREMENTS=<requirements-cuda12.txt, hashed> MIME_7JA_TABLE=<unit_tube_R0.997.npz> \
        modal run scripts/sim2real/m3_modal.py --out DIR [--stages reference,noise,smooth,smooth_local]

A stage whose JSON is already in ``DIR`` is skipped, so a relaunch after a lost connection runs only what is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets"}
WHEEL = os.environ.get("MIME_7JA_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_7JA_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_7JA_TABLE", "")
APP_NAME = "mime-7ja-m3"

app = modal.App(APP_NAME)


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


@app.function(image=image, gpu="A100-40GB", cpu=4.0, memory=16384, max_containers=1, timeout=2400)
def run_stage(stage: str, draws: int = 50) -> dict:
    import sys

    sys.path.insert(0, "/repo/scripts/sim2real")
    import m3_wall_read

    found = m3_wall_read.run_stage(stage, draws)
    return json.loads(json.dumps(found, default=lambda value: getattr(value, "tolist", lambda: str(value))()))


@app.local_entrypoint()
def main(out: str, stages: str = "reference,noise,smooth,smooth_local", draws: int = 50) -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    for stage in stages.split(","):
        path = target / f"m3_{stage}.json"
        if path.exists():
            print(f"{stage}: {path.name} exists, skipped", flush=True)
            continue
        print(f"{stage}: running", flush=True)
        found = run_stage.remote(stage, draws)
        path.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
        print(f"{stage}: wrote {path.name} ({found.get('solves')} solves, {found.get('seconds', 0.0):.0f} s)", flush=True)
