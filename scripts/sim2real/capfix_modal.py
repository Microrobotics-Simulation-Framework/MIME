"""``capfix.py``'s stages on Modal, built to survive a dropped connection: one function call runs every stage in one
container, in the order ``forward``, ``scan_fine``, ``derivative``, ``eigen`` (the cheapest and the preregistered first),
writing each stage's JSON to the Volume ``mime-capfix-results`` as soon as it finishes (``eigen`` after each phase too),
``started.json`` first and ``done.json`` last (``failed.json`` with the traceback on an error), the Volume committed
after each write, so the results do not depend on the laptop. ``modal`` is imported only here and is no dependency of
MIME.

**Cost safety** (the server-side limits, since a detached run cannot be stopped by a laptop that is offline): app
``mime-capfix``; one function, ``timeout`` 1800 s; no retries; no ``min_containers`` or ``keep_warm``;
``max_containers=1``; an A100-40GB. Launched by ``capfix_launch.py`` (``modal run --detach``), which never starts a
second run while one is running or has started.

    MIME_CAPFIX_WHEEL=<mime wheel> MIME_CAPFIX_REQUIREMENTS=<requirements.txt, hashed> \
        MIME_CAPFIX_TABLE=<unit_tube_R0.997.npz> modal run --detach scripts/sim2real/capfix_modal.py --out DIR
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
WHEEL = os.environ.get("MIME_CAPFIX_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_CAPFIX_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_CAPFIX_TABLE", "")
APP_NAME = "mime-capfix"
VOLUME = "mime-capfix-results"
STAGES = ("forward", "scan_fine", "derivative", "eigen")

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


@app.function(image=image, gpu="A100-40GB", cpu=4.0, memory=32768, max_containers=1, timeout=1800, retries=0,
              volumes={"/results": results})
def run_all() -> dict:
    import sys
    import traceback

    sys.path.insert(0, "/repo/scripts/sim2real")
    import capfix

    def save(name: str, payload: dict) -> None:
        plain = json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))
        Path("/results", name).write_text(json.dumps(plain, indent=1, sort_keys=True) + "\n")
        results.commit()

    save("started.json", {"started_unix": time.time(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "stages": list(STAGES)})
    found: dict = {}
    try:
        study = capfix.Study()
        for stage in STAGES:
            payload = capfix.run_stage(stage, study, save=lambda partial, stage=stage: save(f"capfix_{stage}.json",
                                                                                           partial))
            found[stage] = json.loads(json.dumps(payload, default=lambda v: getattr(v, "tolist", lambda: str(v))()))
            save(f"capfix_{stage}.json", found[stage])
    except BaseException:
        save("failed.json", {"traceback": traceback.format_exc(), "finished_stages": sorted(found)})
        raise
    save("done.json", {"done_unix": time.time(), "stages": sorted(found)})
    return found


@app.local_entrypoint()
def main(out: str) -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    found = run_all.remote()
    for stage, payload in found.items():
        (target / f"capfix_{stage}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"{stage}: wrote capfix_{stage}.json", flush=True)
