"""The surrogate generator's runs on Modal CPU containers (float64): the near-wall check, one job per spacing and gap,
each written to the Volume ``mime-7i`` as it finishes, so that a run stopped part way resumes where it stopped.

    MIME_WHEEL=dist/mime_engine-<version>-py3-none-any.whl modal run scripts/surrogate/modal_app.py::near_wall_check \\
        --out DIR

The image is the pinned Python base (``mime.data``'s ``env/bases.json``), MIME's hashed CPU requirements installed with
``--require-hashes --no-deps``, and the wheel built from the checkout (``uv build --wheel``). Each container fetches
the corrected tube table into its own cache by its shipped reference, which checks its sha256. At most eight containers
run at once (five large, three small). The check's spacings and gaps are read from the checkout's environment
(``.venv``), since the ``modal`` command's own does not hold MIME. ``modal`` is imported only here and is no
dependency of MIME."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "src" / "mime" / "data" / "env"
BASE = json.loads((ENV / "bases.json").read_text())["3.14"]["image"]
UV_VERSION = json.loads((ENV / "uv.json").read_text())["version"]
WHEEL = Path(os.environ.get("MIME_WHEEL", "dist/missing.whl"))
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "MIME_ASSETS_DIR": "/tmp/assets"}
#: Spacings at or below this get the larger container: the dense matrices grow with the square of the samples.
LARGE_BELOW = 0.13e-3

app = modal.App("mime-7i-near-wall")
volume = modal.Volume.from_name("mime-7i", create_if_missing=True)
image = (modal.Image.from_registry(BASE)
         .uv_pip_install(requirements=[str(ENV / "requirements-cpu.txt")],
                         extra_options="--require-hashes --no-deps --no-build", uv_version=UV_VERSION)
         .add_local_file(str(WHEEL), f"/wheel/{WHEEL.name}", copy=True)
         .run_commands(f"python -m pip install --no-deps --no-compile /wheel/{WHEEL.name}")
         .env(PRECISION)
         .add_local_file(str(ROOT / "scripts" / "surrogate" / "generate.py"), "/job/generate.py"))
COMMON = dict(image=image, cpu=8.0, timeout=3600, volumes={"/out": volume})


def _key(spacing: float, gap: float) -> str:
    return f"near_wall/s{spacing:.6e}_g{gap:.6e}.json"


def _run(spacing: float, gap: float) -> dict:
    import subprocess
    import sys

    target = Path("/out") / _key(spacing, gap)
    if target.exists():
        return json.loads(target.read_text())
    subprocess.run([sys.executable, "-m", "mime.cli", "assets", "fetch", "unit_tube_R0.997"], check=True)
    scratch = Path("/tmp") / target.name
    subprocess.run([sys.executable, "/job/generate.py", "near-wall", "--spacing", repr(spacing), "--gap", repr(gap),
                    "--out", str(scratch)], check=True)
    held = json.loads(scratch.read_text())
    held["cpu"] = os.cpu_count()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(held))
    volume.commit()
    return held


@app.function(memory=16384, max_containers=3, **COMMON)
def near_wall_small(spacing: float, gap: float) -> dict:
    return _run(spacing, gap)


@app.function(memory=57344, max_containers=5, **COMMON)
def near_wall_large(spacing: float, gap: float) -> dict:
    return _run(spacing, gap)


@app.local_entrypoint()
def near_wall_check(out: str) -> None:
    import subprocess

    listed = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-c",
                             "import json; from mime.surrogates import generator as g; "
                             "print(json.dumps([g.NEAR_WALL_SPACINGS, g.NEAR_WALL_GAPS]))"],
                            capture_output=True, text=True, check=True).stdout
    NEAR_WALL_SPACINGS, NEAR_WALL_GAPS = json.loads(listed)  # noqa: N806 - the generator's constants, read from it
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    small = [(s, g) for s in NEAR_WALL_SPACINGS if s > LARGE_BELOW for g in NEAR_WALL_GAPS]
    large = [(s, g) for s in NEAR_WALL_SPACINGS if s <= LARGE_BELOW for g in NEAR_WALL_GAPS]
    print(f"{len(small)} small jobs and {len(large)} large", flush=True)
    calls = [near_wall_large.spawn(s, g) for s, g in large] + [near_wall_small.spawn(s, g) for s, g in small]
    for (spacing, gap), call in zip(large + small, calls, strict=True):
        try:
            held = call.get()
        except Exception as error:  # noqa: BLE001 - a failed job is reported and the rest gathered
            print(f"spacing {spacing:g} gap {gap:g}: failed: {error!r}", flush=True)
            continue
        (target / Path(_key(spacing, gap)).name).write_text(json.dumps(held))
        sample = held["samples"][0]
        print(f"spacing {spacing:g} gap {gap:g}: {sample['samples']} samples, {sample['seconds']:.0f} s", flush=True)
