"""The tube table's slice on Modal: one CPU container per readable radius (two at once), each radius's result written
to a Volume as it finishes; then each radius's table job timed again on one A100 (float64) and checked against the
CPU's.

    modal run modal_app.py --stage cpu --out DIR
    modal run modal_app.py --stage gpu --out DIR
    modal run modal_app.py --stage retime --out DIR

``modal`` is imported only here and is no dependency of MIME."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import modal

PINS = ("numpy==2.5.4", "scipy==1.18.1")
RADII = (0.997, 0.984, 0.97, 0.958, 0.9)  # the slowest first
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest"}

app = modal.App("mime-tubetable-slice")
volume = modal.Volume.from_name("mime-tubetable-slice", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install(*PINS)
             .add_local_python_source("build", "tube_slice"))
gpu_image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install(*PINS, "jax[cuda12]==0.12.0.1")
             .env(PRECISION).add_local_python_source("build", "tube_slice"))
CPU = dict(image=cpu_image, cpu=8.0, memory=16384, timeout=5400, volumes={"/out": volume}, max_containers=2)
GPU = dict(image=gpu_image, gpu="A100-40GB", cpu=8.0, memory=32768, timeout=2400, volumes={"/out": volume},
           max_containers=1)


def _environment() -> dict:
    import numpy as np
    import scipy

    return {"cpus": os.cpu_count(), "numpy": np.__version__, "scipy": scipy.__version__}


@app.function(**CPU)
def measure(radius: float) -> dict:
    import numpy as np

    import tube_slice

    lines: list[str] = []

    def log(text: str) -> None:
        lines.append(f"{time.time():.0f} {text}")
        print(text, flush=True)

    def save(name: str, payload: dict) -> None:
        buffer = io.BytesIO()
        np.savez(buffer, values=payload["values"], info=json.dumps(payload["info"]))
        Path(f"/out/{name}.npz").write_bytes(buffer.getvalue())
        volume.commit()

    result = tube_slice.measure(radius, log=log, save=save)
    result.update(environment=_environment(), log=lines)
    Path(f"/out/slice_r{radius}.json").write_text(json.dumps(result))
    volume.commit()
    return result


@app.function(**GPU)
def time_on_gpu(radius: float) -> dict:
    import jax
    import numpy as np

    import build
    import tube_slice

    out: dict = {"radius": radius, "devices": [str(d) for d in jax.devices()], **_environment()}
    device = build.jax_target()
    for name, plane in tube_slice.APPROACHES.items():
        held = np.load(f"/out/job_r{radius}_{name}.npz")
        info = json.loads(str(held["info"]))
        tr = build.Truncation(**info["truncation"])
        runs = []
        for _ in range(2):  # the first includes compilation
            values, timing = tube_slice.table_job(radius, plane, tr, info["families"], device=device, budget=3e7)
            runs.append(timing["seconds"])
        cpu = held["values"]
        parity = float(np.max(np.abs(values - cpu)) / np.max(np.abs(cpu)))
        out[name] = {"seconds": runs, "cpu_seconds": info["seconds"], "gpu_vs_cpu": parity}
        print(name, out[name], flush=True)
    Path(f"/out/gpu_r{radius}.json").write_text(json.dumps(out))
    volume.commit()
    return out


def _retime(radius: float, device_name: str) -> dict:
    """Each approach's table job for ``radius`` again, its Gauss-Legendre nodes computed first (a one-off per
    truncation, shared by every source radius) and timed apart."""
    import numpy as np

    import build
    import tube_slice

    out: dict = {"radius": radius, "where": device_name, **_environment()}
    device = build.jax_target() if device_name == "gpu" else None
    for name, plane in tube_slice.APPROACHES.items():
        held = np.load(f"/out/job_r{radius}_{name}.npz")
        info = json.loads(str(held["info"]))
        tr = build.Truncation(**info["truncation"])
        start = time.perf_counter()
        build.legendre(tr.wavenumbers)
        nodes = time.perf_counter() - start
        runs = []
        for _ in range(2 if device is not None else 1):  # on the GPU the first includes compilation
            values, timing = tube_slice.table_job(radius, plane, tr, info["families"], device=device,
                                                  budget=3e7 if device is not None else 4e6)
            runs.append(timing["seconds"])
        entry = {"nodes_seconds": nodes, "job_seconds": runs, "finite": bool(np.all(np.isfinite(values)))}
        if device is None:
            np.save(f"/out/retime_r{radius}_{name}.npy", values)
        else:
            cpu = np.load(f"/out/retime_r{radius}_{name}.npy")
            entry["gpu_vs_cpu"] = float(np.max(np.abs(values - cpu)) / np.max(np.abs(cpu)))
        out[name] = entry
        print(radius, device_name, name, entry, flush=True)
    volume.commit()
    return out


@app.function(**CPU)
def retime_cpu(radius: float) -> dict:
    return _retime(radius, "cpu")


@app.function(**GPU)
def retime_gpu(radius: float) -> dict:
    import jax

    return {**_retime(radius, "gpu"), "devices": [str(d) for d in jax.devices()]}


@app.local_entrypoint()
def main(stage: str = "cpu", out: str = ".", radii: str = "") -> None:
    chosen = [float(r) for r in radii.split(",")] if radii else list(RADII)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    if stage == "cpu":
        for result in measure.map(chosen, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print("FAILED", repr(result), flush=True)
                continue
            (target / f"slice_r{result['radius']}.json").write_text(json.dumps(result, indent=1))
            print(f"radius {result['radius']} done in {result['seconds']:.0f} s", flush=True)
    elif stage == "gpu":
        for result in time_on_gpu.map(chosen, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print("FAILED", repr(result), flush=True)
                continue
            (target / f"gpu_r{result['radius']}.json").write_text(json.dumps(result, indent=1))
            print(f"radius {result['radius']} on the GPU: {result}", flush=True)
    elif stage == "retime":
        for function, label in ((retime_cpu, "cpu"), (retime_gpu, "gpu")):
            for result in function.map(chosen, order_outputs=False, return_exceptions=True):
                if isinstance(result, Exception):
                    print("FAILED", label, repr(result), flush=True)
                    continue
                (target / f"retime_{label}_r{result['radius']}.json").write_text(json.dumps(result, indent=1))
    else:
        raise SystemExit(f"stage is cpu, gpu or retime, not {stage!r}")
