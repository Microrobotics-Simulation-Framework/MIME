"""The proof that the committed build spec rebuilds the shipped tube table (run once on Modal; a report, not CI).

    modal run prove_provenance.py [--indices 0,8,30] [--out REPORT.json]

Each chosen source radius (an index into the table's 31 radii: the axis, an interior one, 0.997 of the radius) is rebuilt
on its own CPU container from ``specs/unit_tube_R0.997.json`` by the builder in this folder, and compared in the
container with the same source radius's block of the shipped asset, which the container fetches itself from the
reference the package ships (its sha256 checked). The report says, per radius, whether the two are bitwise equal and,
if not, by how much: the largest difference over the block's largest entry. The libraries are ``modal_app.py``'s pins.

``modal`` is imported only here and is no dependency of MIME."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PINS = ("numpy==2.5.4", "scipy==1.18.1")  # modal_app.py's
SPEC = "unit_tube_R0.997.json"
ASSET = "unit_tube_R0.997"

app = modal.App("mime-tubeprov")
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install(*PINS)
         .add_local_python_source("build", "full").add_local_dir(str(HERE / "specs"), "/repo/specs"))


@app.function(image=image, cpu=8.0, memory=16384, timeout=1800, max_containers=3)
def rebuild(index: int, url: str, sha256: str) -> dict:
    import numpy as np

    import build
    import full

    start = time.perf_counter()
    target = Path(tempfile.mkdtemp()) / "asset.npz"
    urllib.request.urlretrieve(url, target)  # noqa: S310 - the pinned public dataset revision the package ships
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"the fetched asset hashes to {digest}, not {sha256}")
    fetched = time.perf_counter() - start
    spec = full.load_build_spec(f"/repo/specs/{SPEC}")
    (job,) = [item for item in full.build_specs(spec) if item["key"] == f"table_{index:03d}"]
    start = time.perf_counter()
    values = build.image_grid(job["source"], job["rho"], job["phi"], job["z"], build.Truncation(**job["truncation"]))
    seconds = time.perf_counter() - start
    with np.load(target) as held:
        block = held["data"][:, index]
        source = float(held["rho_grid"][index])
    difference = np.abs(values - block)
    return {"index": index, "source": source, "job_source": job["source"], "truncation": job["truncation"],
            "shape": list(values.shape), "bitwise_equal": bool(np.array_equal(values, block)),
            "entries_differing": int(np.count_nonzero(values != block)), "entries": int(values.size),
            "max_abs_difference": float(difference.max()), "block_max": float(np.abs(block).max()),
            "relative_to_block_max": float(difference.max() / np.abs(block).max()),
            "build_seconds": seconds, "fetch_seconds": fetched, "numpy": np.__version__}


@app.local_entrypoint()
def main(indices: str = "0,8,30", out: str = "") -> None:
    reference = HERE.parents[1] / "src" / "mime" / "data" / "assets" / "unit_tube_tables.json"  # local only
    held = {entry["name"]: entry for entry in json.loads(reference.read_text())["assets"]}[ASSET]
    chosen = [int(i) for i in indices.split(",")]
    results = list(rebuild.starmap([(i, held["locations"][0], held["sha256"]) for i in chosen]))
    for result in sorted(results, key=lambda r: r["index"]):
        print(json.dumps(result), flush=True)
    if out:
        Path(out).write_text(json.dumps(results, indent=1))
