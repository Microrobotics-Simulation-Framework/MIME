"""Writes ``references/mpem_mag_manip.npz`` and its provenance: ETH's ``mag_manip`` 3.0.0 evaluating the three shipped eMNS
calibrations (the files in ``src/mime/data/emns_calibrations/``, checked against their pinned sha256 before use).

**Pinned: mag_manip 3.0.0** (``mag_manip-3.0.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl``, sha256
``1a2b3802132fb2b3a17406e2bf30db45ec313f37fc26cf450241e8fd5bcfa341``, Apache-2.0). It is never a MIME dependency: its
extension is built against NumPy 1.x and fails on NumPy 2, so it runs in an environment of its own, made once::

    uv venv --python 3.10 <env> && uv pip install --python <env>/bin/python "numpy<2" "mag_manip==3.0.0" pyyaml
    <env>/bin/python -I tests/oracles/generate_mpem_references.py

It reproduces the 0.2 line's archive (``tests/data/emns_mpem_oracle.npz`` at the tag ``pre-0.3``, never committed with its
generator) at that archive's points exactly. Per system, from one seed:

- ``points`` (64, 3), uniform in the calibration's declared region, ``currents`` (64, K) uniform in +-5 A, ``moments`` (64,
  3), a direction times 1e-4 to 1e-2 A m^2;
- ``field_matrix`` (64, 3, K), ``getFieldActuationMatrix``; ``actuation`` (64, 8, K), ``getActuationMatrix`` (the field and
  the gradient's five components per unit current); ``field_gradient5`` (64, 8), ``computeFieldGradient5FromCurrents`` at
  the currents (the field, then ``[dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]``);
- ``force`` and ``torque`` (64, 3), ``forceOnDipole`` and ``torqueOnDipole`` on the moments there;
- ``probes`` (48, 3): 32 points drawn in a box twice the region's size about its centre and 16 on its faces, with
  ``in_workspace`` (48,), ``pointInWorkspace`` (bounds inclusive)."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import mag_manip.mag_manip as mm
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CALIBRATIONS = ROOT / "src" / "mime" / "data" / "emns_calibrations"
OUT = Path(__file__).resolve().parent / "references"
PINNED = {"navion_2": "ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56",
          "octomag": "d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0",
          "cardiomag": "31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c"}
SEED = 20261109
POINTS, PROBES, FACES = 64, 32, 16


def column(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, 1)


def main() -> None:
    if mm.__name__ != "mag_manip.mag_manip" or np.__version__.split(".")[0] != "1":
        sys.exit("run in the environment of its own: mag_manip 3.0.0 on NumPy 1.x")
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    for name, pinned in PINNED.items():
        path = CALIBRATIONS / f"{name}.yaml"
        if hashlib.sha256(path.read_bytes()).hexdigest() != pinned:
            sys.exit(f"{path} is not the pinned file")
        model = mm.ForwardModelMPEM()
        model.setCalibrationFile(str(path))
        coils = model.getNumCoils()
        region = np.asarray(yaml.safe_load(path.read_text())["Workspace_Dimensions"], dtype=np.float64)
        low, high = region[:, 0], region[:, 1]
        points = low + rng.random((POINTS, 3)) * (high - low)
        currents = rng.uniform(-5.0, 5.0, (POINTS, coils))
        directions = rng.normal(size=(POINTS, 3))
        moments = directions / np.linalg.norm(directions, axis=1, keepdims=True) * 10.0 ** rng.uniform(-4, -2, (POINTS, 1))
        field_matrix = np.stack([np.asarray(model.getFieldActuationMatrix(column(p))) for p in points])
        actuation = np.stack([np.asarray(model.getActuationMatrix(column(p))) for p in points])
        both = np.stack([np.asarray(model.computeFieldGradient5FromCurrents(column(p), column(c))).ravel()
                         for p, c in zip(points, currents)])
        force = np.stack([np.asarray(mm.forceOnDipole(column(m), column(g[3:]))).ravel() for m, g in zip(moments, both)])
        torque = np.stack([np.asarray(mm.torqueOnDipole(column(m), column(g[:3]))).ravel() for m, g in zip(moments, both)])
        centre, half = (low + high) / 2, (high - low) / 2
        probes = centre + rng.uniform(-2.0, 2.0, (PROBES, 3)) * half
        on_faces = low + rng.random((FACES, 3)) * (high - low)
        axis = rng.integers(0, 3, FACES)
        side = rng.integers(0, 2, FACES)
        on_faces[np.arange(FACES), axis] = np.where(side == 0, low[axis], high[axis])
        probes = np.concatenate([probes, on_faces])
        inside = np.array([bool(model.pointInWorkspace(column(p))) for p in probes])
        for key, value in (("points", points), ("currents", currents), ("moments", moments),
                           ("field_matrix", field_matrix), ("actuation", actuation), ("field_gradient5", both),
                           ("force", force), ("torque", torque), ("probes", probes), ("in_workspace", inside)):
            arrays[f"{name}_{key}"] = value
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "mpem_mag_manip.npz", **arrays)
    provenance = {"generator": "tests/oracles/generate_mpem_references.py", "seed": SEED, "mag_manip": "3.0.0",
                  "mag_manip_wheel_sha256": "1a2b3802132fb2b3a17406e2bf30db45ec313f37fc26cf450241e8fd5bcfa341",
                  "numpy": np.__version__, "python": platform.python_version(),
                  "calibrations_sha256": PINNED,
                  "archive_sha256": hashlib.sha256((OUT / "mpem_mag_manip.npz").read_bytes()).hexdigest()}
    (OUT / "mpem_mag_manip_provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
