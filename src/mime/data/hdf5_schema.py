"""HDF5 training data schema and writer for UMR confinement sweep.

Defines the full dataset schema (collected + deferred) and provides a
writer class that hooks into the sweep loop via a callback pattern.

Deferred datasets are created as empty datasets with a `deferred: True`
attribute. Future sessions slot data in without restructuring.

Schema structure:
    /ground_truth/{ratio}/        — 192^3 Bouzidi results
    /cheap_input/{ratio}/         — 64^3 simple BB results
    /held_out/{ratio}/            — reserved for test point (deferred)
    /provenance                   — git hash, params, versions
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Scalars collected at each check_interval
SCALAR_DATASETS = {
    "drag_torque":          {"shape_per_sample": (3,), "dtype": "float32"},
    "drag_force":           {"shape_per_sample": (3,), "dtype": "float32"},
    "drag_torque_z":        {"shape_per_sample": (), "dtype": "float32"},
    "residual":             {"shape_per_sample": (), "dtype": "float32"},
    "convergence_label":    {"shape_per_sample": (), "dtype": "int32"},
    "step":                 {"shape_per_sample": (), "dtype": "int32"},
    "wall_time":            {"shape_per_sample": (), "dtype": "float64"},
    "max_density_fluctuation": {"shape_per_sample": (), "dtype": "float32"},
    "u_max":                {"shape_per_sample": (), "dtype": "float32"},
}

# Per-fin drag breakdown (collected)
PER_FIN_DATASETS = {
    "drag_torque_per_fin":      {"shape_per_sample": (6,), "dtype": "float32"},
    "drag_torque_per_fin_set":  {"shape_per_sample": (2,), "dtype": "float32"},
}

# Temporal context (collected)
TEMPORAL_DATASETS = {
    "torque_z_last_10":     {"shape_per_sample": (10,), "dtype": "float32"},
    "residual_last_10":     {"shape_per_sample": (10,), "dtype": "float32"},
}

# Deferred datasets (created as empty, filled in future sessions)
DEFERRED_DATASETS = {
    "coarsened_velocity":   {"shape": (32, 32, 32, 3), "dtype": "float32"},
    "coarsened_pressure":   {"shape": (32, 32, 32), "dtype": "float32"},
    "surface_sparse_field": {"shape": (0, 6), "dtype": "float32"},
    "drag_power_spectrum":  {"shape": (0,), "dtype": "float32"},
    "magnetic_torque":      {"shape": (3,), "dtype": "float32"},
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def get_provenance(params: dict) -> dict:
    """Collect provenance metadata."""
    import jax
    from pathlib import Path

    # Git hash: try .mime_git_hash file first (written by launch script
    # before workdir sync — works on cloud where .git/ is not present),
    # then fall back to git rev-parse (works locally).
    git_hash = "unknown"
    hash_file = Path(".mime_git_hash")
    if hash_file.exists():
        git_hash = hash_file.read_text().strip()
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

    return {
        "git_hash": git_hash,
        "params_json": json.dumps(params, default=str),
        "jax_version": jax.__version__,
        "jax_backend": str(jax.default_backend()),
        "numpy_version": np.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "gpu_model": str(jax.devices()[0]) if jax.devices() else "unknown",
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SweepDataWriter:
    """HDF5 writer for confinement sweep data.

    Usage:
        writer = SweepDataWriter("data/sweep.h5")
        writer.create_schema(ratios=[0.15, 0.22, 0.30, 0.40], group="ground_truth")
        writer.write_provenance(params_dict)

        # In sweep loop, at each check_interval:
        writer.append_sample("ground_truth", ratio=0.30, data={
            "drag_torque": torque_array,
            "drag_force": force_array,
            ...
        })

        writer.close()
    """

    def __init__(self, path: str):
        if h5py is None:
            raise ImportError("h5py is required for SweepDataWriter")
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = h5py.File(path, "w")
        self._sample_counts: dict[str, int] = {}

    def create_schema(
        self,
        ratios: list[float],
        group: str = "ground_truth",
        max_samples: int = 500,
    ) -> None:
        """Create the full dataset schema for a set of confinement ratios."""
        for ratio in ratios:
            ratio_key = f"{ratio:.2f}"
            grp = self.f.require_group(f"{group}/{ratio_key}")
            grp.attrs["confinement_ratio"] = ratio

            # Collected scalar datasets (resizable)
            for name, spec in SCALAR_DATASETS.items():
                shape_per = spec["shape_per_sample"]
                if isinstance(shape_per, tuple) and len(shape_per) > 0:
                    full_shape = (max_samples,) + shape_per
                    max_shape = (None,) + shape_per
                    chunks = (min(100, max_samples),) + shape_per
                else:
                    full_shape = (max_samples,)
                    max_shape = (None,)
                    chunks = (min(100, max_samples),)
                ds = grp.create_dataset(
                    name, shape=(0,) + (shape_per if isinstance(shape_per, tuple) and len(shape_per) > 0 else ()),
                    maxshape=max_shape, dtype=spec["dtype"],
                    chunks=chunks,
                )
                ds.attrs["collected"] = True

            # Per-fin datasets
            for name, spec in PER_FIN_DATASETS.items():
                sp = spec["shape_per_sample"]
                ds = grp.create_dataset(
                    name, shape=(0,) + sp, maxshape=(None,) + sp,
                    dtype=spec["dtype"], chunks=(min(100, max_samples),) + sp,
                )
                ds.attrs["collected"] = True

            # Temporal context
            for name, spec in TEMPORAL_DATASETS.items():
                sp = spec["shape_per_sample"]
                ds = grp.create_dataset(
                    name, shape=(0,) + sp, maxshape=(None,) + sp,
                    dtype=spec["dtype"], chunks=(min(100, max_samples),) + sp,
                )
                ds.attrs["collected"] = True

            # Deferred datasets
            for name, spec in DEFERRED_DATASETS.items():
                ds = grp.create_dataset(
                    name, shape=spec["shape"], dtype=spec["dtype"],
                )
                ds.attrs["deferred"] = True
                ds.attrs["collected"] = False

            self._sample_counts[f"{group}/{ratio_key}"] = 0

        # Held-out group (deferred)
        held_out = self.f.require_group("held_out")
        held_out.attrs["deferred"] = True
        held_out.attrs["usage_policy"] = (
            "Reserved for held-out test point (ratio 0.35). "
            "Do not use for training or hyperparameter selection."
        )

    def write_provenance(self, params: dict) -> None:
        """Write provenance metadata to the root group."""
        prov = get_provenance(params)
        grp = self.f.require_group("provenance")
        for k, v in prov.items():
            grp.attrs[k] = v

    def append_sample(self, group: str, ratio: float, data: dict) -> None:
        """Append one sample (one check_interval snapshot) to the datasets."""
        ratio_key = f"{ratio:.2f}"
        path = f"{group}/{ratio_key}"
        grp = self.f[path]
        idx = self._sample_counts.get(path, 0)

        for name, value in data.items():
            if name not in grp:
                continue
            ds = grp[name]
            if ds.attrs.get("deferred", False):
                continue
            arr = np.asarray(value)
            # Resize to accommodate new sample
            new_len = idx + 1
            if ds.shape[0] < new_len:
                new_shape = (new_len,) + ds.shape[1:]
                ds.resize(new_shape)
            ds[idx] = arr

        self._sample_counts[path] = idx + 1

    def close(self) -> None:
        """Close the HDF5 file."""
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
