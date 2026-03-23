#!/usr/bin/env python3
"""Inspect and verify an HDF5 training data file.

Loads the file, checks all datasets for correct shapes and NaN/Inf,
prints a summary table of collected vs deferred datasets, and reports
provenance metadata.

Usage:
    python3 scripts/inspect_training_data.py data/rehearsal_192.h5
    python3 scripts/inspect_training_data.py data/sweep_192.h5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)


def inspect_group(grp, prefix="", errors=None):
    """Recursively inspect an HDF5 group."""
    if errors is None:
        errors = []

    for name in sorted(grp.keys()):
        item = grp[name]
        full_path = f"{prefix}/{name}"

        if isinstance(item, h5py.Group):
            # Check group attributes
            if item.attrs.get("deferred", False):
                print(f"  {full_path}/ [deferred group]")
            else:
                print(f"  {full_path}/")
                inspect_group(item, full_path, errors)
        elif isinstance(item, h5py.Dataset):
            deferred = item.attrs.get("deferred", False)
            collected = item.attrs.get("collected", False)
            shape = item.shape
            dtype = item.dtype
            status = "deferred" if deferred else ("collected" if collected else "unknown")

            # Check for NaN/Inf in collected datasets with data
            nan_count = 0
            inf_count = 0
            if collected and shape[0] > 0 and np.issubdtype(dtype, np.floating):
                data = item[:]
                nan_count = int(np.sum(np.isnan(data)))
                inf_count = int(np.sum(np.isinf(data)))
                if nan_count > 0:
                    errors.append(f"{full_path}: {nan_count} NaN values")
                if inf_count > 0:
                    errors.append(f"{full_path}: {inf_count} Inf values")

            quality = ""
            if nan_count > 0 or inf_count > 0:
                quality = f" [NaN={nan_count}, Inf={inf_count}]"

            print(f"  {full_path}: shape={shape} dtype={dtype} [{status}]{quality}")

    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hdf5_file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Inspecting: {path} ({path.stat().st_size} bytes)")
    print()

    with h5py.File(path, "r") as f:
        # Provenance
        if "provenance" in f:
            prov = f["provenance"]
            print("Provenance:")
            for k in sorted(prov.attrs.keys()):
                v = prov.attrs[k]
                if len(str(v)) > 80:
                    v = str(v)[:77] + "..."
                print(f"  {k}: {v}")
            print()

        # Dataset inventory
        print("Datasets:")
        errors = inspect_group(f)
        print()

        # Summary counts
        n_collected = 0
        n_deferred = 0
        n_samples = 0
        for name in f.keys():
            if isinstance(f[name], h5py.Group):
                for subname in f[name].keys():
                    item = f[name][subname]
                    if isinstance(item, h5py.Group):
                        for dsname in item.keys():
                            ds = item[dsname]
                            if isinstance(ds, h5py.Dataset):
                                if ds.attrs.get("deferred", False):
                                    n_deferred += 1
                                elif ds.attrs.get("collected", False):
                                    n_collected += 1
                                    if ds.shape[0] > n_samples:
                                        n_samples = ds.shape[0]

        print(f"Summary:")
        print(f"  Collected datasets: {n_collected}")
        print(f"  Deferred datasets: {n_deferred}")
        print(f"  Max samples: {n_samples}")
        print(f"  Errors: {len(errors)}")

        if errors:
            print()
            print("ERRORS:")
            for e in errors:
                print(f"  {e}")
            print()
            print("VERDICT: FAIL")
            sys.exit(1)
        else:
            print()
            print("VERDICT: PASS — no NaN/Inf found")


if __name__ == "__main__":
    main()
