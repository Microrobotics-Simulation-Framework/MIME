"""The bits comparison's two controls, on the head: each program of ``hlo.py`` compiled twice (the text must be equal)
and once with the camera's mount one ulp of the program's own dtype along y (``BUMP=1``: the text must differ), each run
pinned as ``.github/probe/bit_compare.py`` pins its runs (one CPU, one thread). ``python scripts/camera_route2/bits_controls.py OUT_DIR``
from the repository's root; writes ``OUT_DIR/bits_controls.json`` and exits 1 where a control fails to discriminate
(a repeat that differs, or a one-ulp move that leaves the text equal)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

PROGRAMS = ("step32", "step64", "run32", "run64", "jvp32", "jvp64", "render32", "render64", "mesh32", "mesh64")
SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def text(program: str, out: str, bump: bool) -> np.ndarray:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} {SINGLE_THREAD}".strip(), "JAX_PLATFORMS": "cpu"}
    if bump:
        environment["BUMP"] = "1"
    else:
        environment.pop("BUMP", None)
    subprocess.run(["taskset", "-c", "0", sys.executable, "scripts/camera_route2/hlo.py", program, out], check=True,
                   env=environment)
    with np.load(out, allow_pickle=False) as found:
        return found["text"]


def main() -> int:
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    held = {}
    for program in PROGRAMS:
        first = text(program, f"{out_dir}/head_{program}.npz", bump=False)
        again = text(program, f"{out_dir}/again_{program}.npz", bump=False)
        bumped = text(program, f"{out_dir}/bumped_{program}.npz", bump=True)
        held[program] = {"repeat_equal": bool(np.array_equal(first, again)), "one_ulp_differs": not np.array_equal(first, bumped)}
        print(program, held[program], flush=True)
    json.dump(held, open(f"{out_dir}/bits_controls.json", "w"), indent=1)
    return 0 if all(v["repeat_equal"] and v["one_ulp_differs"] for v in held.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
