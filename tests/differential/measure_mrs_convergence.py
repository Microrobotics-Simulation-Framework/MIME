"""Record how the de Jongh helix's MRS resistance converges (``test_mrs_convergence.py`` says what is
measured and what the test holds of it):

    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/differential/measure_mrs_convergence.py

writes ``references/mrs_convergence.json``: at each of the five levels, the sample count and the 6 x 6 resistance by
the 0.2 line's rule and by the kind's samples through ``Drag.MRS``'s node, in the screw's frame, float64; with the
versions it ran under. Not collected by pytest; about half a minute on one core."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    from test_mrs_convergence import LEVELS, RECORD, measured

    record = {
        "what": "FL-9's lateral surface in free space: the 0.2 line's triangle rule at each vertex grid, and the "
                "de Jongh kind's lateral samples at the same count through Drag.MRS's node, the screw moved and "
                "turned and its resistance turned back into its frame; viscosity 1e-3 Pa s, blob 0.5 of the mean "
                "spacing, about the screw's centroid, SI",
        "written_by": "tests/differential/measure_mrs_convergence.py",
        "environment": {"jax": importlib.metadata.version("jax"), "jaxlib": importlib.metadata.version("jaxlib"),
                        "x64": True, "device": jax.default_backend(),
                        "platform": f"{sys.platform}-{platform.machine()}",
                        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
        "levels": measured(LEVELS),
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {RECORD.name}: {len(record['levels'])} levels")


if __name__ == "__main__":
    main()
