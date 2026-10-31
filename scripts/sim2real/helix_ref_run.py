"""The helix reference's stages, in the brief's order (the cheapest and most informative first), each solve saved
through ``save(name, payload)`` as it is found and a rerun skipping what the records hold. Run on Modal by
``helix_ref_modal.py``; here on the CPU at coarse sizes (``--dry``) or for the sizes alone (``--sizes``).

1. ``sphere``: the controls. Route (a) at ``SPHERE_A_MM``; route (b) at ``SPHERE_NN_FORCE_MM`` x ``SPHERE_NN_QUAD_MM``;
   the exact regularised references at every eps.
2. ``nn_frozen``: route (b) on A's body at the frozen force set (0.16 mm), every quadrature, the frozen eps first.
3. ``nn_sets``: route (b) at the other force sets, the two finest quadratures.
4. ``a``: route (a) on A's body, coarse to fine, as far as one matrix fits.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python helix_ref_run.py --dry --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helix_ref as hr  # noqa: E402
import operating_point as op  # noqa: E402

Saver = Callable[[str, dict[str, Any]], None]
Holder = Callable[[str], "dict[str, Any] | None"]
STAGES = ("sphere", "nn_frozen", "nn_sets", "a")
#: The frozen eps first (the brief's "route (b) at blob 0.5" before the rest).
EPS_FROZEN_FIRST = (hr.EPS_M[2],) + hr.EPS_M[:2] + hr.EPS_M[3:]


def stage_sphere(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    if held("sphere_references") is None:
        save("sphere_references", {"eps": hr.sphere_references(), "per_panel": 48, "around": 32})
    for spacing in sizes["sphere_a"]:
        name = f"sphere_a_{spacing:.4f}"
        found = hr.nystrom_level("sphere", spacing, lambda p, n=name: save(n, p), held(name), deadline)
        if found.get("stopped") == "deadline":
            return False
    for force in sizes["sphere_nn_force"]:
        name = f"sphere_nn_{force:.4f}"
        found = hr.nn_level("sphere", force, sizes["sphere_nn_quad"], EPS_FROZEN_FIRST, lambda p, n=name: save(n, p),
                            held(name), deadline)
        if found.get("stopped") == "deadline":
            return False
    return True


def stage_nn_frozen(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    name = f"helix_nn_{hr.FROZEN_SPACING_MM:.4f}"
    found = hr.nn_level("helix", hr.FROZEN_SPACING_MM, sizes["nn_quad"], EPS_FROZEN_FIRST,
                        lambda p: save(name, p), held(name), deadline)
    return found.get("stopped") != "deadline"


def stage_nn_sets(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    for force in sizes["nn_force"]:
        if abs(force - hr.FROZEN_SPACING_MM) < 1e-12:
            continue
        name = f"helix_nn_{force:.4f}"
        quads = sizes["nn_quad"][-1:] if force < hr.NN_FINEST_ONLY_BELOW_MM else sizes["nn_quad"][-2:]
        found = hr.nn_level("helix", force, quads, EPS_FROZEN_FIRST, lambda p, n=name: save(n, p), held(name),
                            deadline)
        if found.get("stopped") == "deadline":
            return False
    return True


def stage_a(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    for spacing in sizes["a"]:
        name = f"helix_a_{spacing:.4f}"
        found = hr.nystrom_level("helix", spacing, lambda p, n=name: save(n, p), held(name), deadline)
        if found.get("stopped") == "deadline":
            return False
        if str(found.get("stopped", "")).startswith("does not fit"):
            break
    return True


def stage_cond(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    """The conditioning of route (a)'s matrices at every level it solved and of route (b)'s at the frozen force set
    and the two finest (the finest quadrature), every eps."""
    for spacing in sizes["a"]:
        name = f"helix_cond_{spacing:.4f}"
        nn_eps = hr.EPS_M if spacing in sizes["cond_nn"] else ()
        body_bytes = hr.matrix_bytes(hr.Body("helix", spacing).weights.shape[0])
        if 2 * body_bytes > hr.device_budget():
            break
        found = hr.condition_level(spacing, lambda p, n=name: save(n, p), held(name), deadline, nn_eps)
        if found.get("stopped") == "deadline":
            return False
    return True


def stage_iso(save: Saver, held: Holder, deadline: float | None, sizes: dict[str, Any]) -> bool:
    """The post hoc mechanism test (``helix_ref_iso.py``)."""
    import helix_ref_iso

    if held("iso") is None:
        helix_ref_iso.run(lambda p: save("iso", p))
    return True


RUNNERS = {"iso": stage_iso, "sphere": stage_sphere, "nn_frozen": stage_nn_frozen, "nn_sets": stage_nn_sets, "a": stage_a,
           "cond": stage_cond}
FULL = {"sphere_a": hr.SPHERE_A_MM, "sphere_nn_force": hr.SPHERE_NN_FORCE_MM, "sphere_nn_quad": hr.SPHERE_NN_QUAD_MM,
        "nn_force": hr.NN_FORCE_MM, "nn_quad": hr.NN_QUAD_MM, "a": hr.A_SPACINGS_MM, "cond_nn": (0.16, 0.1024, 0.092)}
#: The CPU dry run: the same code at sizes a laptop solves in seconds.
DRY = {"sphere_a": (0.3, 0.25), "sphere_nn_force": (0.3,), "sphere_nn_quad": (0.12, 0.08),
       "nn_force": (0.5, 0.4), "nn_quad": (0.2, 0.15), "a": (0.5, 0.4), "cond_nn": (0.5,)}


def run_stages(stages: tuple[str, ...], save: Saver, held: Holder, deadline: float | None,
               sizes: dict[str, Any]) -> dict[str, Any]:
    done: dict[str, Any] = {}
    if sizes is DRY:
        # the dry run's force sets are not the frozen one: route (b)'s frozen stage runs its first set instead
        hr.FROZEN_SPACING_MM = sizes["nn_force"][0]
    for stage in stages:
        began = time.time()
        finished = RUNNERS[stage](save, held, deadline, sizes)
        done[stage] = {"finished": finished, "seconds": time.time() - began}
        save("stages", {"done": done, "unix": time.time()})
        if not finished:
            break
    return done


def sizes_table(full: dict[str, Any]) -> dict[str, Any]:
    """Every level's sample count and one matrix's bytes, and each quadrature's points, without a solve."""
    out: dict[str, Any] = {"a": {}, "sphere_a": {}, "nn_quad": {}, "sphere_nn_quad": {}}
    for spacing in full["a"]:
        body = hr.Body("helix", spacing)
        out["a"][f"{spacing}"] = {"samples": len(body.weights), "mean_spacing_m": body.mean_h,
                                  "one_matrix_gb": hr.matrix_bytes(len(body.weights)) / 1e9}
    for spacing in full["sphere_a"]:
        body = hr.Body("sphere", spacing)
        out["sphere_a"][f"{spacing}"] = {"samples": len(body.weights), "mean_spacing_m": body.mean_h}
    helix, sphere = hr.Body("helix", 0.16), hr.Body("sphere", 0.16)
    for q in full["nn_quad"]:
        counts = helix.counts_at(q, hr.ALONG_FACTOR)
        out["nn_quad"][f"{q}"] = {"counts": counts, "points": int(sum(math.prod(c) * (2 if len(c) == 1 else 1)
                                                                      for c in counts))}
    for q in full["sphere_nn_quad"]:
        counts = sphere.counts_at(q)
        out["sphere_nn_quad"][f"{q}"] = {"counts": counts}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--sizes", action="store_true")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    op.require_float64()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    if args.sizes:
        (folder / "sizes.json").write_text(json.dumps(sizes_table(FULL), indent=1) + "\n")
        print((folder / "sizes.json").read_text())
        return

    def save(name: str, payload: dict[str, Any]) -> None:
        (folder / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")

    def held(name: str) -> dict[str, Any] | None:
        path = folder / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    print(run_stages(tuple(args.stages.split(",")), save, held, None, DRY if args.dry else FULL), flush=True)


if __name__ == "__main__":
    main()
