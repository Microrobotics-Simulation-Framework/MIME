"""Phase 7j-a's BEM runs on Modal: the static parts of ``bem.py`` and the BEM rung's full dynamics, float64 at the
highest matmul precision on an A100, the device served recorded with every result. ``modal`` is imported only here
and is no dependency of MIME.

    MIME_7JA_WHEEL=<mime wheel> MIME_7JA_REQUIREMENTS=<requirements.txt, hashed> MIME_7JA_TABLE=<unit_tube_R0.997.npz> \
        modal run modal_app.py --stage static --out DIR [--weights NAME]

Stages: ``probe`` (a few dynamics steps, timed: the cost of a step), ``static`` (the ladder, offsets, blob, margin and
table parts, two at a time), ``budget`` (the sensitivities, on an A100-80GB), ``dynamics`` (the BEM rung's run, closed
and lateral, each twice for the bit-for-bit check), ``repeat`` (the ladder part again, for the same check).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
#: The repository, where the image is defined; inside a container this module is imported from elsewhere, and the
#: image's local files are not read there.
ROOT = HERE.parents[1] if modal.is_local() else HERE
PRECISION = {"JAX_ENABLE_X64": "1", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "MIME_ASSETS_DIR": "/assets"}
WHEEL = os.environ.get("MIME_7JA_WHEEL", "")
REQUIREMENTS = os.environ.get("MIME_7JA_REQUIREMENTS", "")
TABLE = os.environ.get("MIME_7JA_TABLE", "")
#: The BEM rung's dynamics: long enough for the axial transient (script A's settles in a few tenths of a second) and a
#: steady window of whole field periods over its second half.
DYNAMICS_DURATION_S = 1.5

app = modal.App("mime-7ja")


def _image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if not modal.is_local():
        return image
    if WHEEL:
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
GPU = dict(image=image, gpu="A100-40GB", cpu=4.0, memory=16384, max_containers=2)
GPU80 = dict(image=image, gpu="A100-80GB", cpu=4.0, memory=32768, max_containers=1)


def _setup() -> None:
    import sys

    sys.path.insert(0, "/repo/scripts/sim2real")


@app.function(timeout=3600, **GPU)
def static_part(part: str, weights: str) -> dict:
    _setup()
    import bem

    return bem.run_part(part, weights)


@app.function(timeout=5400, **GPU80)
def budget_part(weights: str) -> dict:
    _setup()
    import bem

    return bem.run_part("budget", weights)


@app.function(timeout=10800, **GPU)
def dynamics(patches: str, duration_s: float, tag: str) -> dict:
    _setup()
    import time

    import operating_point as op

    started = time.perf_counter()
    built = op.experiment("bem", patches=patches)
    built = built.build()
    built_seconds = time.perf_counter() - started
    found = op.run_steady(built, op.DT_MS, duration_s)
    found.update(label=f"bem {patches} dynamics {duration_s} s ({tag})", patches=patches, duration_s=duration_s,
                 build_seconds=built_seconds, environment=op.environment(),
                 bem_solves=found["steps"], seconds_per_step=found["run_seconds"] / found["steps"])
    return found


@app.function(timeout=1800, **GPU)
def probe(patches: str, steps: int) -> dict:
    """The cost of the BEM rung's dynamics: ``steps`` steps after one to compile, timed; no speed is read."""
    _setup()
    import time

    import operating_point as op

    from mime.units import ms

    built = op.experiment("bem", patches=patches).build()
    started = time.perf_counter()
    built.run(ms(op.DT_MS))
    first = time.perf_counter() - started
    started = time.perf_counter()
    built.run(ms(op.DT_MS * steps))
    return {"patches": patches, "compile_and_first_step_s": first,
            "seconds_per_step": (time.perf_counter() - started) / steps, "environment": op.environment()}


@app.local_entrypoint()
def main(stage: str, out: str, weights: str = "dejongh_mlp_v2") -> None:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)

    def save(name: str, payload: dict) -> None:
        (target / name).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {name}", flush=True)

    if stage == "probe":
        for found in probe.starmap([("closed", 20), ("lateral", 20)]):
            save(f"probe_{found['patches']}.json", found)
    elif stage == "static":
        parts = ["ladder", "offsets", "blob", "margin", "table"]
        for part, found in zip(parts, static_part.starmap([(p, weights) for p in parts]), strict=True):
            save(f"bem_{part}.json", found)
    elif stage == "repeat":
        save("bem_ladder_repeat.json", static_part.remote("ladder", weights))
    elif stage == "budget":
        save("bem_budget.json", budget_part.remote(weights))
    elif stage == "dynamics":
        runs = [(p, DYNAMICS_DURATION_S, tag) for p in ("closed", "lateral") for tag in ("first", "second")]
        for (patches, _, tag), found in zip(runs, dynamics.starmap(runs), strict=True):
            save(f"bem_dynamics_{patches}_{tag}.json", found)
    else:
        raise SystemExit(f"no stage {stage!r}")
