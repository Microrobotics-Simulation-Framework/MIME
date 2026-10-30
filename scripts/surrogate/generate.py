"""The surrogate generator's command line (phase 7i): a thin script over ``mime.surrogates.generator``.

    python scripts/surrogate/generate.py protocol --spacing 0.15e-3
    python scripts/surrogate/generate.py near-wall --spacing 0.15e-3 [--gap 0.05e-3 ...] --out level.json
    python scripts/surrogate/generate.py near-wall-record DIR --commit SHA --out near_wall_check.json
    python scripts/surrogate/generate.py data --split training --spacing 0.15e-3 --out data.npz --record record.json

``protocol`` prints the protocol at a spacing (the domain, the training margins and each set's size; while the domain's
near-wall edge is open, that, and no margins). ``near-wall`` runs
the near-wall check's cases at one spacing (FL-9 resting in the 1/4 in tube at each gap) and writes their canonical
resistances; ``near-wall-record`` gathers every level's file in ``DIR`` into the check's record, with the edge the
analysis reads from it. ``data`` derives a set's training data through the tube's table and writes the data and its derivation record (the
commit, the hash of each source file it ran and whether the tree was dirty); it refuses, by name, while the domain's
near-wall edge is open, an empty set, a case outside the domain, and a set with a failed check. Each runs in float64 (``JAX_ENABLE_X64=1``), on the CPU unless
``JAX_PLATFORMS`` says otherwise."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

from mime import assets  # noqa: E402
from mime.surrogates import generator  # noqa: E402


def protocol(arguments: argparse.Namespace) -> None:
    print(generator.protocol_json(generator.DOMAIN, generator.Discretisation(arguments.spacing)))


def near_wall(arguments: argparse.Namespace) -> None:
    design = generator.design_named(generator.NEAR_WALL_DESIGN)
    discretisation = generator.Discretisation(arguments.spacing)
    gaps = arguments.gap or list(generator.NEAR_WALL_GAPS)
    started = time.perf_counter()
    evaluator = generator.BEMEvaluator(design, generator.NEAR_WALL_TUBE, discretisation)
    built = time.perf_counter() - started
    found = []
    for gap in gaps:
        start = time.perf_counter()
        sample = evaluator(generator.Case(design, generator.NEAR_WALL_TUBE, gap))
        found.append({"spacing": arguments.spacing, "gap": gap, "samples": sample.samples,
                      "accepted": sample.accepted, "resistance": sample.resistance.tolist(),
                      "checks": [check.as_json() for check in sample.checks()],
                      "seconds": time.perf_counter() - start})
        print(f"spacing {arguments.spacing:g} gap {gap:g}: {sample.samples} samples, accepted {sample.accepted}, "
              f"swim {sample.swim()[2]:.6e} m/rad, {found[-1]['seconds']:.1f} s", flush=True)
    Path(arguments.out).write_text(json.dumps({"build_seconds": built, "samples": found}, indent=1) + "\n")


def near_wall_record(arguments: argparse.Namespace) -> None:
    samples = []
    for path in sorted(Path(arguments.directory).glob("*.json")):
        samples += json.loads(path.read_text())["samples"]
    samples.sort(key=lambda each: (-each["spacing"], each["gap"]))
    record = {"design": generator.NEAR_WALL_DESIGN, "tube": generator.NEAR_WALL_TUBE,
              "tube_radius": generator.TUBES[generator.NEAR_WALL_TUBE], "patches": "closed surface",
              "blob": generator.Discretisation(1.0).blob, "spacings": list(generator.NEAR_WALL_SPACINGS),
              "gaps": list(generator.NEAR_WALL_GAPS), "tolerance": generator.NEAR_WALL_TOLERANCE,
              "table": assets.reference("unit_tube_R0.997").sha256, "commit": arguments.commit, "x64": True,
              "frame": "canonical: the offset's direction, the tube's axis crossed with it, the tube's axis; about the "
                       "centroid; SI",
              "samples": [{key: each[key] for key in ("spacing", "gap", "samples", "accepted", "resistance")}
                          for each in samples]}
    record["excess"] = {use: generator.wall_excess(record, quantities)
                        for use, quantities in generator.NEAR_WALL_USES.items()}
    record["ratios"] = generator.near_wall_ratios(record)
    Path(arguments.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"near-wall edges, in spacings: {record['ratios']}")


#: The source files a derivation of training data runs, whose hashes its record holds.
SOURCES = ("src/mime/surrogates/generator.py", "scripts/surrogate/generate.py", "src/mime/assets/_derivation.py")


def data(arguments: argparse.Namespace) -> None:
    from mime.build import BuildRefused
    from mime.nodes.hydro.bem._tube import UNIT_TUBE

    commit, sources, dirty = assets.working_state(Path(__file__).resolve().parents[2], SOURCES)
    builder = generator.TrainingData()
    discretisation = generator.Discretisation(arguments.spacing)
    parameters = {"split": arguments.split, "discretisation": discretisation.as_json()}
    try:
        content, record = assets.derive(f"{arguments.name}", builder, parameters,
                                        {"wall:tube": assets.reference(UNIT_TUBE)}, commit=commit,
                                        domain=generator.DOMAIN.as_json(), sources=sources, dirty=dirty)
    except BuildRefused as refused:
        sys.exit(f"refused: {refused}")
    Path(arguments.out).write_bytes(content)
    Path(arguments.record).write_text(record.text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    held = commands.add_parser("protocol")
    held.add_argument("--spacing", type=float, required=True)
    held = commands.add_parser("near-wall")
    held.add_argument("--spacing", type=float, required=True)
    held.add_argument("--gap", type=float, action="append")
    held.add_argument("--out", required=True)
    held = commands.add_parser("near-wall-record")
    held.add_argument("directory")
    held.add_argument("--commit", required=True)
    held.add_argument("--out", required=True)
    held = commands.add_parser("data")
    held.add_argument("--split", required=True)
    held.add_argument("--spacing", type=float, required=True)
    held.add_argument("--name", default="screw_training_data")
    held.add_argument("--out", required=True)
    held.add_argument("--record", required=True)
    arguments = parser.parse_args()
    {"protocol": protocol, "near-wall": near_wall, "near-wall-record": near_wall_record, "data": data}[
        arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
