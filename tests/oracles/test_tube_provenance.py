"""The shipped tube table can be rebuilt from the repository: the committed build spec
(``scripts/tube_table/specs/unit_tube_R0.997.json``) equals the provenance arrays of the shipped asset itself.

The table was built at 1.5 times the builder's truncation rule (2500 modes, 16000 Gauss nodes, ``k_max`` 3000, 10834 wall
points), a factor that once lived only in a spec outside the repository, so the builder could not reproduce the shipped
file. The spec is now committed and the builder reads its factor from it. This test holds the spec to the asset: every
field of its ``expected`` block equals the file's arrays; each of its inputs (the grid, the readable radius, the axial
reach, one truncation for every source radius, the factor) is equal to what the file records; and the jobs the builder
makes from the spec have the file's knots and, per source radius, the file's truncation.

Control, failing in every run (20 of 20): the spec with any one field off by one (an integer by 1, a float by one unit in
its last place, a name by a letter, a flag flipped) disagrees with the asset, for every field of the spec."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tube_asset

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "tube_table"
sys.path.insert(0, str(SCRIPTS))
import full  # noqa: E402

SPEC = "unit_tube_R0.997.json"
RUNS = 20
UNIFORM = ("modes", "wavenumbers", "k_max", "angles")


@pytest.fixture(scope="module")
def asset() -> dict[str, np.ndarray]:
    names = ("rho_grid", "dphi_grid", "dz_grid", "readable_rho", "axial_reach", "modes", "wavenumbers", "k_max", "angles",
             "truncation_rule", "knot_rule", "single_truncation", "anomaly", "as_built_defect")
    with np.load(tube_asset.path()) as file:
        return {name: file[name] for name in names}


def leaves(node: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """The path of every value of a nested dict or list in the spec."""
    if isinstance(node, dict):
        return [leaf for key, child in node.items() for leaf in leaves(child, (*path, key))]
    if isinstance(node, list):
        return [leaf for index, child in enumerate(node) for leaf in leaves(child, (*path, str(index)))]
    return [path]


def at(spec: dict, path: tuple[str, ...]) -> object:
    node: object = spec
    for step in path:
        node = node[int(step)] if isinstance(node, list) else node[step]  # pyright: ignore[reportIndexIssue]
    return node


def off_by_one(spec: dict, path: tuple[str, ...]) -> dict:
    """The spec with the value at ``path`` changed by the least change its kind has."""
    out = copy.deepcopy(spec)
    parent = at(out, path[:-1]) if len(path) > 1 else out
    key: str | int = int(path[-1]) if isinstance(parent, list) else path[-1]
    value = parent[key]  # pyright: ignore[reportIndexIssue]
    if isinstance(value, bool):
        changed: object = not value
    elif isinstance(value, int):
        changed = value + 1
    elif isinstance(value, float):
        changed = float(np.nextafter(value, np.inf))
    else:
        changed = f"{value}x"
    parent[key] = changed  # pyright: ignore[reportIndexIssue]
    return out


def disagreements(spec: dict, asset: dict[str, np.ndarray]) -> list[str]:
    """Every way the spec differs from the asset's own arrays, or from what the builder makes of the spec."""
    found: list[str] = []
    expected = spec["expected"]

    def equal(label: str, spec_value: object, held: object) -> None:
        if not np.array_equal(np.asarray(spec_value), np.asarray(held)):
            found.append(f"{label}: spec {spec_value!r}, asset {held!r}")

    equal("asset", spec["asset"], tube_asset.NAME)
    equal("anomaly", spec["anomaly"], str(asset["anomaly"]))
    for name in UNIFORM:
        equal(name, expected[name], np.unique(asset[name])[0] if np.unique(asset[name]).size == 1 else asset[name])
    rule, knot_rule = json.loads(str(asset["truncation_rule"])), json.loads(str(asset["knot_rule"]))
    equal("truncation_factor", spec["truncation_factor"], rule["scale"])
    equal("grid_steps", expected["grid_steps"], knot_rule["steps"])
    equal("near", expected["near"], knot_rule["near"])
    equal("readable_rho", expected["readable_rho"], asset["readable_rho"])
    equal("readable_radius", spec["readable_radius"], asset["readable_rho"][1])
    equal("axial_reach", spec["axial_reach"], asset["axial_reach"])
    single = json.loads(str(asset["single_truncation"]))
    equal("single_truncation", spec["single_truncation"], single is not None)
    equal("knots", [expected["knots"][name] for name in ("rho", "dphi", "dz")],
          [asset[name].size for name in ("rho_grid", "dphi_grid", "dz_grid")])
    if spec["grid"] not in full.GRIDS:
        return [*found, f"grid: {spec['grid']!r} is no grid of the builder"]
    equal("grid steps of the name", full.GRIDS[spec["grid"]], knot_rule["steps"])
    if spec["readable_radius"] != full.R_MAX or spec["axial_reach"] != full.AXIAL_REACH:
        found.append("the builder's readable radius or axial reach is not the spec's")
    for label, knots, held in zip(("rho_grid", "dphi_grid", "dz_grid"), full.knots(spec["grid"]),
                                  (asset["rho_grid"], asset["dphi_grid"], asset["dz_grid"]), strict=True):
        equal(f"builder's {label}", knots, held)
    jobs = {job["key"]: job for job in full.build_specs(spec)}
    for i, source in enumerate(asset["rho_grid"]):
        job = jobs[f"table_{i:03d}"]
        equal(f"job {i} source", job["source"], source)
        for name in UNIFORM:
            equal(f"job {i} {name}", job["truncation"][name], asset[name][i])
    return found


def test_the_committed_spec_equals_the_shipped_assets_provenance(asset):
    assert disagreements(full.load_build_spec(SPEC), asset) == []


def test_the_spec_is_the_one_the_asset_was_built_from(asset):
    spec = full.load_build_spec(SPEC)
    assert not bool(asset["as_built_defect"])
    assert (spec["truncation_factor"], spec["single_truncation"]) == (1.5, True)
    assert (spec["expected"]["modes"], spec["expected"]["wavenumbers"], spec["expected"]["angles"]) == (2500, 16000, 10834)


def test_every_field_of_the_spec_is_checked_and_found_off_by_one_in_every_run(asset):
    spec = full.load_build_spec(SPEC)
    unfound: list[str] = []
    for _ in range(RUNS):
        for path in leaves(spec):
            if not disagreements(off_by_one(spec, path), asset):
                unfound.append(".".join(path))
    assert unfound == []
