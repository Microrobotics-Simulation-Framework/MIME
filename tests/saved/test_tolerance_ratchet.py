"""The fence on ``verify``'s tolerance (designs #263 and #266, with #133's rule that nothing loosens unlogged): every
constant a save writes into its lock, or that judges a folder, is pinned here in a table (:data:`PINS`) of its value, the
measurements it was set from, each with where it was made, and the margin above the largest. Lowering a constant is
free. Raising one fails until this table is changed, by a new measurement and the design row that decided it, in one
reviewed edit. The measurements the repository holds are computed here again from it (the tube demo's arrays across CPU
counts, its resolution, and the T4's spread as the gate's table states it), so the table cannot drift from its data.

A tolerance governs only how another machine's rounding, or a GPU's with deterministic mode off, is judged: the same
environment stays bit for bit, CI's own bit tests are unchanged, and accuracy is the oracles'. What the measurements
cover: one T4 (run to run, and against its CPU), one laptop's CPU at 24, 4 and 2 usable CPUs, one demo, eight steps;
float64 on the CPU only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from test_verify_blocks import (
    T4_SPREAD,
    TUBE_UNITS,
    block_mask,
    cpu_spread_per_block,
    label_of,
    t4_per_block,
    tube,
    tube_resolution,
)

from mime.saved import _scales

COUNTS = "tests/saved/artefacts/tube_demo_counts.npz"
RESOLUTION = "tests/saved/artefacts/tube_demo_resolution.json"
T4 = "the Colab verify-spread run on a T4, 2026-11-07 (T4_SPREAD in tests/saved/test_verify_blocks.py)"


@dataclass(frozen=True)
class Source:
    """One measurement a constant rests on: what it is, its value, and where it was made."""

    what: str
    value: float
    where: str


@dataclass(frozen=True)
class Pin:
    """A constant's value, the margin it stands above its largest measurement by, the design row that decided it and
    the measurements."""

    value: float
    margin: float
    row: str
    sources: tuple[Source, ...]

    @property
    def measured(self) -> float:
        return max((source.value for source in self.sources), default=0.0)


#: Each constant of ``mime.saved._scales`` a lock holds or a folder is judged by, as decided.
PINS = {
    "FLOAT32_RTOL": Pin(1e-5, 9.0, "#263", (
        Source("the largest spread of a block over its own largest magnitude where the rtol term is its bound, between "
               "runs on a T4 and against its CPU: the MRS drag's least eigenvalue", 1.07e-6, T4),
        Source("the same across 24, 4 and 2 usable CPUs: the least eigenvalue", 2.38e-7, COUNTS))),
    "FLOAT64_RTOL": Pin(2e-14, 36.0, "#263", (
        Source("the largest spread of a block over its own largest magnitude where the rtol term is its bound, across "
               "24, 4 and 2 usable CPUs: the MRS drag's least eigenvalue; no GPU measured in float64", 3.33e-16,
               COUNTS),)),
    "SCALED_ATOL": Pin(0.0, 1.0, "#263", ()),
    "RESOLUTION_FACTOR": Pin(30.0, 5.5, "#266", (
        Source("the largest spread of a block over its one-ulp resolution where its rtol term alone would not hold it, "
               "the T4's against the CPU's resolution, float32: the MRS drag's asymmetry", 5.43, f"{T4}; {RESOLUTION}"),
        Source("the same for the ball's angular velocity, the T4's spread put in it as the CPU's is", 4.69,
               f"{T4}; {RESOLUTION}"),
        Source("the same across 24, 4 and 2 usable CPUs in float64: the asymmetry", 1.30, f"{COUNTS}; {RESOLUTION}"))),
}
#: k is fenced from above as well, by review major M2: the tube ball's angular velocity zeroed, flipped or doubled must
#: stay beyond its floor. Its ceiling is the ball's largest angular velocity over that block's float32 resolution, and
#: k stands this far below it.
CEILING_MARGIN = 3.0


def raised(name: str, value: float) -> str | None:
    pin = PINS[name]
    if value <= pin.value:
        return None
    return (f"{name} is raised from {pin.value} to {value}: a tolerance is raised only by a new measurement and the "
            f"design row that decided it, entered in this table (now design {pin.row}, measured {pin.measured})")


@pytest.mark.parametrize("name", list(PINS))
def test_each_constant_is_at_most_its_pin(name: str):
    assert raised(name, getattr(_scales, name)) is None


@pytest.mark.parametrize("name", list(PINS))
def test_control_each_constant_raised_fails_and_lowered_passes(name: str):
    pin = PINS[name]
    assert raised(name, pin.value * 10 if pin.value else 1e-12) is not None
    assert raised(name, pin.value / 10) is None


@pytest.mark.parametrize("name", [name for name, pin in PINS.items() if pin.sources])
def test_each_pin_stands_its_margin_above_its_largest_measurement(name: str):
    pin = PINS[name]
    assert pin.value >= pin.margin * pin.measured, (name, pin.value, pin.margin, pin.measured)


def field_names(precision: str) -> list[str]:
    held = tube(precision, 24)
    return sorted(key.removeprefix("history/") for key, value in held.items()
                  if key.startswith("history/") and value.dtype.kind == "f" and key != "history/ball/contact_points")


def spreads(precision: str, name: str) -> dict[str, tuple[float, float, float]]:
    """``name``'s blocks: each one's largest magnitude, largest difference between two CPU counts, and resolution."""
    arms = [tube(precision, count)[f"history/{name}"].astype(np.float64) for count in (24, 4, 2)]
    units = TUBE_UNITS.get(label_of(name))
    found = {}
    for block, mask in _scales.blocks(units, arms[0].shape[1:]):
        where = block_mask(units, block, arms[0].shape)
        magnitude = float(np.max(np.where(where, np.abs(arms[0]), 0.0)))
        spread = max(float(np.max(np.where(where, np.abs(a - b), 0.0))) for a in arms for b in arms)
        found[block] = (magnitude, spread, tube_resolution(precision).get(label_of(name), {}).get(block) or 0.0)
    return found


def rtol_need(precision: str) -> float:
    """The largest CPU spread across counts of a block over its own largest magnitude, of the blocks whose bound is
    their rtol term."""
    rtol = _scales.FLOAT32_RTOL if precision == "float32" else _scales.FLOAT64_RTOL
    return max(spread / magnitude for name in field_names(precision)
               for magnitude, spread, floor in spreads(precision, name).values()
               if magnitude > 0 and rtol * magnitude >= _scales.RESOLUTION_FACTOR * floor)


def floor_needs(precision: str) -> dict[str, float]:
    """Each block that its rtol term alone would not hold: its spread over its resolution, the larger of the CPU's
    across counts and, in float32, the T4's (spread over the field's blocks as the CPU's is)."""
    rtol = _scales.FLOAT32_RTOL if precision == "float32" else _scales.FLOAT64_RTOL
    reference = tube(precision, 24)
    found: dict[str, float] = {}
    for name in field_names(precision):
        t4 = t4_per_block(reference, name) if precision == "float32" and name in T4_SPREAD else {}
        for block, (magnitude, spread, floor) in spreads(precision, name).items():
            worst = max(spread, t4.get(block, 0.0))
            if worst > rtol * magnitude:
                found[f"{label_of(name)} {block}"] = worst / floor if floor else float("inf")
    return found


def test_the_measurements_the_repository_holds_are_what_its_artefacts_give():
    assert rtol_need("float32") == pytest.approx(PINS["FLOAT32_RTOL"].sources[1].value, rel=0.01, abs=0.0)
    assert rtol_need("float64") == pytest.approx(PINS["FLOAT64_RTOL"].sources[0].value, rel=0.01, abs=0.0)
    assert T4_SPREAD["drag:ball/least_eigenvalue"] == PINS["FLOAT32_RTOL"].sources[0].value
    float32, float64 = floor_needs("float32"), floor_needs("float64")
    factor = PINS["RESOLUTION_FACTOR"].sources
    assert sorted(float32) == ["ball.twist rad/s", f"drag:ball.asymmetry {_scales.UNDECLARED}"], float32
    assert float32[f"drag:ball.asymmetry {_scales.UNDECLARED}"] == pytest.approx(factor[0].value, rel=0.01, abs=0.0)
    assert float32["ball.twist rad/s"] == pytest.approx(factor[1].value, rel=0.01, abs=0.0)
    assert max(float64.values()) == pytest.approx(factor[2].value, rel=0.01, abs=0.0), float64


def test_k_stands_below_the_ceiling_the_angular_velocity_needs_by_its_margin():
    reference = tube("float32", 24)["history/ball/twist"].astype(np.float64)
    ceiling = float(np.max(np.abs(reference[:, 3:]))) / tube_resolution("float32")["ball.twist"]["rad/s"]
    assert _scales.RESOLUTION_FACTOR * CEILING_MARGIN <= ceiling, ceiling


def test_a_save_writes_the_pinned_constants_into_its_lock(tmp_path: Path):
    """The lock is what a folder is judged by: a save writes its precision's rtol, no atol and the factor."""
    from mime.testing.toys import full_toy

    lock = full_toy().save(tmp_path / "toy")
    reference = lock["reference"]
    assert reference["rtol"] == (_scales.FLOAT64_RTOL if lock["environment"]["x64"] else _scales.FLOAT32_RTOL)
    assert reference["atol"] == _scales.SCALED_ATOL
    assert reference.get("resolution_factor") == _scales.RESOLUTION_FACTOR
