"""The faults planted in each new part of A0 on its dynamic rungs, for the comparison's controls: a gain of each loop
of its controller (the joint loop's ``Kd``, the rotor loop's ``Ki``) and the joint loop's gravity term (the share of the
weight it compensates). Each is a size in units of the comparison's resolution for the quantity it moves, so that a
planted fault is above what the comparison can see by a stated factor (``test_a0_dynamic_controls.py`` plants three
resolutions; ``measure_a0_dynamic_controls.py`` sweeps the factor from below one to ten).

A resolution is a gate's band over its measured response to the fault (its relative move per unit of the fault's
relative size, at 1e-2 both ways), and every response here is measured, not assumed (7c1's review found an assumed one
19 times off): ``measure_a0_dynamic_controls.py sensitivities`` records them in
``references/a0_dynamic_controls_measured.json``, and ``test_a0_dynamic_controls.py`` holds the constants below to
that record. A gate's band is its declared tolerance, and for a gate that reads the rotor's single-precision angle (the
rotation rates by phase, the motor's angle rate) that and one unit's move at 3 Hz (the finding,
``references/a0_dynamic_findings.json``): any fault that moves the spin-up's rounding can move such a gate by that much,
so a fault is resolved there only beyond it.

**Restated after the review of PR #77 (its K3).** The band is not granted at 3 Hz (the angle sits 0 units apart
unplanted, and any plant that moves it moves it 4 or more), so ``Plant.resolution`` is the plants' size unit, not the
gate's resolution. Measured by the review's plants (its r3, under the then rule, which at 3 Hz is the bare
tolerances): the rotor loop's Ki has a working threshold between 1e-4 and 2e-4 of its relative change, 0.06 to 0.11 of
the unit here (a draw of +-5e-5 or +1e-4 moves the angle one or two units and passed; -2e-4 moves it five and failed);
it is set by whether the angle's rounding moves, not by a response to the gain. The gravity term is resolved at 2.2e-3
by the body's rotation rate held to its bare tolerance (2.95e-8 over its response 1.32e-5), 2.7 times finer than the
unit's 6.0e-3; the two draws caught "below" the unit are those above 2.2e-3. The joint loop's Kd unit (4.3e-3, by the
tracking error, no band) is its gate's resolution. Every plant (three units) is above its measured threshold. The
second declaration gates Ki by the rotor loop's integral, the phase the gain moves (``a0_dynamic_b_plants.py``)."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from a0_dynamic_compare import TOLERANCES
from a0_loops import planted_gravity

KEY = "3hz"
_FINDINGS = json.loads((Path(__file__).parent / "references" / "a0_dynamic_findings.json").read_text())
#: What a gate that reads the rotor's angle may move beyond its tolerance whatever the fault: one unit's move at 3 Hz.
BAND = {name: value for name, value in _FINDINGS["by_frequency"][KEY]["one_unit_moves"].items() if name in TOLERANCES}
#: The measured responses (the record's ``sensitivities``, at 1e-2 both ways) of the gate each fault is resolved by
#: finest, by part: the relative move of that gate per unit of the planted setting's relative change.
RESPONSES = {"the joint loop's gain": 0.1609, "the rotor loop's gain": -7.211e-5, "the joint loop's gravity term": -0.01022}
#: The gate each fault is resolved by finest (the record's ``sharpest_gate``).
QUANTITY = {"the joint loop's gain": "tracking_error_axial_mean_m", "the rotor loop's gain": "field_rotation_rate_hz",
            "the joint loop's gravity term": "swim_speed_axial_mean_velocity_m_s"}


@dataclass(frozen=True)
class Plant:
    part: str
    name: str
    settings: Callable[[float], dict[str, Any]]
    context: Callable[[float], AbstractContextManager[Any]] = lambda factor: nullcontext()

    @property
    def quantity(self) -> str:
        return QUANTITY[self.part]

    @property
    def resolution(self) -> float:
        """The relative change of the planted setting that moves its gate by its band."""
        return (TOLERANCES[self.quantity] + BAND.get(self.quantity, 0.0)) / abs(RESPONSES[self.part])

    def factor(self, size: float, sign: float) -> float:
        """The planted setting's factor at ``size`` resolutions, of ``sign``."""
        return 1.0 + sign * size * self.resolution

    def apply(self, size: float, sign: float) -> dict[str, Any]:
        """The options of ``script_a0_dynamic.a0_dynamic`` that plant the fault."""
        return self.settings(self.factor(size, sign))

    def planted(self, size: float, sign: float) -> AbstractContextManager[Any]:
        return self.context(self.factor(size, sign))


PLANTS = (
    Plant("the joint loop's gain", "Kd scaled", lambda f: {"rate_gain": 200.0 * f}),
    Plant("the rotor loop's gain", "Ki scaled", lambda f: {"integral": 1e-2 * f}),
    Plant("the joint loop's gravity term", "the weight's share scaled", lambda f: {}, planted_gravity),
)

__all__ = ["BAND", "KEY", "PLANTS", "QUANTITY", "RESPONSES", "Plant"]
