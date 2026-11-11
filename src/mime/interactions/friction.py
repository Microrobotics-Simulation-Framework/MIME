"""Friction at a contact: the laws a ``Contact`` solves in its own sweeps (``Contact(friction=Coulomb(mu=0.15))``), and
the seams for what is left out, each a small declared protocol with a default and a conformance suite
(``mime.testing.friction_law_suite``, ``normal_law_suite``, ``row_kind_suite``), so that adding one is a law of its
own, never a change to the solve:

- :class:`FrictionLaw`, the sliding limit as a function of the normal load and the slip speed: :class:`Coulomb` now
  (one coefficient, static and kinetic alike); a speed-dependent law (Stribeck's, a fitted table) or a static above the
  kinetic later.
- :class:`NormalLaw`, how a point is held along its normal: :class:`Hold` now (no penetration, no rebound); the adhesion
  offset in its limit (:attr:`Hold.adhesion`: the normal force may pull to ``-F_adh`` and friction's limit becomes
  ``mu (N + F_adh)``), restitution and soft or compliant laws later.
- :data:`ROW_KINDS`, the rows a law adds beside each point's normal row: sliding and rolling resistance
  (``Coulomb(rolling=...)``) now; spin declared, not built.
- :data:`POINT_SETS`, how a contact's points are held: one point a group now; line and patch contacts later, as
  distributed point sets with weights.

A contact that writes no friction compiles the program it always did: the friction rows are a structural choice made
when the experiment is built. Friction acts only on the contact's own normal force, never on a lubricating film's
pressure: a body that rides a film above the contact's margin feels exactly none."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar, cast, override

from mime.registry import PHYSICAL, SettingGroup, convention, register_setting_group
from mime.units import LENGTH, Force, Length, Quantity, UnitError, si


@dataclass(frozen=True, kw_only=True)
class FrictionLaw(SettingGroup):
    """The friction a contact holds at its points: the sliding limit as a function of the normal load and the slip
    speed (:meth:`limit`), the row kinds it adds beside each point's normal row (:attr:`rows`, of :data:`ROW_KINDS`),
    and the name and coefficients the body's solve takes (:attr:`law`, :meth:`coefficients`). A law is a setting
    group, written into ``Contact(friction=...)``; every law passes ``mime.testing.friction_law_suite`` (passivity,
    the sticking limits, both dtypes, gradients) and its own oracle."""

    group: ClassVar[str] = "friction_law"
    #: The law's name as the body's solve takes it (``mime.nodes.body._rigid_inertial.FRICTION_LAWS``).
    law: ClassVar[str] = ""
    #: The row kinds the law may add beside each point's normal row, by :data:`ROW_KINDS`' names.
    rows: ClassVar[tuple[str, ...]] = ()

    def held_rows(self) -> tuple[str, ...]:
        """The row kinds this law's values add: :attr:`rows` by default."""
        return self.rows

    def solve_name(self) -> str:
        """The name the body's solve takes for this law's values: :attr:`law` by default."""
        return self.law

    def limit(self, load: float, speed: float) -> float:
        """The most the traction may be (N) at a normal ``load`` (N) and slip ``speed`` (m/s)."""
        raise NotImplementedError

    def coefficients(self) -> Mapping[str, float]:
        """The law's coefficients in SI by name, as the body's solve takes them."""
        raise NotImplementedError

    def problems(self) -> list[str]:
        """What is wrong with the law's values: nothing by default."""
        return []


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Coulomb(FrictionLaw):
    """Coulomb's friction, one coefficient ``mu`` for sticking and sliding alike, on the elliptic cone: a point sticks
    while the traction its motion needs is within ``mu N`` (``N`` the contact's normal force at the point), and slides
    otherwise, its traction ``mu N`` against its slip. Exact sticking, never a regularised creep: a block on a slope
    stays put below ``tan(theta) = mu``. The slip is the body's own point's (``v + omega x r``), so a body rolls without
    slipping where its sliding rows stick. A static coefficient above the kinetic one is another law, which adds a jump
    at breakaway, to follow with its evidence.

    ``rolling``, the rolling resistance ``mu_r`` (a length), adds the rolling row kind: a torque about the surface's
    tangents at each point bounded by ``mu_r N``, so a sphere held still on a slope below ``tan(theta) = mu_r / a``
    stays put and one rolling on a flat slows at ``(5/7) mu_r g / a``. Unwritten, no rolling rows are built (the
    program of the sliding pair alone); written, ``0`` included, they are, so ``mu_r`` is a study's axis, whose
    derivative at zero is the slope into ``mu_r >= 0`` (measured). It does not enter a synchronous rolling speed at all
    (the torque budget absorbs it until step-out): it is identified only from stepped-out data."""

    group = "coulomb"
    law = "coulomb"
    rows = ("sliding", "rolling")
    setting_roles = {"mu": convention(PHYSICAL, "a declared coefficient of the body's and the wall's surfaces in "
                                                "contact, never fitted unless a fit names it: static and kinetic in "
                                                "one"),
                     "rolling": convention(PHYSICAL, "none: no rolling rows, no rolling resistance; written, a length "
                                                     "(0 a study's start), the rolling row kind held")}
    mu: float
    rolling: Annotated[Quantity | None, Length] = None

    @override
    def held_rows(self) -> tuple[str, ...]:
        return ("sliding",) if self.rolling is None else ("sliding", "rolling")

    @override
    def solve_name(self) -> str:
        return "coulomb" if self.rolling is None else "coulomb_rolling"

    @override
    def limit(self, load: float, speed: float) -> float:
        return self.mu * load

    @override
    def coefficients(self) -> Mapping[str, float]:
        """``mu``, and ``mu_r`` where rolling resistance is written (SI: the setting's metres, as phase 1 holds a
        dimensioned value)."""
        found = {"mu": float(self.mu)}
        if self.rolling is not None:
            found["mu_r"] = float(cast(float, si(self.rolling, LENGTH, what="Coulomb's rolling=")))  # a scalar length
        return found

    @override
    def problems(self) -> list[str]:
        mu = cast(object, self.mu)  # a script's value, checked like a script's
        found: list[str] = []
        if not (isinstance(mu, (int, float)) and not isinstance(mu, bool) and math.isfinite(mu) and mu >= 0.0):
            found.append(f"friction's mu is {self.mu!r}, not a coefficient from 0 on: write Coulomb(mu=0.15)")
        if self.rolling is not None:
            try:
                rolling = si(self.rolling, LENGTH, what="Coulomb's rolling=")
            except (UnitError, TypeError):
                return [*found, f"friction's rolling is {self.rolling!r}, not a length: write rolling=um(1.0)"]
            if not (isinstance(rolling, float) and math.isfinite(rolling) and rolling >= 0.0):
                found.append(f"friction's rolling is {self.rolling!r}, not a length from 0 on")
        return found


@dataclass(frozen=True, kw_only=True)
class NormalLaw(SettingGroup):
    """How a contact holds a point along its normal: the least normal force it may take (:meth:`least`, zero for a
    hold that never pulls, ``-F_adh`` with an adhesion), and the load friction's limit reads (:meth:`load`, ``N +
    F_adh``). A law is a setting group, written into ``Contact(normal=...)``; every law passes
    ``mime.testing.normal_law_suite`` and its own oracle (a pull-off force, a rebound height)."""

    group: ClassVar[str] = "normal_law"

    def least(self) -> float:
        """The least normal force a point may take (N)."""
        raise NotImplementedError

    def load(self, normal: float) -> float:
        """The load friction's limit reads at the normal force ``normal`` (N)."""
        raise NotImplementedError

    def problems(self) -> list[str]:
        return []


#: Why an adhesion is not built in W2b, and the seam it plugs into.
ADHESION_RESERVED = ("adhesion (a load-independent pull, the limit mu (N + F_adh)) is reserved: its seam is Hold's "
                     "adhesion, the normal force allowed down to -F_adh, and it comes with its own oracle, the "
                     "pull-off force, and a measured value: none is measured wet for a mm-scale capsule")


@register_setting_group
@dataclass(frozen=True, kw_only=True)
class Hold(NormalLaw):
    """The hold, ``Contact``'s normal law by default: no penetration (a point arriving at the shrunk wall ends the step
    on it) and no rebound (the velocity into the wall is taken away, inelastic), the normal force never pulling.
    ``adhesion``, the pull-off force a point may hold, is the seam an adhesion plugs into: reserved (unwritten, it is
    absent, the default)."""

    group = "hold"
    setting_roles = {"adhesion": convention(PHYSICAL, f"absent: no pull. {ADHESION_RESERVED}")}
    adhesion: Annotated[Quantity | None, Force] = None

    @override
    def least(self) -> float:
        return 0.0

    @override
    def load(self, normal: float) -> float:
        return normal

    @override
    def problems(self) -> list[str]:
        return [] if self.adhesion is None else [ADHESION_RESERVED]


@dataclass(frozen=True)
class RowKind:
    """A kind of friction row a law may add beside a point's normal row: its rows' count and form, what they measure,
    what bounds them, the units of its force and its coefficient, and whether it is built (with the reason and seam
    where it is not)."""

    name: str
    count: int
    form: str
    measures: str
    bound: str
    units: str
    coefficient_units: str
    built: bool
    reason: str = ""


#: The row kinds, by name (decided Fr2): sliding built; rolling resistance with W2b's second pull request; spin
#: declared for soft contact.
ROW_KINDS: Mapping[str, RowKind] = {
    "sliding": RowKind("sliding", 2, "[t; r x t]", "the slip of the body's own point along the surface, v + omega x r",
                       "mu N, on the elliptic cone", "N", "1", True),
    "rolling": RowKind("rolling", 2, "[0; t]", "the body's rotation about the surface's tangents at the point",
                       "mu_r N, mu_r a length, on its own disk", "N m", "m", True),
    "spin": RowKind("spin", 1, "[0; n]", "the body's rotation about the normal at the point",
                    "mu_spin N, mu_spin a length", "N m", "m", False,
                    "spin friction needs a finite contact patch: zero for a rigid point contact, declared for soft "
                    "contact (0.4.0) or a phenomenological row kind where a case needs it"),
}


@dataclass(frozen=True)
class PointSet:
    """How a contact's points are held: each group's points, each with a weight its normal force is shared by; built or
    reserved, with the reason."""

    name: str
    description: str
    built: bool
    reason: str = ""


#: How a contact's points are held, by name: one point a group now; line and patch contacts later, matching the
#: near-contact law's line forms.
POINT_SETS: Mapping[str, PointSet] = {
    "point": PointSet("point", "each group's held point, weight one: up to eight points a body", True),
    "line": PointSet("line", "a line contact as a distributed set of points along it, with weights", False,
                     "a line contact is held at one point a group now (the contact's validity): its distributed set "
                     "comes with the near-contact law's line forms"),
    "patch": PointSet("patch", "a patch of contact as a distributed set of points over it, with weights", False,
                      "comes with soft contact (0.4.0)"),
}

#: The effects friction leaves out, each with the seam it will plug into (the contact's validity lines name them).
LEFT_OUT = {
    "adhesion": "Hold(adhesion=...): the normal force allowed down to -F_adh, friction's limit mu (N + F_adh)",
    "restitution": "a NormalLaw with a coefficient of restitution in place of Hold",
    "soft contact": "a NormalLaw reading a wall kind's declared compliance (0.4.0)",
    "speed-dependent friction": "a FrictionLaw whose limit(load, speed) reads the slip speed (Stribeck's, a fitted "
                                "table), or a static coefficient above the kinetic",
    "line contacts": "POINT_SETS['line'], a distributed point set with weights",
}

__all__ = ["ADHESION_RESERVED", "LEFT_OUT", "POINT_SETS", "ROW_KINDS", "Coulomb", "FrictionLaw", "Hold", "NormalLaw",
           "PointSet", "RowKind"]
