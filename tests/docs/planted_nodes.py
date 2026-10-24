"""Test-only nodes that carry ``NodeMeta`` and a symbol table, for the symbol-table checks' controls to run on: no
MIME node carries either until the pilot's. ``PlantedNode``'s table is clean, and every check passes on it; each
control plants one defect in a copy of it.

Its law, ``F = -c v - mu a L x_{i}``: ``c`` a parameter, ``v`` an input, ``F`` the state, ``mu`` an entry setting, ``a``
a constant, ``L`` a parameter with no units declared, and ``x`` a state indexed by ``i`` in a braced subscript."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from maddening.core.compliance.metadata import NodeMeta
from maddening.core.params import ParamSpec

from mime.core import MimeNode, Symbol, SymbolKind, as_run_array, jnp, node_input

EQUATIONS = r"\mathbf{F} = -c \, \mathbf{v} - \frac{\mu \, a \, L}{\mathrm{d}t} \, x_{i}"


def drag_law(c: float, v: Any) -> Any:
    """A module-level function a row may name."""
    return -c * v


class Unrelated:
    """A class outside every planted node's bases."""

    def helper(self) -> None:
        return None


class PlantedBase(MimeNode):
    def base_helper(self) -> None:
        return None


ROWS = (
    Symbol(r"\mathbf{F}", SymbolKind.STATE, "force", "N; N m", "planted_nodes.PlantedNode.advance"),
    Symbol("c", SymbolKind.PARAMETER, "drag", "kg s^-1", "planted_nodes.drag_law"),
    Symbol(r"\mathbf{v}", SymbolKind.INPUT, "velocity", "m/s", "planted_nodes.PlantedNode.advance"),
    Symbol(r"\mu", SymbolKind.SETTING, "viscosity", "Pa s", "planted_nodes.PlantedNode.advance"),
    Symbol("a", SymbolKind.CONSTANT, "mime.units.VACUUM_PERMEABILITY", "N A^-2", "planted_nodes.PlantedNode.advance"),
    Symbol("L", SymbolKind.PARAMETER, "length", "m", "planted_nodes.PlantedBase.base_helper"),
    Symbol("t", SymbolKind.STATE, "time", "s", "planted_nodes.PlantedNode.advance"),
    Symbol("x", SymbolKind.STATE, "position", "m", "planted_nodes.PlantedNode.advance"),
)
META = NodeMeta(algorithm_id="MIME-NODE-901", governing_equations=EQUATIONS)


class PlantedNode(PlantedBase):
    meta = META
    symbols = ROWS

    def __init__(self, name: str = "planted", timestep: float = 1e-3, *, drag: float = 0.5,
                 length: float = 1e-3) -> None:
        super().__init__(name, timestep, drag=drag, length=length)

    def param_specs(self) -> dict[str, ParamSpec]:
        return {"drag": ParamSpec(units="kg s^-1"), "length": ParamSpec()}

    def initial_state(self) -> dict[str, Any]:
        return {"force": as_run_array(jnp.zeros(6)), "time": as_run_array(0.0), "position": as_run_array(jnp.zeros(3))}

    def boundary_input_spec(self) -> dict[str, Any]:
        return {"velocity": node_input("m/s", (3,))}

    def advance(self, p: Mapping[str, Any], state: dict[str, Any], boundary_inputs: Mapping[str, Any],
                dt: float) -> dict[str, Any]:
        return dict(state)


class PlantedSubNode(PlantedNode):
    def extra(self) -> None:
        return None


def planted(name: str = "Doctored", base: type = PlantedNode, **attributes: Any) -> type:
    """A copy of ``base`` under ``name`` with ``attributes`` replaced: ``meta=``, ``symbols=``, or any method."""
    return type(name, (base,), {"__module__": __name__, **attributes})


def with_equations(equations: str) -> NodeMeta:
    return replace(META, governing_equations=equations)


def with_row(index: int, **changed: str | SymbolKind) -> tuple[Symbol, ...]:
    return tuple(replace(row, **changed) if at == index else row for at, row in enumerate(ROWS))  # type: ignore[arg-type]
