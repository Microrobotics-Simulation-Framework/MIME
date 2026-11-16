"""The symbol table: a node's equations joined to its code, one row per symbol, beside the node's ``NodeMeta``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SymbolKind(Enum):
    """Where a symbol's ``name`` is declared: a parameter in ``param_specs()``, a state field in ``initial_state()``,
    an input in ``boundary_input_spec()``, an output in ``boundary_flux_spec()``, a dimensioned setting of the entry
    that builds the node, or a constant, which no spec declares."""

    PARAMETER = "parameter"
    STATE = "state"
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"
    SETTING = "entry setting"


@dataclass(frozen=True)
class Symbol:
    """One symbol of a node's ``governing_equations``: its LaTeX as the equations write it, where the code declares
    it (``kind`` and ``name``), its units as this row states them, and the dotted path of the function, method or
    property that computes with it, named on the class that defines it, which is the node's class or a base of it,
    or a module-level function."""

    latex: str
    kind: SymbolKind
    name: str
    units: str
    implemented_by: str
    #: ``units`` is text and no unit label (a drive's value, in whatever units its command is declared in). Every other
    #: row's units are read where the row is defined (:func:`mime.units.parse_label`), so a typo is the ``units.label``
    #: refusal at import, never a row nothing compares.
    prose_units: bool = False

    def __post_init__(self) -> None:
        if not self.prose_units:
            from mime.units import parse_label

            parse_label(self.units)
