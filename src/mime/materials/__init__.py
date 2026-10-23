"""The seam to OpenMaterica, with the inline floor: the one place that reads the values of things (a body's mass, a
medium's density) and the only importer of OpenMaterica. Today it reads values declared in the experiment."""

from mime.materials._values import (
    DECLARED_IN_THE_EXPERIMENT,
    VALUE_KINDS,
    Declared,
    Directed,
    ResolvedValue,
    ValueKind,
    declaration_problems,
    resolve,
    value_kind,
)

__all__ = [
    "DECLARED_IN_THE_EXPERIMENT",
    "VALUE_KINDS",
    "Declared",
    "Directed",
    "ResolvedValue",
    "ValueKind",
    "declaration_problems",
    "resolve",
    "value_kind",
]
