"""Entries planted for the end-to-end controls of ``mime docs check``: a fresh process registers one with
:func:`plant`, then runs the real command, so no test's process holds them. Each carries one defect that a single
stage of the check finds, besides ``Documented``, whose node's table is clean."""

from dataclasses import dataclass

from planted_nodes import EQUATIONS, PlantedNode, planted, with_equations

from mime.registry import register_interaction
from mime.testing.toys import ToySpring
from mime.vocabulary import register_namespace

NAMESPACE = "docsplant"
#: ``PlantedNode`` with the token ``a`` gone from its equations, a row still naming it: the symbol-table checks' find.
UnequatedNode = planted("UnequatedNode", meta=with_equations(EQUATIONS.replace(r"\, a \,", r"\,")))


@dataclass(frozen=True, kw_only=True)
class Documented(ToySpring):
    """A spring declaring a node whose symbol table is clean."""

    name = "docsplant.documented"
    builds = {"": (PlantedNode,)}


@dataclass(frozen=True, kw_only=True)
class Unequated(ToySpring):
    """A spring declaring ``UnequatedNode``."""

    name = "docsplant.unequated"
    builds = {"": (UnequatedNode,)}


@dataclass(frozen=True, kw_only=True)
class Undeclared(ToySpring):
    """A spring holding a value at class level that it does not declare a ``ClassVar``: the completeness test's find."""

    name = "docsplant.undeclared"
    table = {"a": 1}


PLANTS = {"documented": Documented, "unequated": Unequated, "undeclared": Undeclared}


def plant(name: str) -> None:
    register_namespace(NAMESPACE, "0.0.1")
    register_interaction(PLANTS[name])
