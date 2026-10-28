"""Entries planted for the end-to-end controls of ``mime docs check``: a fresh process registers one with
:func:`plant`, then runs the real command, so no test's process holds them. Each carries one defect that a single
stage of the check finds, or one value it counts, besides ``Documented``, whose node's table is clean, and the
passing arms ``KeptId`` and ``MimeKnown``. The entries named without the ``docsplant.`` prefix and defined under
``mime`` are MIME's to the check, and so need a hand-written file to pass the docs walk: a control writes it in its
copy of the checkout. :func:`references_before_the_plant` prints, in that process and before the plant, the
references line the check would print: a planted run's baseline, which no entry another test registers can reach."""

from dataclasses import dataclass

from checkout import ROOT
from planted_nodes import (EQUATIONS, KeeperNode, NamedLikeRigidBody, PlantedMimeMetaOnly, PlantedMimeNode, PlantedNode,
                           TakesNextNode, UsurperNode, planted, with_equations)

from mime.docs._check import Report, bibliography_keys, entry_problems, read_lists
from mime.docs._export import load_namespaces
from mime.gradients import GradientStatus
from mime.registry import NOT_ESTABLISHED, GradientRecord, register_interaction, registered
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


@dataclass(frozen=True, kw_only=True)
class RetiredId(ToySpring):
    """A spring declaring a node named as the 0.2 line's RigidBodyNode was, taking that node's retired id."""

    name = "docsplant.retired_id"
    builds = {"": (NamedLikeRigidBody,)}


@dataclass(frozen=True, kw_only=True)
class KeptId(ToySpring):
    """A spring declaring ``KeeperNode``, which a doctored frozen list names as keeping MIME-NODE-012."""

    name = "docsplant.kept_id"
    builds = {"": (KeeperNode,)}


@dataclass(frozen=True, kw_only=True)
class TakenId(ToySpring):
    """A spring declaring ``UsurperNode``, taking MIME-NODE-012, which a doctored frozen list keeps for another."""

    name = "docsplant.taken_id"
    builds = {"": (UsurperNode,)}


@dataclass(frozen=True, kw_only=True)
class LabUnestablished(ToySpring):
    """A lab's spring whose references its code does not establish."""

    name = "docsplant.unestablished"
    references = NOT_ESTABLISHED


@dataclass(frozen=True, kw_only=True)
class LabUnestablishedGradient(ToySpring):
    """A lab's spring, exact with its random keys held fixed, whose gradient record's three fields its code does not
    establish."""

    name = "docsplant.unestablished_gradient"
    differentiability = GradientStatus.FIXED_KEY
    gradient = GradientRecord(where=NOT_ESTABLISHED, reason=NOT_ESTABLISHED, route=NOT_ESTABLISHED)


@dataclass(frozen=True, kw_only=True)
class LabCited(ToySpring):
    """A lab's spring citing a key MIME's bibliography does not hold."""

    name = "docsplant.cited"
    references = ("NoSuchWork1900",)


@dataclass(frozen=True, kw_only=True)
class MimeCited(ToySpring):
    """A spring of MIME's citing a key its bibliography does not hold."""

    name = "docsplant_mime_cited"
    references = ("NoSuchWork1900",)


@dataclass(frozen=True, kw_only=True)
class MimeKnown(ToySpring):
    """A spring of MIME's citing a key its bibliography holds."""

    name = "docsplant_mime_known"
    references = ("Purcell1977",)


@dataclass(frozen=True, kw_only=True)
class MimeUnpaged(ToySpring):
    """A spring of MIME's with no hand-written file: the docs walk's find."""

    name = "docsplant_mime_unpaged"


@dataclass(frozen=True, kw_only=True)
class MimeNoded(ToySpring):
    """A spring of MIME's declaring a node of MIME's that carries ``NodeMeta`` and a symbol table."""

    name = "docsplant_mime_noded"
    builds = {"": (PlantedMimeNode,)}


@dataclass(frozen=True, kw_only=True)
class MimeMetaOnly(ToySpring):
    """A spring of MIME's declaring a node of MIME's that carries ``NodeMeta`` and no symbol table."""

    name = "docsplant_mime_meta_only"
    builds = {"": (PlantedMimeMetaOnly,)}


@dataclass(frozen=True, kw_only=True)
class NextId(ToySpring):
    """A spring declaring a node taking a retired id beside one taking ``MIME-NODE-104``: the next id is past both."""

    name = "docsplant.next_id"
    builds = {"": (NamedLikeRigidBody, TakesNextNode)}


MimeCited.__module__ = MimeKnown.__module__ = MimeUnpaged.__module__ = MimeNoded.__module__ = "mime.docsplant"
MimeMetaOnly.__module__ = "mime.docsplant"
PLANTS = {"documented": Documented, "unequated": Unequated, "undeclared": Undeclared, "retired_id": RetiredId,
          "kept_id": KeptId, "taken_id": TakenId, "lab_unestablished": LabUnestablished,
          "lab_unestablished_gradient": LabUnestablishedGradient, "lab_cited": LabCited, "mime_cited": MimeCited,
          "mime_known": MimeKnown, "mime_unpaged": MimeUnpaged, "mime_noded": MimeNoded, "mime_meta_only": MimeMetaOnly,
          "next_id": NextId}


def plant(name: str) -> None:
    register_namespace(NAMESPACE, "0.0.1")
    register_interaction(PLANTS[name])


def references_before_the_plant() -> None:
    """The check's references line for the entries this process holds now, by the check's own count (the namespaces
    it loads, read as it reads them), printed as ``before the plant: references: ...``."""
    load_namespaces()
    report = Report()
    entry_problems(registered(), read_lists(ROOT)["not_established"], report, bibliography=bibliography_keys(ROOT))
    (line,) = [line for line in report.summary if line.startswith("references:")]
    print(f"before the plant: {line}")
