"""The fragment's closed shape, in the saved document's own types: read without postponed annotations, as the
document's are, so that the run-time check reads each key's ``NotRequired`` (:func:`mime.saved._schema.shape_problems`)."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.saved._schema import (
    BoundEntryDoc,
    ControllerDoc,
    DeclaredDoc,
    MaterialDoc,
    RangeDoc,
    ReasonDoc,
    TermsDoc,
    ThingDoc,
)

FRAGMENT_SCHEMA = "mime.fragment/1"


class FragmentDoc(TypedDict, closed=True):
    """A block's fragment: its form, then each of the document's sections of declarations it holds, in the document's
    own shapes, every one optional."""

    schema: ReadOnly[Literal["mime.fragment/1"]]
    things: NotRequired[dict[str, ThingDoc]]
    interactions: NotRequired[list[BoundEntryDoc]]
    observations: NotRequired[list[BoundEntryDoc]]
    controller: NotRequired[ControllerDoc | None]
    declared: NotRequired[dict[str, DeclaredDoc]]
    ranges: NotRequired[dict[str, RangeDoc]]
    sources: NotRequired[dict[str, ReasonDoc]]
    materials: NotRequired[dict[str, MaterialDoc]]
    terms: NotRequired[dict[str, TermsDoc]]


__all__ = ["FRAGMENT_SCHEMA", "FragmentDoc"]
