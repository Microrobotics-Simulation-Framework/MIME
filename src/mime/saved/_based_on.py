"""The saved experiment an experiment was made from (``Experiment.based_on``), as the new one's document records it
(version 7 of the saved form): the parent's name, the input hash its lock records, the MIME version that saved it, and
the DOI of the record it was published as. That DOI is the one the parent was pulled by, where it was pulled by a DOI,
and otherwise the one its lock names (every bundle ``mime push`` publishes does); a text that is not a DOI's shape is
not recorded. The immediate parent alone: a chain is followed through each parent's own document."""

from __future__ import annotations

import re
import unicodedata
from typing import cast

from mime._terminal import plain
from mime.saved._folder import SavedForm
from mime.saved._schema import BasedOnDoc

#: The longest DOI recorded, in characters: far above any a repository mints, far below one that floods a card.
DOI_LIMIT = 256
#: A DOI as ``mime pull`` takes one: ``10.``, the registrant's number, a slash, and a suffix with no whitespace.
_DOI_SHAPE = re.compile(r"10\.\d{4,9}/\S+")
#: How much of a DOI that is not recorded a warning quotes.
_QUOTED = 80


def doi_problem(doi: str) -> str | None:
    """Why ``doi`` is not recorded as the DOI of the experiment another was based on, or ``None``: a DOI is
    ``10.<registrant>/<suffix>``, with no whitespace and no control or format character, at most :data:`DOI_LIMIT`
    characters long."""
    if any(unicodedata.category(character).startswith("C") for character in doi):
        return "it holds a control or format character"
    if len(doi) > DOI_LIMIT:
        return f"it is {len(doi)} characters long, and a DOI recorded is at most {DOI_LIMIT}"
    if not _DOI_SHAPE.fullmatch(doi):
        return "it is not a DOI's shape, 10.<registrant>/<suffix> with no whitespace"
    return None


def _quoted(text: str) -> str:
    return f"'{plain(text[:_QUOTED])}'" + (f" (the first {_QUOTED} of its {len(text)} characters)"
                                          if len(text) > _QUOTED else "")


def based_on_of(form: SavedForm, pulled_doi: str | None = None) -> tuple[BasedOnDoc, list[str]]:
    """What an experiment made from the saved ``form`` records of it, and what to warn of in doing so. ``pulled_doi``
    is the DOI of the record the form was pulled from, where it was pulled by a DOI: it is recorded, and a lock that
    names another record is warned of, naming both. Otherwise the DOI recorded is the one the lock's ``publication``
    names, if any. A DOI that is not a DOI's shape (:func:`doi_problem`) is not recorded, and the warning says why."""
    lock = form.lock
    name = form.document["experiment"]
    publication = lock.get("publication")
    stated = publication["doi"] if publication is not None else None
    notes: list[str] = []
    doi = stated
    if pulled_doi is not None:
        if stated is not None and stated.casefold() != pulled_doi.casefold():
            notes.append(f"{plain(name)!r} was pulled from the record {_quoted(pulled_doi)}, and its lock names the "
                         f"record {_quoted(stated)}: the new experiment records {_quoted(pulled_doi)}, the record it "
                         f"was pulled from")
        doi = pulled_doi
    problem = doi_problem(doi) if doi is not None else None
    if doi is not None and problem is not None:
        notes.append(f"the DOI of {plain(name)!r}, {_quoted(doi)}, is not recorded: {problem}; the new experiment "
                     f"names it by its input hash")
        doi = None
    held = {"experiment": name, "input_hash": lock["hashes"]["input"], "mime": lock["environment"]["mime"]}
    return cast(BasedOnDoc, held | ({"doi": doi} if doi is not None else {})), notes


def based_on_words(held: BasedOnDoc) -> str:
    """The parent as the card names it: ``Based on <name> (<its DOI, or its input hash where it has none>)``."""
    doi = held.get("doi")
    named = f"DOI {plain(doi)}" if doi is not None else f"input hash {plain(held['input_hash'])}"
    return f"Based on {plain(held['experiment'])} ({named})"


__all__ = ["DOI_LIMIT", "based_on_of", "based_on_words", "doi_problem"]
