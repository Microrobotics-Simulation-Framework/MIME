"""The saved experiment an experiment was made from (``Experiment.based_on``), as the new one's document records it
(version 7 of the saved form): the parent's name, the input hash its lock records, the MIME version that saved it, and
the DOI of the record it was published as, where its lock names one (every bundle ``mime push`` publishes does). The
immediate parent alone: a chain is followed through each parent's own document."""

from __future__ import annotations

from typing import cast

from mime._terminal import plain
from mime.saved._folder import SavedForm
from mime.saved._schema import BasedOnDoc


def based_on_of(form: SavedForm) -> BasedOnDoc:
    """What an experiment made from the saved ``form`` records of it."""
    lock = form.lock
    publication = lock.get("publication")
    held = {"experiment": form.document["experiment"], "input_hash": lock["hashes"]["input"],
            "mime": lock["environment"]["mime"]}
    return cast(BasedOnDoc, held | ({"doi": publication["doi"]} if publication is not None else {}))


def based_on_words(held: BasedOnDoc) -> str:
    """The parent as the card names it: ``Based on <name> (<its DOI, or its input hash where it has none>)``."""
    doi = held.get("doi")
    named = f"DOI {plain(doi)}" if doi is not None else f"input hash {plain(held['input_hash'])}"
    return f"Based on {plain(held['experiment'])} ({named})"


__all__ = ["based_on_of", "based_on_words"]
