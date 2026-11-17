"""Whether MIME runs inside IPython: a notebook (Jupyter, Colab) or IPython's own terminal, where a cell can be run
again. IPython puts ``get_ipython`` among the builtins while it runs code, and ``IPython.get_ipython()`` returns the
running shell at any time (also after a cell has returned); a script, ``python -c`` and Python's own prompt have
neither. The run's live display asks this for its default, and a declaration made a second time asks it
for its rule (``mime.experiment``): one test, here, for both.

The test's reach is IPython's, not the notebook's alone: it is also true for a script run by ``ipython script.py`` or
``%run script.py``, and for a line typed at IPython's terminal prompt, so those runs get the notebook's rule and its
words. That is a decision (a recalled line is the same workflow as a cell run again), which is why the words MIME
says in it name "a notebook or IPython" and not "a notebook" alone."""

from __future__ import annotations

import builtins
import importlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

#: Where the notebook's rule applies, as MIME's words name it (the test's reach is IPython's, see the module).
WHERE = "a notebook or IPython"
#: What a refusal of one declaration made twice adds in a notebook or IPython, where a cell run twice is its usual cause
#: (a line, in IPython's terminal, and a script run through it, which have no cells).
RAN_TWICE = f"; in {WHERE}, a line or cell has probably run twice, changed the second time"
#: What that refusal says to do in a notebook or IPython.
RUN_AGAIN = ("run the lines or cells again from the one that makes the experiment, exp = Experiment(...), or run them "
             "all (Run all, in a notebook)")


def in_notebook() -> bool:
    """Whether an IPython shell is running in this process. The builtin ``get_ipython`` answers while a cell runs; the
    shell instance answers at any time, so a display called after its cell has returned (a thread, a ``post_execute``
    hook, a widget callback) is still in IPython. ``IPython`` is asked only where it is already imported (a process
    that never imported it has no shell, and this does not import it). IPython's terminal shell counts, as it did."""
    if hasattr(builtins, "get_ipython"):
        return True
    shell = getattr(sys.modules.get("IPython"), "get_ipython", None)
    return shell is not None and shell() is not None


def show_in_ipython(shown: object) -> None:
    """Show ``shown`` as IPython's own ``display(shown)`` does, so a call inside an ``if``, a loop or a function shows it
    (an object a call returns shows only as a cell's last expression). The callers (``mime.viz.display``,
    ``diagram.display``, ``figure.display``) check :func:`in_notebook` first and refuse outside IPython
    (``visuals.display-outside-ipython``), before they make what they show."""
    cast(Any, importlib.import_module("IPython.display")).display(shown)


def cell_number(shell: object, filename: str) -> int | None:
    """The number of the cell being run (``In [3]``). ``execution_count`` is not it: read inside a cell it is the
    number of the *next* cell in recent IPython (ipykernel 7, the terminal) and this cell's in older (Colab's), so the
    count of inputs IPython has stored (``In``, whose entry 0 is empty) is read instead, where there is one, and else
    the number an older IPython puts in a cell's file name (``<ipython-input-3-...>``)."""
    namespace = cast(object, getattr(shell, "user_ns", None))
    history = cast(Mapping[object, object], namespace).get("In") if isinstance(namespace, Mapping) else None
    if isinstance(history, Sequence) and len(cast(Sequence[object], history)) > 1:
        return len(cast(Sequence[object], history)) - 1
    named = re.match(r"<ipython-input-(\d+)-", filename)
    return int(named.group(1)) if named else None


def where_called() -> str:
    """Where the script's own code is, as a reader finds it: the cell and its line in a notebook or IPython (``cell
    [3], line 2``), the file and its line elsewhere. The innermost frame that is not a library's, the line that called
    into MIME; ``"somewhere in the script"`` where there is none."""
    from mime.errors._frames import caller_place

    place = caller_place()
    if place is None:
        return "somewhere in the script"
    cell = place.filename.startswith("<ipython-input") or "ipykernel_" in place.filename
    shell = getattr(builtins, "get_ipython", None)
    if cell and shell is not None:
        number = cell_number(shell(), place.filename)
        if number is not None:
            return f"cell [{number}], line {place.line}"
    return place.text()
