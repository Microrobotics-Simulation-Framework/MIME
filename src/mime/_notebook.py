"""Whether MIME runs inside IPython: a notebook (Jupyter, Colab) or IPython's own terminal, where a cell can be run
again. IPython puts ``get_ipython`` among the builtins while it runs code; a script, ``python -c`` and Python's own
prompt never have it. The run's live display asks this for its default, and a declaration made a second time asks it
for its rule (``mime.experiment``): one test, here, for both.

The test's reach is IPython's, not the notebook's alone: it is also true for a script run by ``ipython script.py`` or
``%run script.py``, and for a line typed at IPython's terminal prompt, so those runs get the notebook's rule and its
words. That is a decision (a recalled line is the same workflow as a cell run again), which is why the words MIME
says in it name "a notebook or IPython" and not "a notebook" alone."""

from __future__ import annotations

import builtins

#: Where the notebook's rule applies, as MIME's words name it (the test's reach is IPython's, see the module).
WHERE = "a notebook or IPython"
#: What a refusal of one declaration made twice adds in a notebook or IPython, where a cell run twice is its usual cause.
RAN_TWICE = f"; in {WHERE}, a cell has probably run twice, changed the second time"
#: What that refusal says to do in a notebook or IPython.
RUN_AGAIN = ("run the cells again from the one that makes the experiment, exp = Experiment(...), or run them all "
             "(Run all)")


def in_notebook() -> bool:
    return hasattr(builtins, "get_ipython")
