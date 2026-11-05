"""Whether MIME runs inside IPython: a notebook (Jupyter, Colab) or IPython's own terminal, where a cell can be run
again. IPython puts ``get_ipython`` among the builtins while it runs code; a script, ``python -c`` and Python's own
prompt never have it. The run's live display asks this for its default, and a declaration made a second time asks it
for its rule (``mime.experiment``): one test, here, for both."""

from __future__ import annotations

import builtins

#: What a refusal of one declaration made twice adds in a notebook, where a cell run twice is its usual cause.
RAN_TWICE = "; in a notebook, a cell has probably run twice, changed the second time"
#: What that refusal says to do in a notebook.
RUN_AGAIN = ("run the cells again from the one that makes the experiment, exp = Experiment(...), or run them all "
             "(Run all)")


def in_notebook() -> bool:
    return hasattr(builtins, "get_ipython")
