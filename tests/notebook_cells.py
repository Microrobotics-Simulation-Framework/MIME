"""A notebook simulated where MIME meets IPython, for the tests of ``display`` (``mime.viz.display``, ``diagram.display``,
``figure.display``): ``get_ipython`` among the builtins, a stand-in for ``IPython.display.display`` that records the
``display_data`` bundle IPython would send (``text/plain`` and, from the object's ``_repr_html_``, ``text/html``), and cells
run as IPython runs them (their statements, then the last expression shown unless it is ``None``).

IPython is not a test dependency (a real kernel was run once for the pull request that introduced ``display``'s
behaviour, not in CI), so a test of what a cell shows is made here, with its control: the call that returns the object
shows nothing inside an ``if``."""

import ast
import builtins
import sys
import types

import pytest


class Cells:
    """The outputs of ``display_data`` that IPython's ``display`` would send, and cells run as IPython runs them."""

    def __init__(self) -> None:
        self.outputs: list[dict[str, str]] = []
        self.namespace: dict[str, object] = {}

    def show(self, shown: object) -> None:
        """``IPython.display.display(shown)``: the bundle of the formats the object has."""
        bundle = {"text/plain": repr(shown)}
        if hasattr(shown, "_repr_html_"):
            bundle["text/html"] = shown._repr_html_()
        self.outputs.append(bundle)

    def run(self, source: str) -> None:
        """A cell: its statements run, then its last expression shown when it is not ``None`` (the cell's ``Out``)."""
        tree = ast.parse(source)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<cell>", "exec"), self.namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), self.namespace)
            if value is not None:
                self.show(value)

    @property
    def html(self) -> list[str]:
        return [bundle["text/html"] for bundle in self.outputs if "text/html" in bundle]


def install(monkeypatch: pytest.MonkeyPatch) -> Cells:
    """A notebook in this process: ``get_ipython`` in the builtins and ``IPython.display.display`` recording."""
    cells = Cells()
    monkeypatch.setattr(builtins, "get_ipython", lambda: object(), raising=False)
    ipython, shown = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    shown.display = cells.show  # pyright: ignore[reportAttributeAccessIssue]
    ipython.display = shown  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", shown)
    return cells
