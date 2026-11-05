"""The experiments the views of an experiment (``built.to_mermaid()``, ``exp.tree()``) are tested and documented on: tutorial
A0 as ``examples/tutorial_a0.py`` declares it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mime.experiment import Experiment

ROOT = Path(__file__).resolve().parents[1]


def tutorial_a0() -> Experiment:
    """Tutorial A0's experiment as ``examples/tutorial_a0.py`` declares it, without its run: the file's source up to its
    ``run`` region, executed, with ``script_a`` the examples' (the tests have a script A of their own under that name)."""
    spec = importlib.util.spec_from_file_location("example_script_a", ROOT / "examples" / "script_a.py")
    assert spec is not None and spec.loader is not None
    examples_script_a = importlib.util.module_from_spec(spec)
    sys.modules["example_script_a"] = examples_script_a
    spec.loader.exec_module(examples_script_a)
    source = (ROOT / "examples" / "tutorial_a0.py").read_text(encoding="utf-8").split("# region: run")[0]
    held = sys.modules.get("script_a")
    sys.modules["script_a"] = examples_script_a
    try:
        names = {"__name__": "tutorial_a0_declarations", "__file__": str(ROOT / "examples" / "tutorial_a0.py")}
        exec(compile(source, "examples/tutorial_a0.py", "exec"), names)  # noqa: S102 - the repository's own example
    finally:
        sys.modules.pop("example_script_a", None)
        if held is None:
            sys.modules.pop("script_a", None)
        else:
            sys.modules["script_a"] = held
    return names["close_the_loop"](names["a0"]())
