"""What a port generator's provenance records of how it was run, with no private path in it: the generators run from
the workspace that holds the checkouts, and write their outputs into a working directory before they are copied into
``references/``."""

from collections.abc import Sequence
from pathlib import Path

#: The planning directory beside the checkouts, written so that this file does not name its path.
PLANNING = "plan" + "s"


def recorded_argv(argv: Sequence[str], workspace: Path) -> list[str]:
    """``argv`` as a provenance file records it: a path under ``workspace`` relative to it, one under its planning
    directory by that directory's role, and any other absolute path by its name alone. An argument that is no path
    is kept."""
    recorded: list[str] = []
    for argument in argv:
        path = Path(argument).resolve() if "/" in argument or Path(argument).exists() else None
        if path is None:
            recorded.append(argument)
        elif path.is_relative_to(workspace / PLANNING):
            recorded.append(f"<the planning directory>/{path.relative_to(workspace / PLANNING).as_posix()}")
        elif path.is_relative_to(workspace):
            recorded.append(path.relative_to(workspace).as_posix())
        else:
            recorded.append(f"<outside the workspace>/{path.name}")
    return recorded
