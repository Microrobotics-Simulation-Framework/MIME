"""A private home for the sharing tests: every directory a test could write a user's files into (its home, its config,
cache, data and state directories, on Linux, macOS and Windows) moved into one folder, no token in the environment,
and git's user configuration out of reach, so ``push``'s records, ``pack``'s token scan and ``pull``'s cache never reach
the real ones, whatever the shell running the tests has set."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import platformdirs
import pytest

from mime import credentials

#: Where platformdirs reads a user's directories from: XDG's variables on Linux, the home on macOS (``~/Library``), and
#: on Windows its own overrides first, then the profile's variables.
_XDG = {"XDG_CONFIG_HOME": ".config", "XDG_CACHE_HOME": ".cache", "XDG_STATE_HOME": ".local/state",
        "XDG_DATA_HOME": ".local/share"}
_WINDOWS = ("APPDATA", "LOCALAPPDATA", "WIN_PD_OVERRIDE_APPDATA", "WIN_PD_OVERRIDE_LOCAL_APPDATA")
_GIT = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


def isolate(patch: pytest.MonkeyPatch, home: Path) -> Path:
    """Every user directory under ``home`` (the home itself, XDG's four, Windows' application data), no token in the
    environment, git's user configuration in ``home``, and the network switches at their defaults."""
    home.mkdir(parents=True, exist_ok=True)
    patch.setenv("HOME", str(home))
    for variable, below in _XDG.items():
        patch.setenv(variable, str(home / below))
    for variable in _WINDOWS:
        patch.setenv(variable, str(home / "AppData"))  # Windows: platformdirs reads its overrides first
    for variable in (*credentials.SERVICES.values(), *_GIT):
        patch.delenv(variable, raising=False)
    patch.setenv("GIT_CONFIG_GLOBAL", str(home / "gitconfig"))  # a token's store asks git, which reads the user's
    patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    patch.delenv("MIME_OFFLINE", raising=False)
    return home


@dataclass(frozen=True)
class Outside:
    """The user directories a test would write into if its own isolation failed: those of its module, where the
    module-scoped fixtures run. ``state`` is the state directory there, as platformdirs names it."""

    home: Path
    state: Path

    def written(self) -> list[Path]:
        """Every file under the state directory: a test's records belong in its own, so this stays empty."""
        return sorted(path for path in self.state.rglob("*") if path.is_file()) if self.state.exists() else []


def outside(root: Path) -> Iterator[Outside]:
    with pytest.MonkeyPatch.context() as patch:
        home = isolate(patch, root / "home")
        patch.setenv("MIME_ASSETS_DIR", str(root / "assets-cache"))
        yield Outside(home, Path(platformdirs.user_state_dir("mime")))


__all__ = ["Outside", "isolate", "outside"]
