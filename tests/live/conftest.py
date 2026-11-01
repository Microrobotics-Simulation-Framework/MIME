"""The live tests' servers are stopped on every exit path: ``launch`` starts a process in a process group of its own and
kills the whole group when the test ends, passed, failed, errored or interrupted. A test whose pytest worker is killed
outright runs no finalizer; ``mime serve`` then stops by itself, its launcher gone
(``mime.live._serve.LauncherWatch``)."""

import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from process_groups import stop_group


@pytest.fixture
def launch() -> Iterator[Callable[..., subprocess.Popen]]:
    """Start a process (``subprocess.Popen``'s arguments) in a process group of its own, stopped with its group when
    the test ends."""
    started: list[subprocess.Popen] = []

    def start(args: Sequence[str], **options: Any) -> subprocess.Popen:
        if sys.platform == "win32":
            options.setdefault("creationflags", subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            options.setdefault("start_new_session", True)
        process = subprocess.Popen(list(args), **options)
        started.append(process)
        return process

    try:
        yield start
    finally:
        for process in started:
            stop_group(process)
