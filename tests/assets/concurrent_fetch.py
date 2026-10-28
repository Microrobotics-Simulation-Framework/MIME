"""One of the processes of the concurrency test: fetch every reference it is given, once all of them are released.

Run as ``python concurrent_fetch.py <ready file> <go file> [--unlocked]`` with the references as a JSON list of lock
records on its standard input, and ``MIME_ASSETS_DIR`` set. It touches its ready file after importing (the import is
the slow part), waits for the go file so that every process reaches the cache together, then prints one line:
``ok`` or the exception that stopped it. ``--unlocked`` is the control: the per-asset lock is replaced by nothing.
"""

import contextlib
import json
import sys
import time
from pathlib import Path

from mime import assets
from mime.assets import _fetch


def main(arguments: list[str]) -> int:
    ready, go = Path(arguments[0]), Path(arguments[1])
    if "--unlocked" in arguments:
        _fetch.exclusive = lambda path: contextlib.nullcontext()  # type: ignore[assignment]
    refs = [assets.AssetRef.from_record(record) for record in json.load(sys.stdin)]
    ready.touch()
    while not go.exists():
        time.sleep(0.01)
    try:
        for ref in refs:
            assets.fetch(ref)
            assert assets.locate(ref).stat().st_size == ref.size
    except BaseException as error:  # noqa: BLE001 - whatever stops the fetch is the test's evidence
        print(f"{type(error).__name__}: {str(error)[:300]}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
