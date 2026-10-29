#!/usr/bin/env python
"""How many shards each lane's tests run as, for ``.github/workflows/ci.yml``: its ``shards`` job runs this, and the
shard jobs' matrix, their ``--mime-shard`` and the lanes' ``scripts/check_shards.py`` all read the count it writes to
``$GITHUB_OUTPUT`` as ``count`` (and the matrix as ``list``, JSON). Standard library only.

The count is, in order of precedence:

- the ``shards`` input of a manual run (``workflow_dispatch``), when it is not empty;
- the repository variable ``MIME_CI_SHARDS`` (Settings, Secrets and variables, Actions, Variables), when it is set;
- ``DEFAULT``.

It is a whole number from ``MINIMUM`` to ``MAXIMUM``; anything else is refused, naming where it came from, and the
shards job fails, so no shard job and no lane starts on a count nobody chose. Each shard's job costs the same
job-minutes whatever the count (setup is about 20 s a job), so a larger count shortens a run's wall time and raises the
jobs it holds against the organisation's concurrent limit (60 on the Team plan), and a smaller one does the opposite.
The split itself, ``tests/_sharding.py``, works for any count.
"""

from __future__ import annotations

import argparse
import json
import os
import re

MINIMUM, MAXIMUM, DEFAULT = 1, 12, 6
VARIABLE = "MIME_CI_SHARDS"


class Refused(ValueError):
    """A shard count that cannot be used, naming where it came from."""


def resolve(variable: str, manual: str) -> tuple[int, str]:
    """``(count, where it came from)`` from the repository variable's value and a manual run's input (both ``""``
    when unset); raises ``Refused`` on a value that is not a whole number within the bounds."""
    for source, value in (("the shards input of this manual run", manual),
                          (f"the repository variable {VARIABLE}", variable)):
        text = value.strip()
        if not text:
            continue
        if not re.fullmatch(r"[0-9]+", text):
            raise Refused(f"{source} is {value!r}, which is not a whole number of shards from {MINIMUM} to {MAXIMUM}")
        count = int(text)
        if not MINIMUM <= count <= MAXIMUM:
            raise Refused(f"{source} is {count}, outside the {MINIMUM} to {MAXIMUM} shards a lane may run as")
        return count, source
    return DEFAULT, f"the default, when {VARIABLE} is unset"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variable", default="", help=f"the repository variable {VARIABLE}'s value, empty if unset")
    parser.add_argument("--input", default="", help="a manual run's shards input, empty if none")
    args = parser.parse_args(argv)
    try:
        count, source = resolve(args.variable, args.input)
    except Refused as error:
        print(f"::error title=Shards::{error}")
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"count={count}\nlist={json.dumps(list(range(1, count + 1)))}\n")
    print(f"{count} shards per lane ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
