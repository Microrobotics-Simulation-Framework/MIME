"""Run as ``python -m mime.testing._fresh_load <folder> [--document-only]`` by the round-trip harness: a load in a
process that has seen nothing of the experiment but its folder.

It writes ``fresh_load.json`` (the derived hash it rebuilt) and ``fresh_load.npz`` (every field over the lock's
reference steps) into the folder. With ``--document-only`` it builds the document alone and prints the codes its
build is refused with.
"""

import json
import sys
from pathlib import Path

import numpy as np

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.testing._saved import first_steps


def main(argv: list[str]) -> int:
    folder = Path(argv[0])
    if argv[1:] == ["--document-only"]:
        try:
            Experiment.from_document(folder / DOCUMENT_FILE).build()
        except BuildRefused as refused:
            print(f"refused: {json.dumps(refused.codes())}")
            return 0
        print("built")
        return 0
    experiment = Experiment.load(folder)
    built = experiment.build()
    report = experiment.load_report
    steps = int(json.loads((folder / LOCK_FILE).read_text())["reference"]["steps"])
    history = first_steps(built, steps)
    np.savez(folder / "fresh_load.npz", allow_pickle=False,
             **{f"{node}/{name}": value for node, fields in history.items() for name, value in fields.items()})
    (folder / "fresh_load.json").write_text(json.dumps({
        "derived_hash": built.report.derived_hash,
        "load_report": report.text() if report else None,
    }))
    print(report.text() if report else "no load report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
