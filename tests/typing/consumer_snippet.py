# pyright: strict, reportUnnecessaryTypeIgnoreComment=error
"""A consumer's code, type-checked by CI against the installed wheel rather than the source tree; never run.

Strict pyright refuses the import of a package without ``py.typed``, and ``assert_type`` fails when a public type
changes. A write into the read-only lock must be refused: were it ever accepted, the ignore on that line would be
unnecessary, which the line above makes an error.
"""

from pathlib import Path
from typing import assert_type

import mime
from mime import Experiment
from mime.build import BuildRefused, BuildReport
from mime.experiment import BuiltExperiment
from mime.saved import ExperimentDocument, LoadReport, Lock, VerifyReport
from mime.units import Quantity, mm, s

experiment = Experiment("snippet", up=(0.0, 0.0, 1.0), dt=s(0.01))
assert_type(experiment, Experiment)
assert_type(mm(3.0), Quantity)
assert_type(mime.__version__, str)

document = experiment.document()
assert_type(document, ExperimentDocument)
assert_type(document["experiment"], str)
assert_type(document["scene"], None)

loaded = Experiment.load(Path("a-saved-experiment"))
assert_type(loaded.load_report, LoadReport | None)
built = loaded.build()
assert_type(built, BuiltExperiment)
assert_type(built.report, BuildReport)
assert_type(built.report.run.text(), str)
assert_type(built.verify(), VerifyReport)
assert_type(built.verify().reproduced, bool)

lock = built.save("elsewhere")
assert_type(lock, Lock)
assert_type(lock["hashes"]["derived"], str)
assert_type(lock["environment"]["x64"], bool)
lock["hashes"]["derived"] = "sha256:0"  # type: ignore

try:
    experiment.build()
except BuildRefused as refused:
    assert_type(refused.codes(), list[str])
