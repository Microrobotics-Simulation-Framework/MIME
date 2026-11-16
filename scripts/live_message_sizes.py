"""The largest legitimate messages of ``mime.live/2`` on the repository's rigs, which the transport's limits
(``mime.live._zmtp.LIMITS``) are set from with a stated margin (``docs/live.md``, "Message sizes"); and each rig's
default recording budget. A measurement, never a CI check; sizes depend on the rig and not on the machine.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/live_message_sizes.py

Each rig is saved, loaded and built as ``mime serve`` builds it, and its session opened with a run folder
(``built.live(out=...)``), so that ``hello`` is the one a client receives. It prints one JSON object a rig: ``hello``'s
bytes; a frame's bytes, binary and as JSON (with the state as it starts, and at the longest float text, 24 characters,
in every number); the largest ``set`` the controller's live inputs allow, at the longest float text; a status's bytes;
one recording sample's bytes; and the default ``record_budget``. The rigs: A0's apparatus closed by ``ArmTracking``
(``tests/live_cases.py``'s tracked rig), the toy, and the indented sheet of 33 and of 200 vertices a side, the second
with its points and both fields selected (a surface over the stage's triangle budget, drawn decimated, its clusters in
``hello``'s binding). Not collected by pytest."""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: The longest text a float64 takes in JSON, 24 characters.
LONGEST = -2.2250738585072014e-308
SHEET = ["sheet.points", "sheet.depth", "sheet.rate"]


def measured(name: str, make: Callable[[], Any], select: Sequence[str] | None = None) -> dict[str, Any]:
    from mime.experiment import Experiment
    from mime.live import PROTOCOL

    with tempfile.TemporaryDirectory() as scratch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make().save(Path(scratch) / "saved")
        built = Experiment.load(Path(scratch) / "saved").build()
        session = built.live(pace=None, select=select, out=Path(scratch) / "run")
        values = session.schema.values(built.current_state())
        longest = {path: np.full(np.shape(value), LONGEST) for path, value in values.items()}
        inputs = {label: np.full(held.shape, LONGEST).tolist() for label, held in session.live_inputs.items()}
        row = {"rig": name, "hello": len(json.dumps(session.hello()).encode()),
               "frame_binary": session.schema.frame_bytes,
               "frame_json": len(json.dumps(session.schema.json_frame(0, 0, 0.0, values)).encode()),
               "frame_json_longest": len(json.dumps(session.schema.json_frame(
                   2 ** 32 - 1, 2 ** 63 - 1, LONGEST, longest)).encode()),
               "set_longest": len(json.dumps({"protocol": PROTOCOL, "command": "set", "values": inputs}).encode())
               if inputs else 0,
               "status": len(json.dumps(session.status()).encode()),
               "record_sample": sum(np.asarray(built.current_state()[node][field]).nbytes
                                    for node, field in built.binding().fields()),
               "record_budget": session.record_budget}
        session.stop()
    return row


def main() -> None:
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "viz")]
    from live_cases import tracked
    from test_surfaces import sheet_experiment

    from mime.testing.toys import full_toy

    for row in (measured("a0_tracked", tracked), measured("full_toy", full_toy),
                measured("sheet_33", sheet_experiment), measured("sheet_33_selected", sheet_experiment, SHEET),
                measured("sheet_200_selected", lambda: sheet_experiment(count=200), SHEET)):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
