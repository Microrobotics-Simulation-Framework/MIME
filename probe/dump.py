"""One program's run, every field's history at every step, as an NPZ file, for bit_compare.py: built and run once in a
fresh process, in the tree it runs in (the base's or the head's), 100 steps. ``moved`` is the comparison's control: A0
with a one-step port on its arm where the tree has ports, none where it has not, so the two sides must differ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

program, out = sys.argv[1], Path(sys.argv[2])
tree = Path.cwd()
name, _, dtype = program.rpartition("_")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", dtype == "f64")
# The tutorials import examples/script_a.py, the tests their own script_a: each program its own folder first.
sys.path[:0] = [str(tree / "examples")] if name.startswith("tutorial") else [str(tree / "tests")]
STEPS = 100


def tutorial_a0():
    source = (tree / "examples" / "tutorial_a0.py").read_text(encoding="utf-8").split("# region: run")[0]
    space: dict[str, object] = {"__name__": "tutorial_a0_probe", "__file__": str(tree / "examples" / "tutorial_a0.py")}
    exec(compile(source, "tutorial_a0.py", "exec"), space)
    return [space["close_the_loop"](space["a0"]())]  # type: ignore[operator]


def made() -> list:
    if name == "a0":
        from script_a0 import a0

        return [a0("bits_a0")]
    if name == "script_a":
        from script_a import script_a

        return [script_a()]
    if name == "tutorial_a":
        from script_a import record_arm, script_a

        return [script_a(), record_arm()]
    if name == "tutorial_a0":
        return tutorial_a0()
    if name == "moved":
        from script_a0 import a0

        try:
            from mime import CommandPort
            from mime.units import ms

            return [a0("bits_a0", ports={"arm.joint_positions": CommandPort(latency=ms(0.5), reason="the control")})]
        except ImportError:
            return [a0("bits_a0")]
    raise SystemExit(f"unknown program {program}")


arrays: dict[str, np.ndarray] = {}
for index, experiment in enumerate(made()):
    built = experiment.build()
    dt = float(built.plans.dt)
    from mime.units import s

    history = built.run(s(STEPS * dt), history=True).history
    for node, fields in history.items():
        if node.startswith(("ports:", "measure:")):
            continue  # the control's stage: no field of it is the base's
        for field, values in fields.items():
            arrays[f"{index}/{node}/{field}"] = np.asarray(values)
np.savez(out, **arrays)
print(f"{program}: {len(arrays)} arrays")
