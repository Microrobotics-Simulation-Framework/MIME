"""Every node's step in every switch configuration, traced through ``update`` with no switch context open, written as
jaxpr text, float32 and float64; and the comparison of two such files. Run once on a pull request's merge base and
once on its head: writing a site through ``mime.core.switch`` must change no step's program.

    python tests/saved/step_jaxprs.py dump OUT.json [--plant-eager]
    python tests/saved/step_jaxprs.py compare BASE.json HEAD.json

The merge base is run from an archive of it (``git archive``), its ``src`` first on ``PYTHONPATH``, with these
configurations from the head's tests. ``--plant-eager`` is the comparison's control: every switch function evaluates
its resolution and indicator with no collector open, which adds equations to every step that passes either as a
callable, so the comparison against the base must fail."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import jax  # noqa: E402

DTYPES = {"float32": False, "float64": True}


def plant_eager() -> None:
    from mime.core import switch

    held = switch._collect

    def eager(owner, function, resolution, indicator, branch):
        switch._evaluated(resolution), switch._evaluated(indicator)
        held(owner, function, resolution, indicator, branch)

    switch._collect = eager


def dump(out: Path, eager: bool) -> None:
    from switch_configurations import configurations

    from mime.build import BuildRefused
    from mime.build._maddening import node_state, node_update, parameter_leaves, resolved_inputs, zero_inputs
    from mime.build._probes import _traced

    if eager:
        plant_eager()
    texts: dict[str, str] = {}
    for dtype, x64 in DTYPES.items():
        jax.config.update("jax_enable_x64", x64)
        for configuration, make in configurations().items():
            try:
                graph = make().build().graph
            except BuildRefused as refused:
                print(f"{dtype} {configuration}: refused here, so its steps are in the other file only: {refused.codes()}")
                continue
            for name in graph.schedule:
                node = graph.get_node(name)
                given = {**zero_inputs(node), **resolved_inputs(graph, name)}
                traced = _traced(node_update(node), node_state(graph, name), given, node.delta_t,
                                 parameter_leaves(graph).get(name))
                texts[f"{dtype} {configuration} {name}"] = str(traced)
    out.write_text(json.dumps(texts, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{out}: {len(texts)} steps")


def compare(base: Path, head: Path) -> int:
    old, new = (json.loads(path.read_text(encoding="utf-8")) for path in (base, head))
    missing, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    differ = sorted(key for key in set(old) & set(new) if old[key] != new[key])
    print(f"{len(set(old) & set(new))} steps in both; {len(old) - len(missing) - len(differ)} equal; "
          f"{len(differ)} differ; {len(missing)} only in the base; {len(added)} only in the head")
    for key in differ:
        print(f"  differs: {key}")
    for key in missing:
        print(f"  only in the base: {key}")
    for key in added:
        print(f"  only in the head: {key}")
    return 1 if differ or missing or added else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dumping = commands.add_parser("dump")
    dumping.add_argument("out", type=Path)
    dumping.add_argument("--plant-eager", action="store_true")
    comparing = commands.add_parser("compare")
    comparing.add_argument("base", type=Path)
    comparing.add_argument("head", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "dump":
        dump(arguments.out, arguments.plant_eager)
        return 0
    return compare(arguments.base, arguments.head)


if __name__ == "__main__":
    sys.exit(main())
