"""The port generators record how they were run with no private path in it. ``recorded_argv`` gives a path under the
workspace relative to it, one under the planning directory beside the checkouts by that directory's role, and any other
absolute path by its name alone; the scan's own path patterns judge the result, and the control, the same argv as
given, fails them. Each generator records ``argv`` through it, read from its source, and a doctored copy that records
``sys.argv`` as given fails. The paths are built by concatenation, so that this file does not hold them."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest
from _provenance import PLANNING, recorded_argv

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GENERATORS = sorted(HERE.glob("generate_port_*.py"))
_SPEC = importlib.util.spec_from_file_location("mime_check_confidential", ROOT / "scripts" / "check_confidential.py")
assert _SPEC is not None and _SPEC.loader is not None
check_confidential = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_confidential)

WORKSPACE = Path("/" + "home") / "someone" / "workspace"
ARGV = [str(WORKSPACE / "MIME-030/tests/differential/generate_port_references.py"),
        str(WORKSPACE / PLANNING / "mime030_phase4" / "differential"), "/" + "tmp/x/" + "scratch" + "pad/out", "--x64"]


def path_findings(argv: list[str]) -> list[str]:
    return check_confidential.Scanner(None, frozenset()).findings("argv", json.dumps(argv, indent=1))


def test_the_recorded_argv_holds_no_private_path_and_the_argv_as_given_does():
    recorded = recorded_argv(ARGV, WORKSPACE)
    assert recorded == ["MIME-030/tests/differential/generate_port_references.py",
                        "<the planning directory>/mime030_phase4/differential", "<outside the workspace>/out", "--x64"]
    assert path_findings(recorded) == []
    assert len(path_findings(ARGV)) == 4, "the control: the argv as given carries four private paths"


def argv_problems(source: str) -> list[str]:
    """Each provenance ``argv`` in a generator's ``source`` not recorded through ``recorded_argv``, and none found."""
    values = [node.values[at] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Dict)
              for at, key in enumerate(node.keys) if isinstance(key, ast.Constant) and key.value == "argv"]
    found = [] if values else ["it records no argv"]
    return found + [f"line {value.lineno}: argv is recorded as {ast.unparse(value)}, not through recorded_argv"
                    for value in values if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                                                and value.func.id == "recorded_argv")]


@pytest.mark.parametrize("generator", GENERATORS, ids=[path.stem for path in GENERATORS])
def test_each_generator_records_its_argv_through_recorded_argv(generator):
    source = generator.read_text(encoding="utf-8")
    assert argv_problems(source) == []
    doctored = source.replace("recorded_argv(sys.argv, repo.parent)", "sys.argv")
    assert doctored != source
    (problem,) = argv_problems(doctored)
    assert problem.endswith("argv is recorded as sys.argv, not through recorded_argv")


def test_there_are_four_generators():
    assert [path.name for path in GENERATORS] == ["generate_port_dipole_references.py",
                                                  "generate_port_drag_references.py",
                                                  "generate_port_multibody_references.py",
                                                  "generate_port_references.py"]
