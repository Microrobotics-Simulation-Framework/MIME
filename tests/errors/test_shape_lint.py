"""The one shape, held over every code: the lint that every action of a code the user or the environment acts on names
a MIME argument or command, with its planted controls; a page for every code; and the build report's summary first,
with only the attached rung's validity lines."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mime
from mime.build import REFUSAL_TABLE, RefusalClass
from mime.build._construct import attached_validity
from mime.docs._error_pages import ERRORS_INDEX, error_pages, error_path
from mime.docs._refusal_scan import NAMES_WHAT_TO_CHANGE, action_problems, scan_package
from mime.interactions.hydrodynamic import Drag

PACKAGE = Path(mime.__file__).parent


def test_every_action_names_a_mime_argument_or_command():
    assert action_problems(scan_package(PACKAGE), REFUSAL_TABLE) == []


def planted(tmp_path: Path, source: str) -> list[str]:
    package = tmp_path / "planted"
    package.mkdir(exist_ok=True)
    (package / "module.py").write_text(source, encoding="utf-8")
    table = {"planted.code": frozenset({RefusalClass.USER}), "planted.defect": frozenset({RefusalClass.INTERNAL})}
    return action_problems(scan_package(package), table)


@pytest.mark.parametrize("action", ['"correct the setting"', 'f"give {name} a value"', '"Did you mean x?"',
                                    'did_you_mean(name, names)'])
def test_the_control_an_action_that_names_nothing_to_write_fails_the_lint(tmp_path, action):
    source = (f"def did_you_mean(name, names):\n    return f'Did you mean {{name}}?'\n\n"
              f"def check(name, names):\n    return Refusal('planted.code', RefusalClass.USER, 'what', {action})\n")
    found = planted(tmp_path, source)
    assert len(found) == 1 and "planted.code" in found[0]


@pytest.mark.parametrize("action", ['"write spacing=mm(...)"', 'f"declare it: exp.declare({name!r}, ...)"',
                                    '"run `mime assets fetch <folder>`"', '"pass --into <folder>"',
                                    '"unset MIME_OFFLINE"', 'held', 'default', 'line_of(name)',
                                    'f"Run: {fetch_command(name)}"',
                                    'f"bind {role}= one of them"'])
def test_an_action_that_names_a_mime_argument_or_command_passes_wherever_its_text_is_written(tmp_path, action):
    source = ("def line_of(name):\n    return f'pip install {name}'\n\n"
              "def check(name, role, default='give rate= a multiple of dt'):\n"
              "    held = 'pass seed=0' if name else 'pass seed=1'\n"
              f"    return Refusal('planted.code', RefusalClass.USER, 'what', {action})\n")
    assert planted(tmp_path, source) == []


def test_a_code_only_mimes_defect_is_left_out_and_an_action_the_lint_cannot_read_fails(tmp_path):
    defect = "def check():\n    return Refusal('planted.defect', RefusalClass.INTERNAL, 'what', 'report it')\n"
    assert planted(tmp_path, defect) == []
    unread = "def check(given):\n    return Refusal('planted.code', RefusalClass.USER, 'what', given)\n"
    found = planted(tmp_path, unread)
    assert len(found) == 1 and "cannot read" in found[0]
    tupled = "PROBLEMS = [('planted.code', 'what', 'correct it')]\n"
    assert len(planted(tmp_path, tupled)) == 1, "a problem tuple's action is read too"


def test_the_rule_reads_code_not_prose():
    for text in ("spacing=mm(...)", "exp.declare(...)", "scene[\"/World\"]", "`tube.axis`", "mime pull <doi>",
                 "--gpu A100", "pip install x", "MIME_TRACEBACK=full"):
        assert NAMES_WHAT_TO_CHANGE.search(text), text
    for text in ("correct the setting", "keep one of them", "use a smaller step - or two", "see the docs"):
        assert not NAMES_WHAT_TO_CHANGE.search(text), text


def test_every_code_has_its_page_naming_where_mime_gives_it_and_the_index_lists_every_one():
    pages = {page.path: page.text for page in error_pages(PACKAGE)}
    assert set(pages) == {error_path(code) for code in REFUSAL_TABLE} | {ERRORS_INDEX}
    held = pages[error_path("control.law-traced-branch")]
    assert "**user**" in held and "jnp.where(condition, if_true, if_false)" in held and "mime/build/_in_graph.py" in held
    assert "**unrecognised**" in pages[error_path("error.unrecognised")]
    assert all(f"[`{code}`]({code}.md)" in pages[ERRORS_INDEX] for code in REFUSAL_TABLE)


def test_the_report_keeps_the_attached_rungs_validity_lines_and_the_control_keeps_the_others_for_their_rung():
    def plan(rung: str) -> SimpleNamespace:
        return SimpleNamespace(registered=SimpleNamespace(cls=Drag), entry=SimpleNamespace(rung=rung))

    surrogate, mrs = attached_validity(plan("surrogate")), attached_validity(plan("mrs"))  # type: ignore[arg-type]
    assert surrogate and all(not line.startswith("mrs: ") for line in surrogate)
    assert mrs and all(not line.startswith("surrogate: ") for line in mrs)
    assert len(surrogate) + len(mrs) == len(Drag.validity)


def test_the_report_gives_a_summary_first_and_its_details_on_request():
    from mime import Body, Experiment, State
    from mime.dynamics import RigidInertial
    from mime.scene import Scene, Z
    from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, ms, rad_per_s

    exp = Experiment("drift", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("rod", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("rod.mass", kg(1e-6), reason="a millimetre rod")
    exp.declare("rod.inertia", kg_m2((1e-13, 1e-13, 1e-13)), reason="a millimetre rod")
    report = exp.build().report
    summary, details = report.text(), report.text(details=True)
    assert summary.startswith("build report: experiment 'drift', base step 0.001 s, run as one compiled scan")
    assert "rod.mass = 1e-06 kg  [declared in the experiment]" in summary and "edges, in the order added" not in summary
    assert summary.endswith("report.text(details=True)") and "edges, in the order added" in details
    assert len(summary.splitlines()) < len(details.splitlines())
