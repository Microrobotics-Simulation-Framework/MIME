"""What a stress run runs (``scripts/stress_select.py``) and the workflow that runs it (``.github/workflows/stress.yml``):
the selection from a pull request's diff and description, the refusals, the seeds, which caches are restored; and the
workflow's wiring, read from the file, each control planting one break in a copy and expecting the check to name it."""

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_stress_select", ROOT / "scripts" / "stress_select.py")
assert spec is not None and spec.loader is not None
stress = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stress  # a dataclass reads its module
spec.loader.exec_module(stress)

WORKFLOW = ROOT / ".github" / "workflows" / "stress.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
FILES = {"tests/oracles/test_spinner.py", "tests/oracles/test_dynamic_motor.py", "tests/test_x.py"}


def ask(body: str = "", changed: list[str] | None = None, seed: int = 99) -> stress.Request:
    return stress.select(body, changed or [], lambda path: path in FILES, lambda: seed)


def test_the_changed_tests_are_the_default_selection():
    request = ask(changed=["src/mime/a.py", "tests/test_x.py", "tests/oracles/test_spinner.py", "tests/conftest.py",
                           "tests/oracles/helper.py", "docs/test_not_here.py", "tests/data/test_notes.txt"])
    assert request.targets == ("tests/oracles/test_spinner.py", "tests/test_x.py")
    assert (request.runs, request.python, request.seed) == (20, "3.12", 99)


def test_a_stress_line_adds_to_the_changed_tests_and_only_replaces_them():
    body = "Fixes it.\n\nstress: tests/oracles/test_spinner.py tests/test_x.py::test_y[a-1] runs=30 python=3.14 seed=7\n"
    request = ask(body, changed=["tests/test_x.py"])
    assert request.targets == ("tests/oracles/test_spinner.py", "tests/test_x.py::test_y[a-1]", "tests/test_x.py")
    assert (request.runs, request.python, request.seed) == (30, "3.14", 7)
    assert ask(body + "stress: only\n", changed=["tests/test_x.py"]).targets == request.targets[:2]


def test_the_line_is_read_as_a_markdown_author_writes_it():
    body = "- Stress: `tests/oracles/test_spinner.py`, `tests/oracles/test_dynamic_motor.py` RUNS=5\r\nmore text\r\n"
    request = ask(body)
    assert request.targets == ("tests/oracles/test_spinner.py", "tests/oracles/test_dynamic_motor.py")
    assert request.runs == 5


@pytest.mark.parametrize("runs, wanted", [("1", 1), ("40", 40), ("20", 20)])
def test_runs_within_one_to_forty_are_taken(runs, wanted):
    assert ask(f"stress: tests/test_x.py runs={runs}").runs == wanted


@pytest.mark.parametrize("body", [
    "stress: tests/test_x.py runs=41", "stress: tests/test_x.py runs=0", "stress: tests/test_x.py runs=-3",
    "stress: tests/test_x.py runs=ten", "stress: tests/test_x.py runs=", "stress: tests/test_x.py python=3.11",
    "stress: tests/test_x.py python=3", "stress: tests/test_x.py python=3.12; id", "stress: tests/test_x.py seed=-1",
    "stress: tests/test_x.py seed=99999999999999", "stress: tests/test_x.py runs=5 runs=6",
    "stress: tests/missing.py", "stress: src/mime/a.py", "stress: tests/../setup.py", "stress: /etc/passwd",
    "stress: tests/test_x.py;rm", "stress: -k", "stress: --co", "stress: tests/-x.py", "stress: tests/test_x.py::a b",
    "stress: tests/x/ tests", "stress: run=5", "stress: $(id)", "stress: tests/test_x.pyc",
])
def test_what_cannot_be_used_is_refused(body):
    with pytest.raises(stress.Refused):
        ask(body)


def test_an_empty_selection_is_refused_not_passed():
    with pytest.raises(stress.Refused, match="nothing to run"):
        ask("no stress line here", changed=["src/mime/a.py", "tests/conftest.py"])
    with pytest.raises(stress.Refused, match="nothing to run"):
        ask("stress: only", changed=["tests/test_x.py"])


def test_a_repeated_option_with_the_same_value_is_not_a_conflict():
    assert ask("stress: tests/test_x.py runs=5\nstress: tests/test_x.py runs=5").runs == 5


def test_run_k_uses_the_seed_plus_k_minus_one_and_the_matrix_lists_every_run():
    matrix = ask("stress: tests/test_x.py runs=4 seed=100").matrix()
    assert matrix == {"include": [{"run": 1, "seed": 100}, {"run": 2, "seed": 101}, {"run": 3, "seed": 102},
                                  {"run": 4, "seed": 103}]}


def test_a_fresh_seed_differs_between_requests():
    seeds = {stress.secrets.randbelow(2**31) for _ in range(5)}
    assert len(seeds) > 1


@given(st.text(max_size=300))
def test_whatever_the_description_says_the_result_is_a_valid_request_or_a_refusal(body):
    try:
        request = ask(body, changed=["tests/test_x.py"])
    except stress.Refused:
        return
    assert request.targets
    assert all(stress.TARGET.fullmatch(t) and ".." not in t for t in request.targets)
    assert 1 <= request.runs <= 40 and request.python in stress.SUPPORTED_PYTHONS and 0 <= request.seed < 2**31


@given(st.lists(st.sampled_from(["tests/test_x.py", "tests/oracles/test_spinner.py", "runs=7", "only", "python=3.13"])))
def test_the_order_of_tokens_does_not_change_the_selection(tokens):
    listed = ["tests/oracles/test_dynamic_motor.py", *tokens]
    forward = ask("stress: " + " ".join(listed), changed=["tests/test_x.py"])
    backward = ask("stress: " + " ".join(reversed(listed)), changed=["tests/test_x.py"])
    assert set(forward.targets) == set(backward.targets) and forward.runs == backward.runs


def test_the_supported_pythons_are_the_default_lanes():
    ci = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert tuple(ci["jobs"]["default-tests"]["strategy"]["matrix"]["python-version"]) == stress.SUPPORTED_PYTHONS


def test_the_caches_come_from_the_shards_holding_the_files_most_files_first():
    import _sharding as sharding  # the split CI runs

    dealt = sharding.deal(sharding.load_weights(), 6)
    files = sorted({p for p in dealt})[:40]
    shards = stress.cache_shards(files, 6)
    held = {}
    for path in files:
        held.setdefault(sharding.shard_of(path, 6, dealt), []).append(path)
    assert len(shards) == min(3, len(held)) and all(s.endswith("/6") for s in shards)
    sizes = [len(held[int(s.split("/")[0])]) for s in shards]
    assert sizes == sorted(sizes, reverse=True)
    assert stress.cache_shards([files[0] + "::test_a[1]", files[0] + "::test_b"], 6) == [
        f"{sharding.shard_of(files[0], 6, dealt)}/6"]


# The workflow ---------------------------------------------------------------------------------------------------------

TOKEN = re.compile(r"'[^']*'|[A-Za-z_][\w.]*|==|!=|&&|\|\||\S")
EVENTS = {
    "push 1": {"github.event.pull_request.number": "5", "github.event.action": "synchronize", "github.run_id": "1"},
    "push 2": {"github.event.pull_request.number": "5", "github.event.action": "synchronize", "github.run_id": "2"},
    "stress label": {"github.event.pull_request.number": "5", "github.event.action": "labeled",
                     "github.event.label.name": "stress", "github.run_id": "3"},
    "other label": {"github.event.pull_request.number": "5", "github.event.action": "labeled",
                    "github.event.label.name": "bug", "github.run_id": "4"},
    "another pull request": {"github.event.pull_request.number": "6", "github.event.action": "synchronize",
                             "github.run_id": "5"},
}


def expression(text: str, context: dict[str, str]) -> str:
    """The ``${{ }}`` expressions of ``text`` evaluated for names, literals, ``==``, ``!=``, ``&&`` and ``||``."""
    def value(match: re.Match[str]) -> str:
        code = [{"&&": " and ", "||": " or "}.get(t) or (repr(context.get(t, "")) if re.fullmatch(r"[A-Za-z_][\w.]*", t) else t)
                for t in TOKEN.findall(match.group(1))]
        return str(eval("".join(code), {"__builtins__": {}}, {}))  # noqa: S307 - the workflow file's own expression
    return re.sub(r"\$\{\{(.*?)\}\}", value, text)


def steps(job: dict) -> list[dict]:
    return job["steps"]


def pins(workflow: dict) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for job in workflow["jobs"].values():
        for step in steps(job):
            if "uses" in step and not step["uses"].startswith("./"):
                action, _, pin = step["uses"].partition("@")
                found.setdefault(action, set()).add(pin)
    return found


def stress_problems(workflow: dict, ci: dict) -> list[str]:
    found = []
    trigger = workflow.get("on", workflow.get(True))
    if set(trigger) != {"pull_request"} or set(trigger["pull_request"]["types"]) != {"labeled", "synchronize"}:
        found.append("it must run only on a pull request's labeled and synchronize events")
    jobs = workflow["jobs"]
    if "'stress'" not in jobs["select"].get("if", "") or any("if" in jobs[j] and "stress" in str(jobs[j]["if"])
                                                           for j in ("run", "summary")):
        found.append("only the select job tests for the stress label; the others follow it")
    if "labeled" not in jobs["select"]["if"] or "label.name" not in jobs["select"]["if"]:
        found.append("a label other than stress must not start a stress run")
    block = workflow.get("concurrency", {})
    if str(block.get("cancel-in-progress")) != "True":
        found.append("a newer push must cancel an older stress run")
    group = {name: expression(str(block.get("group", "")), c) for name, c in EVENTS.items()}
    if group["push 1"] != group["push 2"] or group["push 1"] != group["stress label"]:
        found.append("a pull request's stress runs must share a group, so the newer cancels the older")
    if group["other label"] in {group["push 1"], group["another pull request"]} or group["another pull request"] == group["push 1"]:
        found.append("another label or another pull request must not share the group of a stress run")
    for name, job in jobs.items():
        for step in steps(job):
            text = str(step.get("run", ""))
            if re.search(r"\$\{\{\s*github\.(event\.pull_request\.(body|title)|head_ref)", text):
                found.append(f"{name}: the description or title is untrusted and must reach scripts through env, not run:")
    run = jobs["run"]
    if run["needs"] != "select" or run["strategy"]["matrix"] != "${{ fromJSON(needs.select.outputs.matrix) }}" \
            or run["strategy"]["fail-fast"] is not False:
        found.append("run: a matrix from the selection's output, every run completing (fail-fast false)")
    used = [step.get("uses", "") for step in steps(run)]
    if any(u.startswith("actions/cache/save") or u.startswith("actions/cache@") for u in used):
        found.append("run: the compilation cache is restored and never saved")
    if sum(u.startswith("actions/cache/restore@") for u in used) != stress.CACHE_SLOTS:
        found.append(f"run: {stress.CACHE_SLOTS} cache slots, as stress_select.py names them")
    setup = next(step for step in steps(run) if step.get("uses", "").startswith("astral-sh/setup-uv"))
    if setup["with"].get("save-cache") is not False:
        found.append("run: a pull request never saves setup-uv's cache")
    install = [str(step.get("run", "")) for step in steps(run)]
    for needed in ("uv sync --locked --extra verify", "uv pip install --no-deps ./lab_fixture", "--hypothesis-seed=",
                   "-n auto --dist worksteal", "--junitxml"):
        if not any(needed in text for text in install):
            found.append(f"run: no step with {needed!r}")
    ci_restore = [step["with"]["path"] for job in ci["jobs"].values() for step in steps(job)
                  if step.get("uses", "").startswith("actions/cache/restore@")]
    ours = [step["with"]["path"] for step in steps(run) if step.get("uses", "").startswith("actions/cache/restore@")]
    if set(ours) != set(ci_restore) or len(set(ours)) != 1:
        found.append("run: every cache restore uses CI's path, which GitHub keys an entry by, or it finds nothing CI saved")
    if "./.github/actions/unit-tube" not in used:
        found.append("run: the unit tube table is stored as in CI")
    summary = jobs["summary"]
    if "!cancelled()" not in str(summary["if"]) or set(summary["needs"]) != {"select", "run"}:
        found.append("summary: runs after a red run too (!cancelled()), needing select and run")
    last = str(steps(summary)[-1]["run"])
    if "scripts/stress_summary.py" not in last or "||" in last or "set +e" in last:
        found.append("summary: its last step runs stress_summary.py with its exit status intact")
    for name, job in jobs.items():
        if "timeout-minutes" not in job:
            found.append(f"{name}: no timeout-minutes")
    ci_pins = pins(ci)
    for action, pinned in pins(workflow).items():
        if pinned != ci_pins.get(action, pinned):
            found.append(f"{action}: pinned differently from ci.yml")
    if workflow.get("permissions") != {"contents": "read"}:
        found.append("permissions must be contents: read")
    return found


def load() -> tuple[dict, dict]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")), yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_the_workflow_is_wired_as_the_brief_says():
    assert stress_problems(*load()) == []


def _plant(mutate) -> list[str]:
    workflow, ci = load()
    workflow = copy.deepcopy(workflow)
    mutate(workflow)
    return stress_problems(workflow, ci)


PLANTS = {
    "a-manual-trigger": (lambda w: w["on" if "on" in w else True].update(workflow_dispatch={}),
                         "labeled and synchronize events"),
    "the-group-is-per-run": (lambda w: w["concurrency"].update(group="stress-${{ github.run_id }}"),
                             "share a group"),
    "no-cancelling": (lambda w: w["concurrency"].update({"cancel-in-progress": False}), "cancel an older"),
    "every-label-counts": (lambda w: w["jobs"]["select"].update({"if": "contains(x, 'stress')"}),
                           "a label other than stress"),
    "the-body-in-a-script": (lambda w: w["jobs"]["select"]["steps"][-1].update(
        run="echo ${{ github.event.pull_request.body }}"), "untrusted"),
    "the-cache-is-saved": (lambda w: w["jobs"]["run"]["steps"].append(
        {"uses": "actions/cache/save@abc", "with": {}}), "never saved"),
    "a-failed-run-stops-the-rest": (lambda w: w["jobs"]["run"]["strategy"].update({"fail-fast": True}), "fail-fast"),
    "the-summary-skips-after-a-red-run": (lambda w: w["jobs"]["summary"].update({"if": "needs.select.result == 'success'"}),
                                          "!cancelled()"),
    "the-summary-swallows-failure": (lambda w: w["jobs"]["summary"]["steps"][-1].update(
        run="python3 scripts/stress_summary.py x || true"), "exit status intact"),
    "the-seed-dropped": (lambda w: w["jobs"]["run"]["steps"].__setitem__(
        -2, {"name": "Tests", "run": "uv run --no-sync pytest -n auto --dist worksteal --junitxml=x"}), "--hypothesis-seed="),
    "a-restore-into-another-path": (lambda w: w["jobs"]["run"]["steps"][8]["with"].update(path="x/jax-cache-2"),
                                    "CI's path"),
    "a-different-pin": (lambda w: w["jobs"]["run"]["steps"][0].update(uses="actions/checkout@deadbeef"),
                        "pinned differently"),
    "no-timeout": (lambda w: w["jobs"]["run"].pop("timeout-minutes"), "no timeout-minutes"),
}


@pytest.mark.parametrize("name", PLANTS)
def test_control_each_planted_break_in_the_workflow_is_named(name):
    plant, expected = PLANTS[name]
    assert any(expected in problem for problem in _plant(plant)), name


def test_the_selection_and_the_matrix_survive_a_round_trip_through_the_outputs_the_workflow_reads():
    request = ask("stress: tests/test_x.py::test_y[a b]".replace(" ", "") + " runs=3 seed=5")
    assert json.loads(json.dumps(request.targets)) == list(request.targets)
    assert len(json.loads(json.dumps(request.matrix()))["include"]) == 3
