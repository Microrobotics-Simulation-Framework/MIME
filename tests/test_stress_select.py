"""What a stress run runs (``scripts/stress_select.py``) and the workflow that runs it (``.github/workflows/stress.yml``):
the selection from a pull request's diff and description, the refusals, the seeds, which caches are restored; and the
workflow's wiring, read from the file, each control planting one break in a copy and expecting the check to name it."""

import copy
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

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
    assert request.targets == ("tests/test_x.py", "tests/oracles/test_spinner.py")
    assert (request.runs, request.python, request.seed) == (20, "3.12", 99)


def test_a_stress_line_adds_to_the_changed_tests_and_only_replaces_them():
    body = "Fixes it.\n\nstress: tests/oracles/test_spinner.py tests/test_x.py::test_y[a-1] runs=30 python=3.14 seed=7\n"
    request = ask(body, changed=["tests/test_x.py"])
    assert request.targets == ("tests/test_x.py::test_y[a-1]", "tests/test_x.py", "tests/oracles/test_spinner.py")
    assert (request.runs, request.python, request.seed) == (30, "3.14", 7)
    assert ask(body + "stress: only\n", changed=["tests/test_x.py"]).targets == (
        "tests/test_x.py::test_y[a-1]", "tests/oracles/test_spinner.py")


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
    assert matrix == {"include": [{"run": k, "seed": 99 + k, "part": 1, "index": 0} for k in range(1, 5)]}  # one part: one job a run


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


def revisited(targets: tuple[str, ...]) -> list[str]:
    """The directories whose files are not together in ``targets``."""
    runs = [directory for directory, _ in itertools.groupby(str(PurePosixPath(t.split("::")[0]).parent) for t in targets)]
    return sorted({directory for directory in runs if runs.count(directory) > 1})


def test_a_directorys_files_are_run_together_whatever_order_they_are_listed_and_changed_in():
    """pytest 9.1.1 drops a directory's conftest fixtures the second time a run reaches it after a file of a directory
    above it (tests/_collection_guard.py): ``tests/saved``'s float64 mode was lost to 89 tests that way. The planted
    order is the one that did it."""
    body = "stress: tests/oracles/test_spinner.py tests/test_x.py tests/oracles/test_dynamic_motor.py"
    assert revisited(("tests/oracles/test_spinner.py", "tests/test_x.py", "tests/oracles/test_dynamic_motor.py")) == [
        "tests/oracles"]
    request = ask(body, changed=["tests/test_x.py"])
    assert revisited(request.targets) == []
    assert request.targets == ("tests/test_x.py", "tests/oracles/test_spinner.py", "tests/oracles/test_dynamic_motor.py")


@given(st.lists(st.sampled_from(["tests/test_x.py", "tests/oracles/test_spinner.py", "tests/oracles/test_dynamic_motor.py",
                                 "tests/test_x.py::test_y[a]"]), min_size=1))
def test_no_listing_brings_a_directory_back_after_another(listed):
    assert revisited(ask("stress: " + " ".join(listed), changed=["tests/test_x.py"]).targets) == []


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
    for step in steps(run):
        if "TARGETS" in step.get("env", {}) and "matrix.index" not in str(step["env"]["TARGETS"]):
            found.append(f"run: step {step.get('name')!r} reads every target, not its part's (parts output at matrix.index)")
    names = [str(step["with"].get("name", "")) for step in steps(run) if step.get("uses", "").startswith("actions/upload-artifact@")]
    if not names or any("matrix.run" not in name or "matrix.part" not in name for name in names):
        found.append("run: a report's artifact is named by its run and its part, or parts of one run overwrite each other")
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
    if "--parts" not in last:
        found.append("summary: it must be told the parts, or it judges only the first job of each run")
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
    "the-tests-step-runs-every-target": (lambda w: w["jobs"]["run"]["steps"][-2]["env"].update(
        TARGETS="${{ needs.select.outputs.targets }}"), "reads every target"),
    "the-cache-step-reads-every-target": (lambda w: w["jobs"]["run"]["steps"][6]["env"].update(
        TARGETS="${{ needs.select.outputs.targets }}"), "reads every target"),
    "parts-share-an-artifact-name": (lambda w: w["jobs"]["run"]["steps"][-1]["with"].update(name="stress-run-${{ matrix.run }}"),
                                     "overwrite each other"),
    "the-summary-ignores-the-parts": (lambda w: w["jobs"]["summary"]["steps"][-1].update(
        run='python3 scripts/stress_summary.py "$RUNNER_TEMP/runs" --runs "$RUNS" --seed "$SEED"'), "told the parts"),
}


@pytest.mark.parametrize("name", PLANTS)
def test_control_each_planted_break_in_the_workflow_is_named(name):
    plant, expected = PLANTS[name]
    assert any(expected in problem for problem in _plant(plant)), name


def test_the_selection_and_the_matrix_survive_a_round_trip_through_the_outputs_the_workflow_reads():
    request = ask("stress: tests/test_x.py::test_y[a b]".replace(" ", "") + " runs=3 seed=5")
    assert json.loads(json.dumps(request.targets)) == list(request.targets)
    assert len(json.loads(json.dumps(request.matrix()))["include"]) == 3


# The split across jobs ------------------------------------------------------------------------------------------------

def coverage_problems(request: stress.Request) -> list[str]:
    """What is wrong with the way a request splits its targets: a target in no part or in two, a part that is empty, a file
    divided between parts, a matrix that is not every part for every run, or one over GitHub's 256 entries."""
    found = []
    held = [t for part in request.parts for t in part]
    if sorted(held) != sorted(request.targets):
        found.append("the parts do not hold the targets exactly once")
    if any(not part for part in request.parts):
        found.append("a part is empty")
    owners: dict[str, set[int]] = {}
    for number, part in enumerate(request.parts):
        for target in part:
            owners.setdefault(target.split("::", 1)[0], set()).add(number)
    if any(len(parts) > 1 for parts in owners.values()):
        found.append("a file is divided between parts")
    jobs = {(entry["run"], entry["index"]) for entry in request.matrix()["include"]}
    if len(request.matrix()["include"]) != request.runs * len(request.parts) or jobs != {
            (k, i) for k in range(1, request.runs + 1) for i in range(len(request.parts))}:
        found.append("the matrix is not every part for every run, once")
    if len(request.matrix()["include"]) > 256:
        found.append("the matrix is over GitHub's 256 entries")
    return found


#: Seconds in the shard table, for the files below; a heavy file, a few middling ones and a light one.
SECONDS = {"tests/a/test_heavy.py": 1200, "tests/a/test_mid1.py": 600, "tests/b/test_mid2.py": 500, "tests/b/test_mid3.py": 450,
           "tests/c/test_light.py": 5}
UNLISTED = 70


def split_ask(body: str, changed: list[str]) -> stress.Request:
    files = set(SECONDS) | {"tests/c/test_new.py"}
    return stress.select(body, changed, lambda path: path in files, lambda: 7, (SECONDS, UNLISTED))


def test_a_selection_over_a_jobs_seconds_is_dealt_into_parts_by_the_shard_tables_seconds():
    request = split_ask("", list(SECONDS))
    assert coverage_problems(request) == []
    assert len(request.parts) == 3  # 2755 seconds over 1000 a part, rounded up
    loads = [sum(SECONDS[t] for t in part) for part in request.parts]
    assert sorted(loads) == [605, 950, 1200]  # longest first onto the lightest: 1200 | 600 + 5 | 500 + 450
    assert any(part == ("tests/a/test_heavy.py",) for part in request.parts)
    assert len(request.matrix()["include"]) == 20 * 3


def test_a_small_selection_is_one_part_as_before_and_a_listed_file_the_table_lacks_takes_the_assumed_weight():
    assert len(split_ask("", ["tests/c/test_light.py", "tests/b/test_mid3.py"]).parts) == 1
    several = split_ask("stress: tests/c/test_new.py", list(SECONDS))
    assert coverage_problems(several) == []
    assert sum(len(part) for part in several.parts) == 6
    # the file the table lacks counts UNLISTED seconds: 2825 over 1000 is still three parts, 2755 + 70 -> 3
    assert len(several.parts) == 3
    only_new = split_ask("stress: only tests/c/test_new.py", [])
    assert only_new.parts == (("tests/c/test_new.py",),)


def test_node_ids_of_one_file_stay_in_one_part_and_weigh_the_file_once():
    request = split_ask("stress: only tests/a/test_mid1.py::test_a[1] tests/a/test_mid1.py::test_b tests/b/test_mid2.py", [])
    assert coverage_problems(request) == [] and len(request.parts) == 2  # 600 + 500 = 1100 seconds, over one part
    together = [part for part in request.parts if any(t.startswith("tests/a/test_mid1.py") for t in part)]
    assert len(together) == 1 and len(together[0]) == 2


def test_a_part_keeps_each_directorys_files_together():
    request = split_ask("", list(SECONDS))
    for part in request.parts:
        assert revisited(part) == []


def test_the_parts_are_the_same_every_time_the_same_request_is_made():
    assert split_ask("", list(SECONDS)).parts == split_ask("", list(reversed(SECONDS))).parts


@given(st.dictionaries(st.from_regex(r"tests/[a-d]/test_[a-z]{1,6}\.py", fullmatch=True), st.integers(1, 5000), min_size=1,
                       max_size=30), st.integers(1, 40))
def test_whatever_the_seconds_every_file_lands_in_one_part_within_the_dealing_bound_and_the_matrix_fits(table, runs):
    files = sorted(table)
    request = stress.select(f"stress: runs={runs}", files, lambda path: path in table, lambda: 1, (table, 70))
    assert coverage_problems(request) == []
    count = len(request.parts)
    assert count <= max(1, 250 // runs) and count <= len(files)
    loads = [sum(table[t.split("::")[0]] for t in part) for part in request.parts]
    total = sum(table.values())
    # longest-first dealing onto the lightest part: no part exceeds the mean part by more than the heaviest file
    assert max(loads) <= total / count + max(table.values())
    assert count == max(1, min(len(files), -(-total // stress.PART_SECONDS), stress.MAX_JOBS // runs))


def test_a_run_of_forty_is_split_into_no_more_parts_than_the_matrix_holds():
    request = split_ask("stress: runs=40", list(SECONDS))
    assert coverage_problems(request) == [] and len(request.matrix()["include"]) <= 256


def test_control_a_split_that_drops_a_file_or_divides_one_or_double_counts_a_run_is_named_every_time():
    """The invariant above must fail on each plant, or the property passes on a split that loses tests."""
    request = split_ask("", list(SECONDS))
    lost = stress.Request(request.targets, request.runs, request.python, request.seed, request.parts[:-1])
    assert any("exactly once" in problem for problem in coverage_problems(lost))
    twice = stress.Request(request.targets, request.runs, request.python, request.seed, (*request.parts, request.parts[0]))
    assert any("exactly once" in problem for problem in coverage_problems(twice))
    divided = stress.Request(request.targets, request.runs, request.python, request.seed,
                             (("tests/a/test_heavy.py::test_x",), ("tests/a/test_heavy.py",), *request.parts[1:]))
    assert any("divided" in problem for problem in coverage_problems(divided))
    empty = stress.Request(request.targets, request.runs, request.python, request.seed, (*request.parts, ()))
    assert any("empty" in problem for problem in coverage_problems(empty))

    class Short(stress.Request):
        def matrix(self):  # a matrix that runs the first part only: the plant
            return {"include": [{"run": k, "seed": 0, "part": 1, "index": 0} for k in range(1, self.runs + 1)]}

    short = Short(request.targets, request.runs, request.python, request.seed, request.parts)
    assert any("every part for every run" in problem for problem in coverage_problems(short))


def test_a_large_pull_requests_changed_set_is_dealt_by_the_real_table_so_that_no_part_but_a_lone_file_passes_the_budget():
    """The tube tree PR's changed files (a measure of 4806 seconds on the table of the day, which ran 36 to 46 minutes in one
    job) by the committed table: as many parts as its seconds need at the budget, whatever the table's rebuild makes them."""
    changed = ["tests/docs/test_lists.py", "tests/docs/test_none_declared.py", "tests/flow/test_fvm_curved_wall.py",
               "tests/flow/test_fvm_general_wall.py", "tests/flow/test_fvm_tree_declaration.py", "tests/saved/test_golden_outputs.py",
               "tests/saved/test_shape_suite.py", "tests/scene/test_centreline.py", "tests/scene/test_exact_union.py",
               "tests/scene/test_tube_tree.py", "tests/scene/test_union_brute_force.py", "tests/scene/test_union_continuity.py",
               "tests/scene/test_union_reverse.py", "tests/scene/test_union_tie.py", "tests/two_builds/test_tree_walk.py"]
    seconds, unlisted = stress.table_seconds()
    request = stress.select("", changed, lambda path: True, lambda: 1)
    assert coverage_problems(request) == []
    assert len(request.parts) == math.ceil(sum(seconds[f] for f in changed) / stress.PART_SECONDS)
    loads = [sum(seconds.get(t, unlisted) for t in part) for part in request.parts]
    assert sum(loads) == sum(seconds[f] for f in changed)
    assert max(loads) <= max(seconds[f] for f in changed) + 0.2 * stress.PART_SECONDS, loads


def test_the_assumed_weight_of_an_unlisted_file_is_the_shard_deals():
    import _sharding as sharding

    assert stress.table_seconds()[1] == sharding.new_file_weight()
    assert stress.table_seconds()[0] == sharding.load_seconds()
