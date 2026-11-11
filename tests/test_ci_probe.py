"""The probe runner (``.github/workflows/probe.yml``, ``.github/probe/``) and the guard that a probe branch is never merged.

A probe branch is a measurement runner for development: a push to ``probe/**`` runs the branch's ``probe/run.sh`` on a GitHub
runner. Its logs are public, so what is read here is that it (1) runs on a push to ``probe/**`` and on nothing else, (2) first
runs the private-name check exactly as ci.yml's required job does, which the probe job waits for, (3) holds no secret in the
job that runs the script, runs it under a ``timeout`` inside the job's own, and uploads ``probe/out`` for 14 days even when it
failed, (4) reads its arms from ``probe/arms.txt`` as the workflow says, and (5) that ci.yml's required scan job fails a pull
request whose head branch is ``probe/*``. Each step's script is run in bash as GitHub runs it; each control plants one break in
a copy of the workflows and the same check must fail naming it, every run. ``bit_compare.py`` is tested in
``test_probe_bit_compare.py``."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")
GUARD = "A probe branch is never a pull request's head"
BASH = shutil.which("bash") or "bash"


def load(root: Path, name: str) -> dict[str, object]:
    return yaml.safe_load((root / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def step_named(job: dict[str, object], name: str) -> dict[str, object]:
    steps = [step for step in job["steps"] if step.get("name") == name]  # type: ignore[index]
    assert len(steps) == 1, f"{len(steps)} steps are named {name!r}"
    return steps[0]


def run_script(step: dict[str, object], scratch: Path, **env: str) -> subprocess.CompletedProcess[str]:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "step.sh").write_text(str(step["run"]), encoding="utf-8")
    return subprocess.run([BASH, "-e", str(scratch / "step.sh")], cwd=scratch, text=True, capture_output=True, check=False,
                          env={"PATH": "/usr/bin:/bin", **env})


def guard_problems(root: Path, scratch: Path) -> list[str]:
    """What is wrong with the guard in ci.yml's required scan job: it is its first step, and it fails a head branch
    ``probe/<x>`` and passes any other (a branch merely containing ``probe``, a push, which has no head ref)."""
    job = load(root, "ci.yml")["jobs"]["confidential"]  # type: ignore[index]
    if job["name"] != "no private name or path":
        return ["the scan job is not the required check's"]
    if job["steps"][0].get("name") != GUARD:
        return ["the guard is not the scan job's first step"]
    step, found = job["steps"][0], []
    if step.get("if") or step.get("continue-on-error"):
        found.append("the guard step is conditional or may fail without failing the job")
    if step.get("env") != {"HEAD_REF": "${{ github.head_ref }}"}:
        found.append("the guard does not read the pull request's head ref from github.head_ref")
    for index, (head, fails) in enumerate([("probe/x", True), ("probe/a/b", True), ("a-branch", False), ("probe", False),
                                           ("probes/x", False), ("my/probe/x", False), ("", False)]):
        done = run_script(step, scratch / str(index), HEAD_REF=head)
        if (done.returncode != 0) != fails:
            found.append(f"head ref {head!r}: the guard {'passes' if fails else 'fails'}")
    return found


def uncommented(step: dict[str, object]) -> dict[str, object]:
    """The step with the comment lines of its script left out (the probe's says why its one line differs)."""
    lines = str(step["run"]).splitlines()
    return {**step, "run": "\n".join(line for line in lines if not line.strip().startswith("#"))}


def probe_problems(root: Path, scratch: Path) -> list[str]:
    found: list[str] = []
    workflow, ci = load(root, "probe.yml"), load(root, "ci.yml")
    triggers = workflow.get("on", workflow.get(True))
    if triggers != {"push": {"branches": ["probe/**"]}}:
        found.append(f"the triggers are {triggers}; a push to probe/** alone")
    if workflow.get("permissions") != {"contents": "read"}:
        found.append("the workflow's permissions are not contents: read alone")
    concurrency = workflow.get("concurrency") or {}
    if "github.ref" not in str(concurrency.get("group")) or concurrency.get("cancel-in-progress") is not True:
        found.append("a newer push to a branch does not cancel its older run")
    text = (root / ".github" / "workflows" / "probe.yml").read_text(encoding="utf-8")
    found += [f"{used!r} is not pinned by a commit SHA" for used in re.findall(r"uses:\s*(\S+)", text)
              if not used.startswith("./") and not SHA.match(used.split("@")[-1])]
    jobs = workflow["jobs"]
    scan, probe = jobs.get("scan"), jobs.get("probe")
    if not scan or not probe:
        return found + ["the workflow has no `scan` job or no `probe` job"]
    # The scan: ci.yml's step and checkout, byte for byte, and the first job (everything else needs it).
    reference = ci["jobs"]["confidential"]["steps"]
    scan_step = [step for step in scan["steps"] if "check_confidential" in str(step.get("run"))]
    reference_step = [step for step in reference if "check_confidential" in str(step.get("run"))]
    # One line differs: a new branch's first push has no `before`, and the whole history (older commits CI never rescans)
    # would be scanned; its range is what the development branch does not hold.
    expected = [{**step, "run": str(step["run"]).replace('range="$PUSH_AFTER"', 'range="$dev..$PUSH_AFTER"')}
                for step in reference_step]
    if [uncommented(step) for step in scan_step] != [uncommented(step) for step in expected] or len(scan_step) != 1:
        found.append("the scan step is not ci.yml's `no private name or path` step")
    if [s for s in scan["steps"] if "checkout" in str(s.get("uses"))] != [s for s in reference if "checkout" in str(s.get("uses"))]:
        found.append("the scan's checkout is not ci.yml's (fetch-depth 0, no persisted credentials)")
    for name, job in jobs.items():
        if name != "scan" and "scan" not in ([job.get("needs")] if isinstance(job.get("needs"), str) else job.get("needs", [])):
            found.append(f"job {name!r} does not wait on the scan")
    if "secrets." in json.dumps(probe) or any("secrets." in json.dumps(job) for name, job in jobs.items() if name != "scan"):
        found.append("a job other than the scan reads a secret")
    if not 0 < probe.get("timeout-minutes", 0) <= 340:
        found.append("the probe job's timeout-minutes is not at most 340")
    run = step_named(probe, "Run probe/run.sh")
    found += estimate_step_problems(probe, scratch / "estimate")
    limit = re.search(r"timeout\s+(?:--\S+\s+)*(\d+)\s+bash probe/run\.sh", str(run["run"]))
    if not limit or int(limit.group(1)) >= probe["timeout-minutes"] * 60:
        found.append("probe/run.sh does not run under a `timeout` shorter than the job's own")
    uploads = [step for step in probe["steps"] if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    if len(uploads) != 1:
        found.append(f"{len(uploads)} steps upload the artifact; one does")
    else:
        upload = uploads[0]
        if upload.get("if") != "${{ always() }}" or upload["with"].get("path") != "probe/out" \
                or upload["with"].get("retention-days") != 14:
            found.append("the upload is not probe/out, for 14 days, under always()")
    names = step_named(probe, "What the artifact is called")
    for branch, arm, wanted in [("probe/foo", "", "probe-foo"), ("probe/foo", "a1", "probe-foo-a1"),
                                ("probe/a/b c", "x.y", "probe-a-b-c-x.y")]:
        done = run_script(names, scratch / f"name-{branch.replace('/', '_')}-{arm}", BRANCH=branch, PROBE_ARM=arm,
                          GITHUB_OUTPUT=str(scratch / f"out-{branch.replace('/', '_')}-{arm}"))
        output = (scratch / f"out-{branch.replace('/', '_')}-{arm}")
        got = output.read_text(encoding="utf-8").strip() if output.exists() else ""
        if got != f"name={wanted}":
            found.append(f"the artifact of {branch!r} arm {arm!r} is named {got!r}, not name={wanted}")
    found += arm_problems(jobs.get("arms"), scratch / "arms")
    return found


def estimate_step_problems(probe: dict[str, object], scratch: Path) -> list[str]:
    """The memory-estimate step: it precedes the script, runs the estimator on the probe's own lines for its arm and
    requires the runner route, passes a probe that declares nothing with a warning, and fails (naming why) a probe whose
    estimate is beyond a runner, killed or failed. Run in bash with a stand-in ``python`` that exits as told."""
    names = [step.get("name") for step in probe["steps"]]  # type: ignore[index]
    if "Memory estimate and route" not in names:
        return ["the probe job has no step 'Memory estimate and route'"]
    if names.index("Memory estimate and route") > names.index("Run probe/run.sh"):
        return ["the memory estimate does not precede the probe's script"]
    step, found = step_named(probe, "Memory estimate and route"), []
    script = str(step["run"])
    if "mem_estimate.py --from-file probe/estimate.txt" not in script or "--require actions" not in script \
            or '--arm "$PROBE_ARM"' not in script:
        found.append("the estimate step does not run mem_estimate.py on probe/estimate.txt for its arm, requiring actions")
    for index, (declared, code, passes, says) in enumerate([(False, 0, True, "::warning::"), (True, 0, True, ""),
                                                            (True, 1, False, "beyond a runner"), (True, 2, False, "estimate failed"),
                                                            (True, 137, False, "killed")]):
        where = scratch / str(index)
        (where / "probe").mkdir(parents=True)
        (where / "bin").mkdir()
        if declared:
            (where / "probe" / "estimate.txt").write_text("*: --case x:y\n", encoding="utf-8")
        stand_in = where / "bin" / "python"
        stand_in.write_text(f"#!/bin/sh\necho \"called: $@\"\nexit {code}\n", encoding="utf-8")
        stand_in.chmod(0o755)
        (where / "step.sh").write_text(script, encoding="utf-8")
        done = subprocess.run([BASH, "-e", str(where / "step.sh")], cwd=where, text=True, capture_output=True, check=False,
                              env={"PATH": f"{where / 'bin'}:/usr/bin:/bin", "PROBE_ARM": "a1"})
        if (done.returncode == 0) != passes or says not in done.stdout + done.stderr:
            found.append(f"estimate step (declared {declared}, estimator exit {code}): exit {done.returncode}, "
                         f"output {(done.stdout + done.stderr)!r}; wanted {'a pass' if passes else 'a failure'} saying {says!r}")
        if declared and "called: .github/probe/mem_estimate.py --from-file probe/estimate.txt --arm a1 --require actions" \
                not in done.stdout:
            found.append("the estimate step does not hand the estimator the probe's arm")
    return found


def arm_problems(job: dict[str, object] | None, scratch: Path) -> list[str]:
    if not job:
        return ["the workflow has no `arms` job"]
    script = str(step_named(job, "The arms to run")["run"])
    body = script.split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    found = []
    cases = [(None, {"arm": [""]}), ("a\nb\n\n", {"arm": ["a", "b"]}), ("one.arm_1\n", {"arm": ["one.arm_1"]}),
             ("a\na\n", None), ("bad name\n", None), ("$(id)\n", None), ("a;b\n", None), ("\n\n", None)]
    for index, (arms, wanted) in enumerate(cases):
        where = scratch / str(index)
        (where / "probe").mkdir(parents=True)
        if arms is not None:
            (where / "probe" / "arms.txt").write_text(arms, encoding="utf-8")
        (where / "step.py").write_text(body, encoding="utf-8")
        output = where / "out"
        done = subprocess.run(["python3", "-I", str(where / "step.py")], cwd=where, text=True, capture_output=True,
                              check=False, env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(output)})
        got = json.loads(output.read_text(encoding="utf-8").split("matrix=", 1)[1]) if done.returncode == 0 else None
        # A refused file (an empty one too: it names no arm) exits non-zero.
        if got != wanted:
            found.append(f"probe/arms.txt {arms!r}: the matrix is {got}, not {wanted}")
    return found


def test_probe_and_the_guard_pass_in_the_real_workflows(tmp_path):
    assert probe_problems(ROOT, tmp_path / "probe") == []
    assert guard_problems(ROOT, tmp_path / "guard") == []


def planted(tmp_path: Path, name: str, old: str, new: str) -> Path:
    copy = tmp_path / "copy"
    shutil.copytree(ROOT / ".github", copy / ".github")
    path = copy / ".github" / "workflows" / name
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the plant's anchor {old!r} is not in {name} once"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return copy


PROBE_PLANTS = {
    "a pull request trigger": ('    branches: ["probe/**"]\n', '    branches: ["probe/**"]\n  pull_request:\n    branches: ["probe/**"]\n',
                               "the triggers are"),
    "a push to the release branches too": ('    branches: ["probe/**"]\n', '    branches: ["probe/**", "release/**"]\n',
                                           "the triggers are"),
    "an unpinned action": ("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1",
                           "actions/upload-artifact@v7", "is not pinned by a commit SHA"),
    "write permissions": ("permissions:\n  contents: read\n", "permissions:\n  contents: write\n", "permissions are not"),
    "no cancelling": ("  cancel-in-progress: true\n", "  cancel-in-progress: false\n", "does not cancel its older run"),
    "the probe not waiting on the scan": ("    needs: [scan, arms]\n", "    needs: [arms]\n", "does not wait on the scan"),
    "a changed scan step": ("python3 scripts/check_confidential.py ${range:+--messages", "python3 scripts/check_confidential.py ${range:+--diffs",
                            "is not ci.yml's"),
    "a new branch scanned over the whole history": ('range="$dev..$PUSH_AFTER"', 'range="$PUSH_AFTER"', "is not ci.yml's"),
    "a shallow scan": ("          fetch-depth: 0\n          persist-credentials: false\n      - name: No private name",
                       "          fetch-depth: 1\n          persist-credentials: false\n      - name: No private name",
                       "scan's checkout is not ci.yml's"),
    "a timeout beyond the job's": ("--kill-after=60 19200", "--kill-after=60 20400", "`timeout` shorter"),
    "a job timeout over six hours": ("timeout-minutes: 340", "timeout-minutes: 360", "at most 340"),
    "an upload skipped after a failure": ("        if: ${{ always() }}\n        uses: actions/upload-artifact", "        uses: actions/upload-artifact",
                                          "under always()"),
    "a shorter retention": ("retention-days: 14", "retention-days: 90", "under always()"),
    "an arm name reaching the shell unchecked": ("[A-Za-z0-9._-]{1,40}", ".{1,40}", "probe/arms.txt"),
    "the estimate step dropped": ("      - name: Memory estimate and route\n", "      - name: Something else\n", "Memory estimate and route"),
    "the runner route no longer required": ("--arm \"$PROBE_ARM\" --require actions", "--arm \"$PROBE_ARM\"", "does not run mem_estimate.py"),
    "a killed estimate passing": ('if [ "$rc" -ge 128 ]; then', 'if [ "$rc" -ge 1280 ]; then', "killed"),
    "a secret in the probe job": ("      PROBE_ARM: ${{ matrix.arm }}\n", "      PROBE_ARM: ${{ matrix.arm }}\n      K: ${{ secrets.X }}\n",
                                  "reads a secret"),
}


@pytest.mark.parametrize("plant", PROBE_PLANTS)
def test_control_a_break_in_probe_yml_fails_naming_it(tmp_path, plant):
    old, new, problem = PROBE_PLANTS[plant]
    copy = planted(tmp_path, "probe.yml", old, new)
    assert any(problem in each for each in probe_problems(copy, tmp_path / "scratch")), plant


GUARD_PLANTS = {
    "the guard removed": ('      - name: A probe branch is never a pull request\'s head\n        env:\n          HEAD_REF: ${{ github.head_ref }}\n'
                          '        run: |\n          case "$HEAD_REF" in probe/*)', None, "first step"),
    "a pattern that misses": ('case "$HEAD_REF" in probe/*)', 'case "$HEAD_REF" in probes/*)', "head ref 'probe/x'"),
    "an exit that passes": ('is a probe branch: probe branches are never merged or opened as pull requests"; exit 1 ;;',
                            'is a probe branch: probe branches are never merged or opened as pull requests"; exit 0 ;;', "head ref 'probe/x'"),
    "a pattern that catches too much": ('case "$HEAD_REF" in probe/*)', 'case "$HEAD_REF" in *probe*)', "head ref 'my/probe/x'"),
}


@pytest.mark.parametrize("plant", GUARD_PLANTS)
def test_control_a_break_in_the_guard_fails_naming_it(tmp_path, plant):
    old, new, problem = GUARD_PLANTS[plant]
    if new is None:  # drop the whole step: the text from its name to the scan job's checkout
        copy = tmp_path / "copy"
        shutil.copytree(ROOT / ".github", copy / ".github")
        path = copy / ".github" / "workflows" / "ci.yml"
        text = path.read_text(encoding="utf-8")
        start = text.index("      - name: A probe branch is never a pull request's head")
        end = text.index("      - uses: actions/checkout", start)
        path.write_text(text[:start] + text[end:], encoding="utf-8")
    else:
        copy = planted(tmp_path, "ci.yml", old, new)
    assert any(problem in each for each in guard_problems(copy, tmp_path / "scratch")), plant
