"""The release workflow (``.github/workflows/publish.yml``) and its script (``scripts/release.py``), read and run as
``test_ci_wiring.py`` reads and runs CI's.

- **The tag is the version.** ``release.py check-tag`` passes a run of the tag ``v<version>`` whose wheel and sdist are of
  that version, and refuses a run of another tag, a manual run from a branch, a distribution of another version (read
  from its metadata, not its name), none at all, and a source that states no version. In the workflow its step is in
  the build job, after the build and before the upload, with its exit status intact (it is run, with ``GITHUB_REF`` a
  branch's, and the job must stop), and the publishing job needs the build job, so a refused tag publishes nothing.
  The audit of the distributions and the tag's tree is still there.
- **The pages are attached.** ``release.py find-pages`` takes the pages of the newest run of ``ci.yml`` on the tagged
  commit that still holds its ``docs-pages-python-3.12`` artifact (a run on another commit, or one whose artifact
  expired, is passed over), says ``source=build`` when none does, and fails on a download that is not a build's
  pages; ``pack-pages`` writes ``mime-docs-<version>.tar.gz``, one folder ``mime-docs-<version>/`` of ``index.md``,
  ``pages/`` and a ``README.md`` that says where the pages came from, and its sha256 beside it (``.sha256``, in
  ``sha256sum``'s form). The site that renders the pages pins that sha256, so the name and the layout are held. The workflow's docs job is run step by step, its ``gh``, ``uv``
  and scan stubbed: with a CI run holding the pages it builds nothing and attaches that run's pages; with none it
  installs, builds and attaches its own; either way it scans them first, and a failed scan attaches nothing.

The controls plant one break each in a copy of the workflow (the tag step dropped, its exit status neutered, its ref
fixed, the publishing job freed of the build, the audit dropped, the docs built though CI's pages were found, the scan
dropped, the asset renamed, its sha256 left unattached, the job's write permission dropped), and the check must name it, every run: the workflow
and the plants are fixed, so there is no detection rate to measure."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

import mime

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
_SPEC = importlib.util.spec_from_file_location("mime_release_script", ROOT / "scripts" / "release.py")
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = release
_SPEC.loader.exec_module(release)

VERSION = mime.__version__
REPOSITORY = "Microrobotics-Simulation-Framework/MIME"
COMMIT = "c0ffee" * 6 + "c0ff"
RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/202"
Workflow = dict[str, Any]


# -- the distributions and the pages a test makes ---------------------------------------------------------------------

def wheel(directory: Path, version: str, named: str | None = None) -> Path:
    path = directory / f"mime_engine-{named or version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as held:
        held.writestr(f"mime_engine-{version}.dist-info/METADATA", f"Metadata-Version: 2.4\nName: mime-engine\n"
                                                                    f"Version: {version}\n")
    return path


def sdist(directory: Path, version: str) -> Path:
    path = directory / f"mime_engine-{version}.tar.gz"
    with tarfile.open(path, "w:gz") as held:
        data = f"Metadata-Version: 2.4\nName: mime-engine\nVersion: {version}\n".encode()
        member = tarfile.TarInfo(f"mime_engine-{version}/PKG-INFO")
        member.size = len(data)
        held.addfile(member, io.BytesIO(data))
    return path


def source_tree(root: Path, version: str | None = VERSION) -> Path:
    (root / "src" / "mime").mkdir(parents=True)
    line = f'__version__ = "{version}"\n' if version else "VERSION = None\n"
    (root / "src" / "mime" / "__init__.py").write_text(f'"""MIME."""\n\n{line}', encoding="utf-8")
    return root


def pages(directory: Path, marker: str = "ci") -> Path:
    (directory / "pages" / "guide").mkdir(parents=True)
    (directory / "index.md").write_text(f"# MIME ({marker})\n", encoding="utf-8")
    (directory / "pages" / "guide" / "install.md").write_text("# Install\n", encoding="utf-8")
    return directory


# -- check-tag --------------------------------------------------------------------------------------------------------

def test_the_source_states_the_version_the_package_reports():
    assert release.source_version(ROOT) == VERSION


def test_a_release_of_the_versions_tag_with_distributions_of_it_passes(tmp_path):
    root = source_tree(tmp_path / "root")
    found = [wheel(tmp_path, VERSION), sdist(tmp_path, VERSION)]
    assert release.tag_problems(f"refs/tags/v{VERSION}", found, root) == (VERSION, [])


@pytest.mark.parametrize(("ref", "versions", "problem"), [
    ("refs/tags/v0.0.1", (VERSION, VERSION), "the tag v0.0.1 is not v"),
    (f"refs/tags/{VERSION}", (VERSION, VERSION), f"the tag {VERSION} is not v"),
    ("refs/heads/v0.3", (VERSION, VERSION), "this run is of refs/heads/v0.3: a release is published from its tag"),
    ("", (VERSION, VERSION), "this run is of no ref"),
    (f"refs/tags/v{VERSION}", ("0.0.1", VERSION), "is of version 0.0.1, not"),
    (f"refs/tags/v{VERSION}", (VERSION, "0.0.1"), "is of version 0.0.1, not"),
    (f"refs/tags/v{VERSION}", (), "no distribution was given to check"),
], ids=["another-tag", "no-v", "a-branch", "no-ref", "the-wheel-of-another-version", "the-sdist-of-another-version",
        "no-distribution"])
def test_a_release_that_is_not_of_its_versions_tag_is_refused(tmp_path, ref, versions, problem):
    root = source_tree(tmp_path / "root")
    found = [make(tmp_path, version) for make, version in zip((wheel, sdist), versions, strict=False)]
    version, problems = release.tag_problems(ref, found, root)
    assert version == VERSION and len(problems) == 1 and problem in problems[0], problems


def test_a_distributions_version_is_read_from_its_metadata_and_not_its_name(tmp_path):
    root = source_tree(tmp_path / "root")
    named_right = wheel(tmp_path, "0.0.1", named=VERSION)
    assert release.tag_problems(f"refs/tags/v{VERSION}", [named_right, sdist(tmp_path, VERSION)], root)[1] == [
        f"{named_right.name} is of version 0.0.1, not {VERSION}"]


def test_a_source_that_states_no_version_is_refused(tmp_path):
    root = source_tree(tmp_path / "root", None)
    assert release.tag_problems(f"refs/tags/v{VERSION}", [wheel(tmp_path, VERSION)], root) == (
        None, ['src/mime/__init__.py holds no line __version__ = "..."'])


# -- find-pages and pack-pages, with a fake gh -------------------------------------------------------------------------

class FakeGh:
    """``gh`` as the script calls it: the runs of ``ci.yml`` on a commit, each run's artifacts by name, and a download
    that writes ``download`` into the directory asked for."""

    def __init__(self, runs: list[dict[str, Any]], artifacts: dict[int, list[dict[str, Any]]],
                 download: Path | None = None) -> None:
        self.runs, self.artifacts, self.download, self.calls = runs, artifacts, download, []

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(arguments)
        if arguments[0] == "api" and "/actions/workflows/ci.yml/runs?" in arguments[1]:
            return json.dumps({"workflow_runs": self.runs})
        if arguments[0] == "api" and (held := re.search(r"/actions/runs/(\d+)/artifacts\?name=(\S+)", arguments[1])):
            return json.dumps({"artifacts": [a for a in self.artifacts.get(int(held[1]), []) if a["name"] == held[2]]})
        if arguments[:2] == ["run", "download"]:
            target = Path(arguments[arguments.index("--dir") + 1])
            if self.download is not None:
                shutil.copytree(self.download, target, dirs_exist_ok=True)
            return ""
        raise AssertionError(f"gh called with {arguments}")


def a_run(number: int, commit: str = COMMIT, conclusion: str | None = "success") -> dict[str, Any]:
    return {"id": number, "head_sha": commit, "conclusion": conclusion,
            "html_url": f"https://github.com/{REPOSITORY}/actions/runs/{number}"}


def pages_artifact(expired: bool = False) -> list[dict[str, Any]]:
    return [{"name": release.ARTIFACT, "expired": expired}]


def outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if line)


def test_the_newest_run_on_the_commit_that_still_holds_the_pages_is_the_one_taken(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    gh = FakeGh([a_run(201), a_run(203), a_run(202), a_run(204, commit="another")],
                {201: pages_artifact(), 202: pages_artifact(), 203: pages_artifact(expired=True),
                 204: pages_artifact()}, download=pages(tmp_path / "artifact"))
    assert release.find_pages(REPOSITORY, COMMIT, tmp_path / "out", gh) == 0
    assert ["run", "download", "202", "--repo", REPOSITORY, "--name", release.ARTIFACT, "--dir",
            str(tmp_path / "out")] in gh.calls
    assert (tmp_path / "out" / "index.md").read_text(encoding="utf-8") == "# MIME (ci)\n"
    assert outputs(tmp_path / "output") == {"source": "ci", "run": RUN_URL, "conclusion": "success"}


@pytest.mark.parametrize("runs", ["none", "expired", "another-commit", "no-artifact"])
def test_with_no_run_holding_the_pages_they_are_built_in_the_job(tmp_path, monkeypatch, runs):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    held = {"none": ([], {}), "expired": ([a_run(7)], {7: pages_artifact(expired=True)}),
            "another-commit": ([a_run(7, commit="another")], {7: pages_artifact()}),
            "no-artifact": ([a_run(7)], {7: [{"name": "junit-default-python-3.12", "expired": False}]})}[runs]
    gh = FakeGh(*held)
    assert release.find_pages(REPOSITORY, COMMIT, tmp_path / "out", gh) == 0
    assert not any(call[:2] == ["run", "download"] for call in gh.calls)
    assert outputs(tmp_path / "output") == {"source": "build", "run": "", "conclusion": ""}


def test_a_download_that_is_not_a_builds_pages_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    (tmp_path / "artifact").mkdir()
    (tmp_path / "artifact" / "something.txt").write_text("not the pages\n", encoding="utf-8")
    gh = FakeGh([a_run(7)], {7: pages_artifact()}, download=tmp_path / "artifact")
    assert release.find_pages(REPOSITORY, COMMIT, tmp_path / "out", gh) == 1
    assert not (tmp_path / "output").exists()


def unpacked(asset: Path) -> dict[str, str]:
    with tarfile.open(asset) as held:
        return {member.name: (held.extractfile(member).read().decode("utf-8") if member.isfile() else "")  # type: ignore[union-attr]
                for member in held.getmembers()}


@pytest.mark.parametrize("source", ["ci", "build"])
def test_the_asset_is_one_folder_of_the_pages_and_a_readme_saying_where_they_came_from(tmp_path, source):
    asset = tmp_path / f"mime-docs-{VERSION}.tar.gz"
    run = RUN_URL if source == "ci" else ""
    assert release.pack_pages(pages(tmp_path / "pages", source), VERSION, COMMIT, source, run, "success", asset) == 0
    held = unpacked(asset)
    folder = f"mime-docs-{VERSION}"
    assert {name for name in held if not name.endswith(("pages", "guide", folder))} == {
        f"{folder}/index.md", f"{folder}/pages/guide/install.md", f"{folder}/README.md"}
    assert {name.split("/")[1] for name in held if name.count("/") == 1} == {"index.md", "pages", "README.md"}
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert (tmp_path / f"{asset.name}.sha256").read_text(encoding="utf-8") == f"{digest}  {asset.name}\n"
    readme = held[f"{folder}/README.md"]
    assert f"MIME {VERSION}" in readme and f"`v{VERSION}` (commit {COMMIT})" in readme
    assert f"/mime/{release.release_of(VERSION)}/pages/errors/" in readme
    if source == "ci":
        assert RUN_URL in readme and "states the outcome in that run of each test it cites" in readme
    else:
        assert "without the lane's JUnit reports" in readme and "outcome not read (no lane report)" in readme


def test_the_asset_is_refused_for_what_is_not_a_builds_pages_or_ci_pages_with_no_run(tmp_path):
    (tmp_path / "empty").mkdir()
    out = tmp_path / "asset.tar.gz"
    assert release.pack_pages(tmp_path / "empty", VERSION, COMMIT, "build", "", "", out) == 1
    assert release.pack_pages(pages(tmp_path / "pages"), VERSION, COMMIT, "ci", "", "", out) == 1
    assert not out.exists()


# -- the workflow, read and run ----------------------------------------------------------------------------------------

def load(path: Path = WORKFLOW) -> Workflow:
    loaded: Workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def steps_running(job: Workflow, text: str) -> list[tuple[int, Workflow]]:
    return [(number, step) for number, step in enumerate(job.get("steps") or []) if text in str(step.get("run", ""))]


NEUTERING = re.compile(r"\|\|\s*(?:true|:)(?![\w-])|\bset\s+\+e\b")


def expression(text: str, context: dict[str, str]) -> str:
    """``text`` with each ``${{ name }}`` replaced from ``context``; a name it lacks fails, so a plant cannot pass by
    being unread."""
    def value(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name not in context:
            raise AssertionError(f"the expression {name!r} is not one this test evaluates")
        return context[name]
    return re.sub(r"\$\{\{(.*?)\}\}", value, text)


def condition(text: object, context: dict[str, str]) -> bool:
    """A step's ``if``: none, or ``<name> == '<literal>'``, the one form the docs job uses."""
    if text is None:
        return True
    held = re.fullmatch(r"\s*(?:\$\{\{)?\s*([\w.-]+)\s*==\s*'([^']*)'\s*(?:\}\})?\s*", str(text))
    if held is None:
        raise AssertionError(f"the condition {text!r} is not one this test evaluates")
    return context.get(held[1], "") == held[2]


STUB = """#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
log = Path(os.environ["STUB_LOG"])
with log.open("a") as held:
    held.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
config = json.loads(Path(os.environ["STUB_CONFIG"]).read_text())
name, words = Path(sys.argv[0]).name, sys.argv[1:]
if name == "gh" and words[0] == "api" and "/actions/workflows/ci.yml/runs?" in words[1]:
    print(json.dumps({"workflow_runs": config["runs"]}))
elif name == "gh" and words[0] == "api" and "/artifacts?name=" in words[1]:
    run = words[1].split("/actions/runs/")[1].split("/")[0]
    print(json.dumps({"artifacts": config["artifacts"].get(run, [])}))
elif name == "gh" and words[:2] == ["run", "download"]:
    shutil.copytree(config["ci_pages"], words[words.index("--dir") + 1], dirs_exist_ok=True)
elif name == "gh" and words[:2] == ["release", "upload"]:
    for file in words[3:]:
        if file.startswith("--"):
            break
        shutil.copy(file, Path(config["released"]) / Path(file).name)
elif name == "uv" and "build" in words:
    shutil.copytree(config["built_pages"], words[words.index("--out") + 1], dirs_exist_ok=True)
elif name == "check_confidential.py":
    sys.exit(config.get("scan_exit", 0))
"""


def run_docs_job(workflow: Workflow, scratch: Path, ci_holds_pages: bool, scan_exit: int = 0
                 ) -> tuple[int, list[list[str]], Path]:
    """The docs job run step by step in ``bash -e`` as GitHub runs it, its ``uses`` steps skipped, ``gh`` and ``uv``
    stubbed and the scan replaced by a stub, ``release.py`` itself run: its exit status, every stubbed call, and the
    directory the release's assets land in."""
    job = workflow["jobs"]["docs"]
    checkout = scratch / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "release.py", checkout / "scripts" / "release.py")
    bin_ = scratch / "bin"
    bin_.mkdir()
    for name in ("gh", "uv"):
        (bin_ / name).write_text(STUB, encoding="utf-8")
        (bin_ / name).chmod(0o755)
    (checkout / "scripts" / "check_confidential.py").write_text(STUB, encoding="utf-8")
    released = scratch / "released"
    released.mkdir()
    runner_temp = scratch / "runner"
    runner_temp.mkdir()
    config = {"runs": [a_run(202)], "artifacts": {"202": pages_artifact(expired=not ci_holds_pages)},
              "ci_pages": str(pages(scratch / "ci_pages", "ci")), "built_pages": str(pages(scratch / "built", "built")),
              "released": str(released), "scan_exit": scan_exit}
    (scratch / "config.json").write_text(json.dumps(config), encoding="utf-8")
    context = {"github.repository": REPOSITORY, "github.sha": COMMIT, "github.token": "a-token",
               "needs.build.outputs.version": VERSION, "secrets.CONFIDENTIAL_SCAN_KEY": "a-key"}
    env = {name: expression(str(value), context) for name, value in (job.get("env") or {}).items()}
    context |= {f"env.{name}": value for name, value in env.items()}
    log = scratch / "calls.log"
    log.touch()
    status = 0
    for number, step in enumerate(job.get("steps") or []):
        if "run" not in step or not condition(step.get("if"), context):
            continue
        output = scratch / f"output-{number}"
        output.touch()
        script = scratch / f"step-{number}.sh"
        script.write_text(expression(str(step["run"]), context), encoding="utf-8")
        step_env = {name: expression(str(value), context) for name, value in (step.get("env") or {}).items()}
        done = subprocess.run([shutil.which("bash") or "bash", "-e", str(script)], cwd=checkout, capture_output=True,
                              text=True, check=False,
                              env={"PATH": f"{bin_}:{os.environ['PATH']}", "RUNNER_TEMP": str(runner_temp),
                                   "GITHUB_OUTPUT": str(output), "STUB_LOG": str(log),
                                   "STUB_CONFIG": str(scratch / "config.json"), **env, **step_env})
        if step.get("id"):
            context |= {f"steps.{step['id']}.outputs.{name}": value for name, value in outputs(output).items()}
        if done.returncode != 0:
            status = done.returncode
            break
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return status, calls, released


def tag_step_runs(step: Workflow, scratch: Path, ref: str) -> int:
    """The build job's tag step, run in ``bash -e`` with ``release.py`` itself, over a source and two distributions of
    this version, with ``GITHUB_REF`` at ``ref``."""
    checkout = source_tree(scratch / "checkout")
    (checkout / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "release.py", checkout / "scripts" / "release.py")
    (checkout / "dist").mkdir()
    wheel(checkout / "dist", VERSION)
    sdist(checkout / "dist", VERSION)
    (scratch / "step.sh").write_text(str(step["run"]), encoding="utf-8")
    done = subprocess.run([shutil.which("bash") or "bash", "-e", str(scratch / "step.sh")], cwd=checkout,
                          capture_output=True, text=True, check=False,
                          env={"PATH": os.environ["PATH"], "GITHUB_REF": ref, "GITHUB_OUTPUT": str(scratch / "out")})
    return done.returncode


def workflow_problems(path: Path, scratch: Path) -> list[str]:
    """Everything wrong with how ``path`` refuses a release of the wrong tag, keeps the audit, and attaches the pages."""
    workflow = load(path)
    found: list[str] = []
    jobs: dict[str, Workflow] = workflow.get("jobs") or {}
    triggers = workflow.get("on", workflow.get(True)) or {}
    if (triggers.get("release") or {}).get("types") != ["published"]:
        found.append("the workflow does not run on a published release")
    build, publish, docs = jobs.get("build") or {}, jobs.get("publish") or {}, jobs.get("docs") or {}
    tagged = steps_running(build, "scripts/release.py check-tag")
    if len(tagged) != 1:
        found.append(f"{len(tagged)} steps of the build job run release.py check-tag; one does")
    else:
        number, step = tagged[0]
        built = [n for n, _ in steps_running(build, "python -m build")]
        uploads = [n for n, s in enumerate(build.get("steps") or []) if str(s.get("uses", "")).startswith(
            "actions/upload-artifact@")]
        if not (built and uploads and built[0] < number < uploads[0]):
            found.append("the tag step does not run after the build and before the distributions are uploaded")
        if step.get("id") != "tag" or "steps.tag.outputs.version" not in str((build.get("outputs") or {}).get(
                "version", "")):
            found.append("the build job does not output the version the tag step read")
        if step.get("if") or step.get("continue-on-error") or "shell" in step or NEUTERING.search(str(step["run"])):
            found.append("the tag step can be skipped or its failure hidden")
        if tag_step_runs(step, scratch / "tag-branch", "refs/heads/v0.3") == 0:
            found.append("the tag step passes a run from a branch")
        if tag_step_runs(step, scratch / "tag-other", "refs/tags/v0.0.1") == 0:
            found.append("the tag step passes a run of another tag")
        if tag_step_runs(step, scratch / "tag-right", f"refs/tags/v{VERSION}") != 0:
            found.append("the tag step refuses a run of the version's own tag")
    if "build" not in ([publish.get("needs")] if isinstance(publish.get("needs"), str) else publish.get("needs") or []):
        found.append("the publishing job does not need the build job, so a refused tag could still publish")
    if len(steps_running(build, "mime assets audit")) != 1:
        found.append("the build job does not audit the distributions and the tag's tree (mime assets audit)")
    if docs.get("needs") != "build":
        found.append("the docs job does not need the build job, whose tag step gives the version")
    if (docs.get("permissions") or {}).get("contents") != "write" or \
            (docs.get("permissions") or {}).get("actions") != "read":
        found.append("the docs job cannot read CI's runs and write the release's assets (contents: write, "
                     "actions: read)")
        return found
    asset = f"mime-docs-{VERSION}.tar.gz"
    for holds in (True, False):
        where = "with a CI run holding the pages" if holds else "with no CI run holding the pages"
        status, calls, released = run_docs_job(workflow, scratch / f"docs-{holds}", holds)
        builds = [call for call in calls if call[0] == "uv" and "build" in call]
        scans = [call for call in calls if call[0] == "check_confidential.py"]
        if status != 0:
            found.append(f"{where}, the docs job fails")
            continue
        if holds and builds:
            found.append(f"{where}, the docs job builds the pages anyway")
        if not holds and not any(call[1:3] == ["sync", "--locked"] for call in calls if call[0] == "uv"):
            found.append(f"{where}, the docs job does not install the locked releases before it builds")
        if not holds and not builds:
            found.append(f"{where}, the docs job builds no pages")
        if not any("--directory" in call for call in scans):
            found.append(f"{where}, the docs job does not scan the pages before it attaches them")
        uploads = [call for call in calls if call[:3] == ["gh", "release", "upload"]]
        if [call[3] for call in uploads] != [f"v{VERSION}"] or not (released / asset).is_file():
            found.append(f"{where}, the release v{VERSION} is not given {asset}: {[u[3:5] for u in uploads]}")
            continue
        digest = hashlib.sha256((released / asset).read_bytes()).hexdigest()
        if not (released / f"{asset}.sha256").is_file() or \
                (released / f"{asset}.sha256").read_text(encoding="utf-8") != f"{digest}  {asset}\n":
            found.append(f"{where}, the release v{VERSION} is not given {asset}.sha256, the asset's own sha256")
        held = unpacked(released / asset)
        top = {name for name in held if name.count("/") <= 1}
        if top != {f"mime-docs-{VERSION}", *(f"mime-docs-{VERSION}/{name}" for name in ("index.md", "pages",
                                                                                       "README.md"))}:
            found.append(f"{where}, the asset's layout is {sorted(top)}, not mime-docs-{VERSION}/ holding index.md, "
                         f"pages/ and README.md")
        marker = "ci" if holds else "built"
        if held.get(f"mime-docs-{VERSION}/index.md") != f"# MIME ({marker})\n":
            found.append(f"{where}, the asset does not hold the {marker} pages")
    status, calls, released = run_docs_job(workflow, scratch / "docs-scan-fails", True, scan_exit=1)
    if status == 0 or any(call[:3] == ["gh", "release", "upload"] for call in calls):
        found.append("a failed scan of the pages does not stop the docs job before it attaches them")
    return found


def test_the_workflow_refuses_another_tag_keeps_the_audit_and_attaches_the_pages(tmp_path):
    assert workflow_problems(WORKFLOW, tmp_path) == []


TAG_STEP = ("      - name: The tag is v and the package's version, and so is every distribution\n"
            "        id: tag\n"
            "        run: python3 scripts/release.py check-tag --ref \"$GITHUB_REF\" dist/*.whl dist/*.tar.gz\n\n")
BUILD_IF = "        if: steps.ci.outputs.source == 'build'\n        run: uv run --no-sync mime docs build"
SCAN = ("      - name: No private name or path in the pages\n        env:\n"
        "          CONFIDENTIAL_SCAN_KEY: ${{ secrets.CONFIDENTIAL_SCAN_KEY }}\n"
        "        run: python3 scripts/check_confidential.py --directory \"$RUNNER_TEMP/docs-pages\"\n")
#: Each plant: the text replaced in a copy of publish.yml, its replacement, and the problem it must raise.
PLANTS = {
    "the-tag-step-dropped": (TAG_STEP, "", "0 steps of the build job run release.py check-tag; one does"),
    "the-tag-step-neutered": ('dist/*.whl dist/*.tar.gz\n', 'dist/*.whl dist/*.tar.gz || true\n',
                              "the tag step can be skipped or its failure hidden"),
    "the-tag-step-told-its-tag": ('--ref "$GITHUB_REF"', f'--ref "refs/tags/v{VERSION}"',
                                  "the tag step passes a run from a branch"),
    "the-publishing-job-freed": ("    needs: build\n    runs-on: ubuntu-latest\n    environment: pypi\n",
                                 "    runs-on: ubuntu-latest\n    environment: pypi\n",
                                 "the publishing job does not need the build job, so a refused tag could still publish"),
    "the-audit-dropped": ("          mime assets audit dist/*.whl", "          echo dist/*.whl",
                          "the build job does not audit the distributions and the tag's tree (mime assets audit)"),
    "built-though-ci-held-them": (BUILD_IF, "        run: uv run --no-sync mime docs build",
                                  "with a CI run holding the pages, the docs job builds the pages anyway"),
    "the-scan-dropped": (SCAN, "", "with a CI run holding the pages, the docs job does not scan the pages before it "
                                   "attaches them"),
    "the-asset-renamed": ("mime-docs-$VERSION.tar.gz", "docs.tar.gz",
                          f"with a CI run holding the pages, the release v{VERSION} is not given "
                          f"mime-docs-{VERSION}.tar.gz"),
    "the-checksum-dropped": ('\n          "$RUNNER_TEMP/mime-docs-$VERSION.tar.gz.sha256" --clobber', "\n          --clobber",
                             f"with a CI run holding the pages, the release v{VERSION} is not given "
                             f"mime-docs-{VERSION}.tar.gz.sha256"),
    "no-write-permission": ("      contents: write   # gh release upload\n", "",
                            "the docs job cannot read CI's runs and write the release's assets"),
}


@pytest.mark.parametrize("plant", PLANTS, ids=list(PLANTS))
def test_control_a_break_in_the_workflow_fails_naming_it(tmp_path, plant):
    old, new, problem = PLANTS[plant]
    text = WORKFLOW.read_text(encoding="utf-8")
    # the asset's name is planted wherever the job writes or attaches it; every other anchor is in the file once
    assert text.count(old) == (3 if plant == "the-asset-renamed" else 1), f"the plant's anchor {old!r} is not in place"
    copy = tmp_path / "publish.yml"
    copy.write_text(text.replace(old, new), encoding="utf-8")
    found = workflow_problems(copy, tmp_path / "scratch")
    assert any(problem in line for line in found), found
