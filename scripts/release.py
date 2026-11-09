#!/usr/bin/env python
"""The release workflow's steps (``.github/workflows/publish.yml``), on the standard library alone, so that they run
before anything is installed.

``check-tag --ref REF DIST...``: a release's tag is ``v`` and the package's version, and the distributions built from it
are of that version. The version is read from its one source, ``src/mime/__init__.py`` (``__version__``), and from each
distribution's own metadata (a wheel's ``METADATA``, an sdist's ``PKG-INFO``). REF is the run's ``GITHUB_REF``:
``refs/tags/<tag>`` on a release and on a manual run from a tag, ``refs/heads/<branch>`` on a manual run from a
branch, which is refused. Writes ``version=<version>`` to ``$GITHUB_OUTPUT`` when it is set. Exits 1 naming each
disagreement.

``check-ci --repository OWNER/NAME --commit SHA``: the release gate (CONTRIBUTING.md, "CI: which lanes a pull request runs").
Three of CI's five lanes are no required check of a pull request: ``default`` on Python 3.13, ``maddening-main``
and ``openmaterica-main`` run on a pull request only by its diff. So a release is refused unless the newest push or manual run
of ``ci.yml`` on the release head has all five lanes' jobs (``default`` on 3.12, 3.13 and 3.14, and the two ``-main``
lanes) concluded ``success``, each named when it did not, or when there is no such run. Exits 0 or 1.

``find-pages --repository OWNER/NAME --commit SHA --out DIR``: the documentation pages of the commit's own CI run, the
artifact ``docs-pages-python-3.12`` that CI's default lane builds with the lane's JUnit reports (so its evidence counts
page states each cited test's outcome in that run), downloaded into DIR with ``gh`` from the newest run of ``ci.yml`` on
the commit that still holds it. Writes ``source=ci`` and the run's address and conclusion to ``$GITHUB_OUTPUT``, or
``source=build`` when no run holds it (none ran on the commit, or the artifact expired), and the workflow then builds
the pages itself. Exits 0 either way, and 1 when what was downloaded is not a build's pages.

``pack-pages --pages DIR --version V --commit SHA --source ci|build [--run URL --conclusion C] --out FILE``: the release
asset ``mime-docs-<version>.tar.gz``, one folder ``mime-docs-<version>/`` holding the pages as ``mime docs build``
writes them (``index.md`` and ``pages/``) and a ``README.md`` saying what they are, the commit and the tag they are of,
and where they came from, with what pages built without the lane's reports lack; and beside it
``mime-docs-<version>.tar.gz.sha256``, its sha256 in ``sha256sum``'s form, which MICROROBOTICA's docs deploy pins. The
asset's name and layout are fixed: the site renders them. Exits 1 when DIR is not a build's pages.

``tests/test_release_workflow.py`` runs each against planted cases and reads the workflow's wiring of them."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("src/mime/__init__.py")
_VERSION = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.M)
#: CI's workflow, and the artifact of the pages its default lane builds on Python 3.12 with the lane's JUnit reports.
WORKFLOW = "ci.yml"
ARTIFACT = "docs-pages-python-3.12"
#: Where each MIME release's pages are read (``mime.build._refusals.DOCS``): the refusals link their codes' pages there.
DOCS = "https://microrobotica.org/mime"

Gh = Callable[[list[str]], str]

#: The jobs of ``ci.yml`` that stand for its five test lanes, by their names, and the events whose runs hold them all
#: (a pull request's run holds the lanes its diff starts, and a scheduled run only ``usd-core-newest``).
LANE_JOBS = ("default (Python 3.12)", "default (Python 3.13)", "default (Python 3.14)", "maddening-main",
             "openmaterica-main")
FULL_EVENTS = ("push", "workflow_dispatch")


def output(**values: str) -> None:
    """Each value as a step's output, when the step runs on GitHub; printed either way."""
    for name, value in values.items():
        print(f"{name}={value}")
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as held:
            held.writelines(f"{name}={value}\n" for name, value in values.items())


# -- check-tag -------------------------------------------------------------------------------------------------------

def source_version(root: Path = ROOT) -> str | None:
    found = _VERSION.search((root / SOURCE).read_text(encoding="utf-8"))
    return found["version"] if found else None


def distribution_version(path: Path) -> str | None:
    """The ``Version`` a wheel's ``METADATA`` or an sdist's ``PKG-INFO`` states, or ``None``."""
    text = ""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as wheel:
            names = [name for name in wheel.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
            text = wheel.read(names[0]).decode("utf-8") if len(names) == 1 else ""
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as sdist:
            members = [member for member in sdist.getmembers() if re.fullmatch(r"[^/]+/PKG-INFO", member.name)]
            held = sdist.extractfile(members[0]) if len(members) == 1 else None
            text = held.read().decode("utf-8") if held is not None else ""
    found = re.search(r"^Version: (\S+)$", text, re.M)
    return found.group(1) if found else None


def tag_problems(ref: str, distributions: Sequence[Path], root: Path = ROOT) -> tuple[str | None, list[str]]:
    """The package's version, and every way the run's ref and the distributions disagree with it."""
    version = source_version(root)
    if version is None:
        return None, [f'{SOURCE} holds no line __version__ = "..."']
    problems: list[str] = []
    if not ref.startswith("refs/tags/"):
        problems.append(f"this run is of {ref or 'no ref'}: a release is published from its tag, v{version}")
    elif (tag := ref.removeprefix("refs/tags/")) != f"v{version}":
        problems.append(f"the tag {tag} is not v{version}, the version {SOURCE} states: tag the commit that states it")
    if not distributions:
        problems.append("no distribution was given to check")
    for path in distributions:
        held = distribution_version(path)
        if held != version:
            problems.append(f"{path.name} is of version {held}, not {version}")
    return version, problems


# -- find-pages ------------------------------------------------------------------------------------------------------

def run_gh(arguments: list[str]) -> str:
    return subprocess.run(["gh", *arguments], capture_output=True, text=True, check=True).stdout


def run_with_pages(repository: str, commit: str, gh: Gh) -> dict[str, object] | None:
    """The newest run of CI's workflow on ``commit`` that holds the pages' artifact, unexpired, or ``None``."""
    listed = json.loads(gh(["api", f"repos/{repository}/actions/workflows/{WORKFLOW}/runs?head_sha={commit}"
                                   f"&per_page=100"]))
    runs = [run for run in listed.get("workflow_runs", []) if run.get("head_sha") == commit]
    for run in sorted(runs, key=lambda run: int(run["id"]), reverse=True):
        held = json.loads(gh(["api", f"repos/{repository}/actions/runs/{run['id']}/artifacts?name={ARTIFACT}"]))
        if any(artifact.get("name") == ARTIFACT and not artifact.get("expired")
               for artifact in held.get("artifacts", [])):
            return run
    return None


def pages_problems(pages: Path) -> list[str]:
    """Why ``pages`` is not what ``mime docs build`` writes: ``index.md`` and a ``pages/`` of Markdown files."""
    found: list[str] = []
    if not (pages / "index.md").is_file():
        found.append(f"{pages} holds no index.md")
    if not (pages / "pages").is_dir() or not any((pages / "pages").rglob("*.md")):
        found.append(f"{pages} holds no pages/ of Markdown files")
    return found


def find_pages(repository: str, commit: str, out: Path, gh: Gh = run_gh) -> int:
    run = run_with_pages(repository, commit, gh)
    if run is None:
        print(f"no run of {WORKFLOW} on {commit} holds {ARTIFACT}: the pages are built in this job")
        output(source="build", run="", conclusion="")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    gh(["run", "download", str(run["id"]), "--repo", repository, "--name", ARTIFACT, "--dir", str(out)])
    if problems := pages_problems(out):
        print("\n".join(f"error: {problem} (from run {run['html_url']})" for problem in problems), file=sys.stderr)
        return 1
    print(f"the pages of {run['html_url']} ({ARTIFACT})")
    output(source="ci", run=str(run["html_url"]), conclusion=str(run.get("conclusion") or "in progress"))
    return 0


# -- check-ci --------------------------------------------------------------------------------------------------------

def lane_problems(jobs: Sequence[dict[str, Any]]) -> list[str]:
    """Each of the five lanes whose job is missing from ``jobs`` (a run's jobs at their latest attempts, each a mapping
    with a ``name`` and a ``conclusion``) or did not conclude ``success``, named."""
    conclusion = {str(job["name"]): job.get("conclusion") for job in jobs}
    found: list[str] = []
    for lane in LANE_JOBS:
        if lane not in conclusion:
            found.append(f"{lane}: the run has no such job")
        elif conclusion[lane] != "success":
            found.append(f"{lane}: concluded {conclusion[lane] or 'nothing yet (still running)'}")
    return found


def run_jobs(repository: str, run_id: int, gh: Gh) -> list[dict[str, Any]]:
    """The run's jobs at their latest attempts, as ``name`` and ``conclusion``."""
    text = gh(["api", f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", "--paginate",
               "--jq", ".jobs[] | {name, conclusion}"])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def ci_problems(repository: str, commit: str, gh: Gh) -> tuple[str, list[str]]:
    """The newest push or manual run of CI on ``commit``, and every lane of it that is not green."""
    listed = json.loads(gh(["api", f"repos/{repository}/actions/workflows/{WORKFLOW}/runs?head_sha={commit}"
                                   f"&per_page=100"]))
    runs = [run for run in listed.get("workflow_runs", []) if run.get("head_sha") == commit
            and run.get("event") in FULL_EVENTS]
    if not runs:
        return "", [f"no push or manual run of {WORKFLOW} on {commit}: push the commit to a release branch, or run "
                    f"{WORKFLOW} by hand on it, and let it finish"]
    newest = max(runs, key=lambda run: int(run["id"]))
    return str(newest.get("html_url", newest["id"])), lane_problems(run_jobs(repository, int(newest["id"]), gh))


def check_ci(repository: str, commit: str, gh: Gh = run_gh) -> int:
    try:
        url, problems = ci_problems(repository, commit, gh)
    except (subprocess.CalledProcessError, ValueError, KeyError) as error:
        print(f"error: CI's runs on {commit} could not be read ({type(error).__name__}: {error}); nothing is released "
              f"on a gate that was not checked", file=sys.stderr)
        return 1
    if problems:
        print("\n".join(f"error: not every lane is green on {commit} ({url or 'no run'}): {problem}"
                        for problem in problems), file=sys.stderr)
        return 1
    print(f"all five lanes ({', '.join(LANE_JOBS)}) are green on {commit} ({url})")
    return 0


# -- pack-pages ------------------------------------------------------------------------------------------------------

def release_of(version: str) -> str:
    """The release whose documentation the pages are, as the refusals link it: ``0.3.0`` of ``0.3.0.dev1``."""
    found = re.match(r"\d+\.\d+\.\d+", version)
    return found.group(0) if found else "latest"


def readme(version: str, commit: str, source: str, run: str, conclusion: str) -> str:
    where = {
        "ci": (f"They are the artifact `{ARTIFACT}` of the commit's own CI run, {run}, built in its default lane on "
               f"Python 3.12 with the lane's JUnit reports, so the evidence counts page "
               f"(`pages/reference/evidence_counts.md`) states the outcome in that run of each test it cites. The "
               f"run's conclusion when the pages were taken: {conclusion}."),
        "build": ("No CI run of the commit held its pages (none ran on it, or the artifact had expired), so the release "
                  "workflow built them, with the same command in the same locked environment as CI's default lane, "
                  "but without the lane's JUnit reports: the evidence counts page "
                  "(`pages/reference/evidence_counts.md`) reads \"outcome not read (no lane report)\" wherever it cites "
                  "a test, and says nothing of whether the test passed."),
    }[source]
    return (f"# MIME {version}: the documentation pages\n\n"
            f"The pages `mime docs build` writes for MIME {version}, the tag `v{version}` (commit {commit}): "
            f"`index.md` and `pages/`, in Markdown. They are generated from the code and the hand-written sources "
            f"at that commit, and are never committed. Each error MIME prints links to its code's page under "
            f"{DOCS}/{release_of(version)}/pages/errors/.\n\n{where}\n\n"
            f"The release workflow scanned them for a private name or path before attaching them.\n")


def pack_pages(pages: Path, version: str, commit: str, source: str, run: str, conclusion: str, out: Path) -> int:
    if problems := pages_problems(pages):
        print("\n".join(f"error: {problem}" for problem in problems), file=sys.stderr)
        return 1
    if source == "ci" and not run:
        print("error: pages from a CI run name the run (--run)", file=sys.stderr)
        return 1
    folder = f"mime-docs-{version}"
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / folder
        staged.mkdir()
        shutil.copy2(pages / "index.md", staged / "index.md")
        # hidden files, the build's own marker among them, are left out, as CI's artifact upload leaves them out
        shutil.copytree(pages / "pages", staged / "pages", ignore=shutil.ignore_patterns(".*"))
        (staged / "README.md").write_text(readme(version, commit, source, run, conclusion), encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        archive_folder(staged, folder, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    checksum(out).write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"wrote {out} ({source}) and {checksum(out).name}: {digest}")
    return 0


def checksum(asset: Path) -> Path:
    """The asset's sha256 file beside it, in ``sha256sum``'s form (``<hex>  <name>``), which a consumer pins."""
    return asset.with_name(f"{asset.name}.sha256")


def archive_folder(staged: Path, folder: str, out: Path) -> None:
    """``staged`` as ``out``, a gzip tar whose bytes depend on the files' names and contents alone: the same pages give
    the same sha256 on every run, which the site that pins it needs when a failed docs job is re-run. So the members
    are sorted, each has time 0, one owner and one mode, and the gzip header holds neither a time nor a file name."""
    paths = sorted(staged.rglob("*"), key=lambda path: path.relative_to(staged).as_posix())
    with out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, \
            tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in [staged, *paths]:
            member = archive.gettarinfo(path, arcname=f"{folder}/{path.relative_to(staged).as_posix()}"
                                        if path != staged else folder)
            member.mtime, member.uid, member.gid, member.uname, member.gname = 0, 0, 0, "", ""
            member.mode = 0o755 if member.isdir() else 0o644
            if member.isfile():
                with path.open("rb") as content:
                    archive.addfile(member, content)
            else:
                archive.addfile(member)


# -- the command -----------------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release.py", description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-tag")
    check.add_argument("--ref", required=True)
    check.add_argument("distributions", nargs="*", type=Path)
    gate = commands.add_parser("check-ci")
    gate.add_argument("--repository", required=True)
    gate.add_argument("--commit", required=True)
    find = commands.add_parser("find-pages")
    find.add_argument("--repository", required=True)
    find.add_argument("--commit", required=True)
    find.add_argument("--out", required=True, type=Path)
    pack = commands.add_parser("pack-pages")
    pack.add_argument("--pages", required=True, type=Path)
    pack.add_argument("--version", required=True)
    pack.add_argument("--commit", required=True)
    pack.add_argument("--source", required=True, choices=["ci", "build"])
    pack.add_argument("--run", default="")
    pack.add_argument("--conclusion", default="")
    pack.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "check-tag":
        version, problems = tag_problems(arguments.ref, arguments.distributions)
        if problems:
            print("\n".join(f"error: {problem}" for problem in problems), file=sys.stderr)
            return 1
        print(f"the tag is v{version}, and every distribution is of {version}")
        output(version=str(version))
        return 0
    if arguments.command == "check-ci":
        return check_ci(arguments.repository, arguments.commit)
    if arguments.command == "find-pages":
        return find_pages(arguments.repository, arguments.commit, arguments.out)
    return pack_pages(arguments.pages, arguments.version, arguments.commit, arguments.source, arguments.run,
                      arguments.conclusion, arguments.out)


if __name__ == "__main__":
    sys.exit(main())
