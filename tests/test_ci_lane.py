"""The lane check: each CI lane asserts, from the installed distributions' own metadata, that it installed what it
claims. A registry install leaves no ``direct_url.json``; a git install records its URL, branch and commit there
(PEP 610).

Run as a script, it prints the lane's installed versions and origins as a Markdown table, for the job summary.
"""

import json
import os
import platform
import sys
import tomllib
import urllib.request
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from packaging.version import Version

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"
PYPI = "https://pypi.org/simple"
REPOSITORIES = {
    "maddening": "github.com/Microrobotics-Simulation-Framework/MADDENING",
    "openmaterica": "github.com/OpenMaterica/OpenMaterica",
}

# What each lane claims to install: "locked" is the PyPI release uv.lock pins, "git" the repository's main
# branch, "newest" the newest release on PyPI when the lane runs.
LANES = {
    "default": {"maddening": "locked", "openmaterica": "locked", "usd-core": "locked"},
    "openmaterica-main": {"maddening": "locked", "openmaterica": "git", "usd-core": "locked"},
    "maddening-main": {"maddening": "git", "openmaterica": "locked", "usd-core": "locked"},
    "usd-core-newest": {"maddening": "locked", "openmaterica": "locked", "usd-core": "newest"},
}


def direct_url(name: str) -> dict | None:
    text = metadata.distribution(name).read_text("direct_url.json")
    return json.loads(text) if text else None


def without_credentials(url: str) -> str:
    parts = urlsplit(url)
    return parts._replace(netloc=parts.netloc.rpartition("@")[2]).geturl()


def locked(name: str) -> dict:
    (package,) = [package for package in tomllib.loads(LOCK.read_text())["package"] if package["name"] == name]
    return package


def newest_on_pypi(name: str) -> Version:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as response:
        releases = json.load(response)["releases"]
    return max(
        Version(version)
        for version, files in releases.items()
        if files and not Version(version).is_prerelease and not all(file.get("yanked") for file in files)
    )


def mismatches(lane: str) -> list[str]:
    found = []
    for name, claim in LANES[lane].items():
        version, direct = metadata.version(name), direct_url(name)
        if claim == "git":
            vcs = (direct or {}).get("vcs_info", {})
            source = f"{without_credentials((direct or {}).get('url', ''))}@{vcs.get('requested_revision')}"
            if vcs.get("vcs") != "git":
                found.append(f"{name}: not a git install (direct_url.json: {direct})")
            elif REPOSITORIES[name] not in source or vcs.get("requested_revision") != "main":
                found.append(f"{name}: installed from {source}, not {REPOSITORIES[name]}@main")
            continue
        if direct is not None:
            found.append(f"{name}: a direct install from {without_credentials(direct['url'])}, not a PyPI release")
        if claim == "locked":
            package = locked(name)
            if package["source"] != {"registry": PYPI}:
                found.append(f"{name}: uv.lock's source is {package['source']}, not PyPI")
            if version != package["version"]:
                found.append(f"{name}: {version} installed, uv.lock pins {package['version']}")
        elif Version(version) != (newest := newest_on_pypi(name)):
            found.append(f"{name}: {version} installed, the newest on PyPI is {newest}")
    return found


def git_installs(lane: str) -> set[str]:
    return {name for name, claim in LANES[lane].items() if claim == "git"}


def this_lane() -> str:
    lane = os.environ.get("MIME_CI_LANE")
    if lane is None:
        if os.environ.get("CI"):
            pytest.fail("MIME_CI_LANE is unset under CI: every lane names itself, so none can skip the lane check")
        pytest.skip("MIME_CI_LANE is unset outside CI: no lane to check")
    if lane not in LANES:
        pytest.fail(f"MIME_CI_LANE={lane!r} names no lane; the lanes are {sorted(LANES)}")
    return lane


def test_the_lane_installed_what_it_claims():
    assert mismatches(this_lane()) == []


def test_the_claims_of_a_lane_with_other_git_installs_fail_here():
    """The check's control, in every lane. The default and usd-core-newest lanes differ only in usd-core's version,
    which coincide while uv.lock pins the newest usd-core, so neither is held to fail on the other's claims."""
    lane = this_lane()
    others = [other for other in LANES if git_installs(other) != git_installs(lane)]
    assert len(others) >= 2
    for other in others:
        assert mismatches(other), f"{other}'s claims pass in the {lane} lane: the check cannot tell them apart"


def report() -> str:
    import jax

    devices = ", ".join(str(device) for device in jax.devices())
    rows = [
        f"**Lane `{os.environ.get('MIME_CI_LANE', '(unset)')}`**: Python {platform.python_version()}, "
        f"JAX {jax.__version__} on `{jax.default_backend()}` ({devices})",
        "",
        "| distribution | version | origin |",
        "| --- | --- | --- |",
    ]
    for name in ("maddening", "openmaterica", "usd-core", "jax", "jaxlib", "mime-engine"):
        direct = direct_url(name)
        if direct is None:
            origin = "registry (no direct_url.json)"
        elif "vcs_info" in direct:
            vcs = direct["vcs_info"]
            origin = f"git {without_credentials(direct['url'])}@{vcs.get('requested_revision')} = {vcs.get('commit_id')}"
        else:
            origin = f"direct {without_credentials(direct['url'])} ({', '.join(key for key in direct if key != 'url')})"
        rows.append(f"| {name} | {metadata.version(name)} | {origin} |")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.stdout.write(report() + "\n")
