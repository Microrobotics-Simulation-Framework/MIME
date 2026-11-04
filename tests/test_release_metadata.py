"""The files that state a release about MIME agree with the package: ``CITATION.cff`` (the citation GitHub shows) and
``.zenodo.json`` (the record Zenodo's GitHub integration makes of a release, which takes precedence over the CFF there)
name the version ``mime.__version__`` states (the CFF; Zenodo takes the version from the release's tag), the author
``pyproject.toml`` names, its licence and the same keywords; the Zenodo record is software, links the repository and the
documentation, and while the version is a pre-release says it is one, so it claims no more than a development release;
and the changelog's newest release is this version. The controls plant one disagreement each in a copy, and each is
named."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

import mime

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Microrobotics-Simulation-Framework/MIME"


def problems(root: Path, version: str = mime.__version__) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    (author,) = [held["name"] for held in project["authors"]]
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    found: list[str] = []
    if str(citation.get("version")) != version:
        found.append(f"CITATION.cff states version {citation.get('version')}, not {version}")
    names = [f"{held.get('given-names')} {held.get('family-names')}" for held in citation.get("authors", [])]
    if names != [author]:
        found.append(f"CITATION.cff's authors are {names}, not pyproject.toml's {author}")
    if citation.get("license") != project["license"] or zenodo.get("license") != project["license"]:
        found.append(f"the licence is not pyproject.toml's {project['license']} in both")
    creators = [held.get("name") for held in zenodo.get("creators", [])]
    if creators != [f"{held.get('family-names')}, {held.get('given-names')}" for held in citation.get("authors", [])]:
        found.append(f".zenodo.json's creators are {creators}, not CITATION.cff's authors")
    if zenodo.get("keywords") != citation.get("keywords"):
        found.append(".zenodo.json's keywords are not CITATION.cff's")
    if zenodo.get("title") != citation.get("title"):
        found.append(".zenodo.json's title is not CITATION.cff's")
    if "version" in zenodo:
        found.append(".zenodo.json states a version; Zenodo takes it from the release's tag")
    if zenodo.get("upload_type") != "software":
        found.append(".zenodo.json's record is not software")
    linked = {held.get("identifier") for held in zenodo.get("related_identifiers", [])}
    if not {REPOSITORY, citation.get("url")} <= linked:
        found.append(".zenodo.json does not link the repository and the documentation")
    if Version(version).is_prerelease and "development pre-release" not in zenodo.get("description", ""):
        found.append(f".zenodo.json's description does not say that {version} is a development pre-release")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    releases = [line for line in changelog.splitlines() if line.startswith("## [") and "[Unreleased]" not in line]
    if not releases or not releases[0].startswith(f"## [{version}] - "):
        found.append(f"the changelog's newest release is {releases[0] if releases else 'none'}, not {version}")
    return found


def test_the_files_that_state_a_release_agree_with_the_package():
    assert problems(ROOT) == []


def planted(tmp_path: Path, name: str, old: str, new: str) -> Path:
    for file in ("pyproject.toml", "CITATION.cff", ".zenodo.json", "CHANGELOG.md"):
        shutil.copy(ROOT / file, tmp_path / file)
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert text.count(old) == 1, f"the plant's anchor {old!r} is not in {name} once"
    (tmp_path / name).write_text(text.replace(old, new), encoding="utf-8")
    return tmp_path


PLANTS = {
    "the-cff-version": ("CITATION.cff", f"version: {mime.__version__}\n", "version: 0.0.1\n",
                        "CITATION.cff states version 0.0.1"),
    "the-cff-author": ("CITATION.cff", "family-names: Roy", "family-names: Someone", "CITATION.cff's authors are"),
    "the-zenodo-creator": (".zenodo.json", '"name": "Roy, ', '"name": "Someone, ', ".zenodo.json's creators are"),
    "the-zenodo-licence": (".zenodo.json", '"license": "Apache-2.0"', '"license": "MIT"', "the licence is not"),
    "a-zenodo-version": (".zenodo.json", '"upload_type": "software",', f'"upload_type": "software", "version": '
                         f'"{mime.__version__}",', ".zenodo.json states a version"),
    "the-zenodo-claim": (".zenodo.json", "development pre-release", "release",
                         "does not say that"),
    "the-changelog": ("CHANGELOG.md", f"## [{mime.__version__}] - ", "## [0.0.1] - ",
                      "the changelog's newest release is ## [0.0.1]"),
}


@pytest.mark.parametrize("plant", PLANTS, ids=list(PLANTS))
def test_control_a_disagreement_is_named(tmp_path, plant):
    name, old, new, problem = PLANTS[plant]
    found = problems(planted(tmp_path, name, old, new))
    assert any(problem in line for line in found), found
