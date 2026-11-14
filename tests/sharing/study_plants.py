"""The defects the study container's controls plant, each placed at random from a seed: in the study folder before it is
packed, or in the container after (a container rewritten by hand, its manifest made to agree with the change). Each
names the refusal codes that catch it; a control fails on every run where its plant is caught, and passes clean."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mime.saved._container import container_hash

PLANT_VARIABLE = "MIME_P2_PLANT"
SEED_VARIABLE = "DETECTION_SEED"
#: A study folder written once beforehand (a probe writes it), copied in place of running the study in each process.
SOURCE_VARIABLE = "MIME_P2_STUDY_SOURCE"


@dataclass(frozen=True)
class Plant:
    """A defect: what it changes in the study folder (``folder``) or in its container (``container``), and the codes of
    which at least one must be among the refusals that catch it."""

    codes: frozenset[str]
    folder: Callable[[Path, random.Random], None] | None = None
    container: Callable[[Path, random.Random], Path] | None = None


_STRAY_PLACES = (".", "experiment", "experiment/scene", "groups/0", "groups/1", "groups/1/experiment")
_ANY_PLACES = (*_STRAY_PLACES, "figures", "groups/1/experiment/scene")


def _stray(root: Path, draw: random.Random) -> None:
    place = root / draw.choice(_STRAY_PLACES)
    name = draw.choice(("notes", "draft", "scratch", "run", "copy")) + draw.choice((".txt", ".md", ".json", ".npz", ""))
    (place / name).write_bytes(draw.randbytes(draw.randint(1, 64)))


def _env(root: Path, draw: random.Random) -> None:
    place = root / draw.choice(_ANY_PLACES)
    if draw.random() < 0.5:
        (place / ".env").write_text(f"TOKEN={draw.randbytes(16).hex()}\n", encoding="utf-8")
    else:
        (place / ".git").mkdir()
        (place / ".git" / "config").write_text("[core]\n", encoding="utf-8")


def _flip(data: bytes, draw: random.Random) -> bytes:
    at = draw.randrange(len(data))
    return data[:at] + bytes([data[at] ^ (1 << draw.randrange(8))]) + data[at + 1:]


def _group(root: Path, draw: random.Random) -> None:
    path = root / "groups" / str(draw.randrange(2)) / "outputs.npz"
    path.write_bytes(_flip(path.read_bytes(), draw))


def _reindented(path: Path, draw: random.Random) -> None:
    held = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(held, indent=draw.choice((2, 3, 4)), sort_keys=draw.random() < 0.5) + "\n",
                    encoding="utf-8")


def _figure_data(root: Path, draw: random.Random) -> None:
    """The data a figure reads changed in a way every other check takes: a JSON file it reads written again, its
    content the same."""
    _reindented(root / draw.choice(("study.json", "groups/0/report.json", "groups/1/report.json")), draw)


def _input_hash(root: Path, draw: random.Random) -> None:
    other = "sha256:" + draw.randbytes(32).hex()
    if draw.random() < 0.5:
        path = root / "study.json"
        held = json.loads(path.read_text(encoding="utf-8"))
        held["experiment"]["input_hash"] = other
        path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path = root / "study.mime.yaml"
        text = path.read_text(encoding="utf-8")
        start = text.index("sha256:")
        path.write_text(text[:start] + other + text[start + len(other):], encoding="utf-8")


def _machine_path(root: Path, draw: random.Random) -> None:
    person = f"{draw.choice(('ada', 'lin', 'kim', 'noor'))}{draw.randrange(100)}"
    where = "/".join(("", "home", person, "runs", f"{draw.randrange(1000)}.npz"))  # joined, so the file names no home folder
    if draw.random() < 0.5:
        path = root / "study.mime.yaml"
        path.write_text(path.read_text(encoding="utf-8") + f"# made from {where}\n", encoding="utf-8")
    else:
        path = root / "groups" / str(draw.randrange(2)) / "report.json"
        held = json.loads(path.read_text(encoding="utf-8"))
        held["note"] = where
        path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def rewritten(source: Path, change: Callable[[dict[str, bytes], dict[str, object]], None]) -> Path:
    """The container at ``source`` written again with ``change`` made to its members and its manifest, the manifest's
    entries and its own hash made to agree: what a person who rewrites a container by hand can make."""
    with zipfile.ZipFile(source) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(members.pop("manifest.json"))
    change(members, manifest)
    for name, data in members.items():
        if name.startswith("study/"):
            manifest["files"][name.removeprefix("study/")] = {"sha256": hashlib.sha256(data).hexdigest(),
                                                              "size": len(data)}
    manifest["container_hash"] = container_hash(manifest, "container_hash")
    target = source.with_name("rewritten_" + source.name)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for name, data in [("manifest.json", (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode()),
                           *members.items()]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return target


def _forged_group(container: Path, draw: random.Random) -> Path:
    def change(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        name = f"study/groups/{draw.randrange(2)}/outputs.npz"
        members[name] = _flip(members[name], draw)

    return rewritten(container, change)


def _forged_input_hash(container: Path, draw: random.Random) -> Path:
    def change(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        held = json.loads(members["study/study.json"])
        held["experiment"]["input_hash"] = "sha256:" + draw.randbytes(32).hex()
        members["study/study.json"] = (json.dumps(held, indent=1, sort_keys=True) + "\n").encode()

    return rewritten(container, change)


def _reference_outside(container: Path, draw: random.Random) -> Path:
    def change(members: dict[str, bytes], manifest: dict[str, object]) -> None:
        path = draw.choice(("groups/0/extra.npz", "figures/extra.json", "study.extra", "experiment/../x"))
        referenced = manifest["referenced"]
        assert isinstance(referenced, dict)
        referenced[path] = {"name": "anything", "sha256": "0" * 64, "size": 1}

    return rewritten(container, change)


def _renamed(container: Path, draw: random.Random) -> Path:
    target = container.with_name(container.stem + draw.choice(("", "_copy", "-1")) + ".mimebundle")
    container.rename(target)
    return target


PLANTS: dict[str, Plant] = {
    "stray": Plant(frozenset({"study.stray-file", "bundle.stray-file"}), folder=_stray),
    "env": Plant(frozenset({"study.stray-file", "bundle.stray-file"}), folder=_env),
    "group": Plant(frozenset({"study.group-file"}), folder=_group),
    "figure": Plant(frozenset({"plot.data-changed"}), folder=_figure_data),
    "input-hash": Plant(frozenset({"study.input-hash"}), folder=_input_hash),
    "machine-path": Plant(frozenset({"bundle.machine-path"}), folder=_machine_path),
    "forged-group": Plant(frozenset({"study.group-file"}), container=_forged_group),
    "forged-input-hash": Plant(frozenset({"study.input-hash"}), container=_forged_input_hash),
    "reference-outside": Plant(frozenset({"bundle.unsafe"}), container=_reference_outside),
    "renamed": Plant(frozenset({"container.kind"}), container=_renamed),
}

__all__ = ["PLANTS", "PLANT_VARIABLE", "SEED_VARIABLE", "SOURCE_VARIABLE", "Plant", "rewritten"]
