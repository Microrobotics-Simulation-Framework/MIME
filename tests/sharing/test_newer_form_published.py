"""A bundle (``mime.bundle/N``) or a ``sharing.yaml`` (``mime.sharing/N``) written by a newer MIME is refused as
``load.newer-form``, with the file, the version found, the versions this MIME reads and the advice to upgrade. The
controls: a format this MIME reads is not refused, and an id of another family stays the problem it was."""

import json
import re
import zipfile
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy, write_sharing

from mime import push
from mime.build import BuildRefused
from mime.push._sharing import SHARING_FORMAT
from mime.saved._bundle import FORMAT, FORMAT_1, MANIFEST_MEMBER, unknown_format

NEWER = "load.newer-form"


def bundle_of(path: Path, format_: str) -> zipfile.ZipFile:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(MANIFEST_MEMBER, json.dumps({"format": format_}))
    return zipfile.ZipFile(path)


def test_a_bundle_one_format_above_the_newest_is_refused_and_the_formats_it_reads_are_not(tmp_path: Path):
    for known in (FORMAT, FORMAT_1):
        assert unknown_format(bundle_of(tmp_path / "known.mimebundle", known)) is None
    for other in ("other.bundle/3", "mime.bundle/0", "mime.bundle/x"):
        assert unknown_format(bundle_of(tmp_path / "other.mimebundle", other)) is None
    refused = unknown_format(bundle_of(tmp_path / "newer.mimebundle", "mime.bundle/3"))
    assert refused is not None
    (refusal,) = refused.refusals
    assert refusal.code == NEWER and refusal.category.value == "user"
    assert "mime.bundle/3" in refusal.message and f"{FORMAT}, {FORMAT_1}" in refusal.message
    assert "mime-engine" in refusal.action


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("toy"))


def restate(folder: Path, format_: str) -> None:
    path = folder / "sharing.yaml"
    text = re.sub(r"^format: .*$", f"format: {format_}", path.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def test_a_sharing_file_one_format_above_the_newest_is_refused_and_the_one_it_reads_is_not(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_sharing(folder)
    assert push.read_sharing(folder).title
    restate(folder, "mime.sharing/2")
    with pytest.raises(BuildRefused) as refused:
        push.read_sharing(folder)
    (refusal,) = refused.value.refusals
    assert refusal.code == NEWER and refusal.category.value == "user"
    assert "sharing.yaml" in refusal.message and "mime.sharing/2" in refusal.message
    assert SHARING_FORMAT in refusal.message
    assert "mime-engine" in refusal.action
    restate(folder, "other.sharing/2")
    with pytest.raises(BuildRefused) as other:
        push.read_sharing(folder)
    assert other.value.codes() == ["push.sharing"]
