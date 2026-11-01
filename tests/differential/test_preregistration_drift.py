"""A preregistered file changes only by a listed change. ``scripts/sim2real/preregistration_7ja.json`` hashed ten
files before any of 7j-a's computations; ``scripts/sim2real/preregistration_drift.json`` lists each of them that has
since changed, with the hash it has now, the commits that changed it and why (the register's amendment A2).

- **Every file the preregistration hashed** either has its preregistered sha256, or is listed with its current sha256.
- **A listed file whose bytes are not the listed sha256 fails**: a further change must be listed again, with its hash.
- **A listed file that has the preregistered sha256 again fails as stale**: the listing says a change that is gone.
- **The listing agrees with the preregistration**: its preregistered hash is the preregistration's, it names no file the
  preregistration did not hash, and it gives each change a commit and a reason.
- **Controls**, failing on every run, in a temporary copy and never in the repository's files: an unlisted change to a
  preregistered file, a listed file changed again, and a listed file restored.

The preregistered bytes stay verifiable at the commit the listing names (``git show <commit>:<path> | sha256sum``)."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = Path("scripts/sim2real/preregistration_7ja.json")
DRIFT = Path("scripts/sim2real/preregistration_drift.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_files(root: Path) -> dict[str, str]:
    """Each file the preregistration hashed, relative to the repository, with its preregistered sha256."""
    preregistered = json.loads((root / PREREGISTRATION).read_text(encoding="utf-8"))
    scripts = {f"scripts/sim2real/{name}": digest for name, digest in preregistered["scripts_sha256"].items()}
    return scripts | preregistered["tests_read_sha256"]


def problems(root: Path) -> list[str]:
    """What is wrong with the preregistered files under ``root``, one line each; empty when every change is listed."""
    preregistered = hashed_files(root)
    listed = json.loads((root / DRIFT).read_text(encoding="utf-8"))["files"]
    found = [f"{path}: listed as changed, but the preregistration did not hash it" for path in listed
             if path not in preregistered]
    for path, preregistered_sha in preregistered.items():
        file = root / path
        if not file.is_file():
            found.append(f"{path}: preregistered, and no longer there")
            continue
        current = sha256(file)
        entry = listed.get(path)
        if entry is None:
            if current != preregistered_sha:
                found.append(f"{path}: changed since the preregistration and not listed in {DRIFT} "
                             f"(now {current}; list it with its commit and reason)")
            continue
        if entry["preregistered_sha256"] != preregistered_sha:
            found.append(f"{path}: the listing's preregistered sha256 is not the preregistration's")
        if not entry["changed_by"] or not entry["reason"].strip():
            found.append(f"{path}: listed without the commits that changed it or the reason")
        if entry["current_sha256"] == preregistered_sha:
            found.append(f"{path}: listed as changed, but its listed hash is the preregistered one (a stale listing)")
        if current == preregistered_sha:
            found.append(f"{path}: listed as changed, but it has the preregistered bytes again (a stale listing)")
        elif current != entry["current_sha256"]:
            found.append(f"{path}: changed again since it was listed (listed {entry['current_sha256']}, now "
                         f"{current}; list the further change with its hash)")
    return found


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    """The preregistration, the listing and every file they name, copied to a temporary folder."""
    for path in [PREREGISTRATION, DRIFT, *map(Path, hashed_files(ROOT))]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, tmp_path / path)
    assert problems(tmp_path) == []
    return tmp_path


def listed_file(root: Path) -> str:
    return next(iter(json.loads((root / DRIFT).read_text(encoding="utf-8"))["files"]))


def test_every_preregistered_file_has_its_preregistered_bytes_or_a_listed_change():
    assert problems(ROOT) == []


def test_the_listing_is_the_registers_six_changes():
    listed = json.loads((ROOT / DRIFT).read_text(encoding="utf-8"))["files"]
    assert sorted(listed) == [
        "scripts/sim2real/assemble.py", "scripts/sim2real/bem.py", "scripts/sim2real/modal_app.py",
        "scripts/sim2real/operating_point.py", "scripts/sim2real/prereg.py", "tests/script_a.py"]


def test_control_an_unlisted_change_to_a_preregistered_file_fails(copy: Path):
    unlisted = next(path for path in hashed_files(copy) if path not in
                    json.loads((copy / DRIFT).read_text(encoding="utf-8"))["files"])
    with (copy / unlisted).open("ab") as held:
        held.write(b"\n")
    found = problems(copy)
    assert len(found) == 1 and found[0].startswith(f"{unlisted}: changed since the preregistration and not listed")


def test_control_a_listed_file_changed_again_fails(copy: Path):
    path = listed_file(copy)
    with (copy / path).open("ab") as held:
        held.write(b"\n")
    found = problems(copy)
    assert len(found) == 1 and found[0].startswith(f"{path}: changed again since it was listed")


def test_control_a_listed_file_restored_to_the_preregistered_bytes_fails_as_stale(copy: Path):
    path = listed_file(copy)
    # The preregistered bytes are not in the working tree, so the control preregisters the file's current bytes: the
    # preregistration then says what the file has, as when a listed file has been restored.
    registered = json.loads((copy / PREREGISTRATION).read_text(encoding="utf-8"))
    table, key = (("scripts_sha256", path.removeprefix("scripts/sim2real/")) if path.startswith("scripts/sim2real/")
                  else ("tests_read_sha256", path))
    registered[table][key] = sha256(copy / path)
    (copy / PREREGISTRATION).write_text(json.dumps(registered), encoding="utf-8")
    found = problems(copy)
    stale = f"{path}: listed as changed, but it has the preregistered bytes again"
    assert any(line.startswith(stale) for line in found)


def test_control_a_listing_that_names_a_file_the_preregistration_did_not_hash_fails(copy: Path):
    listing = json.loads((copy / DRIFT).read_text(encoding="utf-8"))
    listing["files"]["scripts/sim2real/README.md"] = next(iter(listing["files"].values()))
    (copy / DRIFT).write_text(json.dumps(listing), encoding="utf-8")
    assert any(line.startswith("scripts/sim2real/README.md: listed as changed, but the preregistration did not hash")
               for line in problems(copy))
