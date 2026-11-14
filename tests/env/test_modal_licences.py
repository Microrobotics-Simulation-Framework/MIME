"""The script ``mime modal`` writes carries the licences accepted where it was written, for its run.

A build reads a gated file only once its licence is accepted (``mime.assets.locate``), and nobody can be asked in a
container, which has no ``licences.json``. The script names, as ``ACCEPT_LICENCES``, the ids of the licences of the
experiment's own assets that this machine accepted (by id and title), and accepts them for what it runs
(``_accepting``). Writing and reading the script only: nothing runs on Modal; the stand-in for ``modal`` loads it, and a
fresh interpreter with an empty acceptances file (the container's) reads the file as the run does.

**Controls**, each failing on every run: a licence accepted here is read in the "container" (the script's list and its
acceptance); one not accepted here is refused there, naming the licence (``asset.licence-not-accepted``); and with the
gate gone from ``locate`` the second would read the file (measured in the PR, with the rates).
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from modal_stub import load

import mime.env._modal_script as generator
from mime import assets
from mime.assets import AssetRef
from mime.assets._acceptances import remember
from mime.env import modal_script
from mime.saved import LOCK_FILE
from mime.terms import CustomLicence
from mime.testing.toys import full_toy

DATA = b"a closed scan" * 100
MAKER = CustomLicence("LicenseRef-Maker-NC", "Maker Licence 1.1", "https://example.org/licence", "not_permitted",
                      ("non-commercial use only",))
OTHER = CustomLicence("LicenseRef-Other", "Other Licence 2.0", "https://example.org/other", "not_permitted",
                      ("for teaching only",))

CONTAINER = '''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("modal_app", sys.argv[1])
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
from mime import assets
from mime.build import BuildRefused
ref = assets.AssetRef.from_record(json.loads(sys.argv[2]))
try:
    with assets.asking(None), script._accepting():
        assets.locate(ref)
    print(json.dumps({"located": True}))
except BuildRefused as refused:
    print(json.dumps({"located": False, "codes": refused.codes(), "action": refused.refusals[0].action}))
'''

# The script's own functions, as Modal calls them, with the ``mime`` command replaced by one that reads the gated file
# (as a build does: ``locate``) for the commands that read bytes. A command that reads none (``env check``) does not
# read it, so a wrap missing from ``_mime`` is not hidden by the check ``verify_experiment`` makes first.
WIRING = '''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("modal_app", sys.argv[1])
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
import mime.cli
from mime import assets
from mime.build import BuildRefused
ref = assets.AssetRef.from_record(json.loads(sys.argv[2]))


def reading(arguments):
    if arguments[0] != "env":
        assets.locate(ref)
    return 0


mime.cli.main = reading
found = {}
with assets.asking(None):
    for name, call in {"_mime": lambda: script._mime("run", "/experiment", "--duration", "1"),
                       "fetch_assets": lambda: script.fetch_assets(),
                       "verify_experiment": lambda: script.verify_experiment("", {}, {})}.items():
        try:
            call()
            found[name] = "read"
        except BuildRefused as refused:
            found[name] = refused.codes()[0]
print(json.dumps(found))
'''


def gated(licence: CustomLicence, name: str) -> AssetRef:
    return AssetRef(name, hashlib.sha256(DATA + name.encode()).hexdigest(), len(DATA) + len(name), licence.id,
                    (f"https://example.org/{name}",), licences=(licence,))


@pytest.fixture
def written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A saved toy whose lock names two gated assets, their bytes placed in a cache, and this machine's acceptance of
    one licence; the script written for it. The "container" has the cache and no acceptances."""
    monkeypatch.setattr(generator, "installed_checkout", lambda: None)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MIME_ASSETS_DIR", str(cache))
    monkeypatch.setenv("MIME_OFFLINE", "1")
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path / "here.json"))
    folder = tmp_path / "toy"
    full_toy().save(folder)
    accepted, refused = gated(MAKER, "accepted.bin"), gated(OTHER, "refused.bin")
    for ref in (accepted, refused):
        (cache / ref.sha256).write_bytes(DATA + ref.name.encode())
        with assets.asking(None):
            assert assets.check(ref) == "ok"
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    lock["assets"] = [accepted.to_record(), refused.to_record()]
    (folder / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    remember(list(assets.licences_of(accepted)))
    script = modal_script(folder, out=tmp_path / "out", wheel_sha256=lambda version: "ab" * 32).script
    load(script, tmp_path)  # lays the stand-in for modal beside it
    return script, accepted, refused, tmp_path


def in_the_container(script: Path, ref: AssetRef, tmp: Path) -> dict[str, object]:
    environment = {**os.environ, "PYTHONPATH": str(tmp / "stub"), "MIME_LICENCE_ACCEPTANCES": str(tmp / "container.json")}
    done = subprocess.run([sys.executable, "-c", CONTAINER, str(script), json.dumps(ref.to_record())],
                          capture_output=True, text=True, env=environment, cwd=tmp, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def through_the_scripts_functions(script: Path, ref: AssetRef, tmp: Path) -> dict[str, str]:
    environment = {**os.environ, "PYTHONPATH": str(tmp / "stub"), "MIME_LICENCE_ACCEPTANCES": str(tmp / "container.json")}
    done = subprocess.run([sys.executable, "-c", WIRING, str(script), json.dumps(ref.to_record())],
                          capture_output=True, text=True, env=environment, cwd=tmp, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_the_scripts_own_functions_run_mime_inside_the_acceptance(written):
    """``_mime`` (``mime run``, ``assets fetch``, the study) and ``verify_experiment`` each call the command inside
    ``_accepting()``: a file accepted here is read by all three, and one not accepted here is refused by all three. Each
    wrap is what a planted removal of it fails, on every run."""
    script, accepted, refused, tmp = written
    assert through_the_scripts_functions(script, accepted, tmp) == {
        "_mime": "read", "fetch_assets": "read", "verify_experiment": "read"}
    assert through_the_scripts_functions(script, refused, tmp) == {
        name: "asset.licence-not-accepted" for name in ("_mime", "fetch_assets", "verify_experiment")}


def test_the_script_carries_the_ids_accepted_here_and_only_those(written):
    script, accepted, refused, _ = written
    text = script.read_text(encoding="utf-8")
    assert f"ACCEPT_LICENCES = {[MAKER.id]!r}" in text and OTHER.id not in text


def test_a_gated_asset_accepted_here_is_read_in_the_scripts_run(written):
    script, accepted, _, tmp = written
    assert in_the_container(script, accepted, tmp) == {"located": True}


def test_a_gated_asset_not_accepted_here_is_refused_in_the_scripts_run_naming_the_licence(written):
    script, _, refused, tmp = written
    found = in_the_container(script, refused, tmp)
    assert found["located"] is False and found["codes"] == ["asset.licence-not-accepted"]
    assert OTHER.id in str(found["action"])


def test_an_experiment_with_no_assets_carries_no_licence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(generator, "installed_checkout", lambda: None)
    monkeypatch.setenv("MIME_LICENCE_ACCEPTANCES", str(tmp_path / "here.json"))
    remember([MAKER])
    full_toy().save(tmp_path / "toy")
    script = modal_script(tmp_path / "toy", out=tmp_path / "out", wheel_sha256=lambda version: "ab" * 32).script
    assert "ACCEPT_LICENCES = []" in script.read_text(encoding="utf-8")
