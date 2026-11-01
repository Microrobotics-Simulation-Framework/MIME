"""The saved form's licence, terms and permissions (version 4, designs #198, #199), and the forms before them.

A document of version 4 holds the experiment's own licence, each custom licence it names, each third-party file's
terms by its path with the sha256 they were declared for, and each declared permission with its files by sha256, each
key only where something is declared; a load reproduces them exactly. ``save`` refuses what is incomplete, by its code,
and never an experiment that declares no licence. A folder of versions 1, 2 and 3 declares no licence, terms or
permission, loads as such, and one that holds a key version 4 added is refused, never read as declaring it.

The controls: a version 3 folder whose document was given a licence after it was written must be refused, on every
one of twenty runs, with the upgrade's check removed (where it would load as if it declared that licence); and a
permission bound to bytes the file no longer has must grant nothing, on every run, with the binding by sha256
removed (where it would grant the changed file).
"""

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from terms_cases import NO_REDISTRIBUTION, layered, mesh_bytes, write_mesh

from mime.build import BuildRefused, TermsDeclaration
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, _terms, _versions, notice, pack
from mime.saved._folder import LoadRefused
from mime.saved._versions import document_in_form, upgraded_document, upgraded_lock
from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

PREVIOUS = Path(__file__).parent / "previous"
V3 = PREVIOUS / "layered_toy"
ORIGIN = ("https://example.org/upstream/6a3ebb1/Arm.usda",)


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def messages_of(action: Callable[[], object]) -> str:
    try:
        action()
    except BuildRefused as refused:
        return "\n".join(r.message for r in refused.refusals)
    raise AssertionError("not refused")


def copied(tmp_path: Path, source: Path = V3) -> Path:
    return Path(shutil.copytree(source, tmp_path / source.name))


def a_permission(*files: object, **changes: object) -> Permission:
    held: dict[str, object] = {"grantor": "Ada Maker, founder, Robot Maker Ltd", "grantee": "A. Lab, Uni of Somewhere",
                               "files": files, "scope": Scope(True, False, False), "date": "2026-11-20",
                               "evidence": Evidence("written_permission", "an email of 2026-11-20, held by the grantee")}
    held.update(changes)
    return Permission(**held)  # type: ignore[arg-type]


def third_party(licence: object = NO_REDISTRIBUTION, **changes: object) -> Terms:
    held: dict[str, object] = {"licence": licence, "credit": "Robot Maker Ltd", "role": "presentation",
                               "origin": ORIGIN}
    held.update(changes)
    return Terms(**held)  # type: ignore[arg-type]


# --- what the document holds -------------------------------------------------------------------------------------


def test_a_document_of_version_four_holds_the_licence_the_terms_and_the_permissions(tmp_path: Path):
    mesh = write_mesh(tmp_path / "in", "Arm", 3000)
    pin = hashlib.sha256(mesh.read_bytes()).hexdigest()
    experiment = layered("termed", [(mesh, third_party())], licence="CC-BY-4.0",
                         permissions=[a_permission(pin)])
    experiment.build().save(tmp_path / "termed")
    document = yaml.safe_load((tmp_path / "termed" / DOCUMENT_FILE).read_text())
    assert document["schema"] == "mime.document/4" and document["licence"] == "CC-BY-4.0"
    assert document["licences"] == {NO_REDISTRIBUTION.id: {"title": NO_REDISTRIBUTION.title,
                                                           "url": NO_REDISTRIBUTION.url,
                                                           "redistribution": "not_permitted"}}
    assert document["terms"] == {"scene/layers/Arm.usda": {"sha256": pin, "licence": NO_REDISTRIBUTION.id,
                                                           "credit": "Robot Maker Ltd", "role": "presentation",
                                                           "origin": list(ORIGIN)}}
    (permission,) = document["permissions"]
    assert permission["files"] == [pin] and permission["scope"] == {"redistribute": True, "modify": False,
                                                                     "commercial": False}
    assert permission["evidence"]["kind"] == "written_permission"


def test_a_document_that_declares_nothing_holds_none_of_the_new_keys(tmp_path: Path):
    layered("plain", []).build().save(tmp_path / "plain")
    document = yaml.safe_load((tmp_path / "plain" / DOCUMENT_FILE).read_text())
    assert document["schema"] == "mime.document/4"
    assert not set(_versions.TERMS_KEYS) & set(document)
    assert "referenced_files" not in json.loads((tmp_path / "plain" / LOCK_FILE).read_text())


def test_terms_and_permissions_load_as_they_were_saved_and_save_again_to_the_same_document(tmp_path: Path):
    mesh = write_mesh(tmp_path / "in", "Arm", 3000)
    pin = hashlib.sha256(mesh.read_bytes()).hexdigest()
    layered("termed", [(mesh, third_party())], licence="MIT", permissions=[a_permission(pin)]).build().save(
        tmp_path / "termed")
    loaded = Experiment.load(tmp_path / "termed")
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[2] == "input hash: reproduced exactly"
    loaded.build().save(tmp_path / "again")
    assert (tmp_path / "again" / DOCUMENT_FILE).read_bytes() == (tmp_path / "termed" / DOCUMENT_FILE).read_bytes()
    lock = json.loads((tmp_path / "again" / LOCK_FILE).read_text())
    assert "referenced_files" not in lock, "the permission lets the mesh travel in a bundle"


def test_a_permission_may_name_its_file_by_the_asset_that_declares_it(tmp_path: Path):
    from mime.scene import asset

    mesh = write_mesh(tmp_path / "in", "Arm", 3000)
    experiment = layered("by_asset", [(mesh, third_party())], permissions=[a_permission(asset(mesh))])
    experiment.build().save(tmp_path / "by_asset")
    document = yaml.safe_load((tmp_path / "by_asset" / DOCUMENT_FILE).read_text())
    assert document["permissions"][0]["files"] == [hashlib.sha256(mesh.read_bytes()).hexdigest()]


def test_the_experiment_without_a_licence_is_saved_and_packed_and_says_so(tmp_path: Path):
    layered("unlicensed", []).build().save(tmp_path / "unlicensed")
    pack(tmp_path / "unlicensed", out=tmp_path / "unlicensed.mimebundle")
    text = notice(tmp_path / "unlicensed")
    assert "The experiment declares no licence" in text and "will not publish it until one is declared" in text


# --- what save refuses --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("licence, said", [
    ("CC BY 4.0", "is not an SPDX licence expression"),
    ("cc-by-4.0", "is written 'CC-BY-4.0' in SPDX's canonical spelling"),
    ("NOASSERTION", "nobody stated one"),
    ("MIT AND LicenseRef-Mine", "names LicenseRef-Mine as text"),
])
def test_an_experiments_licence_that_is_not_a_licence_is_refused(tmp_path: Path, licence: str, said: str):
    experiment = layered("bad", [], licence=licence)
    assert codes_of(lambda: experiment.build().save(tmp_path / "bad")) == ["terms.licence"]
    assert said in messages_of(lambda: experiment.build().save(tmp_path / "bad"))


def test_a_licence_declared_twice_is_refused(tmp_path: Path):
    experiment = layered("twice", [], licence="MIT")
    experiment.licence("CC0-1.0")
    assert codes_of(lambda: experiment.build().save(tmp_path / "twice")) == ["terms.licence"]


@pytest.mark.parametrize("terms, code, said", [
    (third_party(licence="Robot Maker's own"), "terms.licence", "is not an SPDX licence expression"),
    (third_party(licence="LicenseRef-Robot-Maker"), "terms.licence", "as text"),
    (third_party(licence=CustomLicence("LicenseRef-Robot-Maker", "", "https://example.org/x")), "terms.custom-licence",
     "has no title"),
    (third_party(credit=""), "terms.file", "give no credit"),
    (third_party(role="drawing"), "terms.file", "give the role 'drawing'"),
    (third_party(origin=("ftp://example.org/arm.usda",)), "terms.origin", "give the origin"),
    (third_party(origin="https://example.org/arm.usda"), "terms.origin", "not a tuple of locations"),
])
def test_incomplete_terms_are_refused_by_their_code(tmp_path: Path, terms: Terms, code: str, said: str):
    experiment = layered("bad_terms", [(write_mesh(tmp_path / "in", "Arm", 3000), terms)])
    assert codes_of(lambda: experiment.build().save(tmp_path / "bad")) == [code]
    assert said in messages_of(lambda: experiment.build().save(tmp_path / "bad"))


def test_two_custom_licences_of_one_id_are_refused(tmp_path: Path):
    other = CustomLicence(NO_REDISTRIBUTION.id, "Another title", NO_REDISTRIBUTION.url, "not_permitted")
    experiment = layered("two", [(write_mesh(tmp_path / "a", "Arm", 3000), third_party()),
                                 (write_mesh(tmp_path / "b", "Hand", 3000), third_party(licence=other))])
    assert codes_of(lambda: experiment.build().save(tmp_path / "two")) == ["terms.custom-licence"]


def test_terms_for_a_file_the_experiment_does_not_save_are_refused(tmp_path: Path):
    experiment = layered("stray", [(write_mesh(tmp_path / "in", "Arm", 3000), None)])
    experiment._declarations.append(TermsDeclaration("scene/layers/Hand.usda", "0" * 64, third_party()))
    assert codes_of(lambda: experiment.build().save(tmp_path / "stray")) == ["terms.file"]
    assert "not a layer or a description" in messages_of(lambda: experiment.build().save(tmp_path / "stray"))


@pytest.mark.parametrize("change, said", [
    ({"grantor": ""}, "names no grantor"), ({"date": "next week"}, "date"),
    ({"scope": Scope(False, True, True)}, "does not include redistribution"),
    ({"evidence": Evidence("public_statement", "an email")}, "https:// link"),
])
def test_an_incomplete_permission_is_refused_naming_the_part(tmp_path: Path, change: dict[str, object], said: str):
    mesh = write_mesh(tmp_path / "in", "Arm", 3000)
    pin = hashlib.sha256(mesh.read_bytes()).hexdigest()
    experiment = layered("bad_permission", [(mesh, third_party())], permissions=[a_permission(pin, **change)])
    assert codes_of(lambda: experiment.build().save(tmp_path / "bad")) == ["terms.permission"]
    assert said in messages_of(lambda: experiment.build().save(tmp_path / "bad"))


def test_a_permission_must_be_a_permission(tmp_path: Path):
    from mime.build import ArgumentTypeRefused

    with pytest.raises(ArgumentTypeRefused) as refused:
        layered("typed", []).permission("Ada said yes")  # type: ignore[arg-type]
    assert refused.value.codes() == ["call.wrong-type"]


# --- a permission is bound to the bytes ---------------------------------------------------------------------------


def a_changed_file_inherits_no_permission(root: Path) -> None:
    """The scenario: a permission for a mesh's bytes; the mesh changes; the permission names the old bytes. Its file
    is judged without it, and a save that still declares it is refused."""
    old = mesh_bytes("Arm", 3000)
    new = mesh_bytes("Arm", 3100)
    old_pin, new_pin = hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()
    document = {"terms": {"scene/layers/Arm.usda": {"sha256": new_pin, "licence": NO_REDISTRIBUTION.id,
                                                     "credit": "Robot Maker Ltd", "role": "presentation"}},
                "licences": {NO_REDISTRIBUTION.id: {"title": NO_REDISTRIBUTION.title, "url": NO_REDISTRIBUTION.url,
                                                     "redistribution": "not_permitted"}},
                "permissions": [{"grantor": "Ada Maker", "grantee": "A. Lab", "files": [old_pin],
                                 "scope": {"redistribute": True, "modify": True, "commercial": True},
                                 "date": "2026-11-20", "evidence": {"kind": "public_statement",
                                                                    "pointer": "https://example.org/yes"}}]}
    (judged,) = _terms.file_verdicts(document, {"scene/layers/Arm.usda": (new_pin, len(new))}, {})  # type: ignore[arg-type]
    assert not judged.verdict.carried and judged.permissions == (), judged
    mesh = root / "in" / "Arm.usda"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(new)
    experiment = layered("changed", [(mesh, third_party())], permissions=[a_permission(old_pin)])
    assert codes_of(lambda: experiment.build().save(root / "changed")) == ["terms.permission"]


def test_a_permission_bound_to_bytes_the_file_no_longer_has_grants_nothing(tmp_path: Path):
    a_changed_file_inherits_no_permission(tmp_path)


def test_a_permission_for_the_files_bytes_lets_it_travel(tmp_path: Path):
    mesh = write_mesh(tmp_path / "in", "Arm", 3000)
    pin = hashlib.sha256(mesh.read_bytes()).hexdigest()
    layered("granted", [(mesh, third_party())], permissions=[a_permission(pin)]).build().save(tmp_path / "granted")
    text = notice(tmp_path / "granted")
    assert "permitted, conditional (non-commercial, no-derivatives), by a declared permission" in text
    assert "A. Lab, Uni of Somewhere declares that Ada Maker, founder, Robot Maker Ltd permitted them to redistribute"\
           in text


# --- the forms before licences ------------------------------------------------------------------------------------


def test_the_fixture_is_a_folder_of_version_three_with_a_layer():
    assert yaml.safe_load((V3 / DOCUMENT_FILE).read_text())["schema"] == "mime.document/3"
    lock = json.loads((V3 / LOCK_FILE).read_text())
    assert lock["schema"] == "mime.lock/3" and list(lock["stage"]["layers"]) == ["scene/layers/arm_visual.usda"]


def test_a_version_three_folder_loads_declaring_no_licence(tmp_path: Path):
    report = Experiment.load(copied(tmp_path)).load_report
    assert report is not None
    assert report.text().splitlines()[1] == ("saved form: mime.document/3, read as version 4; it declares no "
                                             "licence, terms or permission, being of a form before them")
    assert report.text().splitlines()[2] == "input hash: reproduced exactly"


def test_mime_verify_reproduces_a_version_three_folder(tmp_path: Path):
    done = subprocess.run([sys.executable, "-m", "mime.cli", "verify", str(copied(tmp_path))], capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify: reproduced 8 steps" in done.stdout


def test_a_version_three_folder_packs_its_layer_as_its_authors_own(tmp_path: Path):
    folder = copied(tmp_path)
    bundle = pack(folder, out=tmp_path / "v3.mimebundle")
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        assert "experiment/scene/layers/arm_visual.usda" in archive.namelist()
        text = archive.read("NOTICE").decode()
    assert "the folder is of mime.document/3, a form before licences" in text
    assert "scene/layers/arm_visual.usda" in text and "the author's own" in text


def test_the_write_back_of_version_three_is_the_inverse_of_its_upgrade():
    document = yaml.safe_load((V3 / DOCUMENT_FILE).read_text())
    upgraded, version, renamed = upgraded_document(document)
    assert (version, renamed, upgraded["schema"]) == (3, (), "mime.document/4")
    assert document_in_form(upgraded, 3) == document
    held, lock_version = upgraded_lock(json.loads((V3 / LOCK_FILE).read_text()))
    assert lock_version == 3 and held["schema"] == "mime.lock/4"


def a_version_three_folder_given_a_licence_is_refused(root: Path) -> None:
    folder = copied(root)
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump({**document, "licence": "CC-BY-4.0"}, sort_keys=False))
    try:
        Experiment.load(folder)
    except LoadRefused as refused:
        assert refused.codes() == ["load.document-shape"] * len(refused.codes())
        assert "a mime.document/3 document holds licence, which version 4 of the saved form added" in str(refused)
    else:
        raise AssertionError("a version 3 folder given a licence loaded as if it declared one")


def test_a_version_three_folder_given_a_licence_is_refused(tmp_path: Path):
    a_version_three_folder_given_a_licence_is_refused(tmp_path)


def test_a_version_three_lock_holding_a_reference_is_refused(tmp_path: Path):
    folder = copied(tmp_path)
    lock = json.loads((folder / LOCK_FILE).read_text())
    lock["referenced_files"] = {"scene/layers/arm_visual.usda": "arm"}
    (folder / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert "load.lock-shape" in refused.value.codes()
    assert "a mime.lock/3 lock holds referenced_files" in str(refused.value)


@pytest.mark.parametrize("name", ["bem_ball", "magnetic_ball"])
def test_an_earlier_version_given_a_terms_key_is_refused(tmp_path: Path, name: str):
    folder = copied(tmp_path, PREVIOUS / name)
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump({**document, "terms": {}}, sort_keys=False))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert "load.document-shape" in refused.value.codes()


# --- the controls -------------------------------------------------------------------------------------------------

RUNS = 20


@contextmanager
def upgrade_ignores_the_terms_keys() -> Iterator[None]:
    """The plant: an upgrade that reads a version 3 document's licence as if version 3 could declare one."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "terms_keys_held", lambda data: [])
        patch.setattr("mime.saved._folder.terms_keys_held", lambda data: [])
        yield


@contextmanager
def permission_binds_to_any_bytes() -> Iterator[None]:
    """The plant: a verdict that lets every permission cover every file, as one bound to a path would after the file
    changed."""
    real = _terms.third_party_verdict

    def unbound(doc, sha256, permissions, custom):  # type: ignore[no-untyped-def]
        return real(doc, sha256, [{**held, "files": [*held["files"], sha256]} for held in permissions], custom)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_terms, "third_party_verdict", unbound)
        yield


def planted_runs(tmp_path: Path, scenario: Callable[[Path], None], plant: Callable[[], object]) -> int:
    caught = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        with plant():  # type: ignore[attr-defined]
            try:
                scenario(root)
            except AssertionError:
                caught += 1
    return caught


def test_control_a_version_three_folder_read_as_declaring_a_licence_is_caught_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_version_three_folder_given_a_licence_is_refused,
                        upgrade_ignores_the_terms_keys) == RUNS


def test_control_a_permission_bound_to_bytes_that_no_longer_match_is_caught_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_changed_file_inherits_no_permission, permission_binds_to_any_bytes) == RUNS
