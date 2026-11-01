"""The NOTICE and the card (``mime card <folder> [--notice]``), offline, exactly as ``mime push`` will publish them:
generated, byte for byte the same each time, every file the folder names in the NOTICE with its licence, credit,
verdict and declared permission, and the card in its sections, saying what it shows of credit before ``sharing.yaml``
exists.

**The 7n case** (design #199 question 9): a presentation mesh under a rights holder's own licence that forbids
redistribution (an AR4-like file, synthetic) in a saved folder. ``pack`` carries it as a reference; the NOTICE and the
card say so, with its licence's title and link, its role and where it is available from.

What they claim is what holds: the author's own files with no licence declared are ``unknown`` (carried, and a reuser
granted no licence), never "permitted"; "reproducible from this bundle alone" only where nothing travels as a reference;
every condition each file's licence or permission carries, on the card as in the NOTICE; and "what reproduces" as what
``mime verify`` would check, never as checked. A licence naming an id the installed packaging does not list is refused
by both (``terms.unlisted-licence``).

The control plants a NOTICE that drops a file, and the property must catch it on every one of twenty runs.
"""

import hashlib
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from terms_cases import NO_REDISTRIBUTION, OPEN_IDS, UNKNOWN_IDS, V3_FIXTURE, layered, same_reference_run, write_mesh

from mime.build import BuildRefused
from mime.cli import main
from mime.saved import _bundle, _card, _terms, card_text, notice, pack, read_folder
from mime.saved._folder import named_files
from mime.saved._card import CREDIT_HOME, LICENCE_TEXTS, NO_EVIDENCE
from mime.terms import Evidence, Permission, Scope, Terms

ORIGIN = "https://example.org/ar4_ros_driver/6a3ebb1/meshes/ar4_visual.usda"


def ar4_like(root: Path) -> Path:
    """The 7n case: a presentation mesh under a licence that forbids redistribution, beside an MIT description of
    the arm's kinematics (as a URDF would be), in a folder licensed CC-BY-4.0."""
    visual = write_mesh(root / "sources", "ar4_visual", 6000)
    kinematics = write_mesh(root / "sources", "ar4_kinematics", 1200)
    layers = [(visual, Terms(NO_REDISTRIBUTION, "Robot Maker Ltd (the arm's maker)", "presentation", (ORIGIN,))),
              (kinematics, Terms("MIT", "Copyright (c) 2021 A. Driver", "results"))]
    with same_reference_run():
        layered("ar4_like", layers, licence="CC-BY-4.0").build().save(root / "ar4_like")
    return root / "ar4_like"


@pytest.fixture(scope="module")
def ar4_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ar4_like(tmp_path_factory.mktemp("ar4"))


# --- the 7n case -------------------------------------------------------------------------------------------------


def test_pack_carries_the_ar4_like_mesh_as_a_reference(ar4_folder: Path, tmp_path: Path):
    bundle = pack(ar4_folder, out=tmp_path / "ar4.mimebundle")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        held = archive.read("NOTICE").decode()
    assert "experiment/scene/layers/ar4_visual.usda" not in names
    assert "experiment/scene/layers/ar4_kinematics.usda" in names
    assert (ar4_folder / "scene/layers/ar4_visual.usda").read_bytes() not in bundle.read_bytes()
    assert held == notice(ar4_folder), "the bundle's NOTICE is not the folder's"


def test_the_notice_says_the_mesh_is_referenced_with_its_licence_role_and_origin(ar4_folder: Path):
    text = notice(ar4_folder)
    block = text[text.index("scene/layers/ar4_visual.usda\n"):].split("\nscene/scene.usda")[0]
    assert "a third party's; licence LicenseRef-Robot-Maker-NonCommercial" in block
    assert "verdict: not permitted, referenced, by its licence; referenced: not carried in a bundle; the asset " \
           "ar4_visual.usda" in block
    assert "role: presentation (results do not depend on it)" in block and f"origin: {ORIGIN}" in block
    assert f"{NO_REDISTRIBUTION.id}: {NO_REDISTRIBUTION.title}, {NO_REDISTRIBUTION.url}; its redistribution, as " \
           f"declared: not permitted" in text
    assert ("Reproducible exactly, given 1 file the bundle does not carry, which `mime assets fetch <folder>` fetches "
            "and places: scene/layers/ar4_visual.usda (may not be redistributed;") in text
    assert "Naming a rights holder here implies no endorsement by them." in text


def test_the_notice_carries_the_mit_text_under_the_files_credit(ar4_folder: Path):
    text = notice(ar4_folder)
    assert "MIT, as its text asks, for scene/layers/ar4_kinematics.usda:\n\nCopyright (c) 2021 A. Driver\n\n" \
           + LICENCE_TEXTS["MIT"] in text


def test_the_card_says_the_mesh_is_referenced_and_where_credit_will_live(ar4_folder: Path):
    text = card_text(ar4_folder)
    for heading in ("## What it is", "## Evidence: what it bounds and what it estimates", "## What reproduces",
                    "## Validity lines", "## Licences and credit", "## Availability", "## How to reproduce",
                    "## How to cite"):
        assert heading in text, heading
    assert "| `scene/layers/ar4_visual.usda` | a third party's | LicenseRef-Robot-Maker-NonCommercial | not " \
           "permitted | referenced | Robot Maker Ltd (the arm's maker) |" in text
    assert f"- **Creators:** {CREDIT_HOME}" in text and NO_EVIDENCE in text and "validated" not in text
    assert "- **Licence:** CC-BY-4.0" in text
    assert "Reproducible exactly, given 1 file the bundle does not carry" in text


def test_the_card_flags_non_commercial_terms(tmp_path: Path):
    mesh = write_mesh(tmp_path / "sources", "Arm", 900)
    with same_reference_run():
        layered("nc", [(mesh, Terms("CC-BY-NC-4.0", "A. Maker", "results"))]).build().save(tmp_path / "nc")
    text = card_text(tmp_path / "nc")
    assert "**Non-commercial terms** hold for `scene/layers/Arm.usda`." in text
    assert "permitted, **non-commercial** | carried" in text


@pytest.mark.parametrize("licence, flagged", [
    ("CC-BY-SA-4.0", ["Share-alike terms"]), ("CC-BY-ND-3.0", ["No-derivatives terms"]),
    ("CC-BY-NC-SA-2.0", ["Non-commercial terms", "Share-alike terms"]),
    ("CC-BY-NC-ND-4.0", ["Non-commercial terms", "No-derivatives terms"])])
def test_the_card_flags_every_condition_of_every_licence_it_names(tmp_path: Path, licence: str, flagged: list[str]):
    mesh = write_mesh(tmp_path / "sources", "Arm", 900)
    with same_reference_run():
        layered("conditions", [(mesh, Terms(licence, "A. Maker", "results"))]).build().save(tmp_path / "f")
    text = card_text(tmp_path / "f")
    assert [words for words in _card.CONDITION_WORDS.values() if f"**{words}** hold for" in text] == flagged
    assert all(f"**{words}** hold for `scene/layers/Arm.usda`." in text for words in flagged)


def test_the_card_flags_the_conditions_a_permission_leaves(tmp_path: Path):
    mesh = write_mesh(tmp_path / "sources", "Arm", 900)
    permission = Permission("Ada Maker", "A. Lab", (hashlib.sha256(mesh.read_bytes()).hexdigest(),),
                            Scope(True, False, True), "2026-10-20",
                            Evidence("public_statement", "https://example.org/s"))
    with same_reference_run():
        layered("granted", [(mesh, Terms(NO_REDISTRIBUTION, "A. Maker", "presentation", (ORIGIN,)))],
                permissions=[permission]).build().save(tmp_path / "f")
    text = card_text(tmp_path / "f")
    assert "**No-derivatives terms** hold for `scene/layers/Arm.usda`." in text
    assert "permitted, **no-derivatives** | carried" in text and "Non-commercial terms" not in text


def test_the_card_says_what_mime_verify_would_check_and_that_it_checked_nothing(ar4_folder: Path):
    text = card_text(ar4_folder)
    section = text[text.index("## What reproduces"):text.index("## Validity lines")]
    assert "`mime verify <folder>` reruns the first" in section and "it ran no check" in section
    assert "would check" in NO_EVIDENCE and "reproduce" not in NO_EVIDENCE


# --- what the NOTICE claims --------------------------------------------------------------------------------------


def test_the_authors_own_files_without_a_licence_are_unknown_never_permitted(tmp_path: Path):
    with same_reference_run():
        layered("unlicensed", [(write_mesh(tmp_path / "sources", "Arm", 900), None)]).build().save(tmp_path / "f")
    text = notice(tmp_path / "f")
    verdicts = [line for line in text.splitlines() if line.startswith("  verdict: ")]
    assert verdicts and all(line == f"  verdict: unknown, carried, by {_terms.NO_LICENCE_BY}; carried in the bundle"
                            for line in verdicts), verdicts
    assert "permitted" not in "\n".join(verdicts) and "a reuser is granted no licence" in text


@pytest.mark.parametrize("licence, verdict", [
    ("MPL-2.0", "unknown, carried, by the author's own"),
    (NO_REDISTRIBUTION, "not permitted, carried, by the author's own"),
    ("CC-BY-4.0", "permitted, open, by the author's own")])
def test_the_authors_own_files_have_the_status_of_the_experiments_licence(tmp_path: Path, licence: object,
                                                                          verdict: str):
    with same_reference_run():
        layered("licensed", [], licence=licence).build().save(tmp_path / "f")  # type: ignore[arg-type]
    verdicts = {line for line in notice(tmp_path / "f").splitlines() if line.startswith("  verdict: ")}
    assert verdicts == {f"  verdict: {verdict}; carried in the bundle"}, verdicts


def test_a_large_file_that_travels_as_a_reference_is_named_in_the_availability(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch):
    limit = 200_000  # above every file MIME writes for the toy, so that the layer alone is over it
    monkeypatch.setattr(_terms, "INLINE_LIMIT_BYTES", limit)
    monkeypatch.setattr(_bundle, "INLINE_LIMIT_BYTES", limit)
    mesh = write_mesh(tmp_path / "sources", "Big", 2 * limit)
    with same_reference_run():
        layered("big", [(mesh, Terms("CC0-1.0", "A. Maker", "results", (ORIGIN,)))], licence="CC-BY-4.0").build() \
            .save(tmp_path / "f")
    for text in (notice(tmp_path / "f"), card_text(tmp_path / "f")):
        assert "from this bundle alone" not in text
        assert ("given 1 file the bundle does not carry, which `mime assets fetch <folder>` fetches and places: "
                f"scene/layers/Big.usda (over the size a bundle carries; licence CC0-1.0; results (results depend on "
                f"it); available from {ORIGIN})") in text


def test_only_a_bundle_that_carries_every_file_reproduces_from_itself_alone(tmp_path: Path):
    with same_reference_run():
        layered("whole", [(write_mesh(tmp_path / "sources", "Arm", 900), Terms("MIT", "A. Maker", "results"))],
                licence="CC-BY-4.0").build().save(tmp_path / "f")
    assert "Reproducible exactly from this bundle alone." in notice(tmp_path / "f")


def test_the_notice_and_the_card_refuse_a_licence_the_installed_packaging_does_not_list(
        ar4_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import packaging.licenses

    folder = Path(shutil.copytree(ar4_folder, tmp_path / "ar4_like"))
    monkeypatch.delitem(packaging.licenses.LICENSES, "mit")  # as a packaging before MIT was listed would read it
    for action in (lambda: notice(folder), lambda: card_text(folder)):
        with pytest.raises(BuildRefused) as refused:
            action()
        (refusal,) = refused.value.refusals
        assert refusal.code == "terms.unlisted-licence" and "pip install -U packaging" in refusal.action
        assert refusal.message.startswith("scene/layers/ar4_kinematics.usda: 'MIT' names MIT")
    assert main(["card", str(folder)]) == 2 and "terms.unlisted-licence" in capsys.readouterr().err


# --- deterministic -----------------------------------------------------------------------------------------------


def test_the_notice_and_the_card_are_the_same_bytes_each_time_and_in_a_copy(ar4_folder: Path, tmp_path: Path):
    copy = Path(shutil.copytree(ar4_folder, tmp_path / "elsewhere" / "ar4_like"))
    assert notice(ar4_folder) == notice(ar4_folder) == notice(copy)
    assert card_text(ar4_folder) == card_text(copy)
    first = pack(ar4_folder, out=tmp_path / "one.mimebundle")
    assert first.read_bytes() == pack(copy, out=tmp_path / "two.mimebundle").read_bytes()


def test_a_folder_before_licences_says_it_declares_none(tmp_path: Path):
    folder = Path(shutil.copytree(V3_FIXTURE, tmp_path / "layered_toy"))
    assert "none: the folder is of mime.document/3, a form before licences" in notice(folder)
    assert "none declared (mime push will ask for one)" in card_text(folder)


# --- the command ---------------------------------------------------------------------------------------------------


def test_mime_card_prints_the_card_and_with_notice_the_notice(ar4_folder: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["card", str(ar4_folder)]) == 0
    assert capsys.readouterr().out.startswith("# ar4_like\n")
    assert main(["card", str(ar4_folder), "--notice"]) == 0
    assert capsys.readouterr().out.startswith("NOTICE: ar4_like\n")


def test_mime_card_writes_the_file_out_byte_for_byte(ar4_folder: Path, tmp_path: Path):
    assert main(["card", str(ar4_folder), "--out", str(tmp_path / "card.md")]) == 0
    assert (tmp_path / "card.md").read_text(encoding="utf-8") == card_text(ar4_folder)
    assert main(["card", str(ar4_folder), "--notice", "--out", str(tmp_path / "NOTICE")]) == 0
    assert (tmp_path / "NOTICE").read_bytes() == notice(ar4_folder).encode("utf-8")


def test_mime_card_of_a_folder_that_does_not_load_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["card", str(tmp_path / "nothing")]) == 2
    assert "mime card is refused" in capsys.readouterr().err


def test_mime_pack_says_what_travels_as_a_reference_and_the_licence(ar4_folder: Path, tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str]):
    assert main(["pack", str(ar4_folder), "--out", str(tmp_path / "x.mimebundle")]) == 0
    out = capsys.readouterr().out
    assert "referenced, never carried: 1\n  scene/layers/ar4_visual.usda (the asset ar4_visual.usda)" in out
    assert "licence: CC-BY-4.0" in out


def test_mime_pack_of_an_experiment_without_a_licence_says_so_and_packs(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture[str]):
    with same_reference_run():
        layered("unlicensed", []).build().save(tmp_path / "unlicensed")
    assert main(["pack", str(tmp_path / "unlicensed")]) == 0
    out = capsys.readouterr().out
    assert "licence: none declared (packing never refuses that; mime push will ask for one" in out
    assert "predates terms" not in out


def test_mime_pack_of_a_folder_before_terms_with_a_layer_says_it_is_carried_as_yours(tmp_path: Path,
                                                                                    capsys: pytest.CaptureFixture[str]):
    folder = Path(shutil.copytree(V3_FIXTURE, tmp_path / "layered_toy"))
    assert main(["pack", str(folder), "--out", str(tmp_path / "v3.mimebundle")]) == 0
    assert ("this folder predates terms: a third party's file in it is carried as yours; declare its terms and save "
            "again") in capsys.readouterr().out


# --- every file is in the NOTICE -----------------------------------------------------------------------------------

kinds = st.one_of(st.sampled_from(OPEN_IDS), st.sampled_from(UNKNOWN_IDS), st.just(NO_REDISTRIBUTION), st.none())


def every_file_is_in_the_notice(root: Path, licences: list[object]) -> None:
    """The scenario: each file the saved folder names has its own block in the NOTICE, with its sha256."""
    layers = [(write_mesh(root / "sources", f"Mesh{index}", 600 + 37 * index),
               None if licence is None else Terms(licence, "A. Maker", "presentation",  # type: ignore[arg-type]
                                                  ("https://example.org/m.usda",)))
              for index, licence in enumerate(licences)]
    layered("drawn", layers, licence="CC-BY-4.0").build().save(root / "drawn")
    text = notice(root / "drawn")
    lines = text.splitlines()
    form = read_folder(root / "drawn")
    for name in sorted(named_files(form)):
        assert name in lines, f"{name} has no block in the NOTICE"
        digest = hashlib.sha256((root / "drawn" / name).read_bytes()).hexdigest()
        assert f"  sha256 {digest}," in text, f"{name}'s bytes are not in the NOTICE"


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(kinds, min_size=0, max_size=4))
def test_every_file_the_folder_names_is_in_the_notice(tmp_path_factory: pytest.TempPathFactory,
                                                      licences: list[object]):
    root = tmp_path_factory.mktemp("notice")
    with same_reference_run():
        every_file_is_in_the_notice(root, licences)
    shutil.rmtree(root, ignore_errors=True)


RUNS = 20


@contextmanager
def notice_drops_a_file() -> Iterator[None]:
    """The plant: a NOTICE that leaves out the last file of the folder."""
    real = _card.file_lines
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_card, "file_lines", lambda file: [] if file.path == "scene/scene.usda" else real(file))
        yield


def planted_runs(tmp_path: Path, scenario: Callable[[Path, list[object]], None], plant: Callable[[], object]) -> int:
    caught = 0
    cases: list[list[object]] = [[], [NO_REDISTRIBUTION], ["MIT", None], [UNKNOWN_IDS[3], "CC0-1.0", None]]
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        with plant():  # type: ignore[attr-defined]
            try:
                scenario(root, cases[run % len(cases)])
            except AssertionError:
                caught += 1
        shutil.rmtree(root, ignore_errors=True)
    return caught


def test_control_a_notice_that_drops_a_file_is_caught_every_run(tmp_path: Path):
    with same_reference_run():
        assert planted_runs(tmp_path, every_file_is_in_the_notice, notice_drops_a_file) == RUNS


def test_the_control_cases_pass_unplanted(tmp_path: Path):
    with same_reference_run():
        for index, licences in enumerate([[], [NO_REDISTRIBUTION], ["MIT", None], [UNKNOWN_IDS[3], "CC0-1.0", None]]):
            every_file_is_in_the_notice(tmp_path / str(index), licences)
