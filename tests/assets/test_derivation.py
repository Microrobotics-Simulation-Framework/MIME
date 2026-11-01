"""A derived asset's derivation record (design #181): its builder and commit, its inputs by role and hash, its
parameters, its checks and its validity domain; ``derive`` writes it and refuses a build with a failed check; staleness
is an input whose role is now played by an asset with another hash.

- **The shipped records.** The corrected unit-tube table's parameters and domain are what the fetched asset's provenance
  arrays say, its sha256 is the shipped reference's, and its builder and commit are the committed ``full.py`` and
  build spec (with their hashes) that reproduce the table's inputs, the arrays' own commit named in its notes. The
  surrogate's (``dejongh_mlp_v2``) names its weights' sha256 and, as its ``wall:tube`` inputs, the 2026-05-21 tables
  the 0.2 line's generator read.
- **Staleness.** ``dejongh_mlp_v2`` is stale: its tables are not the table the tube's wall kind reads now
  (MIME-ANO-117), and it declares so. A packaged surrogate is stale exactly when it declares it.
- **Controls**, each failing on every run: a record naming another hash for an input is found stale; a check that
  failed produces no asset, and neither does a build that ran no check; a record with a key missing, or another, is
  refused when read; a record naming a role nothing plays (a misspelled ``wall:Tube``) with a wrong hash is refused by
  name, not found fresh, whether asked of ``stale_inputs`` or of ``staleness`` on a surrogate.
- **The build's own state.** ``working_state`` records the commit, each source file's sha256 and a dirty flag; the
  controls: a source changed after its commit, a source never committed, and a tracked file changed elsewhere in the
  tree each make it dirty, and a clean tree does not.

The table's provenance test reads the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job);
it never downloads."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import tube_asset

from mime import assets
from mime.assets import AssetRef, CheckResult, DerivationInput, DerivationRecord, Json, Produced
from mime.build import BuildRefused
from mime.data import asset
from mime.nodes.hydro.mrs._tube import UNIT_TUBE, table_provenance
from mime.surrogates import DEJONGH_MLP_V2, SURROGATES, UNJUDGED_ROLES, current_inputs, staleness

TABLE = assets.reference(UNIT_TUBE)
#: Another table's reference: the unit tube at the first truncation, superseded.
SUPERSEDED = assets.reference("unit_tube_R0.997_1667_modes")


@dataclass(frozen=True)
class Doubling:
    """A builder for the tests: its asset is the parameter ``text`` twice, and its one check passes when ``passes``."""

    name: str = "doubling"
    version: str = "0.0.1"
    passes: bool = True
    checks: bool = True

    def produce(self, parameters: Mapping[str, Json], inputs: Mapping[str, AssetRef]) -> Produced:
        content = (str(parameters["text"]) * 2).encode()
        return Produced(content, (CheckResult("length", float(len(content)), "even", self.passes),) if self.checks
                        else ())


def derived(builder: Doubling = Doubling()) -> tuple[bytes, DerivationRecord]:
    return assets.derive("doubled", builder, {"text": "ab"}, {"wall:tube": TABLE}, commit="0000000",
                         domain={"text": "two letters"}, sources={"doubling.py": "0" * 64}, dirty=False)


def test_derive_records_the_builder_its_inputs_by_role_and_hash_the_parameters_and_the_checks():
    content, record = derived()
    assert content == b"abab" and record.sha256 == hashlib.sha256(content).hexdigest()
    assert (record.builder, record.builder_version, record.commit) == ("doubling", "0.0.1", "0000000")
    assert record.inputs == (DerivationInput("wall:tube", UNIT_TUBE, TABLE.sha256),)
    assert dict(record.parameters) == {"text": "ab"} and record.failed() == ()
    assert dict(record.sources) == {"doubling.py": "0" * 64} and record.dirty is False
    assert DerivationRecord.from_json(json.loads(record.text())) == record


def test_control_a_failed_check_produces_no_asset():
    with pytest.raises(BuildRefused) as refused:
        derived(Doubling(passes=False))
    assert refused.value.codes() == ["asset.derivation-check"]
    assert "length" in str(refused.value) and "doubled" in str(refused.value)


def test_control_a_build_that_ran_no_check_produces_no_asset():
    """An empty case set builds nothing and checks nothing; an asset nothing checked is not produced."""
    with pytest.raises(BuildRefused) as refused:
        derived(Doubling(checks=False))
    assert refused.value.codes() == ["asset.derivation-check"] and "no check at all" in str(refused.value)


@pytest.mark.parametrize("change", ["drop", "add", "reorder"])
def test_control_a_record_with_a_key_missing_added_or_moved_is_refused_when_read(change: str):
    data = json.loads(derived()[1].text())
    if change == "drop":
        del data["domain"]
    elif change == "add":
        data["extra"] = 1
    else:
        data = {"sha256": data.pop("sha256"), **data}
    with pytest.raises(ValueError, match="keys"):
        DerivationRecord.from_json(data)


def test_an_input_is_stale_when_its_role_is_now_played_by_another_hash():
    record = derived()[1]
    assert assets.stale_inputs(record, {"wall:tube": TABLE}) == []
    assert assets.stale_inputs(record, {"wall:tube": TABLE, "training_data": SUPERSEDED}) == [], "a role it lacks"


def test_control_a_record_naming_another_hash_is_stale_and_names_both():
    record = derived()[1]
    planted = replace(record, inputs=(DerivationInput("wall:tube", SUPERSEDED.name, SUPERSEDED.sha256),))
    found = assets.stale_inputs(planted, {"wall:tube": TABLE})
    assert len(found) == 1 and SUPERSEDED.name in found[0] and UNIT_TUBE in found[0]


def test_control_a_misspelled_role_with_a_wrong_hash_is_refused_by_name_not_found_fresh():
    planted = replace(derived()[1], inputs=(DerivationInput("wall:Tube", UNIT_TUBE, "0" * 64),))
    with pytest.raises(BuildRefused) as refused:
        assets.stale_inputs(planted, current_inputs())
    assert refused.value.codes() == ["asset.unknown-role"] and "wall:Tube" in str(refused.value)
    assert assets.stale_inputs(planted, current_inputs(), unjudged=("wall:Tube",)) == [], "declared, it is let by"


def test_control_staleness_on_a_surrogate_refuses_a_misspelled_role_and_judges_the_right_one(monkeypatch):
    """Through ``staleness()`` on a planted surrogate record, as the review planted it: ``wall:Tube`` with a wrong
    hash is refused; ``wall:tube`` with the shipped table's hash is fresh; with another it is stale."""
    provenance = replace(DEJONGH_MLP_V2.provenance, derivation="planted")
    surrogate = replace(DEJONGH_MLP_V2, provenance=provenance)
    base = derived()[1]
    planted: dict[str, DerivationRecord] = {}
    monkeypatch.setattr(assets, "derivation_of", lambda name: planted[name])
    planted["planted"] = replace(base, inputs=(DerivationInput("wall:Tube", UNIT_TUBE, "0" * 64),))
    with pytest.raises(BuildRefused) as refused:
        staleness(surrogate)
    assert refused.value.codes() == ["asset.unknown-role"]
    planted["planted"] = replace(base, inputs=(DerivationInput("wall:tube", UNIT_TUBE, TABLE.sha256),
                                               DerivationInput("training_data", "data", "1" * 64)))
    assert staleness(surrogate) == []
    planted["planted"] = replace(base, inputs=(DerivationInput("wall:tube", UNIT_TUBE, "0" * 64),))
    assert len(staleness(surrogate)) == 1
    assert UNJUDGED_ROLES == ("training_data",)


def _repository(root, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).write_text(text)
    run = lambda *arguments: subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",  # noqa: E731
                                             *arguments], check=True, capture_output=True)
    run("init", "-q")
    run("add", ".")
    run("commit", "-q", "-m", "first")


def test_the_working_state_holds_the_commit_each_sources_hash_and_a_clean_tree_is_not_dirty(tmp_path):
    _repository(tmp_path, {"builder.py": "x = 1\n", "other.txt": "y\n"})
    commit, sources, dirty = assets.working_state(tmp_path, ["builder.py"])
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    assert commit == head and sources == {"builder.py": hashlib.sha256(b"x = 1\n").hexdigest()} and dirty is False
    (tmp_path / "scratch.log").write_text("untracked and not a source\n")
    assert assets.working_state(tmp_path, ["builder.py"])[2] is False


@pytest.mark.parametrize("change", ["source edited", "source untracked", "other tracked file edited"])
def test_control_a_build_from_an_uncommitted_change_is_recorded_dirty(tmp_path, change: str):
    _repository(tmp_path, {"builder.py": "x = 1\n", "other.txt": "y\n"})
    sources = ["builder.py"]
    if change == "source edited":
        (tmp_path / "builder.py").write_text("x = 2\n")
    elif change == "source untracked":
        (tmp_path / "new.py").write_text("z = 1\n")
        sources = ["builder.py", "new.py"]
    else:
        (tmp_path / "other.txt").write_text("changed\n")
    commit, held, dirty = assets.working_state(tmp_path, sources)
    assert dirty is True
    assert held["builder.py"] == hashlib.sha256((tmp_path / "builder.py").read_bytes()).hexdigest()


def test_what_plays_the_tube_wall_role_now_is_the_shipped_table():
    assert current_inputs() == {"wall:tube": TABLE}


# --- the shipped records -----------------------------------------------------------------------------------------


def test_the_shipped_records_are_the_tables_and_the_surrogates_and_name_their_assets_hashes():
    shipped = assets.shipped_derivations()
    assert set(shipped) == {UNIT_TUBE, "dejongh_mlp_v2"}
    assert shipped[UNIT_TUBE].sha256 == TABLE.sha256
    assert shipped["dejongh_mlp_v2"].sha256 == asset("dejongh_mlp_v2").sha256
    for record in shipped.values():
        assert record.failed() == (), f"{record.asset} ships with a failed check"
        assert DerivationRecord.from_json(json.loads(record.text())) == record


ROOT = Path(__file__).resolve().parents[2]
SPEC = "scripts/tube_table/specs/unit_tube_R0.997.json"


def test_the_tables_record_is_what_its_provenance_arrays_say_and_points_at_the_committed_builder():
    """The parameters, domain and anomaly are the arrays'. The builder and commit are not: the arrays name the commit
    the table was built at (in no branch's history) and the record names the committed builder and spec that reproduce
    its inputs (``tests/oracles/test_tube_provenance.py`` holds the spec to the arrays), with the sources' hashes."""
    record = assets.derivation_of(UNIT_TUBE)
    found = table_provenance(str(tube_asset.path()))
    assert dict(record.parameters) == found["parameters"] and dict(record.domain) == found["domain"]
    assert record.inputs == (), "the table is built from the series alone"
    assert found["anomaly"] in record.notes[0]
    assert found["parameters"]["truncation_rule"]["scale"] == 1.5
    assert found["commit"] == "d8bf0a6" and found["commit"] in record.notes[1] and record.commit != found["commit"]
    assert record.builder.startswith("scripts/tube_table/full.py") and SPEC in record.builder
    held = json.loads((ROOT / SPEC).read_text())
    assert held["truncation_factor"] == found["parameters"]["truncation_rule"]["scale"] == 1.5
    assert held["expected"]["modes"] == found["parameters"]["single_truncation"]["modes"]
    assert set(record.sources) >= {SPEC, "scripts/tube_table/full.py", "scripts/tube_table/build.py"}
    assert all((ROOT / path).is_file() for path in record.sources)
    assert record.dirty is None, "not recorded: the shipped file was not built from the recorded commit's tree"


def test_the_specs_hash_in_the_record_is_the_committed_specs_and_a_changed_spec_is_found():
    """The spec is the table's declared build; a change to it must come with a new table and a new record."""
    record = assets.derivation_of(UNIT_TUBE)
    committed = (ROOT / SPEC).read_bytes()
    assert record.sources[SPEC] == hashlib.sha256(committed).hexdigest()
    held = json.loads(committed)
    held["truncation_factor"] = 1.25
    assert record.sources[SPEC] != hashlib.sha256(json.dumps(held, indent=2).encode()).hexdigest()


def test_control_the_tables_record_differs_from_a_planted_provenance():
    record = assets.derivation_of(UNIT_TUBE)
    found = table_provenance(str(tube_asset.path()))
    found["parameters"]["single_truncation"]["modes"] = 1667
    assert dict(record.parameters) != found["parameters"]


def test_the_surrogates_record_names_the_tables_its_training_read_and_its_weights():
    record = assets.derivation_of("dejongh_mlp_v2")
    tables = [each for each in record.inputs if each.role == "wall:tube"]
    shipped = assets.known()
    assert len(tables) == 6 and all(shipped[each.name].sha256 == each.sha256 for each in tables)
    assert all(each.name.startswith("cylinder_wall_r40_") for each in tables)
    assert record.parameters["patches"] == ["lateral"] and record.parameters["reference_point"] == "end"


# --- staleness of the packaged surrogates ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SURROGATES))
def test_a_packaged_surrogate_is_stale_exactly_when_it_declares_it(name: str):
    surrogate = SURROGATES[name]
    found = staleness(surrogate)
    print(f"\n{name}: declared {surrogate.provenance.stale!r}; found {found}")
    assert bool(found) == (surrogate.provenance.stale is not None)


def test_dejongh_mlp_v2_is_stale_through_the_2026_05_21_tables_and_says_why():
    found = staleness(DEJONGH_MLP_V2)
    assert len(found) == 6 and all(UNIT_TUBE in line for line in found)
    assert DEJONGH_MLP_V2.provenance.stale is not None and "MIME-ANO-117" in DEJONGH_MLP_V2.provenance.stale


def test_control_a_surrogate_held_fresh_against_a_record_naming_another_table_is_found_stale():
    """The hold for a surrogate trained through today's table, planted: its record's table hash changed to another's."""
    record = derived()[1]
    assert assets.stale_inputs(record, current_inputs()) == []
    planted = replace(record, inputs=(DerivationInput("wall:tube", UNIT_TUBE, SUPERSEDED.sha256),))
    assert assets.stale_inputs(planted, current_inputs())
