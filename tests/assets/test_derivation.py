"""A derived asset's derivation record (design #181): its builder and commit, its inputs by role and hash, its
parameters, its checks and its validity domain; ``derive`` writes it and refuses a build with a failed check; staleness
is an input whose role is now played by an asset with another hash.

- **The shipped records.** The corrected unit-tube table's is read from the table file's provenance arrays: its builder,
  commit, parameters and domain are what the fetched asset's arrays say, and its sha256 is the shipped reference's. The
  surrogate's (``dejongh_mlp_v2``) names its weights' sha256 and, as its ``wall:tube`` inputs, the 2026-05-21 tables
  the 0.2 line's generator read.
- **Staleness.** ``dejongh_mlp_v2`` is stale: its tables are not the table the tube's wall kind reads now
  (MIME-ANO-117), and it declares so. A packaged surrogate is stale exactly when it declares it.
- **Controls**, each failing on every run: a record naming another hash for an input is found stale; a check that
  failed produces no asset; a record with a key missing, or another, is refused when read.

The table's provenance test reads the cached asset (``mime assets fetch unit_tube_R0.997``; CI stores it once per job);
it never downloads."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

import pytest
import tube_asset

from mime import assets
from mime.assets import AssetRef, CheckResult, DerivationInput, DerivationRecord, Json, Produced
from mime.build import BuildRefused
from mime.data import asset
from mime.nodes.hydro.bem._tube import UNIT_TUBE, table_provenance
from mime.surrogates import DEJONGH_MLP_V2, SURROGATES, current_inputs, staleness

TABLE = assets.reference(UNIT_TUBE)
#: Another table's reference: the unit tube at the first truncation, superseded.
SUPERSEDED = assets.reference("unit_tube_R0.997_1667_modes")


@dataclass(frozen=True)
class Doubling:
    """A builder for the tests: its asset is the parameter ``text`` twice, and its one check passes when ``passes``."""

    name: str = "doubling"
    version: str = "0.0.1"
    passes: bool = True

    def produce(self, parameters: Mapping[str, Json], inputs: Mapping[str, AssetRef]) -> Produced:
        content = (str(parameters["text"]) * 2).encode()
        return Produced(content, (CheckResult("length", float(len(content)), "even", self.passes),))


def derived(builder: Doubling = Doubling()) -> tuple[bytes, DerivationRecord]:
    return assets.derive("doubled", builder, {"text": "ab"}, {"wall:tube": TABLE}, commit="0000000",
                         domain={"text": "two letters"})


def test_derive_records_the_builder_its_inputs_by_role_and_hash_the_parameters_and_the_checks():
    content, record = derived()
    assert content == b"abab" and record.sha256 == hashlib.sha256(content).hexdigest()
    assert (record.builder, record.builder_version, record.commit) == ("doubling", "0.0.1", "0000000")
    assert record.inputs == (DerivationInput("wall:tube", UNIT_TUBE, TABLE.sha256),)
    assert dict(record.parameters) == {"text": "ab"} and record.failed() == ()
    assert DerivationRecord.from_json(json.loads(record.text())) == record


def test_control_a_failed_check_produces_no_asset():
    with pytest.raises(BuildRefused) as refused:
        derived(Doubling(passes=False))
    assert refused.value.codes() == ["asset.derivation-check"]
    assert "length" in str(refused.value) and "doubled" in str(refused.value)


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
    assert assets.stale_inputs(record, {"training_data": SUPERSEDED}) == [], "a role the record has no input for"


def test_control_a_record_naming_another_hash_is_stale_and_names_both():
    record = derived()[1]
    planted = replace(record, inputs=(DerivationInput("wall:tube", SUPERSEDED.name, SUPERSEDED.sha256),))
    found = assets.stale_inputs(planted, {"wall:tube": TABLE})
    assert len(found) == 1 and SUPERSEDED.name in found[0] and UNIT_TUBE in found[0]


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


def test_the_tables_record_is_what_its_provenance_arrays_say():
    record = assets.derivation_of(UNIT_TUBE)
    found = table_provenance(str(tube_asset.path()))
    assert (record.builder, record.commit) == (found["builder"], found["commit"])
    assert dict(record.parameters) == found["parameters"] and dict(record.domain) == found["domain"]
    assert record.inputs == (), "the table is built from the series alone"
    assert found["anomaly"] in record.notes[0]
    assert found["parameters"]["truncation_rule"]["scale"] == 1.5 and "scale" in record.notes[1]


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
