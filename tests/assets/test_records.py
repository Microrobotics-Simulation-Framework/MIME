"""One record type (design #229): a record states what it knows and writes only that, so a record of the five keys is
the one every older lock holds, byte for byte; its tier is the verdict of its licence, never a value written by hand;
it keeps an asset's kind and role; and a manifest is a named group of such records, read in its own format or the
earlier visual set's, and refused in any other. The controls: a record that writes a hand-written tier is refused, a
custom licence that says ``not_permitted`` makes the tier ``referenced`` where the same record without it would be
judged ``unknown`` (still referenced, but never ``not_permitted``), and the earlier visual set's own hand-written
``open`` tier is not believed."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from chain_cases import AR4_VISUALS

from mime import assets
from mime.build import BuildRefused
from mime.terms import CustomLicence

PAYLOAD = b"a record's bytes" * 64
SHA = hashlib.sha256(PAYLOAD).hexdigest()
ANNIN = CustomLicence("LicenseRef-Lab-NonCommercial", "a lab's non-commercial licence", "https://example.org/licence",
                      "not_permitted")


def codes(action) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


def test_a_record_of_the_five_keys_writes_exactly_the_five_an_older_lock_holds():
    ref = assets.AssetRef("blob", SHA, len(PAYLOAD), "CC0-1.0", ("https://example.org/blob",))
    assert json.dumps(ref.to_record()) == json.dumps({"name": "blob", "sha256": SHA, "size": len(PAYLOAD),
                                                      "licence": "CC0-1.0", "locations": ["https://example.org/blob"]})


def test_a_full_record_round_trips_and_keeps_its_kind_role_and_identity():
    ref = assets.AssetRef("scan.usda", SHA, len(PAYLOAD), ANNIN.id, ("https://example.org/scan.usda",),
                          credit="a lab", role="results", kind="derived", derivation="ab" * 32, notice="cite it",
                          licences=(ANNIN,), version="2.1", citation="A. Lab, a scan, 2026",
                          adapters=(("labstore", "mime-lab-fixture"),))
    record = ref.to_record()
    assert list(record) == ["name", "sha256", "size", "licence", "locations", "credit", "role", "kind", "derivation",
                            "notice", "licences", "version", "citation", "adapters"]
    assert assets.AssetRef.from_record(record) == ref


def test_the_tier_is_the_licences_verdict_and_a_custom_licence_that_says_no_makes_it_referenced():
    base = dict(name="mesh.stl", sha256=SHA, size=len(PAYLOAD), locations=("https://example.org/m.stl",))
    assert assets.AssetRef(licence="CC-BY-4.0", **base).tier == "open"
    assert assets.AssetRef(licence="CC-BY-NC-4.0", **base).tier == "conditional"
    unknown = assets.AssetRef(licence=ANNIN.id, **base)
    refused = assets.AssetRef(licence=ANNIN.id, licences=(ANNIN,), **base)
    assert (unknown.tier, unknown.verdict.status) == ("referenced", "unknown")
    assert (refused.tier, refused.verdict.status) == ("referenced", "not_permitted")


def test_control_a_record_that_writes_its_own_tier_is_refused():
    record = dict(assets.AssetRef("blob", SHA, len(PAYLOAD), "CC0-1.0", ("https://example.org/b",)).to_record())
    record["tier"] = "open"
    assert codes(lambda: assets.AssetRef.from_record(record)) == ["asset.reference"]


@pytest.mark.parametrize("change", [
    {"kind": "guessed"}, {"role": "decoration"}, {"derivation": "ab" * 32}, {"credit": " "},
    {"adapters": {"https": "x"}}, {"adapters": {"labstore": "not a name!"}}, {"licences": {"LicenseRef-x": {}}},
    {"locations": ["nowhere://example.org/x"]}, {"locations": ["package:../x"]}, {"locations": ["file:///no/end"]},
    {"locations": ["file:///a/../b/"]},
])
def test_a_malformed_statement_or_location_is_refused_whole(change):
    record = {**assets.AssetRef("blob", SHA, len(PAYLOAD), "CC0-1.0", ("https://example.org/b",)).to_record(),
              **change}
    assert codes(lambda: assets.AssetRef.from_record(record)) == ["asset.reference"]


def test_a_location_of_a_scheme_a_package_provides_is_accepted_where_the_record_names_the_package():
    record = {**assets.AssetRef("blob", SHA, len(PAYLOAD), "CC0-1.0", ("https://example.org/b",)).to_record(),
              "locations": ["s3x://bucket/key"], "adapters": {"s3x": "mime-storage"}}
    assert assets.AssetRef.from_record(record).locations == ("s3x://bucket/key",)


def test_the_ar4s_manifest_is_a_group_of_ordinary_records_each_referenced_by_its_licences_verdict():
    held = assets.read_manifest(AR4_VISUALS)
    assert held.name == "ar4_mk5" and len(held.assets) == 18 == len(held.files)
    assert {ref.tier for ref in held.assets} == {"referenced"} and held.verdict.tier == "referenced"
    assert {(ref.role, ref.kind) for ref in held.assets} == {("presentation", "authored")}
    assert all(ref.credit == held.credit for ref in held.assets), "the manifest's credit fills each record's"
    assert "tier" not in AR4_VISUALS.read_text(encoding="utf-8"), "no tier is written by hand"


def test_a_manifest_round_trips_through_its_json():
    held = assets.read_manifest(AR4_VISUALS)
    assert assets.read_manifest_data(held.to_json(), "again").to_json() == held.to_json()


def earlier_visual_set(tier: str) -> dict[str, object]:
    held = assets.read_manifest(AR4_VISUALS)
    ref = held.assets[0]
    return {"format": "mime.visual_set/1", "name": "old", "title": "an earlier set", "upstream": {"commit": "x"},
            "licence": ref.licence, "licence_url": "https://example.org/l", "tier": tier, "role": "presentation",
            "credit": "someone", "notice": "terms", "archives": [], "archive_checked": "never",
            "files": [{"path": "meshes/a.STL", "sha256": ref.sha256, "size": ref.size, "git_blob": "0" * 40,
                       "location": ref.locations[0]}]}


def test_the_earlier_visual_set_form_is_read_and_its_hand_written_tier_is_not_believed():
    for tier in ("no-redistribution", "open"):
        held = assets.read_manifest_data(earlier_visual_set(tier), "an earlier set")
        assert held.assets[0].tier == "referenced", tier
        assert held.file("meshes/a.STL") is not None


@pytest.mark.parametrize("change", [{"format": "mime.asset_manifest/4"}, {"format": None}, {"tier": "open"},
                                    {"assets": []}, {"files": {"x.STL": {"asset": "nobody"}}}])
def test_a_manifest_of_another_format_or_shape_is_refused_never_misread(change):
    data = copy.deepcopy(assets.read_manifest(AR4_VISUALS).to_json())
    data.update(change)
    assert codes(lambda: assets.read_manifest_data(data, "planted")) == ["asset.manifest"]


def test_mime_ships_records_of_its_own_assets_only(tmp_path: Path):
    """MIME's own data holds the wall tables' records and no manifest: a visual set lives beside its description."""
    from mime.assets._sources import shared

    mine = [each for sourced in shared().records.values() for each in sourced if each.source == "mime"]
    assert mine and all(each.ref.name.startswith(("unit_tube_", "wall_")) or "wall" in each.ref.name for each in mine)
    assert not [held for source, held in shared().manifests if source == "mime"]


def test_mimes_own_tube_table_states_its_kind_and_role_and_names_its_derivation_record_by_hash():
    import importlib.resources

    table = assets.reference("unit_tube_R0.997")
    assert (table.kind, table.role, table.tier) == ("derived", "results", "open")
    shipped = (importlib.resources.files("mime.data") / "derivations" / "unit_tube_R0.997.json").read_bytes()
    assert table.derivation == hashlib.sha256(shipped).hexdigest()
