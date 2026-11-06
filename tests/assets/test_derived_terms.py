"""Derived files take the strictest terms of their inputs (design #229 item 8): the ``AND`` of their verdicts, so one
referenced input makes the derived file referenced; inputs under different conditional licences, whose compatibility
MIME does not judge, make it referenced unless a permission bound to its own hash says otherwise; a derived asset's
inputs are read from its derivation record by hash, and an input no record names has terms nobody stated. A mesh MIME
bakes from a referenced source records its tier as referenced, which the guard finds by hash.

The control, failing on every run: a derived verdict planted as the freest of its inputs (``OR`` in place of ``AND``)
lets a file baked from a referenced mesh travel."""

import hashlib
import json
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mime import assets, terms
from mime.assets import DerivationInput, DerivationRecord
from mime.terms import Scope, derived_verdict, freest, licence_verdict, permission_verdict

RUNS = 20


def verdict(licence: str) -> tuple[str, terms.Verdict]:
    return licence, licence_verdict(licence)


@pytest.mark.parametrize("inputs,tier,conditions", [
    (["CC0-1.0", "MIT"], "open", ()),
    (["CC0-1.0", "CC-BY-NC-4.0"], "conditional", ("non-commercial",)),
    (["CC-BY-NC-4.0", "CC-BY-NC-4.0"], "conditional", ("non-commercial",)),
    (["CC0-1.0", "NOASSERTION"], "referenced", ()),
    (["CC-BY-NC-4.0", "CC-BY-SA-4.0"], "referenced", ()),
    (["MIT", "LicenseRef-Someone"], "referenced", ()),
])
def test_a_derived_file_takes_the_strictest_terms_of_its_inputs(inputs, tier, conditions):
    held = derived_verdict([verdict(each) for each in inputs])
    assert (held.tier, held.conditions) == (tier, conditions)


def test_inputs_under_different_conditional_licences_are_referenced_until_a_permission_on_the_file_says_otherwise():
    held = derived_verdict([verdict("CC-BY-NC-4.0"), verdict("CC-BY-SA-4.0")])
    assert held.tier == "referenced" and held.by == terms.DIFFERING_CONDITIONS_BY
    granted = freest([held, permission_verdict(Scope(redistribute=True, modify=True, commercial=False))])
    assert (granted.tier, granted.conditions) == ("conditional", ("non-commercial",))


def record(inputs: tuple[DerivationInput, ...]) -> DerivationRecord:
    return DerivationRecord("derived.npz", "ef" * 32, "a builder", "1", "0" * 40, {}, False, inputs, {}, (), {})


def test_a_derived_assets_terms_are_read_from_its_derivation_records_inputs_by_hash():
    table = assets.reference("unit_tube_R0.997")
    closed = assets.AssetRef("closed.stl", "12" * 32, 10, "LicenseRef-Maker", ("https://example.org/c",),
                             licences=(terms.CustomLicence("LicenseRef-Maker", "a maker's", "https://example.org/l",
                                                           "not_permitted"),))
    known = {table.sha256: table, closed.sha256: closed}
    one = (DerivationInput("wall:tube", table.name, table.sha256),)
    assert assets.derived_terms(record(one), known).tier == "open"
    two = (*one, DerivationInput("mesh", closed.name, closed.sha256))
    assert assets.derived_terms(record(two), known).tier == "referenced"
    nameless = (*one, DerivationInput("mesh", "nobody", "34" * 32))
    assert assets.derived_terms(record(nameless), known).tier == "referenced", "an input with no record never travels"


def stl(triangles: int) -> bytes:
    body = b"".join(struct.pack("<12fH", 0.0, 0.0, 1.0, *(0.01 * ((i + k) % 5) for k in range(9)), 0)
                    for i in range(triangles))
    return b"made".ljust(80, b"\0") + struct.pack("<I", triangles) + body


def baked_tier(tmp_path: Path, licence: str) -> str:
    """Bake a made mesh under ``licence`` from a manifest in its own folder; the tier its baked record states."""
    from mime.viz import bake

    data = stl(40)
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as patch:
        cache = tmp_path / "cache"
        patch.setenv("MIME_ASSETS_DIR", str(cache))
        patch.setenv("MIME_OFFLINE", "1")
        patch.setenv("MIME_ASSET_LOCATIONS", str(tmp_path / "none.toml"))
        (tmp_path / "none.toml").write_text("", encoding="utf-8")
        cache.mkdir(parents=True, exist_ok=True)
        (cache / hashlib.sha256(data).hexdigest()).write_bytes(data)
        held: dict[str, object] = {
            "format": "mime.asset_manifest/1", "name": "made", "title": "a made mesh", "credit": "a maker",
            "notice": "terms", "archives": [],
            "assets": [{"name": "made.stl", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                        "licence": licence, "locations": ["https://example.org/made.stl"]}],
            "files": {"meshes/made.stl": {"asset": "made.stl"}}}
        if licence.startswith("LicenseRef-"):
            held["licences"] = {licence: {"title": "a maker's", "url": "https://example.org/l",
                                          "redistribution": "not_permitted"}}
        (baked,) = bake(assets.read_manifest_data(held, "made"), say=lambda line: None, accept_licences={licence})
        written = json.loads(baked.layer.with_suffix(".json").read_text(encoding="utf-8"))
        assert written["tier"] == baked.record["tier"]
        findings = assets.audit(baked.layer).findings
        if baked.record["tier"] == "referenced":
            assert [each.kind for each in findings] == ["derived"], "the guard finds the bake by hash"
        return str(baked.record["tier"])


def test_a_mesh_baked_from_a_referenced_source_is_referenced_and_one_from_an_open_source_is_open(tmp_path):
    assert baked_tier(tmp_path / "closed", "LicenseRef-Maker") == "referenced"
    assert baked_tier(tmp_path / "open", "CC-BY-4.0") == "open"


@contextmanager
def freest_of_inputs() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mime.viz._visuals.derived_verdict", lambda inputs: freest([v for _, v in inputs] + [
            licence_verdict("CC0-1.0")]))
        yield


def test_control_a_derived_verdict_planted_as_the_freest_is_caught_every_run(tmp_path):
    caught = 0
    for run in range(RUNS):
        with freest_of_inputs():
            caught += baked_tier(tmp_path / str(run), "LicenseRef-Maker") != "referenced"
    assert caught == RUNS
