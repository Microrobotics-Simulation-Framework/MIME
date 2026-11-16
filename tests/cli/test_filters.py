"""``mime registry --entry <name>`` and ``mime assets list --namespace <source>`` (G9): a plugin author reads what its own
package registers without the 70-odd entries of everyone else's, and a name that holds nothing is refused with the
nearest names, never an empty answer that reads as "none".

The asset records' namespace is their source: ``mime`` for MIME's own, the name of an installed package's
``mime.assets`` entry point for a lab's (``labx`` for the fixture lab). The controls are the unfiltered commands, whose
answers hold what the filtered ones leave out, so a filter that returned everything, or nothing, fails."""

from __future__ import annotations

import json

import pytest

from mime import cli


def run(capsys: pytest.CaptureFixture[str], *words: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = cli.main(["--plain", *words])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_registry_json_filtered_by_entry_holds_that_name_s_entries_of_every_version_and_nothing_else(capsys):
    code, whole, _ = run(capsys, "registry", "--json")
    everything = json.loads(whole)
    code, out, err = run(capsys, "registry", "--json", "--entry", "drag")
    assert code == 0, err
    held = json.loads(out)
    assert {entry["name"] for entry in held["entries"]} == {"drag"}
    expected = [entry for entry in everything["entries"] if entry["name"] == "drag"]
    assert len(expected) >= 2 and held["entries"] == expected, "every version and kind of the name, as the export has them"
    assert held["namespaces"] == everything["namespaces"] and held["not_loaded"] == everything["not_loaded"]
    assert len(everything["entries"]) > 3 * len(held["entries"]), "the control: the unfiltered export holds the rest"


def test_a_plugin_entry_is_found_by_its_prefixed_name(capsys):
    code, out, err = run(capsys, "registry", "--json", "--entry", "labx.height")
    assert code == 0, err
    (entry,) = json.loads(out)["entries"]
    assert entry["name"] == "labx.height" and entry["kind"] == "observation"


def test_the_plain_listing_is_filtered_the_same_way(capsys):
    code, out, err = run(capsys, "registry", "--entry", "drag")
    assert code == 0, err
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines and all(line.split()[1] == "drag" for line in lines), lines


def test_an_entry_that_is_none_is_refused_with_the_nearest_names_and_prints_no_json(capsys):
    code, out, err = run(capsys, "registry", "--json", "--entry", "dra")
    assert code == cli.REFUSED and out.strip() == "", (code, out)
    assert "registry.unknown-entry" in err and "no registered entry is called 'dra'" in err and "Did you mean 'drag'" in err


def test_the_assets_listing_filtered_by_namespace_holds_that_source_s_records_alone(capsys):
    code, whole, err = run(capsys, "assets", "list")
    assert code == 0, err
    code, out, err = run(capsys, "assets", "list", "--namespace", "labx")
    assert code == 0, err
    held = {line.split()[0] for line in out.splitlines() if line.strip()}
    assert {"labx_emns.yaml", "labx_marker.usda"} <= held, held
    everything = {line.split()[0] for line in whole.splitlines() if line.strip()}
    assert held < everything and any(name.startswith("unit_tube") or name.startswith("cylinder") for name in everything - held)
    code, out, err = run(capsys, "assets", "list", "--namespace", "mime")
    assert code == 0, err
    assert held.isdisjoint({line.split()[0] for line in out.splitlines() if line.strip()}), "mime's own, none of the lab's"


def test_a_namespace_that_holds_no_record_is_refused_with_the_sources_there_are(capsys):
    code, out, err = run(capsys, "assets", "list", "--namespace", "labz")
    assert code == cli.REFUSED and out.strip() == "", (code, out)
    assert "asset.unknown" in err and "no asset source is called 'labz'" in err and "Did you mean 'labx'" in err
