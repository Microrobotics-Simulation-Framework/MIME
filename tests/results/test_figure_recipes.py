"""A figure as a recipe: redrawn from the folder's data to the same plotly JSON bytes (a matplotlib figure compared by its
description, never its pixels), a data file whose sha256 changed refusing the redraw by name, a missing install refused
with the line that installs it, and the pages following the 3-D view's and the graph's names and behaviour."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pytest
from conftest import FREQUENCIES, SPEED

from mime.build._refusals import ArgumentRefused, BuildRefused
from mime.results import FigurePage, FigureRecipe, StudyResult
from mime.results._figure import PLOTLY_CDN, PLOTLY_JS_SHA256, PLOTLY_JS_SRI, figure_page, library_tag
from mime.results._study_folder import layout_problems
from mime.units import s

MEAN = "swim_speed:umr.mean_speed"


def test_a_figure_is_redrawn_from_its_recipe_to_the_same_plotly_json_bytes(copy_of_study):
    result = StudyResult.read(copy_of_study)
    for output, name in ((MEAN, "curve"), (SPEED, "overlay"), (MEAN, "histogram")):
        recipe = result.figure(output, name=name, kind="distribution" if name == "histogram" else None, title=f"{name}")
        written = recipe.write_recipe()
        saved = (copy_of_study / "figures" / f"{name}.plotly.json").read_bytes()
        again = FigureRecipe.read(copy_of_study, name)
        assert again.plotly_json().encode() == saved == recipe.plotly_json().encode()
        assert again.description.to_json() == recipe.description.to_json()  # a matplotlib figure: by its description
        assert json.loads(written.read_text())["options"]["title"] == name
    assert layout_problems(copy_of_study) == []


def test_a_recipe_holds_what_it_reads_by_path_and_sha256_and_no_pickle(copy_of_study):
    result = StudyResult.read(copy_of_study)
    result.figure(MEAN, name="curve").write_recipe()
    document = json.loads((copy_of_study / "figures" / "curve.json").read_text())
    assert [r["path"] for r in document["reads"]] == ["study.json", "groups/0/outputs.npz", "groups/0/report.json"]
    assert {r["path"]: r["sha256"] for r in document["reads"]} == {
        path: hashlib.sha256((copy_of_study / path).read_bytes()).hexdigest()
        for path in ("study.json", "groups/0/outputs.npz", "groups/0/report.json")}
    assert document["spec"] == {"kind": "curve", "outputs": [MEAN], "against": None, "members": None}
    assert not [p for p in (copy_of_study / "figures").iterdir() if p.suffix in {".pkl", ".pickle"}]


def test_control_a_data_file_whose_sha256_changed_refuses_the_redraw_by_name(copy_of_study):
    result = StudyResult.read(copy_of_study)
    result.figure(MEAN, name="curve").write_recipe()
    data = copy_of_study / "groups/0/outputs.npz"
    original = data.read_bytes()
    refused = 0
    for position in (0, len(original) // 2, len(original) - 1):
        changed = bytearray(original)
        changed[position] ^= 0x01
        data.write_bytes(bytes(changed))
        with pytest.raises(ArgumentRefused) as error:
            FigureRecipe.read(copy_of_study, "curve")
        refusal = error.value.refusals[0]
        refused += refusal.code == "plot.data-changed" and "groups/0/outputs.npz" in refusal.message
    data.write_bytes(original)
    assert refused == 3
    FigureRecipe.read(copy_of_study, "curve")


def test_a_recipe_is_closed_versioned_and_a_figure_folder_holds_only_recipes(copy_of_study):
    result = StudyResult.read(copy_of_study)
    result.figure(MEAN, name="curve").write_recipe()
    path = copy_of_study / "figures" / "curve.json"
    document = json.loads(path.read_text())
    path.write_text(json.dumps({**document, "extra": 1}))
    with pytest.raises(ArgumentRefused) as closed:
        FigureRecipe.read(copy_of_study, "curve")
    path.write_text(json.dumps({**document, "version": 2}))
    with pytest.raises(ArgumentRefused) as newer:
        FigureRecipe.read(copy_of_study, "curve")
    assert closed.value.refusals[0].code == newer.value.refusals[0].code == "plot.recipe"
    path.write_text(json.dumps(document))
    (copy_of_study / "figures" / "stray.png").write_bytes(b"")
    assert layout_problems(copy_of_study) == ["figures/stray.png"]


def test_a_recipe_of_a_result_in_memory_has_no_folder_to_name_and_is_refused():
    from rigs_plot import run_of

    with pytest.raises(ArgumentRefused) as error:
        run_of([1.0, 2.0, 3.0], 0.001).figure(name="x").write_recipe()
    assert error.value.refusals[0].code == "plot.recipe-folder"


def test_a_missing_install_is_refused_with_the_line_that_installs_it(monkeypatch, study):
    import builtins

    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name.split(".")[0] in {"matplotlib", "plotly"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    for module in [m for m in list(__import__("sys").modules) if m.split(".")[0] in {"matplotlib", "plotly"}]:
        monkeypatch.delitem(__import__("sys").modules, module)
    monkeypatch.setattr(builtins, "__import__", without)
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    for call in (lambda: study.plot(MEAN), lambda: study.plot_interactive(MEAN)):
        with pytest.raises(BuildRefused) as error:
            call()
        refusal = error.value.refusals[0]
        assert refusal.code == "env.missing"
        assert "--extra plot" in refusal.action or "[plot]" in refusal.action


def test_an_interactive_page_follows_the_other_views_names_and_behaviour(study):
    recipe = study.figure(MEAN, name="curve")
    inline, cdn = recipe.page(), recipe.page(cdn=True)
    assert len(inline) > 4_000_000 > len(cdn)
    assert PLOTLY_CDN in cdn and f'integrity="{PLOTLY_JS_SRI}"' in cdn and PLOTLY_CDN not in inline
    assert recipe.page() == recipe.page() and 'id="mime-figure-curve"' in inline
    assert library_tag(True) in cdn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shown = recipe.display()
    assert isinstance(shown, FigurePage) and PLOTLY_CDN in shown.html and "<iframe srcdoc=" in shown._repr_html_()
    assert "FigurePage" in repr(shown) and shown.height == 720
    with pytest.warns(UserWarning, match="notebook"):
        recipe.display(offline=True)


def test_the_pinned_plotly_js_is_the_file_the_installed_plotly_ships():
    from plotly.offline import get_plotlyjs, get_plotlyjs_version

    assert hashlib.sha256(get_plotlyjs().encode("utf-8")).hexdigest() == PLOTLY_JS_SHA256
    assert PLOTLY_CDN.endswith(f"@{get_plotlyjs_version()}/plotly.min.js")
    assert "hash" in figure_page("{}", "t", cdn=True, caption=None, name="n")


def test_a_page_is_written_beside_a_folder_and_never_into_one_and_its_name_is_its_own(study, tmp_path, monkeypatch):
    recipe = study.figure(MEAN, name="curve")
    with pytest.raises(ArgumentRefused) as error:
        recipe.web(study.folder / "curve.html")
    assert error.value.refusals[0].code == "visuals.web-out"
    monkeypatch.chdir(tmp_path)
    written = recipe.web(cdn=True)
    assert written.name == "script_a.curve.figure.html" and written.name != "script_a.html"
    assert written.read_text() == recipe.page(cdn=True)


def test_control_every_file_a_figure_reads_is_hashed_and_a_changed_byte_in_any_is_refused_by_name(copy_of_study):
    result = StudyResult.read(copy_of_study)
    result.figure(MEAN, name="curve").write_recipe()
    reads = [r["path"] for r in json.loads((copy_of_study / "figures" / "curve.json").read_text())["reads"]]
    assert "study.json" in reads and "groups/0/outputs.npz" in reads and "groups/0/report.json" in reads
    refused = trials = 0
    for name in reads:
        data = copy_of_study / name
        original = data.read_bytes()
        for position in (0, len(original) // 2, len(original) - 1):
            changed = bytearray(original)
            changed[position] ^= 0x01
            data.write_bytes(bytes(changed))
            trials += 1
            with pytest.raises(ArgumentRefused) as error:
                FigureRecipe.read(copy_of_study, "curve")
            refusal = error.value.refusals[0]
            refused += refusal.code == "plot.data-changed" and name in refusal.message
        data.write_bytes(original)
    assert refused == trials == 3 * len(reads)
    FigureRecipe.read(copy_of_study, "curve")


def test_a_coordinate_edited_in_the_study_manifest_refuses_the_redraw(copy_of_study):
    result = StudyResult.read(copy_of_study)
    result.figure(MEAN, name="curve").write_recipe()
    path = copy_of_study / "study.json"
    document = json.loads(path.read_text())
    document["points"][0]["values"][0]["value"] = 400.0       # 10 Hz to 400 Hz, which the step-out curve's x reads
    path.write_text(json.dumps(document, indent=1))
    with pytest.raises(ArgumentRefused) as error:
        FigureRecipe.read(copy_of_study, "curve")
    assert error.value.refusals[0].code == "plot.data-changed" and "study.json" in error.value.refusals[0].message


def test_a_runs_recipe_hashes_its_manifest_its_arrays_and_its_report(tmp_path):
    from script_a import reproduction_arm

    from mime.units import ms

    folder = tmp_path / "run"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = reproduction_arm().build()
        built.run_to(folder, s(0.02), record=[SPEED], every=ms(5.0))
    from mime.results import RunResult

    RunResult.read(folder).figure(SPEED, name="speed").write_recipe()
    reads = [r["path"] for r in json.loads((folder / "figures" / "speed.json").read_text())["reads"]]
    assert reads == ["outputs.json", "outputs.npz", "report.json"]
    refused = 0
    for name in reads:
        data = folder / name
        original = data.read_bytes()
        changed = bytearray(original)
        changed[len(original) // 2] ^= 0x01
        data.write_bytes(bytes(changed))
        with pytest.raises(ArgumentRefused) as error:
            FigureRecipe.read(folder, "speed")
        refused += error.value.refusals[0].code == "plot.data-changed" and name in error.value.refusals[0].message
        data.write_bytes(original)
    assert refused == 3
    FigureRecipe.read(folder, "speed")
