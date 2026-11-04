"""What 0.3.0 changed reaches a script that meets it as a coded refusal that says what changed, never a bare Python
error, and the user-facing text says nothing a user cannot act on:

- **``Drag.BEM(...)``** (the rung renamed in 0.3.0) is refused, ``call.renamed``, naming ``Drag.MRS`` and the line to
  write with the settings it was given; ``Drag.MRS(...)`` is the control.
- **An entry made without a required setting** is refused, ``call.missing-setting`` (still a ``TypeError``), naming
  each setting missing and why it has no default: ``Screw`` without ``starts`` (its own reason: earlier development
  versions took 2), ``Sphere`` without ``radius`` (its role's). Every other error of the call is Python's own, as
  before (an unknown setting), and a call with every setting builds the entry.
- **``docs/profiles.md`` no longer teaches ``starts`` as a convention default**, which it is not (the screw declares no
  default for it), and says where ``chosen`` is imported from.
- **Words a user cannot act on are gone**: ``strict``'s refusal and research's listed numerical item say "a later
  release", not a release label.
- **The MRS rung's advice where its resistance turned indefinite never repeats the spacing it ran with**: with the
  body's hull on or through the wall (a free FL-9 tumbling in its tube: -0.0016 mm at 0.829 mm), it gives the margin that
  clears the band at that spacing; nearer than the band, the finer spacing; past it, a smaller spacing than its own.
  The control is the rule it replaced, which repeated the spacing in that case.
- **The traceback file's environment line says what MIME is**: the distribution and its version, and from source the
  version and the tree, where it said "mime not installed"."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mime
from mime.build import ArgumentRefused, ArgumentTypeRefused
from mime.errors import _display
from mime.interactions import Drag
from mime.nodes.hydro.mrs._node import CONFINED_GAP, INTEGRATED_GAP, indefinite_advice
from mime.registry import DefaultKind, declared_roles
from mime.scene import Screw, Sphere
from mime.units import dimensionless, mm, um
from mime.validity import chosen
from mime.validity._profiles import profile_of

ROOT = Path(__file__).resolve().parents[2]
FL9 = {"radius": mm(1.56), "length": mm(7.47), "wavenumber": dimensionless(2.33), "modulation": dimensionless(0.33)}


def test_the_old_alias_of_the_mrs_rung_is_refused_naming_drag_mrs_with_the_settings_it_was_given():
    with pytest.raises(ArgumentRefused) as error:
        Drag.BEM(spacing=um(20.0), blob=0.5)
    (refused,) = error.value.refusals
    assert refused.code == "call.renamed" and refused.message.startswith("Drag.BEM is Drag.MRS since 0.3.0")
    assert refused.action == "write Drag.MRS(spacing=..., blob=...) where Drag.BEM(spacing=..., blob=...) was written"
    assert Drag.MRS(spacing=um(20.0), blob=0.5).rung == "mrs", "the control: the new name builds the rung"


def test_a_screw_without_starts_is_refused_naming_it_and_why_it_has_no_default():
    with pytest.raises(ArgumentTypeRefused) as error:
        Screw(**FL9)
    assert isinstance(error.value, TypeError), "a handler written for Python's error still catches it"
    (refused,) = error.value.refusals
    assert refused.code == "call.missing-setting"
    assert refused.message.startswith("Screw(...) is not given starts, and it has no default: starts, the screw's "
                                      "number of starts")
    assert "took 2 where none was written" in refused.message and "FL-9 has 2" in refused.message
    assert refused.action == "write Screw(..., starts=<an int>), with the value of your own setup"
    assert Screw(**FL9, starts=2).starts == 2, "the control: with starts the screw is made"


def test_an_entry_without_a_required_setting_names_each_by_its_role_and_other_errors_stay_pythons():
    with pytest.raises(ArgumentTypeRefused) as error:
        Sphere()
    (refused,) = error.value.refusals
    assert "radius, a physical setting, a fact about the setup, which MIME does not assume" in refused.message
    assert refused.action == "write Sphere(..., radius=<a quantity of length (m)>), with the value of your own setup"
    with pytest.raises(ArgumentTypeRefused) as several:
        Screw(radius=mm(1.0))
    assert "is not given length, wavenumber, modulation, starts, and they have no default" in \
        several.value.refusals[0].message
    with pytest.raises(TypeError) as unknown:
        Sphere(radius=mm(1.0), bogus=1)
    assert not isinstance(unknown.value, ArgumentTypeRefused) and "bogus" in str(unknown.value)


def test_the_profiles_note_teaches_no_starts_default_and_says_where_chosen_comes_from():
    note = (ROOT / "docs" / "profiles.md").read_text(encoding="utf-8")
    assert declared_roles(Screw)["starts"].default is DefaultKind.NONE, "starts has no default"
    assert not re.search(r"two starts|starts.{0,40}convention|convention.{0,80}starts", note, re.DOTALL)
    assert "from mime.validity import chosen" in note and callable(chosen)


def test_strict_and_researchs_numerical_listing_say_a_later_release_not_a_label():
    _, refused = profile_of("strict")
    assert refused is not None and refused[0] == "profile.reserved"
    assert "a later release" in refused[1] and not re.search(r"\ba\d\b", " ".join(refused))
    source = (Path(mime.__file__).parent / "build" / "_profile.py").read_text(encoding="utf-8")
    listed = re.search(r'what = \("a numerical value with no reason[^)]*\)', source, re.DOTALL)
    assert listed is not None and "a later release" in listed.group(0) and not re.search(r"\ba\d'", listed.group(0))


#: A free FL-9 that tumbled to its tube's wall: the body's least distance from it, and the force spacing (m).
TOUCHING_GAP, TOUCHING_SPACING = -1.579e-6, 0.829e-3


def written_spacings(advice: str) -> list[float]:
    return [float(found) for found in re.findall(r"Drag\.MRS\(spacing=mm\(([0-9.e-]+)\)\)", advice)]


def replaced_rule(gap: float | None, spacing: float, clear: float) -> str:
    """The advice the rung gave before: a spacing of the gap over the clear band, or its own where the gap is none."""
    finer = (gap / clear if gap is not None and gap > 0.0 else spacing) * 1e3
    return f"Drag.MRS(spacing=mm({finer:.3g}))"


@pytest.mark.parametrize("clear", [CONFINED_GAP, INTEGRATED_GAP])
@pytest.mark.parametrize("at_start", [False, True])
def test_the_mrs_advice_never_repeats_the_spacing_it_ran_with(clear, at_start):
    ran_with = float(f"{TOUCHING_SPACING * 1e3:.3g}")
    on_the_wall = indefinite_advice(TOUCHING_GAP, TOUCHING_SPACING, clear, at_start=at_start)
    assert ran_with not in written_spacings(on_the_wall)
    assert "reached the wall" in on_the_wall and f"{clear * TOUCHING_SPACING * 1e3:.3g}" in on_the_wall
    if not at_start:
        assert f"Contact.Projection(margin=mm({clear * TOUCHING_SPACING * 1e3:.3g}))" in on_the_wall
    near = indefinite_advice(0.4 * clear * TOUCHING_SPACING, TOUCHING_SPACING, clear, at_start=at_start)
    assert written_spacings(near) == [pytest.approx(0.4 * TOUCHING_SPACING * 1e3, rel=1e-2)]
    past = indefinite_advice(2.0 * clear * TOUCHING_SPACING, TOUCHING_SPACING, clear, at_start=at_start)
    assert written_spacings(past) and all(spacing < ran_with for spacing in written_spacings(past))
    # the control: the rule it replaced repeats the spacing in that case
    assert written_spacings(replaced_rule(TOUCHING_GAP, TOUCHING_SPACING, clear)) == [ran_with]


def test_the_traceback_files_environment_line_says_what_mime_is(monkeypatch):
    installed = _display._environment()[1]
    assert installed.startswith(f"{_display.DISTRIBUTION} {mime.__version__}") and "mime not installed" not in installed

    import importlib.metadata as metadata

    def absent(name: str):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", absent)
    from_source = _display._mime_line()
    assert from_source in _display._environment()[1]
    assert from_source == (f"mime {mime.__version__} from source at {Path(mime.__file__).resolve().parent}, with no "
                           f"{_display.DISTRIBUTION} distribution installed")
