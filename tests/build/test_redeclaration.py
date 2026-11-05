"""A declaration made a second time (``mime.experiment._redeclared``). The same one again is kept once, with a note, in a
script and in a notebook. A changed one, in a notebook, replaces the earlier one where no other declaration names it,
with a note; where one does, both are kept, with a note, and the build refuses them saying a cell has probably run
twice. In a script a changed one is refused at the build as before, in the script's words.

The notebook is IPython's ``get_ipython`` put among the builtins, as IPython does, so that the one test MIME asks
(``mime._notebook.in_notebook``) answers yes; the script is this process as it is. Each kind of declaration a cell can
make again is a case: the four things, a law, an observation, a declared value, a source, the controller and the
licence; a permission and a scene's reference have tests of their own.

**The control for "the same".** ``same_content`` is held against every single change of one leaf of every declaration
in a pool (each float moved by one unit in the last place, each integer by one, each string, each key renamed, an
exponent of a unit, a quantity's reason and what it counts, a binding): not one may be called the same, while each
declaration made afresh, equal in content and another object, is. The enumeration's resolution is shown by
comparisons planted weaker (a quantity by its own equality, which leaves out its reason, and what it counts where one
of the two does not say; floats to a relative tolerance of 1e-12; a mapping by its values alone): each is caught. The
enumeration is fixed, so each is caught on every run."""

from __future__ import annotations

import builtins
import copy
import dataclasses
import math
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from control_cases import Ticker, spun_magnet
from robot_asset import write_robot
from script_a import ALL_RECORDS, MASS, REASON, WATER_RECORD, fl9, script_a

import mime
import mime._content as content_module
import mime.experiment._experiment as experiment_module
from mime import Apparatus, Body, Experiment, Medium, State, Structure
from mime._content import same_content
from mime._notebook import RAN_TWICE, RUN_AGAIN, in_notebook
from mime.build import BuildRefused, LicenceDeclaration, PermissionDeclaration, Refusal, SourceDeclaration, \
    ValueDeclaration
from mime.build._refusals import ArgumentRefused
from mime.dynamics import RigidInertial
from mime.fields import UniformRotating
from mime.interactions import KEPT, Gravity, Magnetic
from mime.observations import SwimSpeed
from mime.scene import Scene, Tube, X, Z
from mime.scene import asset as file_asset
from mime.scene._scene import Reference
from mime.terms import CustomLicence, Evidence, Permission, Scope
from mime.things import Part
from mime.units import FREQUENCY, A_m2, Hz, Quantity, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, ms, mT, rad_per_s

CentreContact, _ = KEPT["contact", "0.1.0"]
PACKAGE = Path(mime.__file__).parent


@pytest.fixture
def notebook(monkeypatch):
    """IPython's mark, as IPython leaves it among the builtins while it runs a cell."""
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    assert in_notebook()


def where(request, place: str) -> None:
    if place == "notebook":
        request.getfixturevalue("notebook")
    else:
        assert not in_notebook(), "this process is IPython: the script's path cannot be told from the notebook's"


# -- script A's body side, each declaration made by a function, so that making it again makes it afresh -------------

def tube(inner_radius: float = 3.175) -> Structure:
    return Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(inner_radius), wall=mm(1.6), length=mm(100.0)))


def water(material=None) -> Medium:
    return Medium("water", fills="tube.lumen", material=material)


def umr(height: float = -1.0) -> Body:
    return Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((10.0, 0.0, height)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(fl9())})


def field(frequency: float = 10.0) -> Apparatus:
    return Apparatus("field", fields={"magnetic": UniformRotating(rotation_axis="tube.axis",
                                                                  initial_direction=(0.0, 1.0, 0.0))},
                     commands={"frequency": Hz(frequency), "amplitude": mT(1.2)})


VALUES = (("umr.mass", MASS), ("umr.inertia", kg_m2((1.03e-10,) * 3)), ("umr.buoyant_volume", m3(6.0e-8)),
          ("umr.magnetic_moment", A_m2((0.0, 1.68e-3, 0.0))), ("umr.centre_of_buoyancy", m((0.0, 0.0, 0.0))),
          ("water.density", kg_per_m3(1000.0)))


def empty(name: str = "again") -> Experiment:
    return Experiment(name, scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))


def world() -> Experiment:
    """Every thing named by a law or a declared value; no source names a law or the observation."""
    exp = empty()
    exp.add_structure(tube())
    exp.add_medium(water())
    exp.add_body(umr())
    exp.add_apparatus(field())
    for path, value in VALUES:
        exp.declare(path, value, reason=REASON)
    exp.attach(Magnetic.Torque(), body="umr", field="field.magnetic")
    exp.attach(Gravity(), body="umr", medium="water")
    exp.attach(CentreContact.Projection(margin=mm(0.1)), body="umr", surface="tube.inner_surface")
    exp.observe(SwimSpeed("umr", along="tube.axis"))
    return exp


def alone(declare: Callable[[Experiment], object]) -> Callable[[], Experiment]:
    def made() -> Experiment:
        exp = empty("alone")
        declare(exp)
        return exp
    return made


def sourced(path: str, reason: str = "a datasheet") -> Callable[[], Experiment]:
    def made() -> Experiment:
        exp = world()
        exp.source(path, reason=reason)
        return exp
    return made


def licensed() -> Experiment:
    exp = world()
    exp.licence("MIT")
    return exp


def planned(exp: Experiment) -> list[Refusal]:
    try:
        exp.plan()
    except BuildRefused as refused:
        return list(refused.refusals)
    return []


def documented(exp: Experiment) -> list[Refusal]:
    try:
        exp.document()
    except ArgumentRefused as refused:
        return list(refused.refusals)
    return []


@dataclass(frozen=True)
class Case:
    """A declaration a cell can make again. ``again(exp, changed)`` makes it, with one setting changed or as before,
    in an experiment ``held()`` that already holds it, where no other declaration names it; ``named()`` holds it named
    by ``naming`` (``None``: nothing can name it). ``code`` is the build's refusal of the two, read by ``refusals``;
    ``noun`` is how a note names it."""

    name: str
    again: Callable[[Experiment, bool], object]
    held: Callable[[], Experiment]
    code: str
    noun: str
    named: Callable[[], Experiment] | None = None
    naming: str = ""
    refusals: Callable[[Experiment], list[Refusal]] = planned


CASES = (
    Case("structure", lambda exp, changed: exp.add_structure(tube(3.2 if changed else 3.175)),
         alone(lambda exp: exp.add_structure(tube())), "thing.duplicate", "the structure 'tube'", world,
         "the medium 'water'"),
    Case("medium", lambda exp, changed: exp.add_medium(water(WATER_RECORD if changed else None)),
         alone(lambda exp: exp.add_medium(water())), "thing.duplicate", "the medium 'water'", world,
         "the value water.density"),
    Case("body", lambda exp, changed: exp.add_body(umr(-0.5 if changed else -1.0)),
         alone(lambda exp: exp.add_body(umr())), "thing.duplicate", "the body 'umr'", world, "the value umr.mass"),
    Case("apparatus", lambda exp, changed: exp.add_apparatus(field(20.0 if changed else 10.0)),
         alone(lambda exp: exp.add_apparatus(field())), "thing.duplicate", "the apparatus 'field'", world,
         "magnetic.torque:umr"),
    Case("law", lambda exp, changed: exp.attach(CentreContact.Projection(margin=mm(0.2 if changed else 0.1)),
                                                body="umr", surface="tube.inner_surface"),
         world, "id.duplicate", "contact:umr", sourced("contact:umr.margin"), "the source of contact:umr.margin"),
    Case("observation", lambda exp, changed: exp.observe(SwimSpeed("umr", along=(1.0, 0.0, 0.0) if changed
                                                                    else "tube.axis")),
         world, "id.duplicate", "swim_speed:umr", sourced("swim_speed:umr.along"),
         "the source of swim_speed:umr.along"),
    Case("declared value", lambda exp, changed: exp.declare("umr.mass", MASS * 2.0 if changed else MASS,
                                                            reason=REASON),
         world, "value.duplicate", "the value umr.mass"),
    Case("source", lambda exp, changed: exp.source("tube.inner_radius",
                                                   reason="another datasheet" if changed else "a datasheet"),
         sourced("tube.inner_radius"), "source.duplicate", "the source of tube.inner_radius"),
    Case("controller", lambda exp, changed: exp.set_controller(
        Ticker(), reads={"pose": "umr.pose"}, writes={"angle": "spin.motor.angle"}, mode="callback",
        rate=ms(6) if changed else ms(3)), spun_magnet, "controller.more-than-one", "the controller"),
    Case("licence", lambda exp, changed: exp.licence("CC0-1.0" if changed else "MIT"), licensed, "terms.licence",
         "the experiment's licence", refusals=documented),
)
NAMED = tuple(case for case in CASES if case.named is not None)


def ids(case: Case) -> str:
    return case.name


def held_ids(exp: Experiment) -> list[int]:
    return [id(declaration) for declaration in vars(exp)["_declarations"]]


def coded(refusals: list[Refusal], code: str) -> list[Refusal]:
    return [refusal for refusal in refusals if refusal.code == code]


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_each_case_holds_its_declaration_once_and_the_build_does_not_refuse_it(case):
    """The cases' precondition: without a second declaration, none of them is refused as made twice, so a refusal
    found below is the second declaration's."""
    for made in (case.held, *((case.named,) if case.named else ())):
        assert coded(case.refusals(made()), case.code) == []


@pytest.mark.parametrize("place", ["script", "notebook"])
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_the_same_declaration_again_is_kept_once_with_a_note(case, place, request):
    where(request, place)
    exp = case.held()
    before = held_ids(exp)
    with pytest.warns(UserWarning, match=re.escape(f"declares {case.noun} again, the same as before: it is kept "
                                                   f"once")):
        case.again(exp, False)
    assert held_ids(exp) == before
    assert coded(case.refusals(exp), case.code) == []


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_a_changed_declaration_in_a_script_is_added_with_no_note_and_refused_in_the_scripts_words(case):
    where(None, "script")
    exp = case.held()
    count = len(held_ids(exp))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        case.again(exp, True)
    assert len(held_ids(exp)) == count + 1
    refused = coded(case.refusals(exp), case.code)
    assert refused, f"the two {case.name}s were not refused as {case.code}"
    assert all(RAN_TWICE not in refusal.message and refusal.action != RUN_AGAIN for refusal in refused)


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_a_changed_declaration_in_a_notebook_replaces_the_earlier_one_where_nothing_names_it(case, notebook):
    exp = case.held()
    count = len(held_ids(exp))
    with pytest.warns(UserWarning, match=re.escape(f"declares {case.noun} again, not the same as before: in a "
                                                   f"notebook the last one run wins, so it replaces the earlier one")):
        case.again(exp, True)
    assert len(held_ids(exp)) == count
    assert coded(case.refusals(exp), case.code) == []
    with pytest.warns(UserWarning, match="the same as before: it is kept once"):
        case.again(exp, True)  # the changed one is the one held


@pytest.mark.parametrize("case", NAMED, ids=ids)
def test_a_changed_declaration_another_names_is_kept_beside_it_and_refused_saying_a_cell_ran_twice(case, notebook):
    assert case.named is not None
    exp = case.named()
    count = len(held_ids(exp))
    with pytest.warns(UserWarning) as noted:
        case.again(exp, True)
    said = [str(warning.message) for warning in noted]
    assert len(said) == 1 and f"declares {case.noun} again, not the same as before, and " in said[0]
    assert case.naming in said[0] and "so both are kept and the build will refuse them" in said[0]
    assert len(held_ids(exp)) == count + 1
    refused = coded(case.refusals(exp), case.code)
    assert refused and all(refusal.message.endswith(RAN_TWICE) and refusal.action == RUN_AGAIN for refusal in refused)


@pytest.mark.parametrize("place", ["script", "notebook"])
def test_the_same_permission_again_is_kept_once_and_another_is_another_permission(place, request):
    """A permission has no key: an experiment holds one for each grant, so a changed one is not a second of one."""
    where(request, place)

    def permission(grantee: str = "A Lab") -> Permission:
        return Permission("A Maker", grantee, ("0" * 64,), Scope(True, True, True), "2026-11-05",
                          Evidence("written_permission", "an email the lab holds"))

    exp = world()
    exp.permission(permission())
    with pytest.warns(UserWarning, match=re.escape("exp.permission(...) declares a permission again, the same as "
                                                   "before: it is kept once")):
        exp.permission(permission())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exp.permission(permission("Another Lab"))
    assert sum(isinstance(held, PermissionDeclaration) for held in vars(exp)["_declarations"]) == 2


@pytest.mark.parametrize("place", ["script", "notebook"])
def test_the_same_reference_again_is_kept_once_and_another_asset_at_its_prim_is_refused(place, request, tmp_path):
    """The scene cannot tell what binds a referenced asset's prims, so another asset at a prim already referenced is
    kept beside it, and refused at the build: in a notebook saying a cell has probably run twice."""
    where(request, place)
    first = write_robot(tmp_path / "robot.usda")
    other = write_robot(tmp_path / "other.usda", turn_degrees=45.0)
    scene = Scene.new(up=Z, units="m")
    scene.reference(file_asset(first), at="/World/UMR")
    with pytest.warns(UserWarning, match=re.escape("scene.reference(...) references the same asset at /World/UMR "
                                                   "again: it is kept once")):
        scene.reference(file_asset(Path(str(first))), at="/World/UMR")
    assert len(scene.references) == 1
    with warnings.catch_warnings(record=True) as noted:
        warnings.simplefilter("always")
        scene.reference(file_asset(other), at="/World/UMR")
    assert len(scene.references) == 2
    notes = [str(note.message) for note in noted]
    if place == "notebook":
        assert len(notes) == 1 and "both are kept and the build will refuse them" in notes[0]
    else:
        assert notes == []
    refused = coded(planned(Experiment("referenced", scene=scene, gravity=m_per_s2(9.81), dt=ms(0.5))),
                    "scene.reference-path")
    assert refused
    if place == "notebook":
        assert all(refusal.message.endswith(RAN_TWICE) and refusal.action == RUN_AGAIN for refusal in refused)
    else:
        assert all(RAN_TWICE not in refusal.message and refusal.action != RUN_AGAIN for refusal in refused)


def test_the_notebook_test_is_one_helper_that_the_run_display_asks_too(monkeypatch):
    """One test of a notebook, ``in_notebook``: no other module of MIME reads IPython's mark among the builtins, and
    the run's live display takes its default from it."""
    readers = sorted(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")
                     if 'builtins, "get_ipython"' in path.read_text(encoding="utf-8"))
    assert readers == ["_notebook.py"]
    assert experiment_module.in_notebook is in_notebook
    assert not in_notebook() and experiment_module._python_display(None) is None
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    assert in_notebook()


# -- the control for "the same" ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Specimen:
    """One of each kind of value the declarations of today do not hold but a declaration may: a list, a tuple, a
    mapping, a set and a frozenset, arrays of floats, integers and objects, numpy scalars, a complex, a string with
    attributes of its own, and (below) a dataclass holding an attribute beyond its fields."""
    items: list[object]
    pair: tuple[object, ...]
    table: dict[str, object]
    tags: set[str]
    frozen_tags: frozenset[str]
    reals: np.ndarray[Any, Any]
    counts: np.ndarray[Any, Any]
    labelled: np.ndarray[Any, Any]
    single: np.float32
    whole: np.int32
    phase: complex
    moment: np.datetime64
    named: str


class Named(str):
    """A string with attributes of its own."""
    colour: str


def specimens() -> list[object]:
    labelled = np.empty(3, dtype=object)
    labelled[0], labelled[1], labelled[2] = ("tube", 3.175), "robot", 7
    named = Named("umr")
    named.colour = "red"
    plain = Specimen(items=["a", 1, 2.5], pair=("tube", 3, None), table={"radius": 3.175, "name": "tube"},
                     tags={"x", "y", "z"}, frozen_tags=frozenset({"p", "q"}),
                     reals=np.arange(6, dtype=np.float64).reshape(2, 3) + 0.5, counts=np.array([3, 1, 4], dtype=np.int32),
                     labelled=labelled, single=np.float32(1.5), whole=np.int32(7), phase=complex(1.5, -2.5), moment=np.datetime64(5, "s"),
                     named=named)
    noted = copy.deepcopy(plain)
    object.__setattr__(noted, "note", {"made": "by hand", "count": 2})
    return [plain, noted]


def pool() -> list[object]:
    """Declarations of every kind, with the kinds of value a declaration holds: things with parts, records with their
    acceptance and nominal, and magnets (script A's record arm); a law with a group of settings (its drag's
    lubrication) and laws bound by handles and typed paths; an apparatus with a chain, a controller and a field read
    at a part (the controlled magnet); script A's body side, written with strings; a source; a value whose quantity
    carries its reason; a licence of the author's own; a permission; and a scene's reference. Then the specimens, for
    the kinds of value no declaration holds yet (arrays, sets, complex numbers, attributes beyond a dataclass's fields)."""
    held: list[object] = [*vars(script_a(records=ALL_RECORDS))["_declarations"],
                          *vars(spun_magnet(observe=True))["_declarations"], *vars(world())["_declarations"]]
    held += [SourceDeclaration("tube.inner_radius", "a datasheet"),
             ValueDeclaration("umr.mass", kg(8.5e-5, reason="weighed on the lab's balance")),
             LicenceDeclaration(CustomLicence("LicenseRef-Mine", "My licence", "https://example.org/licence")),
             PermissionDeclaration(Permission("A Maker", "A Lab", ("0" * 64,), Scope(True, False, True), "2026-11-05",
                                              Evidence("public_statement", "https://example.org/statement"))),
             Reference(file_asset("robot.usda"), "/World/UMR")]
    return held + specimens()


def _rebuilt_mapping(mapping: Mapping[Any, Any], items: list[tuple[Any, Any]]) -> Mapping[Any, Any]:
    if type(mapping) is dict:
        return dict(items)
    if isinstance(mapping, MappingProxyType):
        return MappingProxyType(dict(items))
    if isinstance(mapping, State):
        return State(**dict(items))
    raise AssertionError(f"the enumeration cannot rebuild a {type(mapping).__name__}: teach it, or the control "
                         f"misses what it holds")


def _rebuilt_sequence(sequence: list[Any] | tuple[Any, ...], items: list[Any]) -> list[Any] | tuple[Any, ...]:
    if type(sequence) is list:
        return items
    if type(sequence) is tuple:
        return tuple(items)
    return type(sequence)(*items)


def _slots(cls: type) -> list[str]:
    names: list[str] = []
    for held in cls.__mro__:
        declared = held.__dict__.get("__slots__", ())
        names += [declared] if isinstance(declared, str) else list(declared)
    return names


def _extras(value: object) -> dict[str, object]:
    """The attributes ``value`` holds beyond its dataclass fields (all of them, for an object that is no dataclass)."""
    fields = {field.name for field in dataclasses.fields(value)} if dataclasses.is_dataclass(value) and not isinstance(
        value, type) else set()
    return {name: item for name, item in vars(value).items() if name not in fields} if hasattr(value, "__dict__") else {}


def changed_once(value: object) -> Iterator[tuple[str, object]]:
    """Every copy of ``value`` with one leaf changed by the least it can be, each with the path to that leaf. A leaf
    of a kind it cannot change (``None``, a class, a function) yields nothing, and a container it cannot rebuild
    fails."""
    if isinstance(value, Enum):
        others = [member for member in type(value) if member is not value]
        if others:
            yield f"={others[0]!r}", others[0]
    elif isinstance(value, bool):
        yield f"={not value!r}", not value
    elif isinstance(value, int):
        yield f"={value + 1}", value + 1
    elif isinstance(value, float):
        yield "+ulp", math.nextafter(value, math.inf)
    elif isinstance(value, complex):
        yield ".real+ulp", complex(math.nextafter(value.real, math.inf), value.imag)
        yield ".imag+ulp", complex(value.real, math.nextafter(value.imag, math.inf))
    elif isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating):
            yield "+ulp", np.nextafter(value, np.inf)
        elif value.dtype.kind in "Mm":
            yield "+1 unit", value + np.timedelta64(1, np.datetime_data(value.dtype)[0])
        else:
            yield "+ulp", value + 1
    elif isinstance(value, str):
        yield "+'x'", value + "x"
        for name, item in _extras(value).items():
            for path, leaf in changed_once(item):
                held = copy.copy(value)
                setattr(held, name, leaf)
                yield f".{name}{path}", held
    elif isinstance(value, PurePath):
        yield "+'x'", type(value)(f"{value}x")
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in enumerate(value.ravel().tolist()):
            for path, leaf in changed_once(item):
                held = value.copy()
                held.reshape(-1)[index] = leaf
                yield f"[{index}]{path}", held
    elif isinstance(value, np.ndarray) and value.size:
        changed = value.copy()
        flat = changed.reshape(-1)
        flat[0] = np.nextafter(flat[0], np.inf) if np.issubdtype(changed.dtype, np.floating) else flat[0] + 1
        yield "[0]+ulp", changed
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            for path, leaf in changed_once(getattr(value, field.name)):
                held = copy.copy(value)
                object.__setattr__(held, field.name, leaf)
                yield f".{field.name}{path}", held
        for name, item in _extras(value).items():
            for path, leaf in changed_once(item):
                held = copy.copy(value)
                object.__setattr__(held, name, leaf)
                yield f".{name}{path}", held
    elif isinstance(value, Mapping):
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            for path, leaf in changed_once(item):
                yield f"[{key!r}]{path}", _rebuilt_mapping(value, [*items[:index], (key, leaf), *items[index + 1:]])
            if isinstance(key, str):
                yield f"[{key!r} renamed]", _rebuilt_mapping(value, [*items[:index], (f"{key}x", item),
                                                                     *items[index + 1:]])
    elif isinstance(value, (list, tuple)):
        items = list(value)
        for index, item in enumerate(items):
            for path, leaf in changed_once(item):
                yield f"[{index}]{path}", _rebuilt_sequence(value, [*items[:index], leaf, *items[index + 1:]])
    elif _slots(type(value)) and not hasattr(value, "__dict__"):
        for name in _slots(type(value)):
            for path, leaf in changed_once(getattr(value, name)):
                held = copy.copy(value)
                object.__setattr__(held, name, leaf)
                yield f".{name}{path}", held


_ADDED = "an item the declaration does not hold"


def _same_size_other_kind(dtype: np.dtype[Any]) -> list[np.dtype[Any]]:
    """The dtypes of the same width that read the same bytes as another kind of number, or in another unit (a numpy
    scalar of a datetime is one type whatever its unit, so only its dtype tells ``5 s`` from ``5 ms``)."""
    width = dtype.itemsize
    if dtype.kind in "Mm":
        return [np.dtype(f"{dtype.kind}8[ms]" if np.datetime_data(dtype)[0] != "ms" else f"{dtype.kind}8[s]")]
    if np.issubdtype(dtype, np.floating):
        return [np.dtype(f"i{width}")]
    if np.issubdtype(dtype, np.signedinteger):
        return [np.dtype(f"u{width}"), *([np.dtype(f"f{width}")] if width in (2, 4, 8) else [])]
    if np.issubdtype(dtype, np.unsignedinteger):
        return [np.dtype(f"i{width}")]
    return []


def changed_in_structure(value: object) -> Iterator[tuple[str, object]]:
    """Every copy of ``value`` with one change to its structure: an item added or dropped, a key added or dropped, a
    set's member added, dropped or put in another's place, an array of the same bytes read as another dtype or
    another shape, an attribute added to (or dropped from) an object that holds attributes beyond its fields, and
    the same inside any of its parts. What it cannot rebuild it passes over (a leaf has no structure), and a
    container of a type it cannot rebuild fails."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for index, item in enumerate(value.ravel().tolist()):
                for path, leaf in changed_in_structure(item):
                    held = value.copy()
                    held.reshape(-1)[index] = leaf
                    yield f"[{index}]{path}", held
        else:
            for dtype in _same_size_other_kind(value.dtype):
                yield f".view({dtype})", np.ascontiguousarray(value).view(dtype)
        if value.ndim and value.shape[0]:
            yield ".reshaped", value.reshape(*value.shape, 1)
            yield ".first axis dropped", value[:-1]
    elif isinstance(value, np.generic):
        for dtype in _same_size_other_kind(value.dtype):
            yield f".view({dtype})", value.view(dtype)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            for path, leaf in changed_in_structure(getattr(value, field.name)):
                held = copy.copy(value)
                object.__setattr__(held, field.name, leaf)
                yield f".{field.name}{path}", held
        for name, item in _extras(value).items():
            for path, leaf in changed_in_structure(item):
                held = copy.copy(value)
                object.__setattr__(held, name, leaf)
                yield f".{name}{path}", held
            held = copy.copy(value)
            del vars(held)[name]
            yield f".-{name}", held
        if hasattr(value, "__dict__"):
            held = copy.copy(value)
            object.__setattr__(held, "added_attribute", _ADDED)
            yield ".+attribute", held
    elif isinstance(value, Mapping):
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            for path, leaf in changed_in_structure(item):
                yield f"[{key!r}]{path}", _rebuilt_mapping(value, [*items[:index], (key, leaf), *items[index + 1:]])
            yield f"[{key!r}] dropped", _rebuilt_mapping(value, [*items[:index], *items[index + 1:]])
        yield "+key", _rebuilt_mapping(value, [*items, (_ADDED, _ADDED)])
    elif isinstance(value, (list, tuple)):
        items = list(value)
        for index, item in enumerate(items):
            for path, leaf in changed_in_structure(item):
                yield f"[{index}]{path}", _rebuilt_sequence(value, [*items[:index], leaf, *items[index + 1:]])
            if type(value) in (list, tuple):  # a named tuple has the length its fields give it
                yield f"[{index}] dropped", _rebuilt_sequence(value, [*items[:index], *items[index + 1:]])
        if type(value) in (list, tuple):
            yield "+item", _rebuilt_sequence(value, [*items, _ADDED])
    elif isinstance(value, (set, frozenset)):
        members = list(value)
        yield "+member", type(value)([*members, _ADDED])
        for index, member in enumerate(members):
            yield f"-{member!r}", type(value)([*members[:index], *members[index + 1:]])
            yield f"{member!r}->other", type(value)([*members[:index], _ADDED, *members[index + 1:]])
    elif isinstance(value, (str, bytes, int, float, complex)) and hasattr(value, "__dict__"):
        held = copy.copy(value)
        setattr(held, "added_attribute", _ADDED)
        yield ".+attribute", held
    elif _slots(type(value)) and not hasattr(value, "__dict__") and not isinstance(value, (PurePath, Enum)):
        for name in _slots(type(value)):
            for path, leaf in changed_in_structure(getattr(value, name)):
                held = copy.copy(value)
                object.__setattr__(held, name, leaf)
                yield f".{name}{path}", held


def every_change(value: object) -> Iterator[tuple[str, object]]:
    """The leaf changes and the changes of structure, of ``value``."""
    yield from changed_once(value)
    yield from changed_in_structure(value)


def called_the_same() -> tuple[int, list[str]]:
    """How many single changes (of a leaf or of the structure) the pool has, and those ``same_content`` calls the same
    as their original, asked both ways round (a comparison that checks one side only misses half the structural
    changes)."""
    count, missed = 0, []
    for declaration in pool():
        for path, changed in every_change(declaration):
            count += 1
            if same_content(declaration, changed) or same_content(changed, declaration):
                missed.append(f"{type(declaration).__name__}{path}")
    return count, missed


def afresh(value: object) -> object:
    """``value`` made again: every dataclass, mapping, list, tuple and slotted object a new one holding its contents
    made again, as a cell run again makes them; a leaf (a number, a string, a class) is itself."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        held = copy.copy(value)
        for field in dataclasses.fields(value):
            object.__setattr__(held, field.name, afresh(getattr(value, field.name)))
        for name, item in _extras(value).items():
            object.__setattr__(held, name, afresh(item))
        return held
    if isinstance(value, Mapping):
        return _rebuilt_mapping(value, [(key, afresh(item)) for key, item in value.items()])
    if isinstance(value, (list, tuple)):
        return _rebuilt_sequence(value, [afresh(item) for item in value])
    if isinstance(value, PurePath):
        return type(value)(str(value))
    if isinstance(value, (set, frozenset)):
        return type(value)(afresh(member) for member in value)
    if isinstance(value, np.ndarray):
        twin = value.copy()
        if value.dtype == object:
            for index, item in enumerate(value.ravel().tolist()):
                twin.reshape(-1)[index] = afresh(item)
        return twin
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, complex):
        return complex(value.real, value.imag)
    if isinstance(value, str) and hasattr(value, "__dict__"):
        held = copy.copy(value)
        held.__dict__.update({name: afresh(item) for name, item in vars(value).items()})
        return held
    if _slots(type(value)) and not hasattr(value, "__dict__") and not isinstance(value, Enum):
        held = copy.copy(value)
        for name in _slots(type(value)):
            object.__setattr__(held, name, afresh(getattr(value, name)))
        return held
    return value


def test_each_declaration_made_afresh_is_the_same_though_it_is_another_object():
    for declaration in pool():
        twin = afresh(declaration)
        assert twin is not declaration and same_content(declaration, twin), declaration


def test_no_single_change_of_a_leaf_or_of_the_structure_of_any_declaration_is_called_the_same():
    count, missed = called_the_same()
    assert missed == []
    paths = [path for declaration in pool() for path, _ in every_change(declaration)]
    reached = {
        "a quantity's value": any(path.endswith(".value+ulp") or ".value[" in path for path in paths),
        "a unit's exponent": any(".dimension.exponents[" in path for path in paths),
        "what a rate counts": any(".angle+'x'" in path for path in paths),
        "a quantity's reason": any(path.endswith(".value.reason+'x'") for path in paths),
        "a law's binding": any(path.startswith(".bindings[") for path in paths),
        "a handle's name": any(".name+'x'" in path and "bindings" in path for path in paths),
        "a typed path": any(path.startswith(".bindings[") and ".parts[" in path for path in paths),
        "a part's material record": any(".material." in path for path in paths),
        "a key renamed": any("renamed]" in path for path in paths),
        "a group of settings": any(".lubrication." in path for path in paths),
        "an item dropped": any(path.endswith(".items[0] dropped") for path in paths),
        "an item added": any(path.endswith(".items+item") for path in paths),
        "a key dropped": any(path.endswith(".table['radius'] dropped") for path in paths),
        "a key added": any(path.endswith(".table+key") for path in paths),
        "a set's member added": any(path.endswith(".tags+member") for path in paths),
        "a set's member dropped": any(".tags-" in path for path in paths),
        "a set's member put in another's place": any(".tags" in path and "->other" in path for path in paths),
        "a frozenset's contents": any(".frozen_tags" in path and "->other" in path for path in paths),
        "an array's dtype": any(path.startswith(".reals.view(") and path.endswith("int64)") for path in paths),
        "an integer array's dtype": any(path.startswith(".counts.view(") for path in paths),
        "an array's shape": any(path.endswith(".reals.reshaped") for path in paths),
        "an array's length": any(path.endswith(".counts.first axis dropped") for path in paths),
        "an object array's contents": any(path.startswith(".labelled[") for path in paths),
        "a numpy scalar's dtype": any(path.startswith(".single.view(") for path in paths),
        "a datetime's unit (one numpy type, two dtypes)": any(path.startswith(".moment.view(") for path in paths),
        "a numpy scalar's value": any(path == ".whole+ulp" for path in paths) and any(path == ".single+ulp" for path in paths),
        "a complex's imaginary part": any(path.endswith(".phase.imag+ulp") for path in paths),
        "an attribute added": any(path.endswith(".+attribute") for path in paths),
        "an attribute dropped": any(path.endswith(".-note") for path in paths),
        "an attribute's value": any(".note[" in path for path in paths),
        "a string's own attribute": any(path.endswith(".named.colour+'x'") for path in paths),
    }
    assert [what for what, found in reached.items() if not found] == [], "the enumeration did not reach them"
    assert count > 500, count


def quantity_by_its_own_equality(real):
    def weaker(a, b, open_pairs):
        if isinstance(a, Quantity) and isinstance(b, Quantity):
            return a == b
        return real(a, b, open_pairs)
    return weaker


def floats_to_a_relative_tolerance(real):
    def weaker(a, b, open_pairs):
        if type(a) is float and type(b) is float:
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
        return real(a, b, open_pairs)
    return weaker


def mappings_by_their_values_alone(real):
    def weaker(a, b, open_pairs):
        if isinstance(a, Mapping) and isinstance(b, Mapping) and type(a) is type(b):
            return len(a) == len(b) and all(real(x, y, open_pairs) for x, y in zip(a.values(), b.values()))
        return real(a, b, open_pairs)
    return weaker


def sequences_by_their_common_prefix(real):
    def weaker(a, b, open_pairs):
        if type(a) in (list, tuple) and type(b) is type(a):
            return all(real(x, y, open_pairs) for x, y in zip(a, b))
        return real(a, b, open_pairs)
    return weaker


def sets_by_their_size(real):
    def weaker(a, b, open_pairs):
        if isinstance(a, (set, frozenset)) and type(b) is type(a):
            return len(a) == len(b)
        return real(a, b, open_pairs)
    return weaker


def numpy_scalars_by_their_bytes(real):
    def weaker(a, b, open_pairs):
        if isinstance(a, np.generic) and type(b) is type(a):
            return a.tobytes() == b.tobytes()
        return real(a, b, open_pairs)
    return weaker


def arrays_by_their_bytes(real):
    def weaker(a, b, open_pairs):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype != object:
            return a.tobytes() == b.tobytes()
        return real(a, b, open_pairs)
    return weaker


@pytest.mark.parametrize("plant", [quantity_by_its_own_equality, floats_to_a_relative_tolerance,
                                   mappings_by_their_values_alone, sequences_by_their_common_prefix, sets_by_their_size,
                                   arrays_by_their_bytes, numpy_scalars_by_their_bytes], ids=lambda plant: plant.__name__)
def test_control_a_weaker_comparison_planted_is_caught_by_the_enumeration(plant, monkeypatch):
    monkeypatch.setattr(content_module, "_same", plant(content_module._same))
    _, missed = called_the_same()
    assert missed, f"{plant.__name__} passed the enumeration, which cannot then tell it from the real comparison"


def test_a_quantity_is_the_same_only_with_its_reason_and_what_it_counts_which_its_equality_can_leave_out():
    assert mm(1.0, reason="a datasheet") == mm(1.0) and not same_content(mm(1.0, reason="a datasheet"), mm(1.0))
    assert Quantity(10.0, FREQUENCY) == Hz(10.0) and not same_content(Quantity(10.0, FREQUENCY), Hz(10.0))
    assert not same_content(Hz(10.0), rad_per_s(10.0))
    assert not same_content(0.0, -0.0) and not same_content(1, 1.0) and not same_content(True, 1)
    assert same_content(mm(1.0), mm(1.0)) and same_content(float("nan"), float("nan"))


def test_two_functions_with_nothing_inside_are_not_the_same():
    """Code is the same only as itself: two functions hold the same nothing a comparison can read."""
    assert not same_content(lambda: 1, lambda: 1)
    assert not same_content(object(), object())


def test_a_string_with_attributes_of_its_own_is_the_same_only_with_those_attributes_the_same():
    left, right = Named("umr"), Named("umr")
    left.colour = right.colour = "red"
    assert same_content(left, right)
    right.colour = "blue"
    assert not same_content(left, right)
    assert not same_content(left, Named("umr"))


def test_a_value_that_holds_itself_is_the_same_as_another_only_where_the_other_holds_itself_at_the_same_place():
    def loop() -> dict[str, object]:
        held: dict[str, object] = {}
        held["inside"] = held
        return held

    inner = loop()
    unrolled: dict[str, object] = {"inside": inner}
    assert same_content(loop(), loop())
    assert not same_content(loop(), unrolled) and not same_content(unrolled, loop())
    pair: list[object] = []
    pair += [pair, loop()]
    assert same_content(pair, copy.deepcopy(pair))
    assert not same_content(pair, [loop(), pair])


def test_a_value_nested_past_the_recursion_limit_is_the_same_only_as_itself_and_raises_nothing():
    def nested(depth: int) -> list[object]:
        held: list[object] = []
        for _ in range(depth):
            held = [held]
        return held

    deep = nested(5000)
    assert same_content(deep, deep)
    assert not same_content(deep, nested(5000))
    assert same_content(nested(20), nested(20))
