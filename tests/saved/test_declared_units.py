"""Every saved field has declared units (design #266): ``verify`` splits a field into blocks by its units, so a field
that declares none is judged whole, and a new field without them fails here. The units are read as a save reads them
(``BuiltExperiment.declared_units``): the node's symbol table, its typed state (``mime.core.Units`` beside a field's
type), the edges that read it, the stage's binding, the observations' outputs and the flags.

**The builds.** Every case of ``tests/saved/cases.py`` that builds, and the two MRS examples the measurements rest on
(``sphere_settling`` and the tube demo, ``sphere_settling_tube``). Left out for the shard's time, each a build of 11 to
13 s on a laptop: ``mrs_free_space`` and ``mrs_solids``, whose drag nodes hold the field set of ``lab_bead``'s, a
separated quadrature in free space (measured on that laptop: before their units were declared the three lacked the
same five fields), and whose bodies hold ``rigid_explicit``'s and ``lab_bead``'s.

Where a node's symbol table and its typed state both declare a field's units, they agree: the symbol table's are read
first, so a typed state that disagreed would never be read and would mislead.

**The controls.** With the typed state's declarations not read, the kept rigid body's twist (``rigid_explicit``,
``rigid_inertial`` 0.2.0, which has no symbol table) and the MRS drag's diagnostics are reported, so the test can fail;
and a typed state that disagrees with a symbol row is reported."""

import pytest
from cases import CASES
from reference_programs import experiment_of

from mime.build._maddening import node_fields_of
from mime.saved._scales import symbol_units

#: The cases left out for their build's time (the module's docstring), their fields held by lighter cases.
HELD_ELSEWHERE = {"mrs_free_space", "mrs_solids"}
BUILDS = [case.name for case in CASES if not case.refused and case.name not in HELD_ELSEWHERE] + [
    "sphere_settling", "sphere_settling_tube"]


def undeclared(built) -> list[str]:
    """Each field of the build's state, every one a save writes, that has no declared units."""
    units = built.declared_units()
    return sorted(f"{node}.{field}" for node, fields in built.current_state().items() for field in fields
                  if (node, field) not in units)


def disagreeing(built, typed_units) -> list[str]:
    """Each field whose node's symbol table and typed state both declare its units, and differently."""
    found = []
    for name in built.graph.node_names:
        node, fields = built.graph.get_node(name), node_fields_of(built.graph, name)
        rows, typed = symbol_units(node, fields), typed_units(node, fields)
        found += [f"{name}.{field}: {rows[field]!r} against {typed[field]!r}" for field in sorted(set(rows) & set(typed))
                  if rows[field] != typed[field]]
    return found


@pytest.mark.parametrize("case", BUILDS)
def test_every_field_a_save_writes_has_declared_units(case: str):
    from mime.core import typed_units

    built = experiment_of(case).build()
    assert undeclared(built) == []
    assert disagreeing(built, typed_units) == []


def test_control_without_the_typed_states_declarations_the_kept_bodys_twist_and_the_drags_diagnostics_are_reported(
        monkeypatch: pytest.MonkeyPatch):
    from mime.core import _field_kinds

    built = experiment_of("rigid_explicit").build()
    assert undeclared(built) == [], "the arm before the plant"
    _field_kinds._units.cache_clear()
    monkeypatch.setattr(_field_kinds, "_units_of", lambda hint: None)
    try:
        missing = undeclared(built)
    finally:
        _field_kinds._units.cache_clear()
    assert "ball.twist" in missing and {"drag:ball.least_eigenvalue", "drag:ball.stopped_pose"} <= set(missing), missing


def test_control_a_typed_state_that_disagrees_with_a_symbol_row_is_reported():
    from mime.core import typed_units

    built = experiment_of("script_a_body").build()
    assert disagreeing(built, typed_units) == [], "the arm before the plant"

    def planted(node, fields):
        return {**typed_units(node, fields), **({"twist": "m/s; deg/s"} if "twist" in fields else {})}

    found = disagreeing(built, planted)
    assert found and all("'m/s; rad/s' against 'm/s; deg/s'" in line for line in found), found
