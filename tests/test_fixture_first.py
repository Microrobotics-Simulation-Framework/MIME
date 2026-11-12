"""The ``fixture_first`` order (``tests/_fixture_first.py``): marked tests first, each group in its own order."""

from types import SimpleNamespace

import _fixture_first as order


def item(nodeid: str, marked: bool = False, fixtures: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(nodeid=nodeid, fixturenames=fixtures,
                           get_closest_marker=lambda name: object() if marked and name == order.MARKER else None)


def test_marked_tests_come_first_and_each_group_keeps_its_order():
    items = [item("a"), item("b", True), item("c"), item("d", True), item("e", True), item("f")]
    order.marked_first(items)  # type: ignore[arg-type]
    assert [i.nodeid for i in items] == ["b", "d", "e", "a", "c", "f"]


def test_control_a_run_with_nothing_marked_keeps_the_collection_as_it_is():
    items = [item(name) for name in "cabd"]
    order.marked_first(items)  # type: ignore[arg-type]
    assert [i.nodeid for i in items] == list("cabd")


def test_a_test_reading_the_fixture_without_the_mark_is_named_and_a_marked_one_or_another_files_is_not():
    items = [item("m.py::t1", True, ("scan",)), item("m.py::t2", False, ("scan", "records")), item("m.py::t3", False, ("tmp_path",)),
             item("other.py::t4", False, ("scan",))]
    assert order.unmarked_readers(items, "m.py", ["scan", "steps"]) == ["m.py::t2"]  # type: ignore[arg-type]
