"""The static schema and the loader's run-time check stay in step. The check is driven by the TypedDicts' own
annotations; these tests show it accepts what the annotations allow and refuses what they do not, for every
TypedDict the document and the lock reach, on instances generated from the same annotations.

The control: each generated instance, once given an unknown key, missing a required key or holding a value of
another type, is refused at the path of the change.
"""

import types
import typing
from typing import Any, Literal, Never, TypeAliasType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typing_extensions import is_typeddict

from mime.saved import ExperimentDocument, Lock, shape_problems
from mime.saved._schema import reachable_typed_dicts, typed_dict_fields

PROPERTY = settings(max_examples=100, deadline=None)
SCHEMAS = [*reachable_typed_dicts(ExperimentDocument), *reachable_typed_dicts(Lock)]
NAMES = st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=6)


def instances(annotation: Any, depth: int = 0) -> st.SearchStrategy[Any]:
    """Values of ``annotation``, built from the annotation as the check reads it."""
    if isinstance(annotation, TypeAliasType):
        return instances(annotation.__value__, depth)
    if annotation is None or annotation is type(None):
        return st.none()
    if annotation is Never:
        return st.nothing()
    if annotation is bool:
        return st.booleans()
    if annotation is int:
        return st.integers()
    if annotation is float:
        return st.floats(allow_nan=False) | st.integers()
    if annotation is str:
        return NAMES
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return st.sampled_from(typing.get_args(annotation))
    if origin in (types.UnionType, typing.Union):
        return st.one_of([instances(m, depth) for m in typing.get_args(annotation)
                          if depth < 3 or typing.get_origin(m) not in (list, dict)])
    if origin is list:
        return st.lists(instances(typing.get_args(annotation)[0], depth + 1), max_size=2)
    if origin is dict:
        item = typing.get_args(annotation)[1]
        return st.just({}) if item is Never else st.dictionaries(NAMES, instances(item, depth + 1), max_size=2)
    if is_typeddict(annotation):
        fields = typed_dict_fields(annotation)
        required = annotation.__required_keys__
        return st.fixed_dictionaries(
            {k: instances(v, depth + 1) for k, v in fields.items() if k in required},
            optional={k: instances(v, depth + 1) for k, v in fields.items() if k not in required})
    raise TypeError(annotation)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_every_typed_dict_is_closed_and_the_check_knows_every_form_in_it(schema):
    assert schema.__closed__ is True
    for annotation in typed_dict_fields(schema).values():
        instances(annotation)
    assert shape_problems({}, schema) == [f"$: missing key {k!r}" for k in sorted(schema.__required_keys__)]


def test_the_lock_is_read_only_throughout():
    for schema in reachable_typed_dicts(Lock):
        assert schema.__readonly_keys__ == frozenset(typed_dict_fields(schema)), schema.__name__


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
@PROPERTY
@given(data=st.data())
def test_what_the_annotations_allow_the_check_accepts_and_a_change_it_refuses(schema, data):
    value = data.draw(instances(schema))
    assert shape_problems(value, schema) == []
    fields = typed_dict_fields(schema)
    assert shape_problems({**value, "unknown_key": 1}, schema) == [
        f"$: unknown key 'unknown_key' ({schema.__name__} holds only {', '.join(sorted(fields))})"]
    for key in sorted(schema.__required_keys__):
        assert shape_problems({k: v for k, v in value.items() if k != key}, schema) == [f"$: missing key {key!r}"]
    for key in sorted(value):
        wrong = object()
        problems = shape_problems({**value, key: wrong}, schema)
        assert problems and all(p.startswith(f"$.{key}") for p in problems), problems
