"""Whether two declarations hold the same content: what two objects built by running one cell twice share, and what
no two different declarations do.

Two values are the same when they are of one type and:
- a float has the same bits (``-0.0`` is not ``0.0``, and ``1`` is not ``1.0`` or ``True``); a string, bytes, an
  integer or ``None`` is equal; an array has one dtype, one shape and the same bytes;
- a dataclass has every field the same, those its equality leaves out included (a quantity's reason, and what it
  counts where one of the two does not say), and any attribute it holds beyond its fields;
- a mapping has the same keys in the same order, each with the same value; a list or tuple the same items in order;
  a set the same items; a path is equal;
- an object of a class that declares its attributes in ``__slots__`` (a handle, a typed path, a prim path) has each
  of them the same.

Anything else (a function, a class, a module, an object of any other kind) is the same only as itself: it may hold
code or state that cannot be read, so two of them are never taken for one. So is a value nested too deep to read (past
Python's recursion limit), and a value that holds itself is the same as another only where the other holds itself at the
same place (a dict that holds itself is not a dict that holds a dict that holds itself).

A scalar of a subclass (a ``str`` or an ``int`` with attributes of its own) is the same only with those attributes the same."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any, cast

import numpy as np

_MISSING = object()


def same_content(a: object, b: object) -> bool:
    try:
        return _same(a, b, {})
    except RecursionError:  # nested too deep to read: the same only as itself
        return a is b


def _slots(cls: type) -> list[str]:
    found: list[str] = []
    for held in cls.__mro__:
        declared = cast(object, vars(held).get("__slots__", ()))
        names = (declared,) if isinstance(declared, str) else cast(tuple[str, ...], declared)
        found += [name for name in names if name not in ("__dict__", "__weakref__")]
    return found


def _same(a: object, b: object, open_pairs: dict[int, int]) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == cast(float, b).hex() and _same_attributes(a, b, open_pairs)
    if a is None or isinstance(a, (str, bytes, int)):
        return bool(a == b) and _same_attributes(a, b, open_pairs)
    if isinstance(a, complex):
        other = cast(complex, b)
        return (a.real.hex() == other.real.hex() and a.imag.hex() == other.imag.hex()
                and _same_attributes(a, b, open_pairs))
    if isinstance(a, np.ndarray):
        return _same_array(cast(np.ndarray[Any, Any], a), cast(np.ndarray[Any, Any], b), open_pairs)
    if isinstance(a, np.generic):
        scalar_a, scalar_b = cast(Any, a), cast(Any, b)
        return bool(scalar_a.dtype == scalar_b.dtype) and scalar_a.tobytes() == scalar_b.tobytes()
    if isinstance(a, PurePath):
        return a == b
    if id(a) in open_pairs:  # a value that holds itself: the same so far, if the other holds itself at this place
        return open_pairs[id(a)] == id(b)
    if id(b) in open_pairs.values():  # the other holds itself where this one does not
        return False
    open_pairs[id(a)] = id(b)
    try:
        return _same_composite(a, b, open_pairs)
    finally:
        del open_pairs[id(a)]


def _same_attributes(a: object, b: object, open_pairs: dict[int, int]) -> bool:
    """The attributes a scalar of a subclass holds of its own (none for a ``str``, an ``int`` or a ``float``)."""
    held_a = vars(a) if hasattr(a, "__dict__") else {}
    held_b = vars(b) if hasattr(b, "__dict__") else {}
    return (not held_a and not held_b) or _same(held_a, held_b, open_pairs)


def _same_array(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], open_pairs: dict[int, int]) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == np.dtype(object):
        return _same(cast(list[object], a.ravel().tolist()), cast(list[object], b.ravel().tolist()), open_pairs)
    return a.tobytes() == b.tobytes()


def _same_composite(a: object, b: object, open_pairs: dict[int, int]) -> bool:
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        names = [field.name for field in dataclasses.fields(a)]
        if not all(_same(getattr(a, name), getattr(b, name), open_pairs) for name in names):
            return False
        extra_a = {k: v for k, v in vars(a).items() if k not in names} if hasattr(a, "__dict__") else {}
        extra_b = {k: v for k, v in vars(b).items() if k not in names} if hasattr(b, "__dict__") else {}
        return _same(extra_a, extra_b, open_pairs)
    if isinstance(a, Mapping):
        items_a = list(cast(Mapping[object, object], a).items())
        items_b = list(cast(Mapping[object, object], b).items())
        return len(items_a) == len(items_b) and all(
            _same(key_a, key_b, open_pairs) and _same(value_a, value_b, open_pairs)
            for (key_a, value_a), (key_b, value_b) in zip(items_a, items_b, strict=True))
    if isinstance(a, (list, tuple)):
        seq_a, seq_b = cast(list[object], list(cast(Any, a))), cast(list[object], list(cast(Any, b)))
        return len(seq_a) == len(seq_b) and all(_same(x, y, open_pairs) for x, y in zip(seq_a, seq_b, strict=True))
    if isinstance(a, (set, frozenset)):
        set_a, set_b = list(cast(Any, a)), list(cast(Any, b))
        return len(set_a) == len(set_b) and all(any(_same(x, y, open_pairs) for y in set_b) for x in set_a) and all(
            any(_same(y, x, open_pairs) for x in set_a) for y in set_b)
    slots = _slots(type(a))
    if slots and not hasattr(a, "__dict__"):
        return all(_same(getattr(a, name, _MISSING), getattr(b, name, _MISSING), open_pairs) for name in slots)
    return False
