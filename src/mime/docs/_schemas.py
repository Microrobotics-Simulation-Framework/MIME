"""The saved form's JSON Schemas, generated from the closed TypedDicts of ``mime.saved._schema``.

The loader is the authority: :func:`mime.saved.shape_problems` reads the same annotations at run time, and a load
refuses what it refuses. A schema is a second reader of that one source, for an editor, and differs from the loader
where JSON Schema cannot say what the loader checks, or where neither checks. Those differences are :data:`GAPS`, held
once: each schema's description lists them, and the test that runs the two readers over one corpus
(``tests/saved/test_json_schema.py``) expects exactly the ones on which they disagree.

The mapping: a closed TypedDict is an object with ``additionalProperties: false`` and its required keys (those not
``NotRequired``) in ``required``; ``ReadOnly`` changes nothing; a ``Literal`` is a ``const`` or an ``enum``; a union is
``anyOf``; ``list[X]`` an array of ``X``; ``dict[str, X]`` an object whose every value is an ``X``; a type alias and a
TypedDict each a ``$defs`` entry, so the recursive ``SettingValue`` refers to itself."""

from __future__ import annotations

import inspect
import json
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Never, TypeAliasType

from typing_extensions import is_typeddict

from mime.saved import DOCUMENT_SCHEMA, LOCK_SCHEMA, ExperimentDocument, Lock
from mime.saved._schema import _unwrap as unwrap  # pyright: ignore[reportPrivateUsage]
from mime.saved._schema import typed_dict_fields

DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True)
class Gap:
    """A known difference between the loader's shape check and a schema: its name, the files it applies to, what the
    loader and the schema each do with an input that shows it (``refuses`` or ``accepts``), and the sentence each
    schema's description prints."""

    name: str
    files: tuple[Literal["document", "lock"], ...]
    loader: Literal["refuses", "accepts"]
    schema: Literal["refuses", "accepts"]
    text: str

    @property
    def disagreement(self) -> bool:
        """Whether the two readers differ on an input that shows it; where they agree, both accept, and only a later
        stage of the load (or nothing) refuses it."""
        return self.loader != self.schema


GAPS: tuple[Gap, ...] = (
    Gap("duplicate-key", ("document",), "refuses", "accepts",
        "A mapping that holds one key twice: the document's YAML loader refuses it, and a schema sees the data a "
        "parser made of the text, in which one of the two values is already gone."),
    Gap("duplicate-key-in-lock", ("lock",), "refuses", "accepts",
        "A mapping of the lock that holds one key twice: MIME's one reader of the lock refuses it (a reader keeping "
        "the first value would read another lock than one keeping the last), and a schema sees the data a parser made "
        "of the text, in which one of the two values is already gone."),
    Gap("integral-float", ("document", "lock"), "refuses", "accepts",
        "An integer written with a zero fractional part (a seed of three written as three point nought): the loader "
        "refuses it as not an integer, and JSON Schema counts it an integer."),
    Gap("non-finite", ("document", "lock"), "accepts", "accepts",
        "A number that is not finite (NaN or an infinity): both readers take it as a number, and strict JSON cannot "
        "hold it. The document's writer refuses one; on a load, only a later check of the value itself refuses it, "
        "where one exists (a run's step must be finite)."),
    Gap("non-string-key", ("document",), "refuses", "accepts",
        "A mapping key that is not a string (YAML reads an unquoted number or date as one): the loader refuses it, "
        "and JSON, where every key is a string, has no word for it."),
    Gap("earlier-form", ("document", "lock"), "accepts", "refuses",
        "A document or a lock of an earlier form of the saved form: the loader reads it under the current form, "
        "upgrading it, and each schema describes the current form alone, so it refuses the earlier form's schema id "
        "(and any key the current form renamed)."),
    Gap("meaning", ("document", "lock"), "accepts", "accepts",
        "A value of the right shape that names nothing or the wrong thing: a unit, a registered entry and its "
        "version, a setting, a record or a file, a lock's file names and package names. The shape is all either "
        "reader checks; the load refuses these later, each by its own code."),
)


class _Converter:
    def __init__(self) -> None:
        self.defs: dict[str, Any] = {}

    def schema(self, annotation: Any) -> Any:
        annotation = unwrap(annotation)
        if isinstance(annotation, TypeAliasType):
            return self._defined(annotation.__name__, lambda: self.schema(annotation.__value__))
        if annotation is None or annotation is type(None):
            return {"type": "null"}
        if annotation is Never:
            return False
        simple = _SIMPLE.get(annotation)
        if simple is not None:
            return {"type": simple}
        origin = typing.get_origin(annotation)
        if origin is Literal:
            allowed = list(typing.get_args(annotation))
            return {"const": allowed[0]} if len(allowed) == 1 else {"enum": allowed}
        if origin in (types.UnionType, typing.Union):
            return {"anyOf": [self.schema(member) for member in typing.get_args(annotation)]}
        if origin is list:
            return {"type": "array", "items": self.schema(typing.get_args(annotation)[0])}
        if origin is dict:
            key, item = typing.get_args(annotation)
            if key is not str:
                raise TypeError(f"the saved form's mappings have string keys, not {key!r}")
            return {"type": "object", "additionalProperties": self.schema(item)}
        if isinstance(annotation, type) and is_typeddict(annotation):
            cls = annotation
            return self._defined(cls.__name__, lambda: self._typed_dict(cls))
        raise TypeError(f"the schema generator does not know the typing form {annotation!r}")

    def _defined(self, name: str, make: Any) -> dict[str, str]:
        if name not in self.defs:
            self.defs[name] = None  # a recursive alias refers to itself while it is made
            self.defs[name] = make()
        return {"$ref": f"#/$defs/{name}"}

    def _typed_dict(self, cls: type) -> dict[str, Any]:
        fields = typed_dict_fields(cls)
        required: frozenset[str] = getattr(cls, "__required_keys__")
        made: dict[str, Any] = {"type": "object"}
        doc = inspect.getdoc(cls)
        if doc and doc != inspect.getdoc(dict):
            made["description"] = " ".join(doc.split())
        made["properties"] = {key: self.schema(annotation) for key, annotation in fields.items()}
        made["required"] = sorted(key for key in fields if key in required)
        made["additionalProperties"] = False
        return made


_SIMPLE: Mapping[Any, str] = {bool: "boolean", int: "integer", float: "number", str: "string"}


def json_schema(root: type, schema_id: str, file: Literal["document", "lock"]) -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of the TypedDict ``root``, identified by its form's id, its description the
    root's own and the known gaps of ``file``."""
    converter = _Converter()
    reference = converter.schema(root)
    gaps = [gap for gap in GAPS if file in gap.files]
    description = (f"{' '.join((inspect.getdoc(root) or '').split())}\n\n"
                   f"Generated from the closed TypedDicts of mime.saved for editors. MIME's loader is the authority: "
                   f"this schema is a second reader of the same annotations and differs from the loader as follows.\n"
                   + "\n".join(f"- {gap.name}: {gap.text}" for gap in gaps))
    return {"$schema": DIALECT, "$id": f"urn:mime:saved-form:{schema_id}", "title": f"MIME saved form, {schema_id}",
            "description": description, **reference, "$defs": dict(sorted(converter.defs.items()))}


def saved_form_schemas() -> dict[str, dict[str, Any]]:
    """The two files' schemas, by the name each is published under."""
    return {"document.schema.json": json_schema(ExperimentDocument, DOCUMENT_SCHEMA, "document"),
            "lock.schema.json": json_schema(Lock, LOCK_SCHEMA, "lock")}


def schema_text(schema: Mapping[str, Any]) -> str:
    return json.dumps(schema, indent=1, ensure_ascii=False) + "\n"


__all__ = ["DIALECT", "GAPS", "Gap", "json_schema", "saved_form_schemas", "schema_text"]
