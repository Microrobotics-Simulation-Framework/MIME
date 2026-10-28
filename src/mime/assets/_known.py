"""The references MIME ships, read from the ``assets/`` folder of ``mime.data``: JSON files of
``{"assets": [record, ...]}``."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from mime.assets._reference import AssetRef
from mime.build import BuildRefused, Refusal, RefusalClass

ASSETS_FOLDER = "assets"


@cache
def known() -> Mapping[str, AssetRef]:
    """Every reference MIME ships, by name."""
    found: dict[str, AssetRef] = {}
    folder = importlib.resources.files("mime.data") / ASSETS_FOLDER
    for file in sorted(folder.iterdir(), key=lambda entry: entry.name):
        if file.name.endswith(".json"):
            for record in json.loads(file.read_text(encoding="utf-8"))["assets"]:
                ref = AssetRef.from_record(record)
                found[ref.name] = ref
    return MappingProxyType(found)


def reference(name: str) -> AssetRef:
    """The shipped reference called ``name``."""
    held = known().get(name)
    if held is None:
        raise BuildRefused([Refusal("asset.unknown", RefusalClass.USER, f"no asset reference is named {name!r}",
                                    f"use one of {sorted(known())}")])
    return held
