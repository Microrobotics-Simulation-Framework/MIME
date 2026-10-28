"""What a page is generated from, gathered once: the registry export, the hand-written sources, the node classes the
entries build, and the lists a page prints a hatch from. A page generator reads this and nothing else, so a page is a
function of its inputs and two builds of one checkout are the same text."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mime.docs._markdown import items, mapping, strings
from mime.vocabulary import MIME_NAMESPACE

type Record = Mapping[str, Any]


def kind_dir(kind: str) -> str:
    return kind.replace(" ", "_")


def entry_path(record: Record) -> str:
    return f"entries/{kind_dir(str(record['kind']))}/{record['name']}.md"


def node_path(name: str) -> str:
    return f"nodes/{name}.md"


def family_path(name: str) -> str:
    return f"families/{name}.md"


def mechanism_path(kind: str) -> str:
    return f"mechanisms/{kind_dir(kind)}.md"


VOCABULARY_PATH = "vocabulary.md"
CONCEPTS_PATH = "concepts.md"
INDEX_PATH = "index.md"


def class_var(record: Record, name: str, default: object = None) -> Any:
    """The value of the ``ClassVar`` ``name`` of an entry in the export; ``default`` where it declares none."""
    held = mapping(mapping(record.get("class_vars")).get(name))
    return held["value"] if "value" in held else default


def built_rungs(record: Record) -> list[str]:
    """The rungs an entry builds, ``""`` for one without rungs: its rungs less the reserved ones."""
    rungs = strings(class_var(record, "rungs", []))
    reserved = set(mapping(class_var(record, "reserved_rungs", {})))
    return [rung for rung in rungs if rung not in reserved] if rungs else [""]


def builds_of(record: Record) -> dict[str, list[str]]:
    """The node classes (dotted paths) each rung builds, by rung."""
    return {str(rung): strings(nodes) for rung, nodes in sorted(mapping(class_var(record, "builds", {})).items())}


def item_of(record: Record, tail: str = "") -> str:
    """The entry as the lists name it: its kind (with ``_`` for a space), name and version, then ``tail``."""
    return " ".join(part for part in (kind_dir(str(record["kind"])), str(record["name"]), str(record["version"]), tail)
                    if part)


@dataclass(frozen=True)
class Context:
    """The inputs of every page.

    ``records`` are the export's entries, ``families`` the registered family objects by name, ``hand`` the hand-written
    text of each page source (its path under ``docs/``, the front matter removed), ``nodes`` the node classes the
    entries build by dotted path, and ``unchecked`` the rungs the node walk cannot compare with a case. ``cited`` are
    the namespaces whose ``references`` are bibliography keys: a lab's are printed as written, since its bibliography
    is its own."""

    records: tuple[Record, ...]
    families: Mapping[str, Any]
    hand: Mapping[str, str]
    nodes: Mapping[str, type]
    unchecked: frozenset[str] = frozenset()
    cited: frozenset[str] = frozenset({MIME_NAMESPACE})
    node_names: Mapping[str, str] = field(init=False)

    def __post_init__(self) -> None:
        names = {dotted: cls.__name__ for dotted, cls in self.nodes.items()}
        object.__setattr__(self, "node_names", names)

    def builders(self, dotted: str) -> list[tuple[Record, str]]:
        """The entries that build the node class, each with its rung."""
        return [(record, rung) for record in self.records for rung, nodes in builds_of(record).items()
                if dotted in nodes]

    def members(self, family: str) -> list[tuple[Record, str]]:
        """The entries that declare ``family``, each with a rung it builds."""
        return [(record, rung) for record in self.records if class_var(record, "family") == family
                for rung in built_rungs(record)]


def count_of(value: object) -> int:
    return len(items(value))
