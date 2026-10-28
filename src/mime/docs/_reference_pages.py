"""The pages that are not about one thing: the mechanism pages (what happens between nodes, one per target kind), the
vocabulary reference, and the index with its front door by goal. Each is generated from the vocabulary and the
registry export; the concepts note is hand-written and copied beside them.

Not rendered at this phase: the lag classes and group settings of the fixtures, which a mechanism page will print
once the fixtures are shipped as data."""

from __future__ import annotations

from collections.abc import Iterable

from mime.docs._context import (
    CONCEPTS_PATH,
    INDEX_PATH,
    VOCABULARY_PATH,
    Context,
    Record,
    class_var,
    entry_path,
    family_path,
    kind_dir,
    mechanism_path,
    node_path,
)
from mime.docs._markdown import Page, Writer, code, link, mapping, strings
from mime.vocabulary import MIME_NAMESPACE, Interface, TargetKind, lookup, names, namespace_of, versions

NO_HAND_WRITTEN = "No hand-written part: the cross-node algorithm of this mechanism is not written yet."
KIND_ORDER = ("interaction", "dynamics", "field provider", "geometry", "observation", "drive", "controller")


def interfaces(kind: object | None = None) -> list[Interface]:
    """Every interface the vocabulary defines (of one kind, if given), by name."""
    found = [held for name in names() if (held := lookup(name)) is not None]
    return [held for held in found if kind is None or held.kind is kind]


def _row(w: Writer, held: Interface) -> list[str]:
    """One interface as a table row. The text of a name in another namespace is printed and not recorded for the number
    rule: its numbers are its namespace's own."""
    own = held.namespace == MIME_NAMESPACE

    def prose(label: str, text: str) -> str:
        return w.prose(label, text) if own else text

    shape = "open (reserved)" if held.shape is None else ("scalar" if held.shape == () else
                                                         " x ".join(str(size) for size in held.shape))
    magnitude = "none" if held.magnitude is None else f"{held.magnitude:g}"
    return [code(held.name), held.kind.value + (", reserved" if held.reserved else ""), held.units,
            held.frame or "none", shape,
            prose(f"{held.name} sign", held.sign) if held.sign else "none", magnitude,
            prose(f"{held.name} description", held.description) if held.description else "none"]


INTERFACE_HEADERS = ["Name", "Kind", "Units", "Frame", "Shape", "Sign", "Typical magnitude", "Description"]
#: The kinds of interface that an entry publishes, and so the vocabulary page says which entries do.
PUBLISHED_KINDS = ("field", "state")
NOT_PUBLISHED = "not a published kind"


def _publishing(records: Iterable[Record], name: str) -> list[Record]:
    return [record for record in records if name in mapping(class_var(record, "publishes", {}))]


def _published_by(held: Interface, records: Iterable[Record]) -> str:
    """The entries that publish the interface, linked; a hatch-free statement where none does, or where its kind is
    not one an entry publishes."""
    publishers = _publishing(records, held.name)
    if publishers:
        return ", ".join(link(VOCABULARY_PATH, entry_path(record), code(record["name"])) for record in publishers)
    return "none registered" if held.kind.value in PUBLISHED_KINDS else NOT_PUBLISHED


def vocabulary_page(context: Context) -> Page:
    """Every interface name with its kind, units, frame, shape, sign, magnitude and description, and the entries that
    publish it; a reserved name, whose shape is still open, is marked."""
    w = Writer()
    w.heading(1, "Vocabulary")
    w.paragraph("Every port that dynamics publish or accept and that entries read or deliver into has a name here. "
                "The build checks every read and every target against this list. A reserved name fixes its kind, "
                "units, frame and sign now and leaves its shape open until it is built; using it is refused.")
    versioned = versions()
    for namespace in sorted(versioned):
        held = [item for item in interfaces() if namespace_of(item.name) == namespace]
        if not held:
            continue
        w.heading(2, f"Namespace {namespace}")
        w.paragraph(f"Vocabulary version {code(versioned[namespace])}.")
        w.table([*INTERFACE_HEADERS, "Published by"],
                [[*_row(w, item), _published_by(item, context.records)] for item in held])
    return w.page(VOCABULARY_PATH)


def _accepting(records: Iterable[Record], name: str) -> list[Record]:
    return [record for record in records if record["kind"] == "dynamics"
            and name in mapping(class_var(record, "accepts", {}))]


def _delivering(records: Iterable[Record], name: str) -> list[Record]:
    return [record for record in records
            if any(mapping(held).get("target") == name for held in mapping(class_var(record, "delivers", {})).values())]


def mechanism_page(kind: TargetKind, context: Context) -> Page:
    """The page of one target kind: how deliveries into it combine, its interfaces, the dynamics that accept it and
    the entries that deliver into it."""
    path = mechanism_path(kind.value)
    w = Writer()
    w.heading(1, f"Mechanism {kind.value}")
    if kind.additive:
        how = "Every delivery into a target of this kind is summed."
    elif kind.exclusive:
        how = "A target of this kind takes at most one delivery."
    else:
        how = ("A target of this kind is not delivered by an edge: the receiving dynamics takes it when it is "
               "constructed.")
    w.paragraph(how)
    held = interfaces(kind)
    w.heading(2, "Interfaces")
    if held:
        w.table(INTERFACE_HEADERS, [_row(w, item) for item in held])
    else:
        w.paragraph("none defined")
    w.heading(2, "Dynamics that accept it")
    accepting = sorted({record["name"]: record for item in held for record in _accepting(context.records, item.name)}
                       .values(), key=lambda record: str(record["name"]))
    if kind is TargetKind.REGISTRATION:
        accepting = [record for record in context.records if record["kind"] == "dynamics"
                     and strings(class_var(record, "registration_kinds", []))]
    if accepting:
        w.bullets(link(path, entry_path(record), code(record["name"])) for record in accepting)
    else:
        w.paragraph("none registered")
    w.heading(2, "Entries that deliver into it")
    delivering = sorted({record["name"]: record for item in held
                         for record in _delivering(context.records, item.name)}.values(),
                        key=lambda record: str(record["name"]))
    if delivering:
        w.bullets(link(path, entry_path(record), code(record["name"])) for record in delivering)
    else:
        w.paragraph("none registered" + (": a registration is taken by the receiving dynamics, from the records "
                                          "the interaction asks it to include" if kind is TargetKind.REGISTRATION
                                          else ""))
    w.heading(2, "The algorithm between nodes")
    hand = context.hand.get(path)
    if hand is None:
        w.paragraph(NO_HAND_WRITTEN)
        w.hatch("no hand-written part", f"mechanism {kind_dir(kind.value)}")
    else:
        w.add(hand.strip("\n"), "")
    return w.page(path)


def _prefixed(target: str) -> str:
    return f"pages/{target}"


def _page_link(to_page: str, label: str) -> str:
    return f"[{label}]({_prefixed(to_page)})"


def index_page(context: Context) -> Page:
    """The generated index, a front door by goal. It is written to the output directory itself, so its links name the
    pages directory."""
    w = Writer()
    w.heading(1, "MIME documentation")
    w.paragraph("Generated from the registry and the vocabulary."
                + (f" Start with {_page_link(CONCEPTS_PATH, 'the concepts')} if the page types are new to you."
                   if CONCEPTS_PATH in context.hand else ""))
    w.heading(2, "By goal")
    w.heading(3, "What can act on a body?")
    w.paragraph("A body takes deliveries into its targets; the mechanism page of each target kind says how they "
                "combine and lists the entries that deliver into it.")
    rows: list[list[str]] = []
    for kind in TargetKind:
        delivering = sorted({str(record["name"]) for item in interfaces(kind) for record in
                             _delivering(context.records, item.name)})
        rows.append([_page_link(mechanism_path(kind.value), code(kind.value)),
                     ", ".join(code(name) for name in delivering) or "none registered"])
    w.table(["Target kind", "Entries that deliver into it"], rows)
    w.heading(3, "Which model, when there are several?")
    w.paragraph("A family collects the laws that answer one question, with the contract each must satisfy and, once "
                "two of its members that can replace one another are built, a comparison. A member is an entry, or "
                "one built rung of an entry that has rungs.")
    if context.families:
        w.table(["Family", "Members", "Built members"],
                [[_page_link(family_path(name), code(name)),
                  ", ".join(sorted({code(record["name"]) for record, _ in context.members(name)})) or "none",
                  str(len(context.members(name)))] for name in sorted(context.families)])
    else:
        w.paragraph("No family is registered.")
    w.heading(3, "What do I write?")
    w.paragraph("Every registered entry has a page, listed by kind.")
    kinds = sorted({str(record["kind"]) for record in context.records},
                   key=lambda kind: (KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER), kind))
    for kind in kinds:
        w.paragraph(f"**{kind}**")
        w.bullets(f"{_page_link(entry_path(record), code(record['name']))} {record['version']}"
                  for record in sorted((r for r in context.records if r["kind"] == kind),
                                       key=lambda r: str(r["name"])))
    w.heading(3, "What does a name mean?")
    w.paragraph(_page_link(VOCABULARY_PATH, "The vocabulary") + " lists every interface name with its units, frame, "
                "shape and sign.")
    w.heading(3, "What does a node do?")
    w.bullets(f"{_page_link(node_path(name), code(name))}" for name in sorted(context.node_names.values()))
    return w.page(INDEX_PATH)


__all__ = ["INTERFACE_HEADERS", "KIND_ORDER", "NO_HAND_WRITTEN", "index_page", "interfaces", "mechanism_page",
           "vocabulary_page"]
