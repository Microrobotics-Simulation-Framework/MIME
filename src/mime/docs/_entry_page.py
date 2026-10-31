"""The entry page: what a user writes, and what the entry declares. Everything below the hand-written part is
generated from the registry export, so an entry of a lab that installs MIME gets the same page as MIME's own: an
entry's page names the version it describes, and prints a hatch where the code establishes nothing.

Not rendered at this phase, and said on the page where it matters: the plan-only items (confinements, registrations
and groups), which come from running ``plan`` on a test fixture, and a worked fragment from the fixture CI builds."""

from __future__ import annotations

import re

from collections.abc import Mapping
from typing import Any

from mime.docs._context import (
    VOCABULARY_PATH,
    Context,
    Record,
    alternatives,
    built_rungs,
    builds_of,
    callback_record,
    callback_status,
    class_var,
    entry_path,
    family_path,
    is_controller,
    item_of,
    mechanism_path,
    node_path,
    rung_record,
    rung_status,
)
from mime.docs._markdown import Page, Writer, anchor, code, items, link, mapping, plural, strings
from mime.docs._symbols import meta_of
from mime.vocabulary import lookup, namespace_of

#: The class variables each section below renders, so that the last section lists the rest.
SHOWN = frozenset({"builds", "callback_gradient", "delivers", "differentiability", "family", "gradient", "name",
                   "native", "outputs", "placement", "publishes", "raises", "reads", "references", "reserved_rungs",
                   "roles", "rung_gradients", "rungs", "support", "validity", "values", "version", "writes"})
PLAN_ADDS = ("Reads, values and deliveries that its `plan` adds for a particular construction are not in these tables: "
             "they list what the class declares, and `plan` is code the build runs when the entry is attached.")
#: A geometry kind's build-side capabilities, each with what it is and who uses it, in the order a page lists them.
CAPABILITY_ROWS = {
    "samples": ("surface samples by named patch, at a resolution the consumer chooses", "the BEM; immersed-boundary "
                                                                                        "markers"),
    "area": ("the area of each patch", "the samples' weights"),
    "mass_properties": ("volume and inertia at a density", "the part resolver"),
    "radial_reach": ("the hull's radial reach about its axis", "a body's extent; contact on the hull"),
    "family_parameters": ("the parameters of the family it is", "a surrogate trained on one family"),
    "tangent": ("a tangent it declares at a surface point", "a law anisotropic along it"),
    "native": ("its native form, one of USD's own primitives", "the stage writer"),
    "drawing": ("a drawing, tessellated at the writer's resolution", "the stage writer, for a kind with no native "
                                                                    "form"),
    "region_primitive": ("its regions as OpenMaterica's primitives: exactly, or (a region with no end along an axis) "
                         "clipped at the kind's own length, or refused",
                         "OpenMaterica's manifest and stage"),
}
NO_HAND_WRITTEN = "No hand-written part: this entry's source has not been written (the docs walk counts it)."
NOT_ESTABLISHED = "not established"
GRADIENT_FIELDS = (("where", "Where"), ("reason", "Why"), ("route", "Route to exact"))
SWITCH_HEADERS = ["Switch", "Indicator", "Orders, per output group", "Route to exact"]
#: A switch's smoothness order: ``-1`` a jump, ``0`` a kink, ``k >= 1`` continuously differentiable ``k`` times.
JUMP, KINK = -1, 0


#: A unit's powers as the page prints them, superscript: a unit's exponent is no number the number rule judges.
_POWERS = str.maketrans({"2": "²", "3": "³"})


def dimension_text(setting: Mapping[str, Any]) -> str:
    """A setting's dimension as its page prints it, ``moment of inertia (kg m²)``: the dimension's name, its unit's
    powers superscript."""
    dimension = mapping(setting.get("dimension"))
    if dimension:
        name = str(dimension.get("dimension", ""))
        return re.sub(r"\^([23])\b", lambda found: found.group(1).translate(_POWERS), name)
    kind = setting.get("kind")
    return f"marked {kind}" if kind else "none"


def _interface(path: str, name: str) -> str:
    """An interface name, linked to the vocabulary page where the vocabulary defines it."""
    if lookup(name) is None:
        return code(name)
    return link(path, VOCABULARY_PATH, code(name), anchor(f"Namespace {namespace_of(name)}"))


def _target(path: str, name: str) -> str:
    """A name a law delivers into or a dynamics accepts, linked to the page of the mechanism of its kind."""
    held = lookup(name)
    return code(name) if held is None else link(path, mechanism_path(held.kind.value), code(name))


def _requires(when: Mapping[str, Any], path: str) -> str:
    parts: list[str] = []
    for role, wanted in sorted(when.items()):
        needs = mapping(wanted)
        terms = [f"flag {flag}" for flag in strings(needs.get("flags"))]
        terms += [f"accepts {_target(path, name)}" for name in strings(needs.get("accepts"))]
        terms += [f"publishes {_interface(path, name)}" for name in strings(needs.get("publishes"))]
        parts.append(f"{role}: " + (", ".join(terms) or "anything"))
    return "; ".join(parts)


def _read_text(read: Mapping[str, Any], path: str) -> str:
    text = f"{read.get('role')} {_interface(path, str(read.get('interface')))}"
    nested = mapping(read.get("at"))
    return f"{text} at the points of {_read_text(nested, path)}" if nested else text


def _node_link(path: str, dotted: str, context: Context) -> str:
    name = context.node_names.get(dotted)
    return link(path, node_path(name), code(name)) if name else code(dotted)


def order_text(order: object) -> str:
    """A switch's smoothness order, with what it means."""
    if order == JUMP:
        return f"{order} (a jump)"
    if order == KINK:
        return f"{order} (a kink)"
    return f"{order} (continuously differentiable)"


def _switches(w: Writer, declared: list[Any]) -> None:
    """The switches of a gradient record, one row each: the owner, the indicator (with the settings a time switch
    reads), the order per output group, and the route to exact. The texts are free text from the code, so they are
    recorded for the number rule; the orders are structured numbers, printed as declared."""
    w.paragraph("Switches the law evaluates, each declared once by its owner:")
    rows: list[list[str]] = []
    for held in map(mapping, declared):
        owner = str(held.get("owner"))
        settings = strings(held.get("settings"))
        indicator = w.prose(f"switch {owner} indicator", str(held.get("indicator")))
        orders = "; ".join(f"{code(group)} {order_text(order)}"
                           for group, order in sorted(mapping(held.get("orders")).items()))
        rows.append([code(owner), indicator + (f". Reads the settings {', '.join(code(s) for s in settings)}"
                                              if settings else ""),
                     orders, w.prose(f"switch {owner} route", str(held.get("route")))])
    w.table(SWITCH_HEADERS, rows)


def _status_text(w: Writer, status: str, held: Mapping[str, Any], subject: str, prefix: str) -> None:
    """A status and its gradient record (``held``, empty for none): where, why, the route and the switches."""
    if not held:
        w.paragraph(f"The derivative through {subject} is {code(status)}, the derivative of what it computes.")
        return
    w.paragraph(f"The derivative through {subject} is {code(status)}: not the exact derivative of what it computes "
                f"everywhere.")
    titles = dict(GRADIENT_FIELDS)
    printed = False
    for label in [*titles, *sorted(set(held) - set(titles) - {"switches"})]:
        if held.get(label) is None:
            continue
        if mapping(held[label]).get("not_established"):
            shown = NOT_ESTABLISHED
        elif isinstance(held[label], str):
            shown = w.prose(f"{prefix}gradient {label}", held[label])
        else:
            shown = w.value(f"{prefix}gradient {label}", held[label], limit=2000)
        w.add(f"- **{titles.get(label, label.replace('_', ' ').capitalize())}**: {shown}")
        printed = True
    if printed:
        w.add("")
    if held.get("switches"):
        _switches(w, items(held["switches"]))


def by_rung(record: Record) -> bool:
    """Whether the entry declares each rung's own status (``rung_gradients``)."""
    return bool(mapping(class_var(record, "rung_gradients", {})))


def _gradient(w: Writer, record: Record) -> None:
    status = str(class_var(record, "differentiability"))
    w.heading(2, "Gradient status")
    if is_controller(record):
        w.paragraph("The derivative depends on the mode the controller is placed in, and a saved experiment's lock "
                    "records that mode's status:")
        w.bullets([f"Inside the compiled step: {code(status)}, its law's own",
                   f"As a callback: {code(callback_status(record))}, the binding's whatever the law"])
        _status_text(w, status, mapping(class_var(record, "gradient")), "this controller inside the compiled step",
                     "mode 1 ")
        _status_text(w, callback_status(record), callback_record(record), "this controller as a callback", "mode 2 ")
        return
    if not by_rung(record):
        _status_text(w, status, mapping(class_var(record, "gradient")), "this entry", "")
        return
    w.paragraph("The derivative depends on the rung attached, and a saved experiment's lock records that rung's. Each "
                "rung's section above gives its own status and the switches its nodes evaluate:")
    w.bullets(f"{anchor_link(rung)}: {code(rung_status(record, rung))}" for rung in built_rungs(record))
    w.paragraph(f"The entry's own status, the least of its rungs', is {code(status)}.")


def _references(w: Writer, record: Record, context: Context, namespace: str) -> None:
    w.heading(2, "References")
    declared = class_var(record, "references", [])
    if mapping(declared).get("not_established"):
        w.paragraph(NOT_ESTABLISHED + ": the entry's code does not say which works it rests on.")
        return
    keys = strings(declared)
    if not keys:
        w.paragraph("none declared")
        w.hatch("none declared", item_of(record, "references"))
        return
    if namespace in context.cited:
        w.paragraph(", ".join(f"[@{key}]" for key in keys))
    else:
        w.paragraph(", ".join(code(key) for key in keys) + " (bibliography keys of this entry's own namespace, "
                    "not checked here)")


def _rungs(w: Writer, record: Record, context: Context, path: str) -> None:
    rungs = strings(class_var(record, "rungs", []))
    reserved = mapping(class_var(record, "reserved_rungs", {}))
    support = [mapping(row) for row in items(class_var(record, "support", []))]
    builds = builds_of(record)
    w.heading(2, "Nodes it builds" if not rungs else "Rungs")
    if not rungs:
        _built(w, record, context, path, "", builds.get("", []))
        if support:
            _support(w, [row for row in support if row.get("rung") is None], "Supported constructions", path)
        return
    w.paragraph("A rung is a section of this page, and the entry's `rung` setting names it; there is no default.")
    for rung in rungs:
        w.heading(3, f"Rung {rung}")
        if rung in reserved:
            w.paragraph(f"Reserved, not built: {w.prose(f'reserved rung {rung}', str(reserved[rung]))}")
            continue
        _built(w, record, context, path, rung, builds.get(rung, []))
        _support(w, [row for row in support if row.get("rung") == rung], "Supported constructions", path)
        if by_rung(record):
            _status_text(w, rung_status(record, rung), rung_record(record, rung), "this rung", f"rung {rung} ")


def _built(w: Writer, record: Record, context: Context, path: str, rung: str, nodes: list[str]) -> None:
    unchecked = item_of(record, rung) in context.unchecked
    if not nodes:
        w.paragraph("Builds no node of its own." + (" declared, unchecked." if unchecked else ""))
    else:
        w.bullets(_node_link(path, dotted, context) for dotted in nodes)
        w.paragraph("Declared in `builds`; the node walk compares it with what a round-trip case builds."
                    if not unchecked else "declared, unchecked: no round-trip case builds this rung, so the node "
                    "walk does not compare the declaration with a build.")
    if unchecked:
        w.hatch("declared, unchecked", item_of(record, rung))


def _support(w: Writer, rows: list[Mapping[str, Any]], title: str, path: str) -> None:
    if not rows:
        return
    w.paragraph(f"{title}, by what each bound thing's dynamics or field provider must offer:")
    w.table(["Requires", "Note"], [(_requires(mapping(row.get("when")), path),
                                    w.prose("support note", str(row["note"])) if row.get("note") else "")
                                   for row in rows])


def _settings(w: Writer, record: Record) -> None:
    fields = mapping(record.get("fields"))
    settings = mapping(record.get("settings"))
    w.heading(2, "Settings")
    if not fields:
        w.paragraph("none: the entry takes no setting.")
        return
    rows: list[list[str]] = []
    for name, held in sorted(fields.items()):
        field = mapping(held)
        declared = mapping(settings.get(name))
        group = field.get("group")
        rows.append([code(name), code(field.get("type", "")), w.value(f"setting {name} default", field.get("default")),
                     dimension_text(declared) if declared else (f"a group, `{group}`" if group else "none")])
    w.table(["Setting", "Type", "Default", "Dimension"], rows)
    grouped = {name: mapping(held) for name, held in settings.items() if "." in name}
    if grouped:
        w.paragraph("Settings inside a group:")
        w.table(["Setting", "Dimension", "May be none"],
                [[code(name), dimension_text(held), "yes" if held.get("optional") else "no"]
                 for name, held in sorted(grouped.items())])


def _capabilities(w: Writer, record: Record) -> None:
    """A geometry kind's capabilities, one row each, from which optional methods it implements; its native form and
    placement settings beside them."""
    held = mapping(record.get("capabilities"))
    if not held:
        return
    w.heading(2, "Capabilities")
    w.paragraph("Generated from which of the optional methods of `mime.registry.Geometry` the kind implements; a "
                "consumer that needs one it lacks is refused when it plans, naming both.")
    rows: list[list[str]] = []
    for name in [*CAPABILITY_ROWS, *sorted(set(held) - set(CAPABILITY_ROWS))]:
        if name not in held:
            continue
        what, used_by = CAPABILITY_ROWS.get(name, (name.replace("_", " "), "not stated"))
        rows.append([code(name), what, used_by, "yes" if held[name] else "no"])
    w.table(["Capability", "What it gives", "Used by", "This kind"], rows)
    native = mapping(class_var(record, "native"))
    lengths = ", ".join([*(f"{code(a)} holds {code(s)}" for a, s in sorted(mapping(native.get("lengths")).items())),
                         *(f"{code(a)} is {code(t)}" for a, t in sorted(mapping(native.get("tokens")).items()))])
    w.paragraph(f"`native`: written as USD's {code(native['primitive'])}, its attributes the kind's settings "
                f"({lengths})." if native else "`native`: none, so the stage writes it in MIME's schema with its "
                                               "drawing.")
    placement = mapping(class_var(record, "placement", {}))
    w.paragraph("`placement`: " + (", ".join(f"{code(name)} is its prim's {role.replace('_', ' ')}"
                                             for name, role in sorted(placement.items()))
                                   + "; on a stage, the prim's transform, never an attribute."
                                   if placement else "none: its own frame is the frame its settings are written in."))
    w.paragraph("Run-time capabilities (signed distance, outward normal, nearest point) are its regions' and "
                "surfaces' shapes, listed below.")


def _output_text(held: object) -> str:
    """What an entry publishes an interface as: the name of its output, which a dynamics gives as a mapping."""
    named = mapping(held)
    return code(named.get("field", "")) if named else code(held)


def _signals(w: Writer, record: Record) -> None:
    """A controller's labels: what it reads and writes under each, bound by the experiment's ``set_controller``."""
    w.heading(2, "Reads and writes")
    for name, what in (("reads", "each label bound by the experiment to a thing's state or an observation's output"),
                       ("writes", "each label bound by the experiment to an apparatus's command, whose drive it "
                                  "replaces")):
        signals = mapping(class_var(record, name, {}))
        w.paragraph(f"{code(name)}: {what}:" if signals else f"{code(name)}: none.")
        if signals:
            w.table(["Label", "Units", "Shape"],
                    [[code(label), w.value(f"{name} {label} units", mapping(held).get("units", "")),
                      code(mapping(held).get("shape", ""))] for label, held in sorted(signals.items())])


def _roles(w: Writer, record: Record, path: str) -> None:
    if record.get("kind") == "controller":
        _signals(w, record)
        return
    roles = mapping(class_var(record, "roles", {}))
    reads = mapping(class_var(record, "reads", {}))
    delivers = mapping(class_var(record, "delivers", {}))
    values = mapping(class_var(record, "values", {}))
    publishes = mapping(class_var(record, "publishes", {}))
    if not (roles or reads or delivers or values or publishes):
        return
    w.heading(2, "Roles, reads and deliveries")
    if roles:
        w.paragraph("Roles, bound by keyword when the entry is attached: "
                    + ", ".join(f"{code(role)} ({kind})" for role, kind in sorted(roles.items())) + ".")
    if reads:
        w.paragraph("Reads:")
        w.table(["Label", "Reads"],
                [[code(label), _read_text(mapping(read), path)] for label, read in sorted(reads.items())])
    if delivers:
        w.paragraph("Delivers:")
        w.table(["Label", "Role", "Into the target"],
                [[code(label), str(mapping(held).get("role")), _target(path, str(mapping(held).get("target")))]
                 for label, held in sorted(delivers.items())])
    if values:
        w.paragraph("Reads values of the things it is bound to:")
        w.table(["Label", "Role", "Value"], [_value_row(label, held) for label, held in sorted(values.items())])
    if publishes:
        w.paragraph("Publishes:")
        w.table(["Interface", "Output"], [[_interface(path, name), _output_text(held)]
                                          for name, held in sorted(publishes.items())])
    if record.get("overrides_plan"):
        w.paragraph(PLAN_ADDS)


def _value_row(label: str, held: object) -> list[str]:
    """A value an interaction reads (a role and a name) or a dynamics reads of its own thing (a name)."""
    named = mapping(held)
    if named:
        return [code(label), str(named.get("role", "its own thing")), code(named.get("name", ""))]
    return [code(label), "its own thing", code(held)]


OUTPUT_HEADERS = ["Output", "When", "Dimension", "Unit", "Shape", "What it is", "Reduced from"]
OUTPUTS_TEXT = ("What a run's result reads by name, with its units (`result[handle].<output>`). A per-step output is "
                "a series, with its time axis, when the run records it (`run(..., record=[handle.<output>])`); an "
                "output at the end is reduced by the graph and always there, over its window. A window holds the "
                "steps whose end lies after its start and not after its end.")
FLAG_HEADERS = ["Flag", "Rung", "Node", "Raised when"]
FLAGS_TEXT = ("Raised by its nodes and reported on every reading, series, export and printout of a run's result over "
              "the window it covers; a flag never blocks a read.")


def _reduced_from(output: Mapping[str, Any]) -> str:
    window = (f"the steps that end after its setting {code(output['after'])}" if output.get("after")
              else "every step of the run")
    if output.get("reduces"):
        return f"{code(output['reduces'])} by the rule {code(output.get('rule'))}, over {window}"
    return f"over {window}" if output.get("kind") == "at_end" else ""


def _outputs(w: Writer, record: Record) -> None:
    """An observation's declared outputs, and the flags any entry's nodes raise."""
    if record.get("kind") == "observation":
        outputs = [mapping(held) for held in items(class_var(record, "outputs", []))]
        w.heading(2, "Outputs")
        if not outputs:
            w.paragraph("none declared")
        else:
            w.paragraph(OUTPUTS_TEXT)
            w.table(OUTPUT_HEADERS, [
                [code(o.get("field", "")), "per step" if o.get("kind") == "per_step" else "at the end",
                 str(mapping(o.get("dimension")).get("dimension", "")), code(mapping(o.get("dimension")).get("unit", "")),
                 code(tuple(o.get("shape") or ())), w.prose(f"output {o.get('field')} description",
                                                            str(o.get("description", ""))), _reduced_from(o)]
                for o in outputs])
    flags = [mapping(held) for held in items(class_var(record, "raises", []))]
    if flags:
        w.heading(2, "Flags")
        w.paragraph(FLAGS_TEXT)
        w.table(FLAG_HEADERS, [[code(f.get("field", "")), code(f["rung"]) if f.get("rung") else "every rung",
                                code(f["node"]) if f.get("node") else "its node",
                                w.prose(f"flag {f.get('field')} meaning", str(f.get("meaning", "")))] for f in flags])


def _validity(w: Writer, record: Record) -> None:
    lines = strings(class_var(record, "validity", []))
    w.heading(2, "Validity")
    if not lines:
        w.paragraph("none declared")
        return
    w.paragraph("Declared validity lines: free text, not parsed, and not yet checked against the operating "
                "point or the region each one describes.")
    w.bullets(w.prose(f"validity line {number}", line) for number, line in enumerate(lines, start=1))
    w.hatch("free text, not parsed", item_of(record, "validity"))


def _family(w: Writer, record: Record, path: str, context: Context) -> None:
    family = class_var(record, "family")
    if family is None:
        return
    w.heading(2, "Family")
    known = str(family) in context.families
    w.paragraph(f"Declared: {link(path, family_path(str(family)), code(family)) if known else code(family)}. The "
                f"family's conformance suite is run by a test of the lab or of MIME; which test, and in which lane, "
                f"is resolved when the evidence resolver lands.")


def _others(w: Writer, record: Record) -> None:
    held = mapping(record.get("class_vars"))
    rest = sorted(name for name in held if name not in SHOWN)
    if not rest:
        return
    w.heading(2, "Other declarations")
    w.table(["Class variable", "Type", "Value"],
            [[code(name), code(mapping(held[name]).get("type", "")),
              w.value(f"class variable {name}", mapping(held[name]).get("value"))] for name in rest])


def at_a_glance(w: Writer, record: Record, context: Context, path: str) -> None:
    """The block at the top: rendered from the other homes (the node page's equation, the family page, the lists), and
    never a copy of them."""
    rungs = built_rungs(record)
    nodes = sorted({dotted for rung_nodes in builds_of(record).values() for dotted in rung_nodes})
    family = class_var(record, "family")
    status = str(class_var(record, "differentiability"))
    w.heading(2, "At a glance")
    lines = [f"{record['kind']} {code(record['name'])}, version {code(record['version'])}, namespace "
             f"{code(record['namespace'])}"]
    if family is not None:
        lines.append("Family: " + (link(path, family_path(str(family)), code(family))
                                   if str(family) in context.families else code(family))
                     + _family_members(context, str(family)))
    if rungs != [""]:
        lines.append("Rungs built: " + ", ".join(anchor_link(rung) for rung in rungs))
    if nodes:
        lines.append("Nodes: " + ", ".join(_node_link(path, dotted, context) for dotted in nodes))
        documented = [dotted for dotted in nodes if dotted in context.nodes
                      and meta_of(context.nodes[dotted]) is not None]
        if documented:
            lines.append("The equations are on the node page: "
                         + ", ".join(link(path, node_path(context.node_names[dotted]), "equations",
                                          "governing-equations") for dotted in documented))
        undocumented = [dotted for dotted in nodes if dotted not in documented]
        if undocumented:
            lines.append("Undocumented, so with no equations to link: "
                         + ", ".join(_node_link(path, dotted, context) for dotted in undocumented))
    if by_rung(record):
        lines.append("Derivative, by rung: " + ", ".join(f"{anchor_link(rung)} {code(rung_status(record, rung))}"
                                                         for rung in rungs))
    elif is_controller(record):
        lines.append(f"Derivative, by mode: inside the compiled step {code(status)}, as a callback "
                     f"{code(callback_status(record))}")
    else:
        lines.append(f"Derivative: {code(status)}")
    lines.append("Validity: " + (plural(len(strings(class_var(record, 'validity', []))), "declared line")
                                 if class_var(record, "validity") else "none declared"))
    w.bullets(lines)


def _family_members(context: Context, family: str) -> str:
    """What the glance says of the family's members: built rungs where its members have rungs, entries where they do
    not, and whether the family page compares them, which it does only for members that can replace one another."""
    members = context.members(family)
    if not members:
        return ""
    if any(rung for _, rung in members):
        count = f", with {plural(len(members), 'built rung')} among its members"
    else:
        count = f", with {plural(len(members), 'member')}, none of them with rungs"
    if len(members) < 2 or family not in context.families:
        return count
    return count + ("; compared on its page" if alternatives(context.families[family])
                    else "; not compared, they are not alternatives")


def anchor_link(rung: str) -> str:
    return f"[{code(rung)}](#{anchor(f'Rung {rung}')})"


def entry_page(record: Record, context: Context) -> Page:
    """The entry page of one entry of the export."""
    path = entry_path(record)
    w = Writer()
    namespace = str(record["namespace"])
    w.heading(1, f"{record['kind']} {record['name']}")
    w.paragraph(f"Describes the entry {code(record['name'])} at version {code(record['version'])}, registered by "
                f"{code(record.get('defined_in', ''))}.")
    at_a_glance(w, record, context, path)
    w.heading(2, "What it models, and when not to use it")
    hand = context.hand.get(path)
    if hand is None:
        w.paragraph(NO_HAND_WRITTEN)
        w.hatch("no hand-written part", item_of(record))
    else:
        w.add(hand.strip("\n"), "")
    _settings(w, record)
    _capabilities(w, record)
    _roles(w, record, path)
    _rungs(w, record, context, path)
    _gradient(w, record)
    _outputs(w, record)
    _validity(w, record)
    _family(w, record, path, context)
    _references(w, record, context, namespace)
    _others(w, record)
    return w.page(path)


__all__ = ["CAPABILITY_ROWS", "FLAGS_TEXT", "NO_HAND_WRITTEN", "OUTPUTS_TEXT", "SHOWN", "at_a_glance", "dimension_text",
           "entry_page"]
