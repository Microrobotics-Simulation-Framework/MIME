"""The family page: which rung. It prints the checkable contract the family's harness holds, the members that declare
the family, and, once two built members can replace each other, a comparison whose every cell is a value or a counted
hatch, never blank. Until a family has two built rungs it prints its contract and its rung list."""

from __future__ import annotations

from typing import Any

from mime.docs._context import Context, Record, class_var, entry_path, family_path
from mime.docs._markdown import Page, Writer, anchor, code, link

NOT_MEASURED = "not measured"
NO_HAND_WRITTEN = "No hand-written part: how to choose among this family's rungs is not written yet."


def _reference(family: Any) -> str:
    """The family's reference member, as it names itself; none where the family has no reference."""
    if family.reference is None:
        return "none"
    try:
        member = family.reference()
    except Exception as error:  # a lab's factory: say it failed, not that the family has none
        return f"not constructible here ({type(error).__name__})"
    rung = getattr(member, "rung", None)
    return f"{member.label()}" + (f", rung {rung}" if rung else "")


def _contract(w: Writer, family: Any) -> None:
    w.heading(2, "Contract")
    w.paragraph("What every member must satisfy, as the family's harness checks it on the member's own node.")
    rule = family.sign
    w.bullets([
        "Sign rule: " + (w.prose("sign rule", rule.description) if rule is not None else "none"),
        "Cases a member passes: " + ", ".join(code(case.value) for case in family.cases),
        "Settings that stand for a material's properties, each of which must reach the output: "
        + (", ".join(code(name) for name in family.material) or "none"),
        "Deliveries a two-way law makes equal and opposite: "
        + (", ".join(f"{code(a[1])} and {code(b[1])}" for a, b in family.pairs) or "none"),
        "Reference member a swap replaces: " + _reference(family),
        "Settings proportional to a material property: " + (", ".join(code(name) for name in family.linear_in) or
                                                              "none"),
    ])


def _members(w: Writer, members: list[tuple[Record, str]], declaring: list[Record], path: str) -> None:
    w.heading(2, "Members")
    if not declaring:
        w.paragraph("No registered entry declares this family.")
        return
    w.table(["Entry", "Rung", "Derivative"],
            [[link(path, entry_path(record), code(record["name"])) + f" {record['version']}",
              link(path, entry_path(record), code(rung), anchor(f"Rung {rung}")) if rung else "none",
              code(class_var(record, "differentiability"))] for record, rung in members])
    reserved = [(record, rung, reason) for record in declaring
                for rung, reason in sorted(dict(class_var(record, "reserved_rungs", {})).items())]
    for record, rung, reason in reserved:
        w.add(f"- reserved, not built: {link(path, entry_path(record), code(rung), anchor(f'Rung {rung}'))} of "
              f"{code(record['name'])}: {w.prose(f'reserved rung {rung}', str(reason))}")
    if reserved:
        w.add("")


def _comparison(w: Writer, family: Any, members: list[tuple[Record, str]], path: str) -> None:
    w.heading(2, "Comparison")
    swappable = any(case.value == "swap" for case in family.cases)
    if not swappable:
        w.paragraph("The members of this family are not alternatives to one another, so they are not compared.")
        return
    if len(members) < 2:
        w.paragraph("A comparison starts when the family has two built rungs; until then the contract and the rung "
                    "list above are the page.")
        return
    rows: list[list[str]] = []
    for record, rung in members:
        who = f"{record['name']} {rung}".strip()
        validity = len(list(class_var(record, "validity", [])))
        for what in ("oracle", "cost", "gpu"):
            w.hatch(NOT_MEASURED, f"family {family.name} {who} {what}")
        rows.append([link(path, entry_path(record), code(who)),
                     f"{validity} declared lines, free text, not parsed" if validity else "none declared",
                     NOT_MEASURED, NOT_MEASURED, code(class_var(record, "differentiability")), NOT_MEASURED])
    w.table(["Member", "Validity", "Checked against", "Cost", "Derivative", "Needs a GPU"], rows)
    w.paragraph("A cell reads `not measured` until a measurement record or a claim holds it, and each is counted.")


def family_page(family: Any, context: Context) -> Page:
    """The page of one registered family (``mime.testing.Family``)."""
    path = family_path(family.name)
    w = Writer()
    w.heading(1, f"Family {family.name}")
    members = context.members(family.name)
    declaring = [record for record in context.records if class_var(record, "family") == family.name]
    w.heading(2, "The contract in words, and how to choose")
    hand = context.hand.get(path)
    if hand is None:
        w.paragraph(NO_HAND_WRITTEN)
        w.hatch("no hand-written part", f"family {family.name}")
    else:
        w.add(hand.strip("\n"), "")
    _contract(w, family)
    _members(w, members, declaring, path)
    _comparison(w, family, members, path)
    return w.page(path)


__all__ = ["NOT_MEASURED", "NO_HAND_WRITTEN", "family_page"]
