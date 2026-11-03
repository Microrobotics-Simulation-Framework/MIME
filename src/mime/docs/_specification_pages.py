"""The reference pages: the saved form's specification with its JSON Schemas, the ``mime:`` USD schema's reference,
and the 0.2 to 0.3 migration table. Each is its hand-written source, or its record, with what the code gives
generated beside it; the checks that hold them are ``mime docs check``'s (``_specification``, ``_usd_reference``,
``_migration``)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mime.docs._context import entry_path
from mime.docs._markdown import Page, Writer, code, link
from mime.docs._migration import RECORD, migration_report
from mime.docs._schemas import GAPS, saved_form_schemas, schema_text
from mime.docs._specification import SPECIFICATIONS, statements
from mime.docs._usd_reference import SOURCE as USD_SOURCE
from mime.docs._usd_reference import documented, reference_page_rows
from mime.registry import Registered

SAVED_FORM_PAGE = "reference/saved_form.md"
USD_PAGE = "reference/mime_usd.md"
MIGRATION_PAGE = "reference/migration.md"
SCHEMA_DIRECTORY = "reference"
RENAMES = Path("docs/ratchets/renames.txt")


def saved_form_pages(root: Path) -> list[Page]:
    """The specification's page, its hand-written statements and the schemas' gaps and coverage beside them, and the
    two schemas."""
    source = root / SPECIFICATIONS[0]
    if not source.is_file():
        return []
    text = source.read_text(encoding="utf-8")
    held, _ = statements(text, SPECIFICATIONS[0])
    schemas = saved_form_schemas()
    w = Writer()
    w.add(text.strip("\n"), "")
    w.heading(2, "The schemas and their known gaps")
    w.paragraph("Generated from the closed TypedDicts: "
                + ", ".join(link(SAVED_FORM_PAGE, f"{SCHEMA_DIRECTORY}/{name}", code(name)) for name in schemas)
                + ". Each is a second reader of the shape the loader checks, for an editor. Where a schema and the "
                  "loader differ on an input, the loader decides; the differences known are these, and the test that "
                  "runs both over one corpus expects exactly the ones on which they disagree.")
    w.table(["Gap", "Files", "Loader", "Schema", "What it is"],
            [[code(gap.name), ", ".join(gap.files), gap.loader, gap.schema, w.prose(f"gap {gap.name}", gap.text)]
             for gap in GAPS])
    w.heading(2, "Coverage")
    covered = [s for s in held if s.covered]
    w.paragraph(f"{len(held)} normative statements: {len(covered)} cite the tests that hold them, and "
                f"{len(held) - len(covered)} are not yet covered"
                + (": " + ", ".join(code(s.key) for s in held if not s.covered) + "." if len(covered) < len(held)
                   else "."))
    pages = [w.page(SAVED_FORM_PAGE)]
    pages += [Page(f"{SCHEMA_DIRECTORY}/{name}", schema_text(schema)) for name, schema in schemas.items()]
    return pages


def usd_reference_page(root: Path) -> list[Page]:
    """The ``mime:`` schema's page: its hand-written meanings, then each name's constant and each geometry kind's
    settings, generated."""
    source = root / USD_SOURCE
    if not source.is_file():
        return []
    text = source.read_text(encoding="utf-8")
    names, _ = documented(text)
    rows, kinds = reference_page_rows(names)
    w = Writer()
    w.add(text.strip("\n"), "")
    w.heading(2, "The names in the code")
    w.paragraph("Each documented name with the constant that names it where the code has one; the others are built "
                "where they are written (`mime:<name>` from a prim's attributes, `mime:geometry:<setting>` from a "
                "kind's settings).")
    w.table(["Name", "Constant"], [[name, constant] for name, constant, _ in rows])
    w.heading(2, "Each geometry kind on a stage")
    w.paragraph("A kind with a native USD form is that primitive; any other writes its shape settings, in SI, as "
                "`mime:geometry:<setting>`.")
    w.table(["Kind", "Version", "Written as"], [[code(name), version, written] for name, version, written in kinds])
    return [w.page(USD_PAGE)]


def migration_page(root: Path, records: Iterable[Registered]) -> list[Page]:
    """The migration table, from its record, and the renames of entries during 0.3's development."""
    if not (root / RECORD).is_file():
        return []
    rows, _ = migration_report(root, records)
    w = Writer()
    w.heading(1, "From 0.2 to 0.3")
    w.paragraph("Each public name of the 0.2 line, the node classes and effect models a 0.2 script built its graph "
                "from, with the entry a 0.3 experiment attaches or declares in its place, or why it was removed. A 0.3 "
                "experiment declares things and attaches registered entries, which build their nodes. Generated from "
                f"`{RECORD.as_posix()}`, whose names are held to a list frozen from the tag `pre-0.3`.")
    table: list[list[str]] = []
    for row in sorted(rows, key=lambda r: r.name):
        if row.to is not None:
            kind, name = row.to.split(" ", 1)
            target = link(MIGRATION_PAGE, entry_path({"kind": kind.replace("_", " "), "name": name}),
                          code(row.to)) + (f", rung {code(row.rung)}" if row.rung else "")
        else:
            target = "removed"
        table.append([code(row.name), code(row.module), target, row.removed or row.note or ""])
    w.table(["0.2 name", "0.2 module", "In 0.3", "Note"], table)
    renames = [line.split("#", 1)[0].strip() for line in (root / RENAMES).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")] if (root / RENAMES).is_file() else []
    w.heading(2, "Renamed during 0.3's development")
    if renames:
        w.table(["Kind", "Earlier name", "Name now"],
                [[kind, code(old), code(new)] for kind, old, new in
                 ((r.split(" ", 1)[0], *r.split(" ", 1)[1].split(" -> ")) for r in renames)])
    else:
        w.paragraph("None.")
    return [w.page(MIGRATION_PAGE)]


__all__ = ["MIGRATION_PAGE", "SAVED_FORM_PAGE", "USD_PAGE", "migration_page", "saved_form_pages",
           "usd_reference_page"]
