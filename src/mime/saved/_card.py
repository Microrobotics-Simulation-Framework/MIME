"""The ``NOTICE`` and the card of a saved folder, generated, never hand-copied, and byte for byte the same each time
(designs #198, #199; ``PUSH_HUB_DESIGN`` section 3.2).

**The NOTICE** is plain text: the experiment's own licence, then every file the folder names with its sha256 and size,
whose it is, its licence, its verdict and where its bytes live, and for a third party's file its credit, role, origin
and the permissions that name its bytes, in #198's words; the custom licences with their links, and the text of each
licence that asks for it to travel with every copy (MIT and the BSD licences) under each such file's credit; and the
availability statement. A bundle carries it (format 2), and ``mime push`` will upload it beside the bundle.

**The card** is Markdown, offline (``mime card <folder>``): what the experiment is, what it bounds and estimates (no
evidence record until a1), what ``mime verify <folder>`` would check (the card runs no check), its validity lines, its
licences and credit with every condition each file's licence or permission carries (non-commercial, share-alike,
no-derivatives), its availability, and how to reproduce and cite it, exactly as ``mime push`` writes it as a record's
description. **The record it is published as** (its version, DOI and, once minted, concept DOI, and so its pull and cite
lines) comes from the ``publication`` of the lock it is written for: ``mime card`` reads the folder's own lock, and
``mime push`` gives it the staged lock's, so a published record's description says what the record is. A first version's
concept DOI is minted when it is published, so its card names it from version 2 on. Both refuse a licence that names an id the installed packaging does not list
(``terms.unlisted-licence``), whose verdict this packaging cannot give. **Credit** (the creators with
their ORCIDs, the contributors and their roles) is kept in ``sharing.yaml`` beside the saved form, which ``mime push
<folder> --to <service> --init`` writes: the card names its creators (and its contributors) where it holds one, and
says where credit lives where it does not; each third-party file's credit is under its licences either way.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mime._terminal import plain
from mime.assets import size_text
from mime.build import BuildRefused, BuildReport
from mime.saved._folder import named_files, read_folder
from mime.saved._schema import ExperimentDocument, Lock, PermissionDoc, PublicationRecord
from mime.saved._terms import FileTerms, FolderTerms, folder_terms
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS
from mime.terms import (
    CUSTOM_PREFIX,
    NO_DERIVATIVES,
    NON_COMMERCIAL,
    NOTICE_TEXT_REQUIRED,
    SHARE_ALIKE,
    CustomLicence,
    ids_of,
)

SPDX_PAGE = "https://spdx.org/licenses/{}.html"

_MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

_BSD_START = """Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
"""
_BSD_THIRD = """
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
"""
_BSD_END = """
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
#: The text each licence of :data:`mime.terms.NOTICE_TEXT_REQUIRED` asks to travel with every copy, under the copyright
#: notice (the file's credit). As SPDX's list gives them: written from memory, then compared word for word with
#: SPDX's license-list-data texts on 2026-11-01 (whitespace aside, the copyright line excluded): identical.
LICENCE_TEXTS: Mapping[str, str] = {"MIT": _MIT, "BSD-2-Clause": _BSD_START + _BSD_END,
                                    "BSD-3-Clause": _BSD_START + _BSD_THIRD + _BSD_END}
_OWNER_WORDS = {"yours": "the author's own", "third party": "a third party's", "record": "OpenMaterica's record"}
NO_LICENCE = ("The experiment declares no licence: its own files are its author's, and a reuser is granted no licence "
              "to them. Saving and `mime pack` never refuse that; `mime push` will not publish it until one is "
              "declared: exp.licence(\"CC-BY-4.0\"), then save the experiment again.")
NO_EVIDENCE = ("No evidence record: MIME's validity layer arrives in a1. This card states only what the build declares "
               "and what `mime verify <folder>` would check; making the card ran no check.")
#: Each condition a licence or a permission can carry, as the card names the terms it flags.
CONDITION_WORDS = {NON_COMMERCIAL: "Non-commercial terms", SHARE_ALIKE: "Share-alike terms",
                   NO_DERIVATIVES: "No-derivatives terms"}
CREDIT_HOME = ("Not stated here: credit (the creators with their ORCIDs, the contributors and their roles) is kept in "
               "`sharing.yaml` beside the saved form, which `mime push <folder> --to <service> --init` writes; this "
               "folder has none yet. Each third-party file's credit is under Licences and credit.")


def _licence_link(identifier: str, custom: Mapping[str, CustomLicence]) -> str:
    held = custom.get(identifier)
    if held is not None:
        return f"{identifier}: {held.title}, {held.url}; its redistribution, as declared: " \
               f"{held.redistribution.replace('_', ' ')}"
    return f"{identifier}: {SPDX_PAGE.format(identifier)}"


def own_licence_text(judged: FolderTerms) -> str:
    if judged.licence is not None:
        return (f"{judged.licence}. It covers the experiment's own files: those MIME wrote, and those its author "
                f"brought without declaring their terms.")
    if judged.version != CURRENT:
        return (f"none: the folder is of {DOCUMENT_SCHEMAS[judged.version]}, a form before licences, so it declares "
                f"no licence, terms or permission. " + NO_LICENCE)
    return "none. " + NO_LICENCE


def _where(file: FileTerms) -> str:
    if file.referenced is not None:
        return (f"referenced: not carried in a bundle; the asset {file.referenced}, fetched by `mime assets fetch "
                f"<folder>`")
    if not file.verdict.carried:
        return "referenced: not carried in a bundle, and its terms give no origin, so `mime pack` refuses the folder"
    return "carried in the bundle"


def _role(file: FileTerms) -> str:
    return ("presentation (results do not depend on it)" if file.role == "presentation"
            else "results (results depend on it)")


def file_lines(file: FileTerms) -> list[str]:
    """One file of the NOTICE: its path, then its facts, indented."""
    licence = file.licence if file.licence is not None else "none declared"
    if file.owner == "record":
        licence += ", of its values (quoted evidence is not covered by it, as the record states)"
    lines = [plain(file.path),
             f"  sha256 {file.sha256}, {size_text(file.size)}; {_OWNER_WORDS[file.owner]}; licence {plain(licence)}",
             f"  verdict: {file.verdict.words()}, by {file.verdict.by}; {_where(file)}"]
    if file.owner == "third party":
        lines += [f"  credit: {plain(file.credit or '')}", f"  role: {_role(file)}",
                  f"  origin: {', '.join(plain(each) for each in file.origin) or 'none given'}",
                  "  permissions: " + (", ".join(f"number {index + 1}" for index in file.permissions)
                                       if file.permissions else "none declared")]
    return lines


def permission_words(index: int, held: PermissionDoc, paths: Sequence[str]) -> list[str]:
    """A declared permission in #198's words: who declares what, for which files, on what evidence."""
    scope = held["scope"]
    words = ", ".join(("redistribute" if scope["redistribute"] else "no redistribution",
                       "modify" if scope["modify"] else "no modification",
                       "commercial use" if scope["commercial"] else "non-commercial"))
    evidence = held["evidence"]
    shown = (f"a public statement, {plain(evidence['pointer'])}" if evidence["kind"] == "public_statement" else
             f"a written permission held privately ({plain(evidence['pointer'])}): the document itself is not "
             f"published, and is available from the grantee on request")
    count = len(held["files"])
    return [f"{index}. {plain(held['grantee'])} declares that {plain(held['grantor'])} permitted them to redistribute "
            f"these {count} file{'s' if count != 1 else ''}, dated {plain(held['date'])}; scope: {words}; evidence: "
            f"{shown}.",
            *(f"   {plain(path)}" for path in paths),
            *(f"   sha256 {each}" for each in held["files"]),
            "   MIME checked that the declaration is complete and bound it to these bytes; it does not and cannot "
            "check that it is true."]


def _why_referenced(file: FileTerms) -> str:
    return "may not be redistributed" if not file.verdict.carried else "over the size a bundle carries"


def availability(judged: FolderTerms) -> str:
    """#198's availability statement: reproducible exactly from the bundle alone only where the bundle carries every
    file; otherwise given each file it does not carry (one its terms do not let it carry, and one over the size it
    carries), with why, its licence, its role and where it is available from."""
    held = [file for file in judged.files if file.referenced is not None or not file.verdict.carried]
    if not held:
        return "Reproducible exactly from this bundle alone."
    parts = [f"{plain(file.path)} ({_why_referenced(file)}; licence {plain(file.licence or 'none')}; {_role(file)}; "
             f"available from {plain(file.origin[0]) if file.origin else 'nowhere declared'})" for file in held]
    return (f"Reproducible exactly, given {len(held)} file{'s' if len(held) != 1 else ''} the bundle does not carry, "
            f"which `mime assets fetch <folder>` fetches and places: " + "; ".join(parts)
            + ". Not checked since the folder was saved: the availability check comes later.")


def notice_text(judged: FolderTerms, document: ExperimentDocument, lock: Lock) -> str:
    """The NOTICE of a saved folder, from its terms as judged now: deterministic, a line for every file it names."""
    name = plain(document["experiment"])
    title = f"NOTICE: {name}"
    lines = [title, "=" * len(title), "",
             f"Generated from the folder MIME {plain(lock['environment']['mime'])} saved "
             f"({DOCUMENT_SCHEMAS[judged.version]}); a bundle carries it, and mime push will upload it beside the "
             f"bundle. Naming a rights holder here implies no endorsement by them.", "",
             "The experiment's licence", "------------------------", own_licence_text(judged), "",
             "Files", "-----",
             "Each file of the folder: the sha256 and size of its bytes, whose it is, its licence, and its verdict: "
             "permitted, not permitted or unknown (OpenMaterica's words for a source's redistribution status), and "
             "its tier, where its bytes live (open and conditional: carried in the bundle, a conditional file's "
             "conditions with it; carried: the author's own, carried because the author packs it, under a licence "
             "that grants only what it states, or none; referenced: never carried, whatever its size).", ""]
    for file in judged.files:
        lines += file_lines(file)
    lines += ["", "Permissions", "-----------"]
    if not judged.permissions:
        lines.append("None declared.")
    for index, held in enumerate(judged.permissions, start=1):
        paths = [file.path for file in judged.files if file.sha256 in held["files"]]
        lines += permission_words(index, held, paths)
    lines += ["", "Licences", "--------"]
    used = sorted({identifier for file in judged.files if file.licence is not None
                   for identifier in ids_of(file.licence)})
    lines += [plain(_licence_link(identifier, judged.custom)) for identifier in used] or ["None named."]
    for identifier in sorted(NOTICE_TEXT_REQUIRED & set(used)):
        for file in judged.files:
            if file.licence is not None and identifier in ids_of(file.licence) and file.owner != "yours":
                lines += ["", f"{identifier}, as its text asks, for {plain(file.path)}:", "",
                          plain(file.credit or "(no credit declared)"), "", LICENCE_TEXTS[identifier]]
    lines += ["", "Availability", "------------", availability(judged), ""]
    return "\n".join(lines)


def judged_folder(folder: str | os.PathLike[str]) -> tuple[FolderTerms, ExperimentDocument, Lock]:
    """A saved folder's terms, judged now; a file that travels as a reference and is not placed yet is described by
    its asset. Reads the folder and runs nothing it names."""
    root = Path(os.path.abspath(folder))
    form = read_folder(root, absent_referenced=True)
    judged = folder_terms(root, form.document, form.lock, named_files(form), form.version)
    refusals = judged.unlisted_refusals("mime card")
    if refusals:
        raise BuildRefused(refusals)
    return judged, form.document, form.lock


def notice(folder: str | os.PathLike[str]) -> str:
    """The NOTICE of the saved folder, as ``mime pack`` puts it in the bundle."""
    return notice_text(*judged_folder(folder))


def _table_row(file: FileTerms) -> str:
    licence = file.licence or "none declared"
    flag = "".join(f", **{condition}**" for condition in file.verdict.conditions)
    where = "referenced" if file.referenced is not None or not file.verdict.carried else "carried"
    credit = plain(file.credit) if file.credit else ""
    return (f"| `{plain(file.path)}` | {_OWNER_WORDS[file.owner]} | {plain(licence)} | "
            f"{file.verdict.status.replace('_', ' ')}{flag} | {where} | {credit} |")


def _credit_lines(root: Path) -> list[str]:
    """The creators and contributors ``sharing.yaml`` names, where the folder holds one (refused as push refuses it,
    since the card is the record's description); where it holds none, where credit lives."""
    from mime.push._sharing import SHARING_FILE, read_sharing

    if not (root / SHARING_FILE).is_file():
        return [f"- **Creators:** {CREDIT_HOME}"]
    sharing = read_sharing(root)
    lines = [f"- **Creators:** {'; '.join(person.words() for person in sharing.creators)}"]
    if sharing.contributors:
        lines.append(f"- **Contributors:** {'; '.join(person.words() for person in sharing.contributors)}")
    return lines


@dataclass(frozen=True)
class CardFacts:
    """What a card states that one push settles once: the build's report, and the folder's terms, document and lock.
    ``sharing.yaml`` and the publication are read as the card is written, since an edit kept from the site and a reserved
    DOI change them during a push."""

    root: Path
    report: BuildReport
    judged: FolderTerms
    document: ExperimentDocument
    lock: Lock


def card_facts(folder: str | os.PathLike[str]) -> CardFacts:
    """The facts of a saved folder's card: it loads and builds the experiment (``Experiment.load``) for what the build
    declares, and reads the rest from the saved form."""
    from mime.experiment import Experiment

    root = Path(os.path.abspath(folder))
    report = Experiment.load(root).build().report
    judged, document, lock = judged_folder(root)
    return CardFacts(root, report, judged, document, lock)


def _publication_lines(publication: PublicationRecord | None) -> list[str]:
    if publication is None:
        return ["- **Version:** not published (no record yet)",
                "- **DOI:** none yet; `mime push` reserves one when the experiment is published"]
    concept = publication.get("concept_doi")
    return [f"- **Version:** {publication['version']}", f"- **DOI:** {plain(publication['doi'])}",
            *([f"- **Concept DOI (every version):** {plain(concept)}"] if concept else [])]


def _pull_line(publication: PublicationRecord | None) -> str:
    if publication is None:
        return "mime pull <the record's DOI, or the bundle file>"
    if publication["service"] == "zenodo":
        return f"mime pull {plain(publication['doi'])}"
    return f"mime pull --from {plain(publication['service'])} {plain(publication['record'])}"


def _cite_line(publication: PublicationRecord | None) -> str:
    if publication is None:
        return "No DOI yet: cite the record's DOI once `mime push` publishes it; `mime cite` comes later."
    concept = publication.get("concept_doi")
    every = f", and every version by its concept DOI, {plain(concept)}" if concept else ""
    return f"Cite this version by its DOI, {plain(publication['doi'])}{every}; `mime cite` comes later."


def card_text(folder: str | os.PathLike[str]) -> str:
    """The card of a saved folder as Markdown, naming the record its lock's ``publication`` names, or none.
    Deterministic within one environment."""
    facts = card_facts(folder)
    return card_of(facts, facts.lock.get("publication"))


def card_of(facts: CardFacts, publication: PublicationRecord | None) -> str:
    """The card from its ``facts``, as the description of the record ``publication`` (``None``: of no record yet)."""
    root, report, judged, document, lock = facts.root, facts.report, facts.judged, facts.document, facts.lock
    name = plain(document["experiment"])
    licences = sorted({file.licence for file in judged.files if file.licence is not None})
    lines = [f"# {name}", "", *_publication_lines(publication),
             "- **Licence:** " + (plain(judged.licence) if judged.licence
                                  else "none declared (mime push will ask for one)"),
             f"- **Licences of its files:** {', '.join(plain(each) for each in licences) or 'none declared'}",
             *_credit_lines(root), "",
             "## What it is", "",
             f"Saved by MIME {plain(lock['environment']['mime'])} as {DOCUMENT_SCHEMAS[judged.version]}, built under "
             f"the {document['profile']} profile.", ""]
    if report.profile is not None:
        held = report.profile
        lines.append(f"- The {held.profile.value} profile leaves {len(held.pending)} item"
                     f"{'s' if len(held.pending) != 1 else ''} pending and lists {len(held.listed)}; every output is "
                     f"stamped with them, and `mime pending <folder>` lists each with the lines to paste.")
    lines += [f"- {thing['kind']} `{plain(thing_name)}`" for thing_name, thing in sorted(document["things"].items())]
    lines += [f"- {plain(entry['kind'])} {plain(entry['name'])} {plain(entry['version'])}, from "
              f"{plain(entry['distribution'])} {plain(entry['distribution_version'])}"
              for entry in sorted(lock["entries"], key=lambda e: (e["kind"], e["name"], e["version"]))]
    lines += ["", "## Evidence: what it bounds and what it estimates", "", NO_EVIDENCE, "",
              "## What reproduces", ""]
    reference = lock["reference"]
    environment = lock["environment"]
    precision = "float64" if environment["x64"] else "float32"
    lines += [f"`mime verify <folder>` reruns the first {reference['steps']} steps of every field and checks them "
              f"against the reference outputs the lock pins: bit for bit in the environment the lock was written in "
              f"(Python {plain(environment['python'])} on {plain(environment['platform'])}, "
              f"{plain(environment['device'])}, {precision}, JAX {plain(environment['jax'])}), and within rtol "
              f"{reference['rtol']!r} and atol {reference['atol']!r} in another. This card states what it would "
              f"check; it ran no check.", "",
              "## Validity lines", ""]
    validity = [f"- validity of {plain(owner)} (declared, not yet checked): {plain(line)}"
                for owner, held in sorted(report.validity.items()) for line in held]
    validity += [f"- limits of {plain(owner)} (stated by its record): {plain(line)}"
                 for owner, held in sorted(report.limits.items()) for line in held]
    validity += [f"- {plain(line.strip())}" for line in report.watch_lines()]
    lines += validity or ["None declared."]
    lines += ["", "## Licences and credit", "", f"The experiment's licence: {own_licence_text(judged)}", "",
              "| File | Whose | Licence | Verdict | Where | Credit |", "|---|---|---|---|---|---|",
              *(_table_row(file) for file in judged.files), ""]
    for condition, words in CONDITION_WORDS.items():
        held = [file for file in judged.files if condition in file.verdict.conditions]
        own = [file for file in held if file.owner == "yours"]
        named = [f"`{plain(file.path)}`" for file in held if file.owner != "yours"]
        named += [f"the experiment's own files ({len(own)}, under its licence)"] if own else []
        if named:
            lines += [f"**{words}** hold for {', '.join(named)}.", ""]
    lines += [plain(_licence_link(identifier, judged.custom)) for identifier in
              sorted({each for licence in licences for each in ids_of(licence) if each.startswith(CUSTOM_PREFIX)})]
    for index, held in enumerate(judged.permissions, start=1):
        paths = [file.path for file in judged.files if file.sha256 in held["files"]]
        lines += permission_words(index, held, paths)
    lines += ["", "## Availability", "", availability(judged), "", "## How to reproduce", "", "```",
              f"pip install mime-engine=={plain(lock['environment']['mime'])}"]
    distributions = sorted({(entry["distribution"], entry["distribution_version"]) for entry in lock["entries"]
                            if entry["distribution"] != "mime-engine"})
    lines += [f"pip install {plain(each)}=={plain(version)}" for each, version in distributions]
    lines += [_pull_line(publication), f"mime verify ./{name}/", "```", "", "## How to cite", "",
              _cite_line(publication), ""]
    return "\n".join(lines)


__all__ = ["CONDITION_WORDS", "CREDIT_HOME", "LICENCE_TEXTS", "NO_EVIDENCE", "NO_LICENCE", "CardFacts", "availability",
           "card_facts", "card_of", "card_text",
           "file_lines",
           "judged_folder", "notice", "notice_text", "own_licence_text", "permission_words"]
