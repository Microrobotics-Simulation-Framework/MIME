"""The ``NOTICE`` and the card of a saved folder, generated, never hand-copied, and byte for byte the same each time
(designs #198, #199; ``PUSH_HUB_DESIGN`` section 3.2).

**The NOTICE** is plain text: the experiment's own licence, then every file the folder names with its sha256 and size,
whose it is, its licence, its verdict and where its bytes live, and for a third party's file its credit, role, origin
and the permissions that name its bytes, in #198's words; the custom licences with their links, and the text of each
licence that asks for it to travel with every copy (MIT and the BSD licences) under each such file's credit; and the
availability statement. A bundle carries it (format 2), and ``mime push`` will upload it beside the bundle.

**The card** is Markdown, offline (``mime card <folder>``): what the experiment is, what it bounds and estimates (no
evidence record until a1), what reproduces, its validity lines, its licences and credit, its availability, and how to
reproduce and cite it, exactly as ``mime push`` will write it as a record's description. **Credit** (the creators with
their ORCIDs, the contributors and their roles) is kept in ``sharing.yaml`` beside the folder, which ``mime push
--init`` writes and which is not built yet: until it is, the card says so where the creators go, and shows the credit
the folder holds, each third-party file's, under its licences.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from mime._terminal import plain
from mime.assets import size_text
from mime.saved._folder import named_files, read_folder
from mime.saved._schema import ExperimentDocument, Lock, PermissionDoc
from mime.saved._terms import FileTerms, FolderTerms, folder_terms
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS
from mime.terms import CUSTOM_PREFIX, NON_COMMERCIAL, NOTICE_TEXT_REQUIRED, CustomLicence, ids_of

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
#: notice (the file's credit). As SPDX's list gives them (recalled, and compared there by the maintainer).
LICENCE_TEXTS: Mapping[str, str] = {"MIT": _MIT, "BSD-2-Clause": _BSD_START + _BSD_END,
                                    "BSD-3-Clause": _BSD_START + _BSD_THIRD + _BSD_END}
_OWNER_WORDS = {"yours": "the author's own", "third party": "a third party's", "record": "OpenMaterica's record"}
NO_LICENCE = ("The experiment declares no licence: its own files are shared under no licence it states. Saving and "
              "`mime pack` never refuse that; `mime push` will not publish it until one is declared: "
              "exp.licence(\"CC-BY-4.0\"), then save the experiment again.")
NO_EVIDENCE = ("No evidence record: MIME's validity layer arrives in a1. This card states only what the build declares "
               "and what the reference steps reproduce.")
CREDIT_HOME = ("Not stated here: credit (the creators with their ORCIDs, the contributors and their roles) is kept in "
               "`sharing.yaml` beside the folder, which `mime push --init` will write; it is not built yet. Each "
               "third-party file's credit is under Licences and credit.")


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


def availability(judged: FolderTerms) -> str:
    """#198's availability statement: reproducible exactly from the bundle alone, or given each file MIME cannot
    carry, with its licence, its role and where it is available from."""
    held = [file for file in judged.files if not file.verdict.carried]
    if not held:
        return "Reproducible exactly from this bundle alone."
    parts = [f"{plain(file.path)} (licence {plain(file.licence or 'none')}; {_role(file)}; available from "
             f"{plain(file.origin[0]) if file.origin else 'nowhere declared'})" for file in held]
    return (f"Reproducible exactly, given {len(held)} file{'s' if len(held) != 1 else ''} MIME cannot carry: "
            + "; ".join(parts) + ". Not checked since the folder was saved: the availability check comes later.")


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
             "conditions with it; referenced: never carried, whatever its size).", ""]
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
    return folder_terms(root, form.document, form.lock, named_files(form), form.version), form.document, form.lock


def notice(folder: str | os.PathLike[str]) -> str:
    """The NOTICE of the saved folder, as ``mime pack`` puts it in the bundle."""
    return notice_text(*judged_folder(folder))


def _table_row(file: FileTerms) -> str:
    licence = file.licence or "none declared"
    flag = " **non-commercial**" if NON_COMMERCIAL in file.verdict.conditions else ""
    where = "carried" if file.verdict.carried else "referenced"
    credit = plain(file.credit) if file.credit else ""
    return (f"| `{plain(file.path)}` | {_OWNER_WORDS[file.owner]} | {plain(licence)} | "
            f"{file.verdict.status.replace('_', ' ')}{flag} | {where} | {credit} |")


def card_text(folder: str | os.PathLike[str]) -> str:
    """The card of a saved folder as Markdown: it loads and builds the experiment (``Experiment.load``) for what the
    build declares, and reads the rest from the saved form. Deterministic within one environment."""
    from mime.experiment import Experiment

    root = Path(os.path.abspath(folder))
    experiment = Experiment.load(root)
    report = experiment.build().report
    judged, document, lock = judged_folder(root)
    name = plain(document["experiment"])
    licences = sorted({file.licence for file in judged.files if file.licence is not None})
    lines = [f"# {name}", "",
             "- **Version:** not published (no record yet)",
             "- **DOI:** none yet; `mime push` reserves one when the experiment is published",
             "- **Licence:** " + (plain(judged.licence) if judged.licence
                                  else "none declared (mime push will ask for one)"),
             f"- **Licences of its files:** {', '.join(plain(each) for each in licences) or 'none declared'}",
             f"- **Creators:** {CREDIT_HOME}", "",
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
    lines += [f"The first {reference['steps']} steps of every field, bit for bit in the environment the lock was "
              f"written in (Python {plain(environment['python'])} on {plain(environment['platform'])}, "
              f"{plain(environment['device'])}, {precision}, JAX {plain(environment['jax'])}), and within rtol "
              f"{reference['rtol']!r} and atol {reference['atol']!r} in another: `mime verify <folder>`.", "",
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
    commercial = [file.path for file in judged.files if NON_COMMERCIAL in file.verdict.conditions]
    if commercial:
        lines += [f"**Non-commercial terms** hold for {', '.join(f'`{plain(each)}`' for each in commercial)}.", ""]
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
    lines += ["mime pull <the record's DOI, or the bundle file>", f"mime verify ./{name}/", "```", "",
              "## How to cite", "",
              "No DOI yet: cite the record's DOI once `mime push` publishes it; `mime cite` comes later.", ""]
    return "\n".join(lines)


__all__ = ["CREDIT_HOME", "LICENCE_TEXTS", "NO_EVIDENCE", "NO_LICENCE", "availability", "card_text", "file_lines",
           "judged_folder", "notice", "notice_text", "own_licence_text", "permission_words"]
