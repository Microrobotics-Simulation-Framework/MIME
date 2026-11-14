"""The ``NOTICE`` and the card of a saved folder, generated, never hand-copied, and byte for byte the same each time
(designs #198, #199; ``PUSH_HUB_DESIGN`` section 3.2).

**The NOTICE** is plain text: the experiment's own licence, then every file the folder names with its sha256 and size,
whose it is, its licence, its verdict and where its bytes live, and for a third party's file its credit, role, origin
and the permissions that name its bytes, in #198's words; the custom licences with their links, and the text of each
licence that asks for it to travel with every copy (MIT and the BSD licences) under each such file's credit; and the
availability statement. A bundle carries it (format 2), and ``mime push`` will upload it beside the bundle. **The
lock's assets count as well as the folder's files**: an asset the experiment reads whose bytes the folder does not hold
(a wall's table, a lab's record; ``FolderTerms.assets``) is never carried by a bundle, whatever its licence, so the
NOTICE lists it under its own heading with its licence, verdict and where it is available from, its licence is among
the licences named, and the availability statement says "from this bundle alone" only where the bundle needs none. An
asset MIME's own data holds says what a reuser needs besides the bytes: that MIME, at the version that saved the folder
(whether MIME's own data holds an asset is read from the MIME installed where the NOTICE is written: the lock does not
record it, so one folder can read differently under another install; a pip line is given for a released version only).
The meshes the stage draws (the lock's ``visuals`` manifests) count the same way, fetched by ``mime visuals fetch``; where
every asset the bundle lacks is declared of role presentation and every file is carried, the statement says the results
reproduce from the bundle alone as far as the records declare, and that showing the experiment needs the assets.
An id no SPDX list the installed packaging holds names, and no licence the folder or its lock defines, is named with no
link and as unknown, never given an invented SPDX page. **A file nothing names a download location for** (saved form
14: a record with no location, or a file that may not be redistributed whose terms give no origin) is said plainly:
the availability statement opens "Not reproducible from public sources" where results depend on it, with where to
request it, and says only that showing the experiment needs it where it is of role presentation.

**The card** is Markdown, offline (``mime card <folder>``): what the experiment is, what it bounds and estimates (no
evidence record until a1), what ``mime verify <folder>`` would check (the card runs no check), its validity lines, its
licences and credit with every condition each file's licence or permission carries (non-commercial, share-alike,
no-derivatives), its availability, and how to reproduce and cite it, exactly as ``mime push`` writes it as a record's
description; and, for an experiment made from another (``Experiment.based_on``), a line naming that one, which push's
description leaves out until push states the relation between the two records (rehearsed on the sandbox first). **The record it is published as** (its version, DOI and, once minted, concept DOI, and so its pull and cite
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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mime._shell import quoted
from mime._terminal import plain
from mime.assets import size_text
from mime.assets._cache import NO_ROUTE
from mime.build import BuildRefused, BuildReport
from mime.saved._based_on import based_on_words
from mime.saved._folder import named_files, nondeterministic_gpu, read_folder
from mime.saved._scales import RESOLUTION_FACTOR, SCALED_FROM
from mime.saved._schema import ExperimentDocument, Lock, PermissionDoc, PublicationRecord
from mime.saved._terms import FileTerms, FolderTerms, LockAsset, folder_terms
from mime.saved._versions import DOCUMENT_SCHEMAS, TERMS_ADDED
from mime.terms import (
    CUSTOM_PREFIX,
    NO_DERIVATIVES,
    NON_COMMERCIAL,
    NOTICE_TEXT_REQUIRED,
    SHARE_ALIKE,
    NO_ASSERTION,
    CustomLicence,
    canonical,
    ids_of,
)

SPDX_PAGE = "https://spdx.org/licenses/{}.html"
#: A released version's spelling (``0.3.0``): what ``pip install mime-engine==`` can name.
_RELEASED = re.compile(r"\d+(\.\d+)*")

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
NO_EVIDENCE = ("No evidence record: MIME's validity layer arrives in a later release. This card states only what the "
               "build declares and what `mime verify <folder>` would check; making the card ran no check.")
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
    if identifier.startswith(CUSTOM_PREFIX) or canonical(identifier) != identifier:
        return (f"{identifier}: no link: it is no id the SPDX list of the installed packaging holds, and neither the "
                f"folder nor its lock defines it; its terms are unknown")
    return f"{identifier}: {SPDX_PAGE.format(identifier)}"


def own_licence_text(judged: FolderTerms) -> str:
    if judged.licence is not None:
        return (f"{judged.licence}. It covers the experiment's own files: those MIME wrote, and those its author "
                f"brought without declaring their terms.")
    if judged.version < TERMS_ADDED:
        return (f"none: the folder is of {DOCUMENT_SCHEMAS[judged.version]}, a form before licences, so it declares "
                f"no licence, terms or permission. " + NO_LICENCE)
    return "none. " + NO_LICENCE


def _route(request: str | None) -> str:
    return plain(request) if request else NO_ROUTE


def _where(file: FileTerms) -> str:
    if file.referenced is not None and file.by_request:
        return (f"referenced: not carried in a bundle; the asset {file.referenced}, with no download location (by "
                f"request); request it from: {_route(file.request)}")
    if file.referenced is not None:
        return (f"referenced: not carried in a bundle; the asset {file.referenced}, fetched by `mime assets fetch "
                f"<folder>`")
    if not file.verdict.carried:
        return ("referenced: not carried in a bundle, and its terms give no origin; the lock names no reference for it, "
                "so `mime pack` asks for the experiment to be saved again")
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


def _asset_role(held: LockAsset) -> str:
    role = held.ref.role
    return "role not stated" if role is None else ("presentation (results do not depend on it)"
                                                    if role == "presentation" else "results (results depend on it)")


def _asset_origin(held: LockAsset) -> str:
    if held.ref.by_request:
        return f"no download location (by request); request it from: {_route(held.ref.request)}"
    return ", ".join(plain(each) for each in held.ref.locations)


def _mime_needed(saved_by: str) -> str:
    return f"MIME {plain(saved_by)}, the version that saved the folder" if saved_by else "the MIME that saved the folder"


def _install_line(saved_by: str) -> str:
    """The pip line for a released version only: a development version is named, with no line that would not find it."""
    return f" (`pip install mime-engine=={plain(saved_by)}`)" if _RELEASED.fullmatch(saved_by) else ""


def _fetched_by(held: LockAsset) -> str:
    return "mime visuals fetch <folder>" if held.visual else "mime assets fetch <folder>"


def asset_lines(held: LockAsset, saved_by: str) -> list[str]:
    """One asset of the lock that no file of the folder stands for, as the NOTICE says it: its facts, indented, and
    where its bytes are available from; for one MIME's own data holds, the MIME a reuser needs."""
    ref = held.ref
    whose = "MIME's own record" if held.mime_own else "an asset record the lock holds"
    lines = [plain(ref.name),
             f"  sha256 {ref.sha256}, {size_text(ref.size)}; {whose}; licence {plain(ref.licence)}",
             f"  verdict: {held.verdict.words()}, by {held.verdict.by}; not carried in a bundle: the lock names it"
             + (" among the meshes the stage draws" if held.visual else "")
             + (f", with no download location: a copy placed in the asset cache is taken by its sha256 "
                f"(`{_fetched_by(held)}`)" if held.ref.by_request else f", and `{_fetched_by(held)}` fetches it"),
             f"  role: {_asset_role(held)}", f"  available from: {_asset_origin(held)}"]
    if ref.credit:
        lines.append(f"  credit: {plain(ref.credit)}")
    if held.mime_own:
        lines.append(f"  needs: {_mime_needed(saved_by)}{_install_line(saved_by)}, besides the bytes: MIME's own data holds "
                     f"this record")
    if ref.notice:
        lines.append(f"  notice: {plain(ref.notice)}")
    return lines


def _why_referenced(file: FileTerms) -> str:
    return "may not be redistributed" if not file.verdict.carried else "over the size a bundle carries"


def _by_request_parts(judged: FolderTerms) -> tuple[list[str], list[str]]:
    """Each file and asset nothing names a download location for, as the statement names it: those results may depend
    on (any role but presentation), then those of role presentation."""
    results: list[str] = []
    shown: list[str] = []
    for file in judged.files:
        if file.by_request:
            (shown if file.role == "presentation" else results).append(
                f"{plain(file.path)} (licence {plain(file.licence or 'none')}; request it from: {_route(file.request)})")
    for each in judged.assets:
        if each.ref.by_request:
            (shown if each.ref.role == "presentation" else results).append(
                f"{plain(each.ref.name)} (licence {plain(each.ref.licence)}; request it from: "
                f"{_route(each.ref.request)})")
    return results, shown


def _files_words(count: int) -> str:
    return (f"{count} file{'s' if count != 1 else ''} {'their publishers do' if count != 1 else 'its publisher does'} "
            f"not make available for download")


def by_request_warnings(judged: FolderTerms) -> list[str]:
    """``pack``'s warning for each file and asset nothing names a download location for (:class:`ByRequestWarning`):
    what the bundle carries of it, what that costs, and where to request it, or that no route is given."""
    found: list[tuple[str, str | None, str | None]] = [
        (f"{plain(file.path)} (licence {plain(file.licence or 'none')}: {file.verdict.words()})", file.role,
         file.request) for file in judged.files if file.by_request]
    found += [(f"the asset {plain(each.ref.name)} (licence {plain(each.ref.licence)})", each.ref.role, each.ref.request)
              for each in judged.assets if each.ref.by_request]
    lines: list[str] = []
    for what, role, request in found:
        cost = ("showing the experiment needs it, though no result depends on it (role presentation)"
                if role == "presentation" else "the results that depend on it are not reproducible from public sources")
        route = (f"request it from: {plain(request)}" if request else
                 "no route to request it is given: one is strongly recommended, a contact or a link to the paper's "
                 "data-availability statement (AssetRef(..., locations=(), request=\"...\"))")
        lines.append(f"{what} has no download location: the bundle carries its record (its sha256 and size), never "
                     f"its bytes, and {cost}; {route}. MIME does not recommend this: where its exact bytes can be "
                     f"fetched, name where (its record's locations, or its terms' origin, Terms(..., "
                     f"origin=(\"https://...\",)))")
    return lines


def availability(judged: FolderTerms) -> str:
    """#198's availability statement: reproducible exactly from the bundle alone only where the bundle needs nothing
    else; otherwise given each file it does not carry (one its terms do not let it carry, and one over the size it
    carries) and each asset the lock names that no file stands for (a bundle never carries those), with why, its
    licence, its role and where it is available from. A file nothing names a download location for (saved form 14) is
    said first and plainly: results that depend on one are not reproducible from public sources, and where to request
    it; one of role presentation is said last, as needed only to show the experiment."""
    results, shown = _by_request_parts(judged)
    held = [file for file in judged.files
            if (file.referenced is not None or not file.verdict.carried) and not file.by_request]
    assets = [each for each in judged.assets if not each.ref.by_request]
    clauses = _fetched_clauses(judged, held, assets)
    if results:
        count = len(results)
        text = (f"Not reproducible from public sources: the results depend on {_files_words(count)}: "
                f"{'; '.join(results)}. With {'each' if count != 1 else 'the file'} in place, `mime verify <folder>` "
                f"checks it by its sha256.")
        if clauses:
            text += " Besides, it needs " + "; and ".join(clauses) + _CLOSING
    elif not clauses:
        text = "Reproducible exactly from this bundle alone" + (" as far as the results go." if shown else ".")
    elif not held and all(each.ref.role == "presentation" for each in assets):
        text = ("Reproducible exactly from this bundle alone as far as the results go: what the bundle does not carry "
                "is declared of the display only (role presentation: no result depends on it), and showing the "
                "experiment needs " + "; and ".join(clauses) + _CLOSING)
    else:
        text = "Reproducible exactly, given " + "; and ".join(clauses) + _CLOSING
    if shown:
        text += (f" Showing the experiment needs {_files_words(len(shown))} (role presentation: no result depends on "
                 f"{'them' if len(shown) != 1 else 'it'}): {'; '.join(shown)}.")
    return text


_CLOSING = ". Not checked since the folder was saved: the availability check comes later."


def _fetched_clauses(judged: FolderTerms, held: Sequence[FileTerms], assets: Sequence[LockAsset]) -> list[str]:
    """The statement's clauses for the files and the assets the bundle does not carry that a location names
    (``held``, ``assets``): each group with how it is fetched, and each one's why, licence, role and origin."""
    clauses: list[str] = []
    if held:
        parts = [f"{plain(file.path)} ({_why_referenced(file)}; licence {plain(file.licence or 'none')}; "
                 f"{_role(file)}; available from {plain(file.origin[0]) if file.origin else 'nowhere declared'})"
                 for file in held]
        clauses.append(f"{len(held)} file{'s' if len(held) != 1 else ''} the bundle does not carry, which `mime "
                       f"assets fetch <folder>` fetches and places: " + "; ".join(parts))
    for visual in (False, True):
        group = [each for each in assets if each.visual == visual]
        if not group:
            continue
        parts = [f"{plain(each.ref.name)} (not carried in a bundle; licence {plain(each.ref.licence)}; "
                 f"{_asset_role(each)}; available from {_asset_origin(each)}"
                 + (f"; MIME's own record, which needs {_mime_needed(judged.saved_by)}" if each.mime_own else "") + ")"
                 for each in group]
        what = "the stage draws (named in the lock's visuals)" if visual else "the lock names"
        clauses.append(f"{len(group)} asset{'s' if len(group) != 1 else ''} {what} that the bundle does not carry, "
                       f"which `{_fetched_by(group[0])}` fetches" + (" and bakes" if visual else "") + ": "
                       + "; ".join(parts))
    return clauses


def _licences_named(judged: FolderTerms) -> list[str]:
    """Every licence the folder's files and the lock's assets state."""
    return [*(file.licence for file in judged.files if file.licence is not None),
            *(held.ref.licence for held in judged.assets if held.ref.licence != NO_ASSERTION)]


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
    if judged.assets:
        lines += ["", "Assets the lock names", "---------------------",
                  "Each asset the experiment reads whose bytes the folder does not hold: a bundle carries this record of "
                  "it, never its bytes, whatever its licence; the same facts as for a file, and where it is available "
                  "from.", ""]
        for held in judged.assets:
            lines += asset_lines(held, judged.saved_by)
    lines += ["", "Permissions", "-----------"]
    if not judged.permissions:
        lines.append("None declared.")
    for index, held in enumerate(judged.permissions, start=1):
        paths = [file.path for file in judged.files if file.sha256 in held["files"]]
        lines += permission_words(index, held, paths)
    lines += ["", "Licences", "--------"]
    used = sorted({identifier for licence in _licences_named(judged) for identifier in ids_of(licence)})
    definitions = judged.licence_definitions()
    lines += [plain(_licence_link(identifier, definitions)) for identifier in used] or ["None named."]
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
    where = ("by request" if file.by_request else
             "referenced" if file.referenced is not None or not file.verdict.carried else "carried")
    credit = plain(file.credit) if file.credit else ""
    return (f"| `{plain(file.path)}` | {_OWNER_WORDS[file.owner]} | {plain(licence)} | "
            f"{file.verdict.status.replace('_', ' ')}{flag} | {where} | {credit} |")


def _asset_row(held: LockAsset) -> str:
    flag = "".join(f", **{condition}**" for condition in held.verdict.conditions)
    whose = "MIME's own record" if held.mime_own else "an asset the lock names"
    return (f"| `{plain(held.ref.name)}` | {whose} | {plain(held.ref.licence)} | "
            f"{held.verdict.status.replace('_', ' ')}{flag} | {'by request' if held.ref.by_request else 'referenced'} | "
            f"{plain(held.ref.credit or '')} |")


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
        return f"mime pull {quoted(plain(publication['doi']))}"
    return f"mime pull --from {quoted(plain(publication['service']))} {quoted(plain(publication['record']))}"


def _cite_line(publication: PublicationRecord | None) -> str:
    if publication is None:
        return "No DOI yet: cite the record's DOI once `mime push` publishes it; `mime cite` comes later."
    concept = publication.get("concept_doi")
    every = f", and every version by its concept DOI, {plain(concept)}" if concept else ""
    return f"Cite this version by its DOI, {plain(publication['doi'])}{every}; `mime cite` comes later."


def card_text(folder: str | os.PathLike[str]) -> str:
    """The card of a saved folder as Markdown, naming the record its lock's ``publication`` names, or none, and the
    experiment it was based on, where it was (``Experiment.based_on``). Deterministic within one environment."""
    facts = card_facts(folder)
    return card_of(facts, facts.lock.get("publication"), based_on=True)


def card_of(facts: CardFacts, publication: PublicationRecord | None, *, based_on: bool = False) -> str:
    """The card from its ``facts``, as the description of the record ``publication`` (``None``: of no record yet).
    ``based_on`` names the experiment it was based on, where its document records one: ``mime card`` names it, and
    ``mime push``'s description does not, until push states the relation between the two records (a change to what
    push sends, rehearsed on the sandbox first)."""
    root, report, judged, document, lock = facts.root, facts.report, facts.judged, facts.document, facts.lock
    name = plain(document["experiment"])
    licences = sorted(set(_licences_named(judged)))
    parent = document.get("based_on") if based_on else None
    lines = [f"# {name}", "", *([f"- **{based_on_words(parent)}**"] if parent is not None else []),
             *_publication_lines(publication),
             "- **Licence:** " + (plain(judged.licence) if judged.licence
                                  else "none declared (mime push will ask for one)"),
             f"- **Licences of its files{' and assets' if judged.assets else ''}:** "
             f"{', '.join(plain(each) for each in licences) or 'none declared'}",
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
    written = (f"(Python {plain(environment['python'])} on {plain(environment['platform'])}, "
               f"{plain(environment['device'])}, {precision}, JAX {plain(environment['jax'])})")
    if judged.version >= SCALED_FROM:
        within = (f"within rtol {reference['rtol']!r} of the largest magnitude of each block of a field (its parts "
                  f"by unit: a pose's position and quaternion, a twist's velocity and angular velocity), or "
                  f"{reference.get('resolution_factor', RESOLUTION_FACTOR)!r} times the block's resolution, its "
                  f"divergence over the reference steps rerun from a start one ulp off; held points on the rows "
                  f"whose force acts; with no absolute part")
    else:
        within = f"within rtol {reference['rtol']!r} and atol {reference['atol']!r}, element by element"
    if nondeterministic_gpu(environment):
        claim = (f"within tolerance everywhere, the GPU it was made on {written} included, and never bit for bit: its "
                 f"reference was made with XLA's deterministic mode off. That tolerance is {within}")
    else:
        claim = f"bit for bit in the environment the lock was written in {written}, and {within} in another"
    lines += [f"`mime verify <folder>` reruns the first {reference['steps']} steps of every field and checks them "
              f"against the reference outputs the lock pins: {claim}. This card states what it would check; it ran "
              f"no check.", "",
              "## Validity lines", ""]
    validity = [f"- validity of {plain(owner)} (declared, not yet checked): {plain(line)}"
                for owner, held in sorted(report.validity.items()) for line in held]
    validity += [f"- limits of {plain(owner)} (stated by its record): {plain(line)}"
                 for owner, held in sorted(report.limits.items()) for line in held]
    validity += [f"- {plain(line.strip())}" for line in report.watch_lines()]
    lines += validity or ["None declared."]
    lines += ["", "## Licences and credit", "", f"The experiment's licence: {own_licence_text(judged)}", "",
              "| File | Whose | Licence | Verdict | Where | Credit |", "|---|---|---|---|---|---|",
              *(_table_row(file) for file in judged.files), *(_asset_row(held) for held in judged.assets), ""]
    for condition, words in CONDITION_WORDS.items():
        held = [file for file in judged.files if condition in file.verdict.conditions]
        own = [file for file in held if file.owner == "yours"]
        named = [f"`{plain(file.path)}`" for file in held if file.owner != "yours"]
        named += [f"the asset `{plain(each.ref.name)}`" for each in judged.assets if condition in each.verdict.conditions]
        named += [f"the experiment's own files ({len(own)}, under its licence)"] if own else []
        if named:
            lines += [f"**{words}** hold for {', '.join(named)}.", ""]
    lines += [plain(_licence_link(identifier, judged.licence_definitions())) for identifier in
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
           "by_request_warnings", "card_facts", "card_of", "card_text",
           "file_lines",
           "judged_folder", "notice", "notice_text", "own_licence_text", "permission_words"]
