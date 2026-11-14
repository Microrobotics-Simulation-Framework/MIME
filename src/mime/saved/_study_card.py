"""The NOTICE and the card of a study: what a study folder holds, under which terms, and how it is reproduced.

The **NOTICE** a study container carries lists the study's own files (what its run wrote: MIME-written results of its
base experiment, under that experiment's licence), each with its sha256 and size, then each of its saved folders' NOTICE
(the base, and each group's own build) as ``mime pack`` writes a bundle's. The **card** is read from the study's
manifest and its saved folders, and runs nothing: per study its base (name, input hash, profile), its axes and points,
what its run kept, its groups, the licences of its files and assets, the packages to install, what it leaves out and the
commands that pull, verify and run it again. A publishing folder's card is its record's ``[record]`` and a section per
study.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mime._shell import quoted
from mime._terminal import plain
from mime.assets import size_text
from mime.build import BuildRefused
from mime.results._study_folder import BASE_DIR, DONE, STUDY_FORMAT, STUDY_VERSION
from mime.saved._card import NO_EVIDENCE, notice_text, own_licence_text
from mime.saved._container import files_of, sha256_of
from mime.saved._folder import SavedForm, named_files, read_folder
from mime.saved._schema import PublicationRecord
from mime.saved._terms import FolderTerms, folder_terms

#: What a study folder never holds, said on its card: the study's spec run again makes the rest.
LEFT_OUT = ("A study folder holds what its run kept and nothing else: each member's recorded outputs at the rate the "
            "spec asked, its manifest and report, and the saved experiments it ran; no run folder, checkpoint or raw "
            "output of a member is in it.")


def _own_files(root: Path) -> list[tuple[str, str, int]]:
    """The study's own files below ``root`` (none in a saved folder, ``mime.toml`` aside), each with its sha256 and size."""
    from mime.saved._bundle import SHARING_FILES
    from mime.saved._studies import STUDY, saved_folder_of

    found: list[tuple[str, str, int]] = []
    for member, path in files_of(root, STUDY):
        if member in SHARING_FILES or saved_folder_of(member) is not None:
            continue
        data = path.read_bytes()
        found.append((member, sha256_of(data), len(data)))
    return found


def _folder_words(where: str) -> str:
    return "the study's base" if where == BASE_DIR else f"group {where.split('/')[1]}'s own build"


def study_notice_text(root: Path, document: Mapping[str, Any], judged: Mapping[str, FolderTerms],
                      forms: Mapping[str, SavedForm]) -> str:
    """The NOTICE of the study folder ``root``: deterministic, a line for each of its own files, then each of its saved
    folders' NOTICE."""
    title = f"NOTICE: the study {plain(root.name)}"
    licence = forms[BASE_DIR].document.get("licence")
    lines = [title, "=" * len(title), "",
             f"Generated from the study folder MIME {plain(str(document['lineage']['mime']))} wrote ({STUDY_FORMAT} "
             f"version {STUDY_VERSION}); a study container carries it, and mime push uploads it beside the container. "
             f"Naming a rights holder here implies no endorsement by them.", "",
             "The study's own files", "---------------------",
             f"What the study's run wrote (its manifest and spec, each group's outputs and report, its figures and "
             f"comparison layer): MIME-written results of the base experiment, under its licence "
             f"({plain(licence) if licence else 'none declared'}), each with the sha256 and size of its bytes.", ""]
    lines += [f"- {plain(member)}: sha256 {digest}, {size_text(size)}" for member, digest, size in _own_files(root)]
    for where, form in forms.items():
        heading = f"The saved experiment {where}/ ({_folder_words(where)})"
        lines += ["", heading, "-" * len(heading), "", *notice_text(judged[where], form.document, form.lock).splitlines()]
    return "\n".join(lines) + "\n"


def _study_facts(root: Path) -> tuple[dict[str, Any], dict[str, SavedForm], dict[str, FolderTerms]]:
    """A study folder read for its card and NOTICE: its manifest, and each saved folder's form and terms, judged now. A
    file that travels as a reference and is not placed yet is described by its asset; nothing is run."""
    from mime.results._study_read import read_study
    from mime.saved._studies import saved_folders

    document = read_study(root).record
    forms = {where: read_folder(root / where, absent_referenced=True) for where in saved_folders(document)}
    judged = {where: folder_terms(root / where, form.document, form.lock, named_files(form), form.version)
              for where, form in forms.items()}
    refusals = [refusal for terms in judged.values() for refusal in terms.unlisted_refusals("mime card")]
    if refusals:
        raise BuildRefused(refusals)
    return document, forms, judged


def study_notice(folder: str | os.PathLike[str]) -> str:
    """The NOTICE of a study folder, as ``mime pack`` puts it in the study container; of a publishing folder, each of its
    studies' in turn."""
    from mime.saved._studies import is_publishing_folder, publishing_entries

    root = Path(os.path.abspath(folder))
    if is_publishing_folder(root):
        return "\n".join(study_notice(study) for study in publishing_entries(root)[0])
    document, forms, judged = _study_facts(root)
    return study_notice_text(root, document, judged, forms)


def _value(held: Mapping[str, Any]) -> str:
    value: object = held["value"]
    unit = held["unit"]
    shown = f"({', '.join(map(str, cast(list[object], value)))})" if isinstance(value, list) else str(value)
    return plain(shown) + (f" {plain(str(unit))}" if unit else "")


def _axes_lines(document: Mapping[str, Any]) -> list[str]:
    points = cast(list[dict[str, Any]], document["points"])
    lines: list[str] = []
    for index, axis in enumerate(cast(list[dict[str, Any]], document["axes"])):
        values: list[str] = []
        for point in points:
            shown = _value(cast(Mapping[str, Any], point["values"][index]))
            if shown not in values:
                values.append(shown)
        lines.append(f"- axis `{plain(str(axis['path']))}` ({plain(str(axis['kind']))}): {', '.join(values)}")
    return lines


def _run_words(document: Mapping[str, Any]) -> str:
    run = cast(Mapping[str, Any], document["run"])
    every = run.get("every")
    kept = ", ".join(f"`{plain(str(each))}`" for each in cast(list[object], run["record"])) or "nothing"
    return (f"each member ran {_value(cast(Mapping[str, Any], run['duration']))} ({run['steps']} steps) and kept "
            f"{kept}" + (f", every {_value(cast(Mapping[str, Any], every))}" if every else ", at every step"))


def _section(root: Path, level: str) -> tuple[list[str], str]:
    """A study's part of a card (what it is, its terms, what it leaves out, its packages) and the MIME that ran it."""
    document, forms, judged = _study_facts(root)
    base = cast(Mapping[str, Any], document["experiment"])
    rows = cast(list[dict[str, Any]], document["members"])
    done = sum(1 for row in rows if row["status"] == DONE)
    groups = cast(list[dict[str, Any]], document["groups"])
    lines = [f"{level} {plain(root.name)}", "",
             f"- **Base experiment:** `{plain(str(base['name']))}`, input hash {plain(str(base['input_hash']))}, built "
             f"under the {plain(str(base['profile']))} profile",
             f"- **Licence:** {own_licence_text(judged[BASE_DIR])}",
             f"- **Points:** {len(cast(list[object], document['points']))}, {done} of {len(rows)} members run, "
             f"{len(cast(list[object], document['dropped']))} dropped; {len(groups)} group"
             f"{'s' if len(groups) != 1 else ''} (each compiles once)",
             f"- **Run:** {_run_words(document)}", *_axes_lines(document), "",
             f"Run by MIME {plain(str(document['lineage']['mime']))}. {LEFT_OUT}", ""]
    for where, terms in judged.items():
        lines.append(f"- The saved experiment `{plain(where)}/` ({_folder_words(where)}): "
                     + "; ".join(f"`{plain(file.path)}` {plain(file.licence or 'no licence')}, "
                                 f"{file.verdict.words()}" + (", referenced" if file.referenced is not None
                                                              or not file.verdict.carried else "")
                                 for file in terms.files))
        lines += [f"  - the asset `{plain(held.ref.name)}` {plain(held.ref.licence)}, {held.verdict.words()}, "
                  f"referenced" for held in terms.assets]
    distributions = sorted({(entry["distribution"], entry["distribution_version"]) for form in forms.values()
                            for entry in form.lock["entries"]})
    lines += ["", f"Packages: {', '.join(f'{plain(name)} {plain(version)}' for name, version in distributions)}.", ""]
    return lines, str(document["lineage"]["mime"])


def _reproduce(names: Sequence[str], mime_versions: Sequence[str], publication: PublicationRecord | None) -> list[str]:
    if publication is None:
        pull = "mime pull <the record's DOI, or the .mimestudy file>"
    elif publication["service"] == "zenodo":
        pull = f"mime pull {quoted(plain(publication['doi']))}"
    else:
        pull = f"mime pull --from {quoted(plain(publication['service']))} {quoted(plain(publication['record']))}"
    several = len(names) > 1
    into = " --into ./studies/" if several else ""
    folders = [f"./studies/{name}/" if several else f"./{name}/" for name in names]
    lines = ["```", *(f"pip install mime-engine=={plain(version)}" for version in sorted(set(mime_versions))),
             pull + into, *(f"mime verify {folder}" for folder in folders),
             *(f"mime study run {folder}study.mime.yaml --out ./{name}_again/" for folder, name in zip(folders, names,
                                                                                                    strict=True)),
             "```", ""]
    return lines


_REPRODUCES = ("`mime verify <folder>` checks each study's files against its manifest (each group's outputs by sha256, "
               "each figure's data, the base's input hash), then reruns its base experiment and each group's own build "
               "against their reference outputs, as it verifies a saved experiment. This is an acceptance: the files "
               "are the ones the author's run wrote and its experiments reproduce here; it bounds nothing about physics, "
               "and it does not run the study again. `mime study run <folder>/study.mime.yaml --out <a new folder>` "
               "does, and `StudyResult.compare` compares the two.")


def study_card_text(folder: str | os.PathLike[str], publication: PublicationRecord | None = None) -> str:
    """The card of a study folder, or of a publishing folder (its record's ``[record]``, then a section per study), as
    Markdown, as the description of the record ``publication`` (``None``: of no record yet). Deterministic within one
    environment; runs nothing."""
    from mime.push._manifest import MANIFEST_FILE, read_manifest
    from mime.saved._studies import is_publishing_folder, publishing_entries

    root = Path(os.path.abspath(folder))
    studies = publishing_entries(root)[0] if is_publishing_folder(root) else [root]
    record = read_manifest(root).record if (root / MANIFEST_FILE).is_file() else None
    title = plain(record.title) if record is not None else plain(root.name)
    lines = [f"# {title}", ""]
    if publication is not None:
        lines += [f"- **Version:** {publication['version']}", f"- **DOI:** {plain(publication['doi'])}"]
    else:
        lines += ["- **Version:** not published (no record yet)",
                  "- **DOI:** none yet; `mime push` reserves one when the studies are published"]
    if record is not None:
        lines.append(f"- **Creators:** {'; '.join(person.words() for person in record.creators)}")
        if record.contributors:
            lines.append(f"- **Contributors:** {'; '.join(person.words() for person in record.contributors)}")
    lines += [f"- **Studies:** {len(studies)}, one `.mimestudy` each", "", "## Evidence: what it bounds and what it "
              "estimates", "", NO_EVIDENCE, "", "## What reproduces", "", _REPRODUCES, "", "## The studies", ""]
    versions: list[str] = []
    for study in studies:
        section, version = _section(study, "###")
        lines += section
        versions.append(version)
    lines += ["## How to reproduce", "", *_reproduce([study.name for study in studies], versions, publication),
              "## How to cite", "",
              (f"Cite this version by its DOI, {plain(publication['doi'])}; `mime cite` comes later." if publication
               else "No DOI yet: cite the record's DOI once `mime push` publishes it; `mime cite` comes later."), ""]
    if record is not None and record.also_cite:
        lines += [f"If you use this, please also cite: {'; '.join(each.words() for each in record.also_cite)}.", ""]
    return "\n".join(lines)


__all__ = ["LEFT_OUT", "study_card_text", "study_notice", "study_notice_text"]
