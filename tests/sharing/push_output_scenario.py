"""The session whose output ``mime push`` and ``mime pull`` are held to: a toy experiment and a toy with a layer under a
licence of its own (a declared permission lets it travel, and one whose licence forbids redistribution travels as a
reference), pushed to the offline fake, edited on its "site", published and pulled, command after command.

``build`` makes the two folders; ``session`` runs the same commands wherever it is asked to: in a test against a captured
stream, and by the script that re-runs P2b's samples at two commits. What it returns is the transcript of what a pipe received, each command with its
standard output, its standard error and its exit code, with what differs from run to run (the temporary folder, the fake's
port, the sizes the bundle's bytes make) written as a placeholder."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from push_cases import write_sharing
from terms_cases import NO_REDISTRIBUTION, layered, same_reference_run, write_mesh
from zenodo_records import FakeRecords

from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

CREATOR = "Carberry, Josiah"
ORCID = "0000-0002-1825-0097"

#: The line pooch logs as a download starts, there or not as the process set its log level: not MIME's to keep.
_POOCH_LOG = "Downloading data from "

type Run = Callable[[list[str]], tuple[int, str, str]]


def build(work: Path) -> None:
    """In ``work``, the toy and the toy with the three layers, each with its ``sharing.yaml``."""
    from mime.testing.toys import full_toy

    toy = full_toy()
    toy.licence("CC-BY-4.0")
    toy.save(work / "mime_sandbox_toy")
    own = CustomLicence("LicenseRef-Example-Maker-1.0", "Example Maker Licence 1.0",
                        "https://example.org/example-maker/LICENSE.txt")
    kinematics = write_mesh(work / "made", "arm_kinematics", 1200)
    granted = write_mesh(work / "made", "arm_shell", 2400)
    visual = write_mesh(work / "made", "arm_visual", 6000)
    layers = [(kinematics, Terms("MIT", "Copyright (c) 2021 A. Driver (synthetic)", "results")),
              (granted, Terms(own, "Example Maker Ltd (synthetic)", "presentation")),
              (visual, Terms(NO_REDISTRIBUTION, "Robot Maker Ltd (synthetic)", "presentation",
                             ("https://example.org/robot-maker/6a3ebb1/arm_visual.usda",)))]
    permission = Permission("Example Maker Ltd (synthetic)", CREATOR, (hashlib.sha256(granted.read_bytes()).hexdigest(),),
                            Scope(True, False, False), "2026-11-01",
                            Evidence("public_statement", "https://example.org/example-maker/permission"))
    with same_reference_run():
        layered("mime_sandbox_terms", layers, licence="CC-BY-4.0", permissions=[permission]).build().save(
            work / "mime_sandbox_terms")
    for name in ("mime_sandbox_toy", "mime_sandbox_terms"):
        write_sharing(work / name, creators=f"{{family: Carberry, given: Josiah, orcid: {ORCID}}}",
                      extra=f'uploader: {{name: "{CREATOR}"}}\n')


def edit_on_site(fake: FakeRecords, record: str) -> None:
    """The draft's title and its creator's given name changed on the site."""
    metadata = fake.records[record].body["metadata"]  # type: ignore[index]
    metadata["title"] = "A title corrected on the site"  # type: ignore[index]
    metadata["creators"][0]["person_or_org"]["given_name"] = "J."  # type: ignore[index]


def session(work: Path, fake: FakeRecords, run: Run) -> str:
    """The commands, run through ``run`` (which returns the exit code, standard output and standard error of ``mime`` with
    those words); the transcript, normalised."""
    toy, terms = work / "mime_sandbox_toy", work / "mime_sandbox_terms"
    to = ["--to", "zenodo-sandbox"]
    steps: list[Callable[[], list[str]]] = [
        lambda: ["push", str(toy), *to],
        lambda: ["push", str(toy), *to],
        lambda: (edit_on_site(fake, "629525"), ["push", str(toy), *to, "--keep-site-edits"])[1],
        lambda: ["push", str(terms), *to],
        lambda: ["push", str(terms), *to, "--status"],
        lambda: ["push", str(terms), *to, "--full-plan", "--publish", "--confirm", "10.5072/zenodo.629527"],
        lambda: ["push", str(toy), *to, "--publish", "--confirm", "10.5072/zenodo.629525"],
        lambda: ["pull", "--from", "zenodo-sandbox", "629525", "--into", str(work / "pulled")],
        lambda: ["push", str(toy), *to, "--init"],
    ]
    held: list[str] = []
    for step in steps:
        words = step()
        code, out, err = run(words)
        err = "".join(line for line in err.splitlines(keepends=True) if not line.startswith(_POOCH_LOG))
        held.append("$ mime " + " ".join(words) + "\n" + out + ("[stderr]\n" + err if err else "") + f"[exit {code}]\n")
    return normalised("".join(held), work, fake.site)


#: What differs from run to run or from machine to machine: a size, a checksum of the bundle (its lock records the
#: machine), the card's length, the day, the release, and the file a traceback was written to.
_PLACEHOLDERS = [(re.compile(r"\b\d+(?:\.\d+)? (?:kB|MB|bytes)\b"), "<size>"),
                 (re.compile(r"md5:[0-9a-f]{32}"), "md5:<md5>"),
                 (re.compile(r"\([\d,]+ characters\)"), "(<n> characters)"),
                 (re.compile(r"Publication date: \d{4}-\d{2}-\d{2}"), "Publication date: <date>"),
                 (re.compile(r"mime-engine \S+"), "mime-engine <release>"),
                 (re.compile(r"/mime/\d[^/]*/"), "/mime/<release>/"),
                 (re.compile(r"The full traceback is in \S+;"), "The full traceback is in <file>;")]


def normalised(text: str, work: Path, site: str) -> str:
    text = text.replace(str(work), "<tmp>").replace(site, "<fake>")
    for pattern, placeholder in _PLACEHOLDERS:
        text = pattern.sub(placeholder, text)
    return text
