# SPDX-License-Identifier: Apache-2.0
"""Fail where a tracked file, a commit message or a branch's name names one of the maintainer's private projects, or
carries a private path.

    python scripts/check_confidential.py [--key-file KEY] [--messages RANGE] [--diffs RANGE] [--public REF]
                                         [--ref-name NAME]
    python scripts/check_confidential.py [--key-file KEY] --directory DIR
    python scripts/check_confidential.py --key-file KEY --digest < TERMS > scripts/confidential_digests.txt

Copied from OpenMaterica's `scripts/check_confidential.py` (its `main` at a3052d4c, the script as last changed in
903e513f), with `confidential_digests.txt` byte for byte (as of f8775b0a): the digests are made under the same key, so
the same file applies. Changed for MIME: without a key the scan fails, so it cannot pass having checked nothing, except
on a pull request from a fork, which GitHub gives no secrets: told apart by the event, never by the key's absence
alone, it checks the private paths and warns that the names were not. The pre-push mode and its hook are left out; in
their place, `--diffs` reads what each commit of a range adds, so a name added and removed again within a pull
request is found, and every path is read as well as its content. And `--directory DIR` reads the files under DIR
instead of the tracked tree, each by its path under DIR and its content, with the same key and the same fork
exception: it is for what is built and never committed (the generated documentation pages), which the tracked tree
does not hold and which can carry text found in no tracked file. It reads that directory alone, so it takes none of
the options that read the repository.

The names cannot be listed in this public repository, and plain hashes of them would not hide them: they are short
words, which hashing a word list recovers. So `confidential_digests.txt` holds their HMAC-SHA256 under a secret key,
64 hexadecimal digits from `CONFIDENTIAL_SCAN_KEY` or a key file. Its `check` line is the digest of a fixed phrase: a
key other than the one the digests were made with is refused, rather than finding nothing. Text is lowercased and
split into words of letters and digits; each word, each pair of adjacent words and each hyphenated run of words is
hashed, its words joined by hyphens. A finding gives its place and the term's first letter and length, never the
term.

The private paths need no key: an absolute home directory, a session scratchpad, Claude Code's temporary directory,
and the private planning directory, which only `CLAUDE.md` names, one level up. The tracked files are those of the
repository in the working directory, read from its working tree (`tracked_contents`, the one place files are read),
each by its path and its content. With `--diffs RANGE`, each commit in the range is read against its first parent (a
root commit against the empty tree): every path it adds or changes, and the lines it adds, with the line after each
(where a pair of words split by the line break is placed); a binary content, or a file the commit adds, is read whole.
`--public REF` leaves out of both ranges the commits reachable from REF, a branch already public and scanned.

A content holding no NUL byte is read as UTF-8, a byte that is not UTF-8 read as a replacement, so every ASCII word of
a text in any ASCII-compatible encoding is read. Any other content, a binary file or a text in UTF-16 or UTF-32, is
read by its printable runs of four ASCII characters or more, as bytes and as UTF-16 and UTF-32 code units of either
byte order, each run a line, and its findings are marked `(binary)`. Not read: a run shorter than four characters,
text in other characters than ASCII (which no term or path is written in), and whatever a content holds compressed,
such as a PNG's compressed text chunks, a PDF's streams or an archive's members.
"""

import argparse
import hmac
import json
import os
import re
import subprocess
import sys
from collections.abc import Collection, Iterable, Iterator, Sequence
from pathlib import Path

DIGESTS = Path(__file__).resolve().parent / "confidential_digests.txt"
KEY_VARIABLE = "CONFIDENTIAL_SCAN_KEY"
# OpenMaterica's phrase: the digest file's check line is its digest under the key, so it stays as the file was made.
CHECK_PHRASE = "openmaterica confidential scan key check"

_WORD = re.compile(r"[a-z0-9]+")
_HYPHENATED = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
_KEY = re.compile(r"[0-9a-fA-F]{64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PRINTABLE = rb"[\t\n\r\x20-\x7e]"
# A printable run of ASCII in a binary content: as bytes, and as UTF-16 and UTF-32 code units of either order.
_RUNS = tuple(
    re.compile(rb"(?:" + unit + rb"){4,}")
    for unit in (
        _PRINTABLE,
        _PRINTABLE + rb"\x00",
        rb"\x00" + _PRINTABLE,
        _PRINTABLE + rb"\x00\x00\x00",
        rb"\x00\x00\x00" + _PRINTABLE,
    )
)
# Each written so that its own source does not match it: this file is scanned too.
PATHS = {
    "an absolute home directory path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "a session scratchpad path": re.compile(r"/scratch(?:pad)\b"),
    "a path into Claude Code's temporary directory": re.compile(r"/tmp/claud(?:e)"),
    "a path into the private planning directory": re.compile(r"\bplan(?:s)/"),
}
# CLAUDE.md says where the private working files are: in the planning directory beside the repository.
ALLOWED_PATHS = {"CLAUDE.md": re.compile(r"\.\./plan(?:s)/")}


class ScanError(ValueError):
    """A key, a digest file or a range this script cannot use, or a run that cannot check the names."""


def parse_key(text: str) -> bytes:
    if not _KEY.fullmatch(text.strip()):
        raise ScanError(f"a key is 64 hexadecimal digits ({KEY_VARIABLE} or --key-file)")
    return bytes.fromhex(text.strip())


def load_key(key_file: Path | None) -> bytes | None:
    """The key from `key_file`, else from the environment; None when neither gives one."""
    if key_file is not None:
        try:
            return parse_key(key_file.read_text(encoding="utf-8"))
        except OSError as error:
            raise ScanError(f"--key-file: {error}") from error
    from_environment = os.environ.get(KEY_VARIABLE, "")
    return parse_key(from_environment) if from_environment else None


def fork_pull_request() -> bool:
    """Whether this is GitHub Actions' run for a pull request from a fork: the event `pull_request`, and its payload's
    head repository not this repository (`GITHUB_REPOSITORY`; a fork deleted since has none). Any other run, a
    payload without its pull request's head among them, is not."""
    path, repository = os.environ.get("GITHUB_EVENT_PATH"), os.environ.get("GITHUB_REPOSITORY")
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request" or not path or not repository:
        return False
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pull_request = event.get("pull_request") if isinstance(event, dict) else None
    head = pull_request.get("head") if isinstance(pull_request, dict) else None
    if not isinstance(head, dict):
        return False
    head_repository = head.get("repo")
    name = head_repository.get("full_name") if isinstance(head_repository, dict) else None
    return not (isinstance(name, str) and name.lower() == repository.lower())


def normalised(term: str) -> str:
    return "-".join(_WORD.findall(term.lower()))


def digest(key: bytes, term: str) -> str:
    return hmac.digest(key, term.encode("utf-8"), "sha256").hex()


def digest_file(key: bytes, terms: Iterable[str]) -> str:
    """The text of `confidential_digests.txt` for `terms`, sorted, so that it keeps no order of the terms."""
    digests = sorted({digest(key, normalised(term)) for term in terms if normalised(term)})
    header = "# Written by scripts/check_confidential.py --digest (scripts/README.md). Do not edit by hand.\n"
    return header + f"check {digest(key, CHECK_PHRASE)}\n" + "".join(f"{line}\n" for line in digests)


def read_digests(text: str, key: bytes) -> frozenset[str]:
    """The digests of `text`, a digest file, once its check line shows that they were made with `key`."""
    digests: set[str] = set()
    check = None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("check "):
            check = line.removeprefix("check ")
        elif _DIGEST.fullmatch(line):
            digests.add(line)
        else:
            raise ScanError(f"not a digest: {line!r}")
    if check is None or not hmac.compare_digest(check, digest(key, CHECK_PHRASE)):
        raise ScanError("the key is not the one the digests were made with")
    return frozenset(digests)


def candidates(text: str) -> Iterator[tuple[int, str]]:
    """Each word, pair of adjacent words and hyphenated run of words in `text`, normalised, with its line; a
    pair split by a line break is on the second word's line."""
    previous = ""
    for number, line in enumerate(text.lower().splitlines(), 1):
        for hyphenated in _HYPHENATED.findall(line):
            yield number, hyphenated
        for word in _WORD.findall(line):
            yield number, word
            if previous:
                yield number, f"{previous}-{word}"
            previous = word


def masked(term: str) -> str:
    return f"{term[0]}..., {len(term)} characters"


class Scanner:
    def __init__(self, key: bytes | None, digests: frozenset[str]) -> None:
        self.key = key
        self.digests = digests
        self.confidential: dict[str, bool] = {}

    def is_confidential(self, candidate: str) -> bool:
        if self.key is None:
            return False
        if candidate not in self.confidential:
            self.confidential[candidate] = digest(self.key, candidate) in self.digests
        return self.confidential[candidate]

    def findings(
        self, place: str, text: str, allowed: re.Pattern[str] | None = None, only: Collection[int] | None = None
    ) -> list[str]:
        """Each private path and confidential term in `text`, as `place:line: what`, in line order; on the lines
        `only` names, when it is given."""
        found: list[tuple[int, str]] = []
        for number, line in enumerate(text.splitlines(), 1):
            checked = allowed.sub("", line) if allowed else line
            found += [(number, kind) for kind, pattern in PATHS.items() if pattern.search(checked)]
        for number, candidate in candidates(text):
            if self.is_confidential(candidate):
                found.append((number, f"a confidential term ({masked(candidate)})"))
        kept = [finding for finding in found if only is None or finding[0] in only]
        return [f"{place}:{number}: {what}" for number, what in sorted(kept, key=lambda finding: finding[0])]


def is_binary(data: bytes) -> bool:
    return b"\0" in data


def text_of(data: bytes) -> str:
    """The text the scan reads in a content (the module's docstring says what it leaves out). Of runs that
    overlap, the first and longest is kept: UTF-16 read a byte off is UTF-16 of the other order."""
    if not is_binary(data):
        return data.decode("utf-8", errors="replace")
    found = sorted((match.start(), -len(match[0]), match[0]) for run in _RUNS for match in run.finditer(data))
    runs: list[str] = []
    end = 0
    for start, negative_length, run in found:
        if start >= end:
            runs.append(run.replace(b"\0", b"").decode("ascii"))
            end = start - negative_length
    return "\n".join(runs)


def git(*arguments: str) -> bytes:
    try:
        return subprocess.run(["git", *arguments], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise ScanError(
            f"git {' '.join(arguments)}: {error.stderr.decode(errors='replace').strip()}"
        ) from error


def names_of(output: bytes) -> str:
    """git's output of paths or ref names, which are bytes: those that are not UTF-8 kept apart, so that a
    path read from it opens the same file."""
    return output.decode("utf-8", "surrogateescape")


def shown(name: str) -> str:
    """A path or ref name as a finding prints it, on one line: bytes that are not UTF-8 and control
    characters escaped."""
    text = name.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in text)


def tracked_contents() -> Iterator[tuple[str, bytes]]:
    """Each tracked file of the working tree, by its path, and its content."""
    root = Path(names_of(git("rev-parse", "--show-toplevel")).strip())
    for name in names_of(git("-C", str(root), "ls-files", "-z")).split("\0"):
        path = root / name
        if name and path.is_file():
            yield name, path.read_bytes()


def directory_contents(directory: Path) -> Iterator[tuple[str, bytes]]:
    """Each file under `directory` (a symbolic link is not followed), by its path under it, and its content."""
    if not directory.is_dir():
        raise ScanError(f"--directory: {directory} is not a directory")
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path.relative_to(directory).as_posix(), path.read_bytes()


def content_findings(
    scanner: Scanner,
    name: str,
    data: bytes,
    allowed: re.Pattern[str] | None,
    *notes: str,
    only: Collection[int] | None = None,
) -> list[str]:
    """The findings in the content `data` of `name`, its place followed by `notes` in parentheses, `binary`
    among them for a binary content, whose lines are its runs; on the lines `only` names, when it is given."""
    notes = (*notes, "binary") if is_binary(data) else notes
    return scanner.findings(f"{name} ({', '.join(notes)})" if notes else name, text_of(data), allowed, only)


def path_findings(scanner: Scanner, place: str, name: str) -> list[str]:
    """The findings in a path, which is public as its content is."""
    return scanner.findings(f"{place} {shown(name)}", name)


def blob_contents(blobs: Sequence[str]) -> list[bytes]:
    """The contents of `blobs`, by their ids, read by one `git cat-file --batch`."""
    if not blobs:
        return []
    try:
        output = subprocess.run(
            ["git", "cat-file", "--batch"], input="".join(f"{blob}\n" for blob in blobs).encode(),
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ScanError(f"git cat-file: {error.stderr.decode(errors='replace').strip()}") from error
    contents: list[bytes] = []
    at = 0
    for _ in blobs:
        end = output.index(b"\n", at)
        header = output[at:end].split()
        if len(header) != 3:
            raise ScanError(f"git cat-file: {output[at:end].decode(errors='replace')}")
        size = int(header[2])
        contents.append(output[end + 1 : end + 1 + size])
        at = end + 2 + size
    return contents


SUBMODULE = "160000"


def commit_changes(*revisions: str) -> Iterator[tuple[str, str, bytes | None, bytes | None]]:
    """Each path a commit in `revisions` (`git log`'s revisions) adds or changes, against its first parent or, for a
    root commit, the empty tree: the commit's shortened hash, the path, and its content before and after the commit
    (None where there is none, or a submodule's)."""
    empty = git("hash-object", "-t", "tree", os.devnull).decode().strip()
    for line in git("log", "--format=%H %P", "--end-of-options", *revisions).decode().splitlines():
        commit, *parents = line.split()
        against = parents[0] if parents else empty
        listed = git("diff-tree", "-r", "-z", "--no-renames", "--diff-filter=d", against, commit).split(b"\0")
        changes = [
            (listed[at].decode().lstrip(":").split(), names_of(listed[at + 1])) for at in range(0, len(listed) - 1, 2)
        ]
        wanted = [
            blob
            for (old_mode, new_mode, old, new, _), _ in changes
            for mode, blob in ((old_mode, old), (new_mode, new))
            if mode != SUBMODULE and blob.strip("0")
        ]
        contents = dict(zip(wanted, blob_contents(wanted), strict=True))
        for (_, _, old, new, _), name in changes:
            yield commit[:12], name, contents.get(old), contents.get(new)


def added_lines(before: bytes | None, after: bytes) -> set[int] | None:
    """The lines of `after` that `before` does not hold, by number, each with the line after it, where a pair of words
    split by the line break is placed; None for every line, where there was no content before or either is binary."""
    if before is None or is_binary(before) or is_binary(after):
        return None
    held = set(text_of(before).splitlines())
    added = {number for number, line in enumerate(text_of(after).splitlines(), 1) if line not in held}
    return added | {number + 1 for number in added}


def diff_findings(scanner: Scanner, *revisions: str) -> list[str]:
    """The findings in what each commit of `revisions` adds: its paths, and the lines or the contents it adds."""
    found: list[str] = []
    named: set[str] = set()
    for commit, name, before, after in commit_changes(*revisions):
        if name not in named:
            named.add(name)
            found += path_findings(scanner, f"commit {commit} path", name)
        if after is not None:
            only = added_lines(before, after)
            place, allowed = f"commit {commit} {shown(name)}", ALLOWED_PATHS.get(name)
            found += content_findings(scanner, place, after, allowed, only=only)
    return found


def commit_messages(*revisions: str) -> Iterator[tuple[str, str]]:
    """Each commit's message in `revisions`, `git log`'s arguments, by its shortened hash."""
    for entry in git("log", "-z", "--format=%H%n%B", *revisions).decode("utf-8", "replace").split("\0"):
        commit, _, message = entry.partition("\n")
        if commit:
            yield commit[:12], message


def ref_findings(scanner: Scanner, ref: str) -> list[str]:
    """The findings in a ref's name, which is public from the moment it is pushed, as `ref <name>`."""
    short = next(
        (ref.removeprefix(prefix) for prefix in ("refs/heads/", "refs/tags/") if ref.startswith(prefix)), ref
    )
    return scanner.findings(f"ref {shown(short)}", short)


def scan(
    scanner: Scanner,
    revisions: str | None,
    ref_names: Sequence[str] = (),
    diffs: str | None = None,
    public: Sequence[str] = (),
    directory: Path | None = None,
) -> list[str]:
    found: list[str] = []
    for name, data in tracked_contents() if directory is None else directory_contents(directory):
        found += path_findings(scanner, "path", name)
        found += content_findings(scanner, shown(name), data, ALLOWED_PATHS.get(name))
    published = [f"^{ref}" for ref in public]
    for commit, message in commit_messages("--end-of-options", revisions, *published) if revisions else ():
        found += scanner.findings(f"commit {commit} message", message)
    found += diff_findings(scanner, diffs, *published) if diffs else []
    for ref in ref_names:
        found += ref_findings(scanner, ref)
    return found


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find private names and paths in tracked files and commits.")
    parser.add_argument("--key-file", type=Path, help=f"the key, instead of {KEY_VARIABLE}")
    parser.add_argument("--messages", metavar="RANGE", help="scan the messages of the commits in RANGE too")
    parser.add_argument(
        "--diffs", metavar="RANGE", help="scan the paths and the lines each commit in RANGE adds too"
    )
    parser.add_argument(
        "--public",
        action="append",
        default=[],
        metavar="REF",
        help="leave out of both ranges the commits reachable from REF, a branch already public (repeatable)",
    )
    parser.add_argument(
        "--ref-name", action="append", default=[], metavar="NAME", help="scan a ref's name too (repeatable)"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        metavar="DIR",
        help="scan the files under DIR instead of the tracked tree (built pages, never committed); no other option",
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="write the digest file for the terms on standard input, one a line",
    )
    options = parser.parse_args(arguments)
    try:
        key = load_key(options.key_file)
        if options.digest:
            if key is None:
                raise ScanError("--digest needs the key")
            sys.stdout.write(digest_file(key, sys.stdin.read().splitlines()))
            return 0
        if options.directory is not None and (options.messages or options.diffs or options.ref_name or options.public):
            raise ScanError("--directory reads that directory alone: it takes none of --messages, --diffs, --public "
                            "or --ref-name")
        digests = read_digests(DIGESTS.read_text(encoding="utf-8"), key) if key else frozenset[str]()
        found = scan(
            Scanner(key, digests), options.messages, options.ref_name, options.diffs, options.public, options.directory
        )
    except ScanError as error:
        print(error, file=sys.stderr)
        return 2
    for finding in found:
        print(finding, file=sys.stderr)
    if key is None and not fork_pull_request():
        print(
            f"no key ({KEY_VARIABLE} or --key-file), so the confidential names were not checked, and a run without "
            "the key fails: only a pull request from a fork, which has no secrets, runs without it",
            file=sys.stderr,
        )
        return 2
    if key is None:
        print(
            "warning: a pull request from a fork has no secrets, so the confidential names were not checked; the "
            "private paths were, and the push after its merge scans with the key",
            file=sys.stderr,
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
