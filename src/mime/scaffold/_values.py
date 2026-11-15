"""What ``mime scaffold`` asks for, by kind: each key with what it is for, its check and its own value, and where each value
came from (``--set``, the person's ``defaults.toml``, scaffold's own, or what the walk asked).

The keys are dotted for the table they land in, as ``mime.toml`` names them (``block.id``, ``record.title``). A value is
text, except ``record.keywords`` (a list) and ``record.creators`` and ``plugin.authors`` (lists of people): ``--set`` reads
a list from commas, and people from ``family, given[, orcid]`` separated by semicolons."""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as installed_version
from typing import cast

from packaging.version import Version

from mime._terminal import plain
from mime.blocks._definition import id_problem, version_problem
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._defaults import namespace_problem
from mime.push._sharing import bare_orcid, orcid_problem
from mime.terms import licence_problem

KINDS = ("experiment", "bundle", "study", "block", "plugin")


def _key(table: str, name: str) -> str:
    """A key as ``--set`` names it: the table of ``mime.toml`` it lands in, then its own name."""
    return f"{table}.{name}"


EXPERIMENT_NAME, EXPERIMENT_LICENCE = _key("experiment", "name"), _key("experiment", "licence")
BUNDLE_LICENCE, STUDY_LICENCE = _key("bundle", "licence"), _key("study", "licence")
TITLE, DESCRIPTION = _key("record", "title"), _key("record", "description")
CREATORS, KEYWORDS = _key("record", "creators"), _key("record", "keywords")
BLOCK_ID, BLOCK_VERSION, BLOCK_LICENCE = _key("block", "id"), _key("block", "version"), _key("block", "licence")
DISTRIBUTION, NAMESPACE, PACKAGE = _key("plugin", "distribution"), _key("plugin", "namespace"), _key("plugin", "package")
PLUGIN_LICENCE, AUTHORS = _key("plugin", "licence"), _key("plugin", "authors")
#: Where a value came from.
SET, DEFAULTS, OWN, ASKED = "--set", "defaults.toml", "scaffold's own", "asked"

_RECORD: Mapping[str, str] = {
    TITLE: "the record's title, as Zenodo shows it",
    DESCRIPTION: "a paragraph shown above the card, which is the record's description (Enter: the card alone)",
    CREATORS: "each person who made it, as the record names them (everyone named has agreed to be named)",
    KEYWORDS: "searchable words, separated by commas",
}
#: Each kind's keys, in the order the walk asks them, with what each is for.
KEYS: Mapping[str, Mapping[str, str]] = {
    "experiment": {
        EXPERIMENT_NAME: "the experiment's name, which its saved form carries",
        EXPERIMENT_LICENCE: "the licence the experiment declares, which covers its saved files",
    },
    "bundle": {BUNDLE_LICENCE: "the experiment's own licence: the one it declares", **_RECORD},
    "study": {STUDY_LICENCE: "the study's own licence: the one its base experiment declares", **_RECORD},
    "block": {
        BLOCK_ID: "the block's id, <namespace>.<name>, which names its container and its record",
        BLOCK_VERSION: "the block's version, which a change to its content raises",
        BLOCK_LICENCE: "the block's own licence, for its content",
        **_RECORD,
    },
    "plugin": {
        DISTRIBUTION: "the distribution's name, as pip installs it",
        NAMESPACE: "the namespace of your entries: the word before the dot in each entry's name",
        PACKAGE: "the Python package that holds the entries",
        PLUGIN_LICENCE: "the licence of the plugin's code",
        AUTHORS: "each person who wrote the code",
    },
}
#: The keys a person's defaults hold, as the key of ``defaults.toml`` they fill: ``creators``, ``keywords`` (under
#: ``[sharing]``), ``[licences]`` ``content`` and ``code``, and ``[scaffold]`` ``namespace``.
DEFAULT_OF: Mapping[str, str] = {
    CREATORS: "creators", AUTHORS: "creators", KEYWORDS: "keywords",
    EXPERIMENT_LICENCE: "content", BLOCK_LICENCE: "content", PLUGIN_LICENCE: "code",
    NAMESPACE: "namespace",
}
LIST_KEYS = (KEYWORDS,)
PEOPLE_KEYS = (CREATORS, AUTHORS)
#: A distribution's name as PEP 508 writes it.
_DISTRIBUTION = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
_PACKAGE = re.compile(r"[a-z_][a-z0-9_]*")
#: An experiment's name, as the build reads one.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Held:
    """A value, and where it came from."""

    value: object
    source: str


def text_of(value: object) -> str:
    """A value as it is said, one line."""
    if isinstance(value, list):
        return ", ".join(text_of(each) for each in cast(list[object], value))
    if isinstance(value, dict):
        person = cast(dict[str, str], value)
        return ", ".join(part for part in (person.get("family"), person.get("given")) if part)
    return str(value)


def _people(text: str) -> list[dict[str, str]] | str:
    """People from ``family, given[, orcid]; ...``, or why not."""
    found: list[dict[str, str]] = []
    for each in (part.strip() for part in text.split(";") if part.strip()):
        words = [word.strip() for word in each.split(",")]
        if not 1 <= len(words) <= 3 or not words[0]:
            return f"{plain(repr(each))} is not `family, given[, orcid]`"
        person = {"family": words[0]}
        if len(words) > 1 and words[1]:
            person["given"] = words[1]
        if len(words) == 3 and words[2]:
            if (problem := orcid_problem(words[2])) is not None:
                return plain(problem)
            person["orcid"] = bare_orcid(words[2])
        found.append(person)
    return found


def licence_check(declared: str | None) -> Callable[[str], str | None]:
    """A licence check: an SPDX expression, and the one the experiment declares where it declares one."""
    def check(given: str) -> str | None:
        if (problem := licence_problem(given)) is not None:
            return plain(problem)
        if declared is not None and given != declared:
            return (f"the experiment declares {declared}, bound to its files: give that, or declare {plain(given)} "
                    f"where the experiment is built, exp.licence(...), and save it again")
        return None
    return check


def _plain_text(given: str) -> str | None:
    return None if given.strip() and given == given.strip() and given.isprintable() else \
        "a name is one line of printable text, with no space at its ends"


def distribution_problem(given: str) -> str | None:
    return None if _DISTRIBUTION.fullmatch(given) else \
        f"{plain(repr(given))} is not a distribution's name (letters, digits, '.', '-' and '_', starting and ending " \
        f"with a letter or digit)"


def experiment_problem(given: str) -> str | None:
    return None if _IDENTIFIER.fullmatch(given) else \
        f"{plain(repr(given))} is not an identifier (letters, digits and underscores, starting with a letter)"


def own_experiment_name(folder: str) -> str:
    """An experiment's name from its folder's: the folder's, with each run of other characters an underscore."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", folder).strip("_")
    return name if name and not name[0].isdigit() else f"experiment_{name}".rstrip("_")


def package_problem(given: str) -> str | None:
    return None if _PACKAGE.fullmatch(given) and not keyword.iskeyword(given) else \
        f"{plain(repr(given))} is not a Python package's name (lower-case letters, digits and '_', not a keyword)"


def taken_namespaces() -> dict[str, str]:
    """Every namespace a plugin may not take, each with who holds it: the first word of one of MIME's refusal codes or of
    one of MIME's registered entries (the words :func:`mime.vocabulary.register_namespace` refuses, read from the same
    two places), a namespace registered in this process, and one an installed distribution declares in its
    ``mime.entries`` entry points. Read from the registry and the environment at the time of asking, never listed here."""
    from mime.build import REFUSAL_TABLE
    from mime.registry import registered
    from mime.saved._resolve import namespace_providers
    from mime.vocabulary import MIME_NAMESPACE, versions

    taken: dict[str, str] = {}
    for first in sorted({code.split(".", 1)[0] for code in REFUSAL_TABLE}):
        taken[first] = "the first word of MIME's refusal codes"
    for record in registered():
        if record.namespace == MIME_NAMESPACE:
            taken[record.name.split(".", 1)[0]] = "the first word of MIME's entries"
    for name, point in namespace_providers().items():
        taken.setdefault(name, f"a namespace {point.dist.name if point.dist is not None else 'a package'} declares here")
    for name in versions():
        taken.setdefault(name, "a namespace registered in this environment")
    return taken


def free_namespace(stem: str) -> str:
    """A namespace near ``stem`` that nothing holds: ``lab_<stem>``, then that with a number."""
    taken = taken_namespaces()
    candidate = f"lab_{stem}"
    number = 2
    while candidate in taken:
        candidate = f"lab_{stem}{number}"
        number += 1
    return candidate


def namespace_clash(name: str) -> str | None:
    """Why a plugin may not take ``name`` as its namespace (MIME's own first word, or one this environment already holds),
    naming a free one near it; ``None`` where it is free. :func:`~mime.push._defaults.namespace_problem` checks the form."""
    held = taken_namespaces().get(name)
    if held is None:
        return None
    return (f"{plain(name)} is {held}: a plugin that takes it breaks `mime registry` for the whole environment; "
            f"try {free_namespace(name)}")


def namespace_check(given: str) -> str | None:
    """A plugin's namespace as the registry would take it: one lower-case word that nothing holds."""
    return namespace_problem(given) or namespace_clash(given)


def check_of(key: str, declared: str | None = None) -> Callable[[str], str | None]:
    """The check a text value of ``key`` is held to."""
    if key.endswith(".licence"):
        return licence_check(declared)
    return {BLOCK_ID: id_problem, BLOCK_VERSION: version_problem, NAMESPACE: namespace_check,
            DISTRIBUTION: distribution_problem, PACKAGE: package_problem,
            EXPERIMENT_NAME: experiment_problem}.get(key, _plain_text)


def parse(key: str, text: str, declared: str | None) -> object:
    """A ``--set`` value for ``key``: its text, a list, or people; refused (``scaffold.set``) where its check refuses."""
    problem: str | None = None
    value: object = text
    if key in LIST_KEYS:
        value = [word.strip() for word in text.split(",") if word.strip()]
    elif key in PEOPLE_KEYS:
        found = _people(text)
        value, problem = (found, None) if isinstance(found, list) else (None, found)
    elif key != DESCRIPTION:
        problem = check_of(key, declared)(text)
    if problem is not None:
        raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"--set {key}=...: {problem}",
                                    f"give {key} {_what(key)} with --set, or leave it out")])
    return value


def _what(key: str) -> str:
    return {CREATORS: "people as `family, given[, orcid]`, separated by semicolons",
            AUTHORS: "people as `family, given[, orcid]`, separated by semicolons",
            KEYWORDS: "words separated by commas"}.get(key, "a different value")


def sets_of(kind: str, given: Sequence[str], declared: str | None) -> dict[str, object]:
    """The ``--set key=value`` arguments read: each key one of ``kind``'s, once, its value held to its check. An unknown
    key is refused (``scaffold.set``), naming the known."""
    known = KEYS[kind]
    found: dict[str, object] = {}
    for each in given:
        key, equals, text = each.partition("=")
        if not equals:
            raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"--set {plain(each)} has no `=`",
                                        f"write --set key=value, with a key of a {kind}: {', '.join(known)}")])
        if key not in known:
            raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"--set {plain(key)} is not a key of a {kind}",
                                        f"the keys of a {kind} are {', '.join(known)}: `--set <key>=<value>`")])
        if key in found:
            raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"--set {key} is given twice",
                                        "give each key once, `--set <key>=<value>`")])
        found[key] = parse(key, text, declared)
    return found


def next_minor(installed: str) -> str:
    """The next minor release after ``installed`` (``0.3.0.dev1``: ``0.4``)."""
    release = Version(installed).release
    return f"{release[0]}.{release[1] + 1}"


def mime_range() -> str:
    """The versions of MIME a new block or plugin names: from this one to the next minor, as PEP 440 writes them."""
    installed = installed_version("mime-engine")
    return f">={installed},<{next_minor(installed)}"


def own_plugin_names(folder: str) -> dict[str, str]:
    """A plugin's distribution, package and namespace from its folder's name, where nothing else gives them."""
    distribution = folder if _DISTRIBUTION.fullmatch(folder) else re.sub(r"[^A-Za-z0-9._-]+", "-", folder).strip("-._")
    package = re.sub(r"[^a-z0-9]+", "_", distribution.lower()).strip("_") or "plugin"
    if not _PACKAGE.fullmatch(package) or keyword.iskeyword(package):
        package = f"mime_{package}"
    stem = package.removeprefix("mime_").split("_")[0]
    namespace = stem if namespace_problem(stem) is None else "mylab"
    return {"distribution": distribution or package.replace("_", "-"), "package": package, "namespace": namespace}


def person_name(person: Mapping[str, str]) -> str:
    """A person as a distribution's ``authors`` names them: ``Given Family``."""
    return " ".join(part for part in (person.get("given"), person.get("family")) if part)


__all__ = ["ASKED", "AUTHORS", "BLOCK_ID", "BLOCK_LICENCE", "BLOCK_VERSION", "BUNDLE_LICENCE", "CREATORS", "DESCRIPTION",
           "DISTRIBUTION", "EXPERIMENT_LICENCE", "EXPERIMENT_NAME", "KEYWORDS", "NAMESPACE", "PACKAGE", "PLUGIN_LICENCE",
           "STUDY_LICENCE", "TITLE", "DEFAULTS", "DEFAULT_OF", "KEYS", "KINDS", "LIST_KEYS", "OWN", "PEOPLE_KEYS", "SET", "Held",
           "check_of", "licence_check", "mime_range", "namespace_check", "namespace_clash", "own_experiment_name",
           "own_plugin_names", "parse", "person_name", "sets_of", "taken_namespaces", "text_of"]
