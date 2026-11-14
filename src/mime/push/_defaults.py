"""A person's defaults: ``defaults.toml`` in MIME's config directory (``platformdirs``, beside the credentials file), never
in a shared folder.

It holds the values of ``[record]`` a person repeats from record to record (their name, ORCID iD and affiliation as a
creator at the top, which a plugin's authors take too; the contributors, the keywords, the uploader and the funding under
``[sharing]``), the licences a new record's content and a scaffolded plugin's code take (``[licences] content``, ``code``),
and the namespace a scaffold prefills (``[scaffold] namespace``). ``mime push <folder> --init`` fills a new ``mime.toml``
from them (its licence from ``content`` where the experiment declares none), ``--setup`` offers each for a new one and asks
at its end whether to save what was set, ``mime defaults`` shows them and ``mime defaults edit`` walks them. They are held
to the checks ``[record]`` is held to for the same keys (an ORCID iD by its check digit), and a token MIME holds is never
saved in them (``defaults.file``): they go into every record's metadata. Nor do they hold licence acceptances or asset
mirrors, which keep their own files.

**Moved once from ``sharing-defaults.yaml``:** with ``defaults.toml`` absent and the old file present, the first read writes
``defaults.toml`` from it (checked, whole or not at all) and says so; the old file is kept and never read again. Where the
config directory cannot be written, the old file is read and the line says why.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import platformdirs
import yaml

from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._sharing import Reader, funding_of, keyword_list, people, uploader_of
from mime.push._toml import key_lines, value_text
from mime.saved._versions import newer_form_refusal
from mime.terms import licence_problem

type Say = Callable[[str], None]

DEFAULTS_FILE = "defaults.toml"
DEFAULTS_FORMAT = "mime.defaults/1"
#: The file the defaults were kept in before ``defaults.toml``, read once to move them.
OLD_DEFAULTS_FILE = "sharing-defaults.yaml"
_OLD_FORMAT = "mime.sharing-defaults/1"
#: The keys of ``[record]`` a person repeats from record to record: ``creators`` at the top, the rest under ``[sharing]``.
SHARING_KEYS = ("contributors", "keywords", "uploader", "funding")
DEFAULT_KEYS = ("creators", *SHARING_KEYS)
#: ``[licences]``: a scaffolded plugin's licence (``code``), and a new record's content's (``content``).
LICENCE_KEYS = ("code", "content")
#: A namespace, as the registry's: one lower-case word, and not ``mime``.
_NAMESPACE = re.compile(r"[a-z][a-z0-9_]*")
_TOP = ("format", "creators", "sharing", "licences", "scaffold")

_HEAD = ("# Your own defaults for MIME: the values you repeat from record to record, which `mime push <folder> --init`\n"
         "# and `--setup` fill a new mime.toml from, the licences a new record's content and a scaffolded plugin's code\n"
         "# take, and the namespace a scaffold starts from. `mime defaults` shows them, `mime defaults edit` changes\n"
         "# them, or edit this file by hand. Never a token, a licence acceptance or an asset mirror.\n")


@dataclass(frozen=True)
class Defaults:
    """A person's defaults as read: the ``[record]`` values by key (only those that hold one), ``[licences]`` by key,
    and the namespace."""

    values: Mapping[str, object] = field(default_factory=lambda: cast(Mapping[str, object], {}))
    licences: Mapping[str, str] = field(default_factory=lambda: cast(Mapping[str, str], {}))
    namespace: str | None = None

    def empty(self) -> bool:
        return not self.values and not self.licences and self.namespace is None


def defaults_path() -> Path:
    """Where a person's defaults live: MIME's config directory, never a shared folder."""
    return Path(platformdirs.user_config_dir("mime")) / DEFAULTS_FILE


def old_defaults_path() -> Path:
    return defaults_path().with_name(OLD_DEFAULTS_FILE)


def _filled(value: object) -> bool:
    return value not in (None, "", [], {})


def namespace_problem(text: str) -> str | None:
    """Why ``text`` is not a namespace (one lower-case word, as the registry's, and not ``mime``), or ``None``."""
    if _NAMESPACE.fullmatch(text) and text != "mime":
        return None
    return f"{plain(text)} is not a namespace: one lower-case word (letters, digits, _), starting with a letter, not mime"


def checked(held: Mapping[str, object], path: Path) -> tuple[Defaults, list[Refusal]]:
    """``defaults.toml``'s values (``held``), each key held to its check; every problem as a refusal."""
    top = Reader(path, defaults=True, table="")
    top.mapping(held, "", _TOP)
    if held.get("format") != DEFAULTS_FORMAT:
        top.problem("format", f"is {plain(repr(held.get('format')))}, and this MIME reads {DEFAULTS_FORMAT}",
                    f'format = "{DEFAULTS_FORMAT}"')
    people(top, held, "creators")
    values: dict[str, object] = {key: held[key] for key in ("creators",) if _filled(held.get(key))}
    sharing = Reader(path, defaults=True, table="sharing")
    table = sharing.mapping(held.get("sharing", {}), "", SHARING_KEYS)
    if table is not None:
        people(sharing, table, "contributors")
        keyword_list(sharing, table)
        uploader_of(sharing, table)
        funding_of(sharing, table)
        values.update({key: table[key] for key in SHARING_KEYS if _filled(table.get(key))})
    named = Reader(path, defaults=True, table="licences")
    licences = named.mapping(held.get("licences", {}), "", LICENCE_KEYS) or {}
    for key in LICENCE_KEYS:
        if (licence := named.text(licences, key, "")) is not None and (problem := licence_problem(licence)) is not None:
            named.problem(key, plain(problem), f'{key} = "CC-BY-4.0"')
    scaffold = Reader(path, defaults=True, table="scaffold")
    held_scaffold = scaffold.mapping(held.get("scaffold", {}), "", ("namespace",)) or {}
    namespace = scaffold.text(held_scaffold, "namespace", "")
    if namespace is not None and (problem := namespace_problem(namespace)) is not None:
        scaffold.problem("namespace", problem, 'namespace = "labx"')
    refusals = [*top.refusals, *sharing.refusals, *named.refusals, *scaffold.refusals]
    return Defaults(values, {key: cast(str, licences[key]) for key in LICENCE_KEYS if _filled(licences.get(key))},
                    namespace), refusals


def _read(path: Path) -> Defaults:
    from mime.push._manifest import read_toml

    _, held = read_toml(path)
    if (newer := newer_form_refusal(held.get("format"), "defaults", (DEFAULTS_FORMAT,), path.name)) is not None:
        raise BuildRefused([newer])
    found, refusals = checked(held, path)
    if refusals:
        raise BuildRefused(refusals)
    return found


def _old(path: Path) -> Defaults:
    """The defaults ``sharing-defaults.yaml`` holds, checked as ``defaults.toml``'s are."""
    from mime.push._manifest import toml_shaped

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise BuildRefused([Refusal("defaults.file", RefusalClass.USER,
                                    f"{path} is not YAML that reads ({type(error).__name__}), so your defaults cannot "
                                    f"be moved from it", "correct it by hand, or delete it and save them again from "
                                                         "`mime defaults edit`")]) from None
    held = cast(dict[str, object], toml_shaped(cast(object, loaded))) if isinstance(loaded, dict) else {}
    unknown = [plain(str(key)) for key in held if key not in ("format", *DEFAULT_KEYS)]
    as_toml: dict[str, object] = {"format": DEFAULTS_FORMAT if held.get("format") == _OLD_FORMAT else held.get("format"),
                                  **{key: held[key] for key in ("creators",) if key in held},
                                  "sharing": {key: held[key] for key in SHARING_KEYS if key in held}}
    found, refusals = checked({key: value for key, value in as_toml.items() if value is not None}, path)
    refusals += [Refusal("defaults.file", RefusalClass.USER, f"{path.name}, {key}: is not a key it holds",
                         f"delete it from {path}, or delete the file and walk your defaults afresh, `mime defaults "
                         f"edit`") for key in unknown]
    if refusals:
        raise BuildRefused(refusals)
    return found


def read_all(say: Say | None = None) -> Defaults:
    """The person's defaults, checked; none where none are saved. A file that does not read, or holds a value its
    checks refuse, is refused with every reason at once (``defaults.file``, ``push.creators``). With ``defaults.toml``
    absent and ``sharing-defaults.yaml`` present, they are moved once, and ``say`` is told."""
    path, old = defaults_path(), old_defaults_path()
    if path.is_file():
        return _read(path)
    if not old.is_file():
        return Defaults()
    found = _old(old)
    try:
        _write(path, found)
    except OSError as error:
        if say is not None:
            say(f"read your sharing defaults from {old}: {path} could not be written ({type(error).__name__})")
        return found
    if say is not None:
        say(f"moved your sharing defaults from {old} to {path}; {old.name} is kept and no longer read: you may "
            f"delete it")
    return found


def read_defaults(say: Say | None = None) -> dict[str, object]:
    """The ``[record]`` values of the person's defaults, by key (only those that hold one)."""
    return dict(read_all(say).values)


def defaults_text(found: Defaults) -> str:
    """The file that holds ``found``: the creators at the top, each table only where it holds a value."""
    lines = [f'format = "{DEFAULTS_FORMAT}"']
    if "creators" in found.values:
        lines += key_lines("creators", found.values["creators"])
    for table, held in (("sharing", {key: found.values[key] for key in SHARING_KEYS if key in found.values}),
                        ("licences", dict(found.licences)),
                        ("scaffold", {"namespace": found.namespace} if found.namespace else {})):
        if held:
            lines += ["", f"[{table}]", *(line for key, value in held.items() for line in key_lines(key, value))]
    return _HEAD + "\n".join(lines) + "\n"


def _write(path: Path, found: Defaults) -> Path:
    """Write ``found`` at ``path``, whole or not at all, once its text reads back as it and holds no token MIME stores."""
    from mime.credentials import held_tokens

    text = defaults_text(found)
    again, refusals = checked(tomllib.loads(text), path)
    if refusals or again != found:
        raise BuildRefused(refusals or [Refusal("defaults.file", RefusalClass.INTERNAL,
                                                f"{path}: the defaults written do not read back as they were given",
                                                "report it: nothing was written")])
    if any(each.occurs_in(text.encode("utf-8")) for each in held_tokens()):
        raise BuildRefused([Refusal("defaults.file", RefusalClass.USER,
                                    f"{path}: would hold a token MIME stores, which never goes into a record's metadata; "
                                    f"nothing was written", "remove the token from the value it is in, and save them "
                                                            "again, `mime defaults edit`")])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return path


def write_defaults(values: Mapping[str, object] | None = None, *, licences: Mapping[str, str] | None = None,
                   namespace: str | None = None, whole: bool = False) -> Path:
    """Save the person's defaults, checked first (a value its checks refuse, or a token MIME holds, is refused and
    nothing is written): ``values`` (the keys of :data:`DEFAULT_KEYS` among them) in place of the saved ``[record]``
    values, ``licences`` and ``namespace`` in place of theirs where given, the rest as saved; ``whole`` writes exactly
    what is given."""
    held = Defaults() if whole else read_all()
    kept = {key: value for key, value in (values or {}).items() if key in DEFAULT_KEYS and _filled(value)}
    return _write(defaults_path(), Defaults(
        kept if whole or values is not None else held.values,
        {key: value for key, value in (licences or {}).items() if value} if whole or licences is not None
        else held.licences,
        namespace if whole or namespace is not None else held.namespace))


def defaults_lines(say: Say | None = None) -> list[str]:
    """What ``mime defaults`` (and ``mime push --defaults``) says: where the defaults live, and each, as the file holds
    it, one a line."""
    path, old = defaults_path(), old_defaults_path()
    found = read_all(say)
    after = ([f"{old} is no longer read (your defaults are in {DEFAULTS_FILE}): you may delete it"]
             if old.is_file() and path.is_file() else [])
    if found.empty():
        return [f"no defaults are saved; they would be in {path}",
                "`mime defaults edit` asks for them, and `mime push <folder> --setup` asks at its end whether to save "
                "what it set; or write that file by hand: creators at the top, [sharing] contributors, keywords, "
                "uploader and funding, [licences] code and content, [scaffold] namespace", *after]
    lines = [f"your defaults, in {path} (`mime push <folder> --init` and `--setup` fill a new mime.toml from them; "
             f"`mime defaults edit` changes them):"]
    lines += [f"  creators = {value_text(found.values['creators'])}"] if "creators" in found.values else []
    lines += [f"  [sharing] {key} = {value_text(found.values[key])}" for key in SHARING_KEYS if key in found.values]
    lines += [f"  [licences] {key} = {value_text(value)}" for key, value in found.licences.items()]
    lines += [f"  [scaffold] namespace = {value_text(found.namespace)}"] if found.namespace else []
    return [*lines, *after]


__all__ = ["DEFAULTS_FILE", "DEFAULTS_FORMAT", "DEFAULT_KEYS", "LICENCE_KEYS", "OLD_DEFAULTS_FILE", "SHARING_KEYS",
           "Defaults", "checked", "defaults_lines", "defaults_path", "defaults_text", "namespace_problem", "old_defaults_path",
           "read_all", "read_defaults", "write_defaults"]
