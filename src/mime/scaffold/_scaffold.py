"""``mime scaffold <kind> <path>``: a new folder of a kind, made from templates, or ``mime.toml`` added to an existing one.

The five kinds are an experiment (``experiment.py``, saved into ``saved/``), a bundle and a study (the ``mime.toml`` of a
saved or study folder, written by :func:`mime.push._draft.new_manifest`, the writer ``mime push --init`` uses too), a block
(``mime.toml``, ``block.mime.yaml``, ``cases/``) and a plugin (``pyproject.toml``, its entries, a test). Every value of every
file comes from ``--set``, else the person's ``defaults.toml``, else scaffold's own, and the walk asks only where none
of ``--set``, ``--yes`` or ``--dry-run`` is given: Enter keeps the value shown. Scaffold never overwrites (``scaffold.exists``),
asks nothing under ``--yes`` and chooses no licence and no namespace (where none is given it writes a marker every check
refuses, ``mime.terms.PLACEHOLDER_LICENCE`` and ``PLACEHOLDER_NAMESPACE``), searches everything it writes for a token MIME holds
before writing a byte, and writes nothing under ``--dry-run``, which draws the tree and where each value came from."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._said import DONE, QUESTION_CONFIRM, Said
from mime._shell import quoted
from mime._terminal import plain
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.credentials import held_tokens
from mime.push._defaults import DEFAULT_KEYS, Defaults, read_all, write_defaults
from mime.push._manifest import MANIFEST_FILE, OLD_FILE, STUDY, record_lines
from mime.push._setup import Walk, asked, filled, folder_facts
from mime.terms import PLACEHOLDER_LICENCE, PLACEHOLDER_NAMESPACE
from mime.push._toml import value_text
from mime.scaffold._templates import fill
from mime.scaffold._values import (
    ASKED,
    AUTHORS,
    BLOCK_ID,
    BLOCK_LICENCE,
    BLOCK_VERSION,
    BUNDLE_LICENCE,
    CREATORS,
    DESCRIPTION,
    DISTRIBUTION,
    EXPERIMENT_LICENCE,
    EXPERIMENT_NAME,
    KEYWORDS,
    NAMESPACE,
    PACKAGE,
    PLUGIN_LICENCE,
    STUDY_LICENCE,
    TITLE,
    DEFAULT_OF,
    DEFAULTS,
    KEYS,
    KINDS,
    OWN,
    PEOPLE_KEYS,
    SET,
    Held,
    check_of,
    mime_range,
    namespace_clash,
    own_experiment_name,
    own_plugin_names,
    sets_of,
    text_of,
)

type Say = Callable[[str], None]
type Typed = Callable[[str], str]
type Ask = Callable[[str], bool]
#: What ``--yes`` writes where a namespace or a licence is wanted that neither ``--set`` nor the person's defaults gave: a
#: marker every check refuses (:data:`mime.terms.PLACEHOLDER_NAMESPACE`), never a name or a licence chosen for them.
OWN_NAMESPACE = PLACEHOLDER_NAMESPACE
OWN_CODE_LICENCE = PLACEHOLDER_LICENCE
#: The keys a walk keeps the value shown of at Enter: a record always has each.
_NEEDED = frozenset({EXPERIMENT_NAME, EXPERIMENT_LICENCE, BUNDLE_LICENCE, STUDY_LICENCE, BLOCK_ID,
                     BLOCK_VERSION, BLOCK_LICENCE, TITLE, DISTRIBUTION, NAMESPACE,
                     PACKAGE, PLUGIN_LICENCE})
#: The part of a ``[record]`` a person's defaults and ``--set`` give a new one, as ``mime.toml`` names the keys.
_PLACEHOLDERS = (PLACEHOLDER_LICENCE, PLACEHOLDER_NAMESPACE)
_RECORD_KEYS = {TITLE: "title", DESCRIPTION: "description", CREATORS: "creators",
                KEYWORDS: "keywords"}


@dataclass(frozen=True)
class Scaffolded:
    """What a scaffold wrote (or, under ``--dry-run``, would): the folder, each file's path below it with its note, and
    each value with where it came from."""

    kind: str
    root: Path
    files: tuple[tuple[str, str], ...]
    values: Mapping[str, Held]
    written: bool


@dataclass(frozen=True)
class _Plan:
    """The files of one scaffold, each by its path below the folder: its text, or the file it is copied from."""

    files: dict[str, str | Path]
    notes: dict[str, str]
    #: Directories made empty, and the files made after the others are written (``saved/``).
    empty: tuple[str, ...] = ()
    after: str | None = None


def _walks(yes: bool, dry_run: bool) -> bool:
    """Whether scaffold asks: not under ``--yes`` or ``--dry-run``."""
    return not (yes or dry_run)


def _exists(kind: str, root: Path, given: str) -> None:
    """Never overwrite: a path that exists (for a bundle or a study, a folder that holds ``mime.toml`` or ``sharing.yaml``)
    is refused, naming what edits one."""
    holds = [name for name in (MANIFEST_FILE, OLD_FILE) if (root / name).is_symlink() or (root / name).exists()]
    if kind in ("bundle", "study"):
        if not holds:
            return
        why = f"{quoted(given)} already holds {' and '.join(holds)}"
    else:
        if not (root.is_symlink() or root.exists()):
            return
        why = (f"{quoted(given)} exists and holds {' and '.join(holds)}" if holds else f"{quoted(given)} exists")
    message = f"{why}: scaffold never overwrites"
    if kind in ("bundle", "study"):
        raise BuildRefused([Refusal("scaffold.exists", RefusalClass.USER, message,
                                    f"`mime push {quoted(given)} --setup` edits its mime.toml in place")])
    edit = f"; `mime push {quoted(given)} --setup` edits its mime.toml in place" if holds else ""
    raise BuildRefused([Refusal("scaffold.exists", RefusalClass.USER, message,
                                f"scaffold into another name, `mime scaffold {kind} <new path>`{edit}")])


def _declared_licence(source: Path) -> str | None:
    """The licence the saved experiment in ``source`` declares, ``None`` where its document declares none."""
    from mime.saved import read_folder

    licence = read_folder(source, absent_referenced=True).document.get("licence")
    return str(licence) if licence else None


def _no_terminal(kind: str, given: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "scaffold.no-terminal", RefusalClass.USER,
        "scaffold asks for each value one at a time, and there is no terminal to ask in; nothing was written",
        f"use --yes with --set key=value for the keys you need (`mime scaffold {kind} {quoted(given)} --yes "
        f"--set <key>=<value>`), or run it in a terminal; --dry-run shows what --yes would write")])


def _own(kind: str, root: Path, defaults: Defaults, facts: tuple[str, str, str | None] | None,
         declared: str | None) -> dict[str, Held]:
    """The value of each key where nothing is set: the person's default where they have one, scaffold's own else."""
    content = defaults.licences.get("content")
    fallback = Held(content, DEFAULTS) if content else Held(PLACEHOLDER_LICENCE, OWN)
    found: dict[str, Held] = {}
    if kind == "experiment":
        found = {EXPERIMENT_NAME: Held(own_experiment_name(root.name), OWN), EXPERIMENT_LICENCE: fallback}
    elif kind in ("bundle", "study"):
        assert facts is not None
        found = {f"{kind}.licence": Held(declared, OWN) if declared else fallback, TITLE: Held(facts[1], OWN)}
    elif kind == "block":
        namespace = Held(defaults.namespace, DEFAULTS) if defaults.namespace else Held(OWN_NAMESPACE, OWN)
        name = re.sub(r"[^a-z0-9_]+", "_", root.name.lower()).strip("_")
        name = name if re.match(r"[a-z]", name) else f"block_{name}".rstrip("_")
        found = {BLOCK_ID: Held(f"{namespace.value}.{name}", namespace.source), BLOCK_VERSION: Held("0.1.0", OWN),
                 BLOCK_LICENCE: Held(declared, OWN) if declared else fallback, TITLE: Held(root.name, OWN)}
    else:
        names = own_plugin_names(root.name)
        found = {DISTRIBUTION: Held(names["distribution"], OWN), PACKAGE: Held(names["package"], OWN),
                 NAMESPACE: (Held(defaults.namespace, DEFAULTS) if defaults.namespace
                                      else Held(names["namespace"], OWN)),
                 PLUGIN_LICENCE: (Held(defaults.licences["code"], DEFAULTS) if "code" in defaults.licences
                                    else Held(OWN_CODE_LICENCE, OWN))}
    if kind in ("bundle", "study", "block"):
        found[DESCRIPTION] = Held("", OWN)
    for key, short in ((CREATORS, "creators"), (KEYWORDS, "keywords"),
                       (AUTHORS, "creators")):
        if key in KEYS[kind] and filled(defaults.values.get(short)):
            found[key] = Held(defaults.values[short], DEFAULTS)
    return found


def _ask(walk: Walk, key: str, head: str, declared: str | None, needed: str | None = None) -> object:
    """A new value of ``key``, asked as ``mime push --setup`` asks the fields it shares; ``None`` where none was given."""
    if key in PEOPLE_KEYS:
        return asked(walk, "creators", head, None)
    if key == KEYWORDS:
        return asked(walk, "keywords", head, None)
    if key.endswith(".licence"):
        return asked(walk, "licence", head, declared)
    check = None if key == DESCRIPTION else check_of(key, declared)
    return walk.value(f"{head}\n{key}: ", check, needed=needed)


def _walked(kind: str, walk: Walk, current: Mapping[str, Held], sets: Mapping[str, object],
            declared: str | None) -> dict[str, Held]:
    """Every key of ``kind`` in the walk's order: ``--set``'s as given, the others asked (Enter keeps the value shown)."""
    chosen: dict[str, Held] = {}
    for key, why in KEYS[kind].items():
        if key in sets:
            chosen[key] = Held(sets[key], SET)
            continue
        head = f"{key}: {why}"
        held = current.get(key)
        offered, refused = held, None
        if held is not None and held.source == OWN and any(each in str(held.value) for each in _PLACEHOLDERS):
            offered = None  # a marker is never offered as a value to keep: the person is asked
        if key == NAMESPACE and held is not None and (refused := namespace_clash(str(held.value))) is not None:
            head, offered = f"{head}\nthe value found, {text_of(held.value)}, cannot be used: {refused}", None
        if offered is not None and filled(offered.value) and walk.keep(head, offered.value):
            chosen[key] = offered
            continue
        given = _ask(walk, key, "" if offered is not None and filled(offered.value) else head, declared, needed=refused)
        if given is not None:
            chosen[key] = Held(given, ASKED)
        elif held is not None and key in _NEEDED:
            chosen[key] = held
    return chosen


def _values(kind: str, root: Path, defaults: Defaults, facts: tuple[str, str, str | None] | None, sets: Mapping[str, object],
            walk: Walk | None, declared: str | None) -> dict[str, Held]:
    """The value of each key, with its source: ``--set``, else the person's defaults, else scaffold's own, and under a
    walk each asked in turn."""
    current = _own(kind, root, defaults, facts, declared)
    if walk is not None:
        chosen = _walked(kind, walk, current, sets, declared)
    else:
        chosen = {key: Held(sets[key], SET) if key in sets else current[key] for key in KEYS[kind]
                  if key in sets or key in current}
    return chosen


def _record(held: Mapping[str, Held], defaults: Defaults) -> dict[str, object]:
    """The ``[record]`` values a ``mime.toml`` is written with: the person's defaults, and the keys set or asked over them."""
    found = dict(defaults.values)
    for key, short in _RECORD_KEYS.items():
        if key in held and held[key].source in (SET, ASKED):
            found[short] = held[key].value
    return found


def _title(value: object) -> str:
    """A name as a document's heading may hold it: one line, with no quote or backslash."""
    return re.sub(r"[^\w .,:;()/-]+", "", str(value)).strip() or "a MIME experiment"


def _bundle(kind: str, root: Path, facts: tuple[str, str, str | None], held: Mapping[str, Held], defaults: Defaults,
            say: Say) -> _Plan:
    from mime.push._draft import new_manifest

    given: dict[str, object] = {short: held[key].value for key, short in _RECORD_KEYS.items()
                                if key in held and held[key].source in (SET, ASKED)}
    if held[f"{kind}.licence"].source in (SET, ASKED):
        given["licence"] = held[f"{kind}.licence"].value
    text = new_manifest(root, facts, say, given)
    what = "[study] licence and [record]" if kind == STUDY else "[bundle] licence and [record]"
    return _Plan({MANIFEST_FILE: text}, {MANIFEST_FILE: f"{what}: the record's metadata, `mime push --setup` edits it"})


def _experiment(held: Mapping[str, Held]) -> _Plan:
    name, licence = str(held[EXPERIMENT_NAME].value), held[EXPERIMENT_LICENCE]
    lines = ("    # The licence the saved files are shared under: it is bound to them when the experiment is saved.\n"
             + (f"    experiment.licence({licence.value!r})" if licence.source != OWN or licence.value != PLACEHOLDER_LICENCE
                else "    # Not chosen: a licence is yours to choose, and nothing is published without one. Write it here, an\n"
                     "    # SPDX licence expression, such as CC-BY-4.0 or MIT OR Apache-2.0:\n"
                     "    # experiment.licence(\"<your licence>\")"))
    text = fill("experiment", "experiment.py.txt", {"title": _title(name), "name": repr(name), "licence_lines": lines})
    return _Plan({"experiment.py": text},
                 {"experiment.py": "a minimal experiment over MIME's toy entries: edit it, then `python experiment.py`",
                  "saved/": "the saved experiment `build().save()` writes: what `mime verify`, `run` and `pack` read"},
                 after="saved/")


def _block(root: Path, held: Mapping[str, Held], defaults: Defaults, source: Path | None, sockets: Mapping[str, str],
           ) -> _Plan:
    from mime.blocks import fragment_from

    drafted = fragment_from(source, **sockets) if source is not None else None
    record = _record(held, defaults)
    title = str(held[TITLE].value)
    table = (drafted.sockets_table() if drafted is not None else
             "# What a user binds, by kind (structure, medium, body, apparatus). For example:\n"
             "# [sockets]\n# medium = { kind = \"medium\" }")
    toml = fill("block", "mime.toml.txt", {
        "title": str(held[BLOCK_ID].value), "id": value_text(held[BLOCK_ID].value),
        "version": value_text(held[BLOCK_VERSION].value), "licence": value_text(held[BLOCK_LICENCE].value),
        "mime_range": value_text(mime_range()), "sockets": table,
        "record": "\n".join(record_lines(record, title))})
    fragment = fill("block", "block.mime.yaml.txt", {})
    notes = {MANIFEST_FILE: "[block], [requires], [sockets], [cases], [record]: `mime check` lists what is still to write",
             "block.mime.yaml": "the declarations: empty, its comments name what it may hold",
             "cases/": "where the block's cases go, which `mime verify` runs"}
    files: dict[str, str | Path] = {MANIFEST_FILE: toml, "block.mime.yaml": fragment}
    if drafted is not None and source is not None:
        files["block.mime.yaml"] = ("# Written by `mime scaffold block --from`, from a saved experiment: its sockets are "
                                    "declared in mime.toml's [sockets].\n" + drafted.text)
        notes["block.mime.yaml"] = ("the declarations of the saved experiment" +
                                    (f", its {', '.join(sockets)} as sockets" if sockets else ""))
        for file in drafted.files:
            files[file] = source / file
            notes[file] = "a file the fragment reads, copied from the saved experiment"
    return _Plan(files, notes, empty=("cases",))


def _plugin(held: Mapping[str, Held], defaults: Defaults) -> _Plan:
    from mime.scaffold._plugin import plugin_files

    files, notes = plugin_files(held, defaults)
    return _Plan(dict(files), notes)


def _a(kind: str) -> str:
    return f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"


def _tree(root: str, files: Mapping[str, str], empty: Sequence[str] = ()) -> list[str]:
    """``files`` (each path with its note) drawn as a tree under ``root``, ASCII so that a pipe reads it as a terminal does."""
    nested: dict[str, object] = {}
    for path in [*files, *(f"{each}/" for each in empty if f"{each}/" not in files)]:
        node = nested
        parts = [part for part in path.split("/") if part]
        for index, part in enumerate(parts):
            last = index == len(parts) - 1 and not path.endswith("/")
            node = cast(dict[str, object], node.setdefault(part if last else f"{part}/", {}))
    notes = dict(files)

    def draw(node: Mapping[str, object], prefix: str, at: str) -> list[str]:
        lines: list[str] = []
        names = sorted(node, key=lambda name: (not name.endswith("/"), name))
        for index, name in enumerate(names):
            last = index == len(names) - 1
            path = f"{at}{name}"
            note = notes.get(path, "")
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{name}" + (f"   {note}" if note else ""))
            lines += draw(cast(Mapping[str, object], node[name]), prefix + ("    " if last else "|   "), path)
        return lines

    return [f"{root}/", *draw(nested, "", "")]


def _tokens(plan: _Plan, held: Mapping[str, Held]) -> None:
    """Refuse (``scaffold.set``) before a byte is written if any file would hold a token MIME stores; name the value."""
    tokens = held_tokens()
    if not tokens:
        return
    texts = {path: content.encode("utf-8") for path, content in plan.files.items() if isinstance(content, str)}
    for path, data in texts.items():
        if any(each.occurs_in(data) for each in tokens):
            by_key = [key for key, value in held.items()
                      if any(each.occurs_in(text_of(value.value).encode("utf-8")) for each in tokens)]
            where = f" (in the value of {', '.join(by_key)})" if by_key else ""
            message = (f"{path} would hold a token MIME stores{where}, which never goes into a file that is shared; "
                       f"nothing was written")
            if any(held[key].source == SET for key in by_key):
                raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, message,
                                            "remove the token from the value you gave with --set, and run scaffold again")])
            if any(held[key].source == DEFAULTS for key in by_key):
                raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, message,
                                            "remove the token from your defaults (`mime defaults` shows them), and run "
                                            "scaffold again")])
            raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, message,
                                        "the token is in the folder's name or a value taken from it: `mime scaffold <kind> "
                                        "<another folder name>`")])


def _written(kind: str, root: Path, plan: _Plan) -> None:
    """Write the plan: each file created, never replaced; the folder removed again if a step fails after it was made."""
    made = kind not in ("bundle", "study")
    if made:
        root.mkdir(parents=True)
    written: list[Path] = []
    try:
        for path, content in plan.files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "xb") as handle:
                written.append(target)
                handle.write(content.encode("utf-8") if isinstance(content, str) else content.read_bytes())
        for each in plan.empty:
            (root / each).mkdir(exist_ok=True)
        if plan.after == "saved/":
            _save_experiment(root, cast(str, plan.files["experiment.py"]))
    except BaseException:
        if made:
            shutil.rmtree(root, ignore_errors=True)
        else:
            for each in written:
                each.unlink(missing_ok=True)
        raise


def _save_experiment(root: Path, text: str) -> None:
    """Run ``build()`` of the experiment just written and save it into ``saved/``, as running the file does."""
    from mime.experiment import Experiment

    namespace: dict[str, object] = {"__name__": "mime_scaffold"}
    exec(compile(text, str(root / "experiment.py"), "exec"), namespace)  # noqa: S102 - the template just filled, MIME's own
    cast(Callable[[], Experiment], namespace["build"])().save(root / "saved")


def _next(kind: str, given: str, held: Mapping[str, Held]) -> list[tuple[str, str]]:
    path = quoted(given)
    if kind == "experiment":
        saved = quoted(f"{given}/saved")
        return [("check it", f"mime verify {saved}"), ("run it", f"mime run {saved} --duration 1s --out {quoted(given + '_run')}"),
                ("share it", f"mime scaffold bundle {saved}")]
    if kind in ("bundle", "study"):
        return [("check it", f"mime check {path}"), ("edit it", f"mime push {path} --setup")]
    if kind == "block":
        return [("check it", f"mime check {path}"), ("see it", f"mime show {path}")]
    return [("install it", f"pip install -e '{given}[test]'"), ("test it", f"cd {path} && JAX_PLATFORMS=cpu pytest"),
            ("list it", f"mime registry | grep {held[NAMESPACE].value}")]


def _sources(held: Mapping[str, Held]) -> list[str]:
    lines: list[str] = []
    for source in (SET, DEFAULTS, ASKED, OWN):
        keys = [key for key, value in held.items() if value.source == source]
        if keys:
            lines.append(f"  from {source if source != ASKED else 'your answers'}: {', '.join(keys)}")
    return lines


def _offer(held: Mapping[str, Held], ask: Ask, say: Say) -> None:
    """Ask whether to save what the walk was told as the person's defaults: the keys ``defaults.toml`` holds, where they
    differ from what it holds."""
    current = read_all()
    values: dict[str, object] = {}
    licences = dict(current.licences)
    namespace = current.namespace
    changed: list[str] = []
    for key, value in held.items():
        if value.source != ASKED or (key not in DEFAULT_OF and key != BLOCK_ID):
            continue
        which = DEFAULT_OF.get(key, "namespace")
        if which in DEFAULT_KEYS:
            if current.values.get(which) != value.value:
                values[which] = value.value
                changed.append(which)
        elif which in ("content", "code"):
            if current.licences.get(which) != value.value:
                licences[which] = str(value.value)
                changed.append(f"licences.{which}")
        else:
            found = str(value.value).partition(".")[0] if key == BLOCK_ID else str(value.value)
            if current.namespace != found:
                namespace = found
                changed.append("namespace")
    if not changed or not ask(Said(f"Save these as your defaults ({', '.join(changed)}), which scaffold and --setup fill "
                                   f"a new file from?", QUESTION_CONFIRM)):
        return
    saved = write_defaults({**current.values, **values}, licences=licences, namespace=namespace)
    say(Said(f"saved your defaults in {saved}: `mime defaults` shows them", DONE))


def scaffold(kind: str, path: str | os.PathLike[str], *, yes: bool = False, sets: Sequence[str] = (),
             dry_run: bool = False, source: str | os.PathLike[str] | None = None, sockets: Sequence[str] = (),
             typed: Typed | None = None, ask: Ask | None = None, say: Say = print) -> Scaffolded:
    """Scaffold a ``kind`` (one of :data:`~mime.scaffold._values.KINDS`) at ``path``: a new folder, or for a bundle and a study
    the saved or study folder it adds ``mime.toml`` to. ``yes`` or ``dry_run`` ask nothing; otherwise ``typed`` and ``ask``
    are the questions' ways (none: ``scaffold.no-terminal``). ``sets`` are ``key=value`` words; ``source`` and
    ``sockets`` (``name=thing``) are a block's ``--from`` and ``--socket``. Refused before anything is written."""
    given = os.fspath(path)
    if kind not in KINDS:
        raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"{plain(repr(kind))} is not a kind scaffold writes",
                                    f"the kinds are {', '.join(KINDS)}: `mime scaffold --help`")])
    if (source is not None or sockets) and kind != "block":
        raise BuildRefused([Refusal(
            "scaffold.set", RefusalClass.USER,
            f"--from and --socket write a block's fragment from a saved experiment, and {given!r} is scaffolded as a {kind}",
            f"leave them out, or scaffold a block: `mime scaffold block {quoted(given)} --from <saved folder> --socket "
            f"<name>=<thing>`")])
    if sockets and source is None:
        raise BuildRefused([Refusal(
            "scaffold.set", RefusalClass.USER,
            "--socket names a thing of the saved experiment that becomes a socket, and no --from names one",
            "add --from <saved folder>")])
    root = Path(os.path.abspath(given))
    _exists(kind, root, given)
    walking = _walks(yes, dry_run)
    if walking and (typed is None or ask is None):
        raise _no_terminal(kind, given)
    facts = None
    if kind in ("bundle", "study"):
        facts = folder_facts(root)
        if facts[0] != kind:
            raise BuildRefused([Refusal(
                "manifest.kind", RefusalClass.USER,
                f"a {kind} was asked for, and {quoted(given)} is a {facts[0]} folder: its mime.toml would name "
                f"[{kind}], which a {facts[0]} folder's cannot",
                f"scaffold what it is, `mime scaffold {facts[0]} {quoted(given)}`")])
    declared = facts[2] if facts is not None else None
    if kind == "block" and source is not None:
        declared = _declared_licence(Path(os.fspath(source)))
    chosen_sockets: dict[str, str] = {}
    for each in sockets:
        name, equals, thing = each.partition("=")
        if not equals or not name or not thing:
            raise BuildRefused([Refusal("scaffold.set", RefusalClass.USER, f"--socket {plain(each)} is not name=thing",
                                        "write --socket <socket's name>=<the saved experiment's thing it stands for>")])
        chosen_sockets[name] = thing
    explicit = sets_of(kind, sets, declared)
    defaults = read_all(say)
    walk = Walk(cast(Typed, typed)) if walking else None
    if walk is not None:
        say(f"Scaffolding {_a(kind)} in {quoted(given)}, one field at a time: Enter skips a field, or keeps the value shown.")
    held = _values(kind, root, defaults, facts, explicit, walk, declared)
    if kind == "plugin" and (clash := namespace_clash(str(held[NAMESPACE].value))) is not None:
        raise BuildRefused([Refusal(
            "scaffold.set", RefusalClass.USER,
            f"--set plugin.namespace: {clash}" if held[NAMESPACE].source == SET else
            f"the plugin's namespace, {held[NAMESPACE].value}, cannot be used: {clash}"
            + (" (it is the first word of the folder's name)" if held[NAMESPACE].source == OWN else
               " (it is your defaults' [scaffold] namespace)" if held[NAMESPACE].source == DEFAULTS else ""),
            "choose the namespace of your entries with --set plugin.namespace=<a free word>"
            + ("; `mime defaults` shows your defaults' namespace" if held[NAMESPACE].source == DEFAULTS else ""))])
    if kind == "block" and source is not None and declared is None:
        licence = held[BLOCK_LICENCE]
        say(f"the saved experiment declares no licence: block.licence is {licence.value}, from {licence.source}")
    if kind == "experiment" or kind == "plugin":
        plan = _experiment(held) if kind == "experiment" else _plugin(held, defaults)
    elif kind == "block":
        plan = _block(root, held, defaults, Path(os.fspath(source)) if source is not None else None, chosen_sockets)
    else:
        assert facts is not None
        plan = _bundle(kind, root, facts, held, defaults, say)
    _tokens(plan, held)
    shown = {path: plan.notes.get(path, "") for path in (*plan.files, *plan.notes)}
    files = tuple(sorted(shown.items()))
    named = f" {held[BLOCK_ID].value}" if kind == "block" else ""
    if dry_run:
        say(Said(f"Would scaffold {_a(kind)}{named} in {quoted(given)} (nothing written)", DONE))
        for line in _tree(given, shown, plan.empty):
            say(line)
        for line in _sources(held):
            say(line)
        return Scaffolded(kind, root, files, held, written=False)
    _written(kind, root, plan)
    if walking and ask is not None:
        _offer(held, ask, say)
    say(Said(f"Scaffolded {_a(kind)}{named} in {quoted(given)}", DONE))
    for line in _tree(given, shown, plan.empty):
        say(line)
    for line in _sources(held):
        say(line)
    say("Next:")
    for label, command in _next(kind, given, held):
        say(f"  {label + ':':<12}{command}")
    return Scaffolded(kind, root, files, held, written=True)


__all__ = ["Scaffolded", "scaffold"]
