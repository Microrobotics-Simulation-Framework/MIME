"""A person's sharing defaults: the values of ``sharing.yaml`` they repeat from record to record (their name, ORCID iD
and affiliation as a creator, the contributors, the keywords, the uploader, the funding), kept once in MIME's config
directory (``platformdirs``, beside the credentials file) and never in an experiment's folder, which a bundle and a
record are made from.

``mime push <folder> --to <service> --init`` writes a new ``sharing.yaml`` filled from them, ``--setup`` offers each as
the value to keep where it makes a new one (an existing file's empty fields are asked as empty) and asks at its end
whether to save what was set as them, and ``mime push --defaults`` shows them and where they live. They are held to
the checks ``sharing.yaml`` is held to for the same keys (an ORCID iD by its check digit), and a token MIME holds is
never saved in them: they go into every record's metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import platformdirs
import yaml

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.push._sharing import DEFAULT_KEYS, defaults_refusals, value_line

DEFAULTS_FILE = "sharing-defaults.yaml"
DEFAULTS_FORMAT = "mime.sharing-defaults/1"

_HEAD = ("# Your own values for sharing.yaml, which `mime push <folder> --to <service> --init` fills a new one from and\n"
         "# `--setup` offers for a new one: the keys a person repeats from record to record, in its shapes.\n"
         "# `mime push --defaults` shows them; `--setup` asks to save what it set; edit or delete this file by hand.\n")


def defaults_path() -> Path:
    """Where a person's sharing defaults live: MIME's config directory, never an experiment's folder."""
    return Path(platformdirs.user_config_dir("mime")) / DEFAULTS_FILE


def _refused(path: Path, problem: str) -> BuildRefused:
    return BuildRefused([Refusal("push.sharing", RefusalClass.USER, f"{path}: {problem}",
                                 f"correct it by hand, or delete it and save them again from `mime push <folder> "
                                 f"--setup`")])


def read_defaults() -> dict[str, object]:
    """The person's sharing defaults, checked, by key (only those that hold a value); empty where none are saved. A file
    that does not read, or holds a value ``sharing.yaml`` would refuse, is refused (``push.sharing``, ``push.creators``)
    with every reason at once."""
    path = defaults_path()
    if not path.is_file():
        return {}
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as error:
        raise _refused(path, f"is not YAML that reads ({type(error).__name__})") from None
    if loaded is None:
        return {}
    refusals = defaults_refusals(loaded, path, DEFAULTS_FORMAT)
    if refusals:
        raise BuildRefused(refusals)
    held = cast(dict[str, object], loaded)
    return {key: held[key] for key in DEFAULT_KEYS if held.get(key) not in (None, "", [], {})}


def defaults_text(values: Mapping[str, object]) -> str:
    """The file that holds ``values`` as a person's defaults, each key on a line of its own."""
    return _HEAD + "\n".join([f"format: {DEFAULTS_FORMAT}", *(value_line(key, values[key]) for key in DEFAULT_KEYS
                                                                if key in values)]) + "\n"


def write_defaults(values: Mapping[str, object]) -> Path:
    """Save ``values`` (the keys of :data:`~mime.push._sharing.DEFAULT_KEYS` among them) as the person's defaults, checked
    first: a value ``sharing.yaml`` would refuse, or a token MIME holds, is refused and nothing is written."""
    from mime.credentials import held_tokens

    path = defaults_path()
    kept = {key: values[key] for key in DEFAULT_KEYS if values.get(key) not in (None, "", [], {})}
    text = defaults_text(kept)
    refusals = defaults_refusals(yaml.safe_load(text), path, DEFAULTS_FORMAT)
    if refusals:
        raise BuildRefused(refusals)
    if any(each.occurs_in(text.encode("utf-8")) for each in held_tokens()):
        raise _refused(path, "would hold a token MIME stores, which never goes into a record's metadata; nothing was "
                             "written")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return path


def defaults_lines() -> list[str]:
    """What ``mime push --defaults`` says: where the defaults live, and each, as ``sharing.yaml`` would hold it."""
    path = defaults_path()
    values = read_defaults()
    if not values:
        return [f"no sharing defaults are saved; they would be in {path}",
                "`mime push <folder> --setup` asks, at its end, whether to save what it set as them; or write that "
                "file by hand, with sharing.yaml's keys creators, contributors, keywords, uploader and funding"]
    return [f"your sharing defaults, in {path} (`--init` fills a new sharing.yaml from them, and `--setup` offers each "
            "for a new one):",
            *(f"  {value_line(key, values[key])}" for key in DEFAULT_KEYS if key in values)]


__all__ = ["DEFAULTS_FILE", "DEFAULTS_FORMAT", "defaults_lines", "defaults_path", "defaults_text", "read_defaults",
           "write_defaults"]
