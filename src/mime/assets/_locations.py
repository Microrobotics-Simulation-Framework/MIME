"""Where this machine gets an asset: the user's or lab's locations file.

A lab keeps its own copies (a mounted share, an internal server) and says so once, in one file, rather than in every
record. The file maps an asset, by its sha256 or by its name, to extra locations, and names mirrors tried for every
asset. They are tried after the cache and the files placed in it, and **before** the record's own locations, since a
lab's mirror is usually why the file exists. The file never changes a pin, a record or a saved folder: a saved lock
holds only the record's own locations, so an internal server's address never travels with a shared experiment.

The file is TOML, at ``MIME_ASSET_LOCATIONS`` when that is set, else ``locations.toml`` in the user's config directory
(``platformdirs``)::

    # tried for every asset, by its sha256 or a name it may be placed under
    mirrors = ["file:///mnt/lab-share/mime-assets/"]
    # the certificate bundle of the institution's https servers
    ca_bundle = "/etc/ssl/certs/lab-ca.pem"

    [assets]
    "unit_tube_R0.997" = ["https://files.lab.example.org/mime/unit_tube_R0.997.npz"]
    "d2b4dbe4bf19caac61c74c5d4d2d77e57c6318c4f96a664a22221fa75151741c" = ["https://mirror.example.org/d2b4..."]

``MIME_CA_BUNDLE`` overrides ``ca_bundle``. A file that cannot be read, or holds a location MIME does not read, is
refused (``asset.locations-file``), naming the entry: a lab's mistake is never read as no mirror.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mime.build import BuildRefused, Refusal, RefusalClass

LOCATIONS_VARIABLE = "MIME_ASSET_LOCATIONS"
CA_BUNDLE_VARIABLE = "MIME_CA_BUNDLE"
LOCATIONS_FILE = "locations.toml"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Locations:
    """What the locations file says: mirrors for every asset, extra locations by sha256 and by name, and the
    certificate bundle, each where it says one. Empty where there is no file."""

    path: Path | None = None
    mirrors: tuple[str, ...] = ()
    by_sha256: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    by_name: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    ca_bundle: Path | None = None

    def extra(self, name: str, sha256: str) -> tuple[str, ...]:
        """The extra locations of an asset, in the order they are tried: its own by sha256, by name, then the
        mirrors; each once."""
        return tuple(dict.fromkeys((*self.by_sha256.get(sha256, ()), *self.by_name.get(name, ()), *self.mirrors)))


def locations_path() -> Path:
    """The locations file: ``MIME_ASSET_LOCATIONS`` when set, else ``locations.toml`` in the user's config directory."""
    chosen = os.environ.get(LOCATIONS_VARIABLE)
    if chosen:
        return Path(chosen)
    import platformdirs

    return Path(platformdirs.user_config_dir("mime")) / LOCATIONS_FILE


def _refused(path: Path, problem: str) -> BuildRefused:
    return BuildRefused([Refusal(
        "asset.locations-file", RefusalClass.USER, f"the locations file {path}: {problem}",
        "correct it (`mirrors = [\"file:///<directory>/\"]`, `[assets] \"<name or sha256>\" = [\"https://...\"]`, "
        "`ca_bundle = \"<file>\"`), or point MIME_ASSET_LOCATIONS at another file")])


def _location_list(path: Path, where: str, value: object) -> tuple[str, ...]:
    from mime.assets._reference import location_problem

    if not isinstance(value, list) or not all(isinstance(each, str) for each in cast(list[object], value)):
        raise _refused(path, f"{where} is a list of locations")
    found = tuple(cast(list[str], value))
    for location in found:
        problem = location_problem(location)
        if problem is not None:
            raise _refused(path, f"{where} holds {location!r}: {problem}")
    return found


def read_locations() -> Locations:
    """The locations file as this machine has it, or an empty one where there is none."""
    path = locations_path()
    bundle_text = os.environ.get(CA_BUNDLE_VARIABLE) or None
    if not path.is_file():
        if os.environ.get(LOCATIONS_VARIABLE):
            raise _refused(path, f"{LOCATIONS_VARIABLE} names it, and it is not a file")
        return Locations(ca_bundle=_bundle(path, bundle_text))
    try:
        held: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise _refused(path, f"cannot be read as TOML ({error})") from None
    unknown = sorted(set(held) - {"mirrors", "assets", "ca_bundle"})
    if unknown:
        raise _refused(path, f"its keys are mirrors, assets and ca_bundle, not {', '.join(unknown)}")
    mirrors = _location_list(path, "mirrors", held.get("mirrors", []))
    listed = held.get("assets", {})
    if not isinstance(listed, dict):
        raise _refused(path, "[assets] is a table of a name or a sha256 to a list of locations")
    by_sha256: dict[str, tuple[str, ...]] = {}
    by_name: dict[str, tuple[str, ...]] = {}
    for key, value in cast(dict[str, object], listed).items():
        found = _location_list(path, f"[assets] {key!r}", value)
        (by_sha256 if _SHA256.fullmatch(key) else by_name)[key] = found
    stated = held.get("ca_bundle")
    if stated is not None and not isinstance(stated, str):
        raise _refused(path, "ca_bundle is the path of a certificate bundle (.pem)")
    return Locations(path, mirrors, by_sha256, by_name, _bundle(path, bundle_text or stated))


def _bundle(path: Path, text: str | None) -> Path | None:
    if text is None:
        return None
    bundle = Path(text).expanduser()
    if not bundle.is_file():
        raise BuildRefused([Refusal(
            "asset.locations-file", RefusalClass.USER,
            f"the certificate bundle {bundle} ({CA_BUNDLE_VARIABLE}, or ca_bundle in {path}) is not a file",
            "point MIME_CA_BUNDLE at the institution's .pem bundle, or unset it")])
    return bundle


__all__ = ["CA_BUNDLE_VARIABLE", "LOCATIONS_FILE", "LOCATIONS_VARIABLE", "Locations", "locations_path",
           "read_locations"]
