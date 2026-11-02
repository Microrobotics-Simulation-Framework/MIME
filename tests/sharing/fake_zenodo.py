"""A fake of Zenodo's record API on the assets tests' local server, shared by the tests of ``pull``: ``requests.get`` is
told that ``doi.org`` and ``zenodo.org`` are that server, so pooch's own DOI code runs against the fake record."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import requests
from assets_rig import Rig

RECORD = 4242
DOI = f"10.5281/zenodo.{RECORD}"


@contextmanager
def zenodo(rig: Rig, files: dict[str, bytes], *, legacy: bool = True, checksums: dict[str, str] | None = None,
           concept: str | None = None, version: str | None = None) -> Iterator[None]:
    """A fake of Zenodo's record API on the rig's server. ``requests.get`` is told that ``doi.org`` and ``zenodo.org`` are
    that server, and answers a request to the landing page with the address pooch expects to be redirected to. With
    ``concept``, the record is the latest version of that concept DOI, which resolves to it as Zenodo's concept DOIs
    resolve to their latest version (the record states its DOI, its concept's and its ``version``)."""
    listed = []
    for name, data in files.items():
        checksum = (checksums or {}).get(name, hashlib.md5(data).hexdigest())
        if legacy:
            listed.append({"key": name, "checksum": f"md5:{checksum}", "size": len(data),
                           "links": {"self": f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"}})
            rig.server.files[f"/zenodo/api/records/{RECORD}/files/{name}/content"] = data
        else:
            listed.append({"filename": name, "checksum": checksum, "filesize": len(data)})
            rig.server.files[f"/zenodo/records/{RECORD}/files/{name}?download=1"] = data
    held: dict[str, object] = {"id": RECORD, "files": listed}
    if concept is not None:
        held.update({"doi": DOI, "conceptdoi": concept, "metadata": {"version": version}})
    rig.server.files[f"/zenodo/api/records/{RECORD}"] = json.dumps(held).encode()
    rig.server.files["/landing"] = b"<html></html>"
    real = requests.get

    def routed(url: str, *args: object, **keywords: object) -> requests.Response:
        if url in (f"https://doi.org/{DOI}", f"https://doi.org/{concept}"):
            local, reported = rig.server.url("/landing"), f"https://zenodo.org/records/{RECORD}"
        elif url.startswith("https://zenodo.org/"):
            local, reported = rig.server.url("/zenodo/" + url.removeprefix("https://zenodo.org/")), url
        else:
            return real(url, *args, **keywords)  # type: ignore[arg-type]
        response = real(local, *args, **keywords)  # type: ignore[arg-type]
        response.url = reported
        return response

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests, "get", routed)
        yield
