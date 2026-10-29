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
def zenodo(rig: Rig, files: dict[str, bytes], *, legacy: bool = True, checksums: dict[str, str] | None = None
           ) -> Iterator[None]:
    """A fake of Zenodo's record API on the rig's server. ``requests.get`` is told that ``doi.org`` and ``zenodo.org`` are
    that server, and answers a request to the landing page with the address pooch expects to be redirected to."""
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
    rig.server.files[f"/zenodo/api/records/{RECORD}"] = json.dumps({"id": RECORD, "files": listed}).encode()
    rig.server.files["/landing"] = b"<html></html>"
    real = requests.get

    def routed(url: str, *args: object, **keywords: object) -> requests.Response:
        if url == f"https://doi.org/{DOI}":
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
