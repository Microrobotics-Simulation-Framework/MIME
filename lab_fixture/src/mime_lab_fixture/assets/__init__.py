"""The lab's records and its own storage adapter, from outside MIME.

Its records (``records.json`` beside this module) are found through the ``mime.assets`` entry point ``labx``: a
marker layer whose bytes are inside this package (``package:labx/marker.usda``). ``LabStore`` is a thin adapter for a
lab's authenticated store, ``labstore://<host>[:<port>]/<path>``, registered through the ``mime.asset_adapters`` entry
point: it opens a stream over plain HTTP on this machine with the token MIME's resolver hands it for that host, and
answers a redirect, a missing file and a refused login in MIME's words. Everything else (the hash, the size cap,
retries, which host gets the token) is MIME's resolver's."""

from __future__ import annotations

import urllib.error
import urllib.request
from email.message import Message
from typing import IO, cast, override

from mime.assets import LoginRefused, NotFound, Opened, Redirect, Request, Transient

SCHEME = "labstore"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: Message,
                         newurl: str) -> None:
        return None


class LabStore:
    """The lab's store: ``labstore://host:port/path`` read as ``http://host:port/path``, on this machine only."""

    scheme = SCHEME
    needs_credential = True
    ranged = False

    def open(self, location: str, request: Request) -> Opened | Redirect:
        address = "http://" + location.removeprefix(f"{SCHEME}://")
        headers = {}
        if request.credential is not None:
            headers["Authorization"] = f"Bearer {request.credential.token.reveal()}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            response = opener.open(urllib.request.Request(address, headers=headers), timeout=request.timeout)
        except urllib.error.HTTPError as error:
            status, target = error.code, error.headers.get("Location") if error.headers is not None else None
            error.close()
            if status in (301, 302, 303, 307, 308) and target:
                return Redirect(f"{SCHEME}://" + str(target).partition("://")[2])
            if status in (404, 410):
                raise NotFound(f"the store has no file there ({status})") from None
            if status in (401, 403):
                raise LoginRefused(f"the store refused the login ({status})") from None
            raise Transient(f"the store answered {status}") from None
        except urllib.error.URLError as error:
            raise Transient(f"the store could not be reached ({error.reason})") from None
        length = response.headers.get("Content-Length")
        return Opened(cast(IO[bytes], response), int(length) if length and length.isdigit() else None)
