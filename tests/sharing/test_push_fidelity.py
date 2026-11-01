"""The fake's fidelity: the sandbox probe's requests of 2026-11-01, replayed here against the offline fake in their
order, each answer compared with the sandbox's (``zenodo_probe_answers.json``): its status, its content type, the
rate-limit headers it carries, the JSON type of each field push reads (``id``, ``parent.id``, ``pids.doi``,
``is_published``, ``status``, ``links.self_html``, ``metadata.rights``, ``errors``, the legacy ``conceptrecid``, a file
entry's ``key``, ``status``, ``size`` and ``checksum``), and the few values push compares (a status, a checksum's
``md5:`` prefix, the DOI's prefix, how many rights a draft kept, the field an ``errors`` names).

**What the replay cannot show**, listed so that none is skipped by accident (``NOT_REPLAYED``): the preview page and its
secret link (HTML for a person, which push never reads), and the legacy deposit API side by side (push never uses it).
Nor what the probe did not do (publishing, versions, communities, a token's scopes, an answer to a token that is not
accepted): the fake's guesses there are marked in ``zenodo_records.py``, and only the sandbox lane
(``scripts/zenodo_sandbox_lane.py``, by hand) checks them.

**The sandbox lane's answers** are replayed too: its first run found that a DOI's registration needs a publisher
(``Missing publisher field required for DOI registration.``), which the probe's requests had always sent; the fake
now says the same of a draft without one, and of each other field DataCite requires.

The control: an answer of the fake changed in one field's type (the DOI's identifier a number) is caught by the
comparison, as is a status changed (the re-``PUT`` of a completed file answered without its ``errors``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from zenodo_records import FakeRecords

from mime.push import RDM_TYPE, Answer, Dropped

RECORDED = json.loads((Path(__file__).with_name("zenodo_probe_answers.json")).read_text(encoding="utf-8"))
ANSWERS = RECORDED["answers"]
#: What the sandbox lane's runs recorded beyond the probe: each a request and the answer push read.
LANE = RECORDED["lane"]
TOKEN = "a-probe-token-" + "t" * 26
#: The probe's requests the replay leaves out, by its log's tag: the preview page and its secret link (13, 41, 44 to
#: 46; the log's redaction made 44 unreadable, and it has no 42 or 43), and the legacy deposit API (33 to 40, 48, 51
#: to 53).
NOT_REPLAYED = frozenset({13, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 48, 51, 52, 53})
DRAFT_PATHS = ("id", "parent", "parent.id", "parent.pids", "pids", "pids.doi", "pids.doi.identifier", "is_published",
               "status", "links", "links.self_html", "metadata", "metadata.title", "metadata.creators",
               "metadata.rights", "files", "files.entries", "errors", "conceptrecid", "metadata.license")
ENTRY_PATHS = ("key", "status", "size", "checksum", "errors")
CUSTOM = {"title": {"en": "MIME Example Licence 1.0"}, "description": {"en": "A custom licence."},
          "link": "https://example.org/licence-1.0"}


def _kind(value: object) -> str:
    return {dict: "object", list: "array", str: "string", int: "integer", float: "number", bool: "boolean",
            type(None): "null"}[type(value)]


def _at(held: object, path: str) -> object:
    for part in path.split("."):
        if not isinstance(held, dict) or part not in held:
            return KeyError
        held = held[part]
    return held


def _shape(held: object, paths: tuple[str, ...]) -> dict[str, str]:
    return {path: "absent" if (value := _at(held, path)) is KeyError else _kind(value) for path in paths}


def shape_of(answer: Answer | str) -> dict[str, object]:
    """An answer as the probe's fixture records one (``derive``'s reading, written out here again)."""
    if isinstance(answer, str):
        return {"status": answer}
    found: dict[str, object] = {"status": answer.status,
                                "content_type": answer.headers.get("content-type", "").split(";")[0],
                                "rate_limit_headers": sorted(h for h in answer.headers if h.startswith(
                                    "x-ratelimit-") or h == "retry-after")}
    body = answer.body.decode("utf-8")
    try:
        held = json.loads(body) if body.strip() else None
    except ValueError:
        held = None
    if isinstance(held, dict):
        if answer.status >= 400:
            found["fields"] = _shape(held, ("status", "message", "errors"))
        elif "entries" in held:
            found["fields"] = _shape(held, ("entries",))
            found["entries"] = [{**_shape(e, ENTRY_PATHS), "status_value": e.get("status"),
                                 "checksum_prefix": str(e.get("checksum", ""))[:4] or None} for e in held["entries"]]
        elif "key" in held:
            found["fields"] = _shape(held, ENTRY_PATHS)
            found["status_value"] = held.get("status")
            found["checksum_prefix"] = str(held.get("checksum", ""))[:4] or None
            found["errors"] = held.get("errors")
        else:
            found["fields"] = _shape(held, DRAFT_PATHS)
            found["status_value"] = held.get("status")
            doi = _at(held, "pids.doi.identifier")
            found["doi_prefix"] = doi[:8] if isinstance(doi, str) else None
            rights = _at(held, "metadata.rights")
            found["rights"] = len(rights) if isinstance(rights, list) else None
            errors = held.get("errors")
            found["errors"] = [e.get("field") for e in errors] if isinstance(errors, list) else None
    else:
        found["body"] = "empty" if not body else "text"
    return found


def comparable(recorded: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in recorded.items() if key != "method"}


def draft(rights: list[object], *, date: bool = True) -> dict[str, object]:
    metadata: dict[str, object] = {"resource_type": {"id": "dataset"}, "title": "MIME zenodo probe", "publisher": "Zenodo",
                                   "creators": [{"person_or_org": {"type": "personal", "family_name": "Probe"}}],
                                   "rights": rights, "description": "Probe draft; never published."}
    if date:
        metadata["publication_date"] = "2026-11-01"
    return {"metadata": metadata, "access": {"record": "public", "files": "public"}, "files": {"enabled": True}}


def replay(fake: FakeRecords) -> dict[int, Answer | str]:
    """The probe's requests in its order, by tag, each answered by ``fake``."""
    record: list[str] = []
    found: dict[int, Answer | str] = {}

    def send(tag: int, method: str, path: Callable[[], str], body: object = None, *, rdm: bool = True,
             raw: bytes | None = None) -> None:
        headers = {"Authorization": f"Bearer {TOKEN}", **({"Accept": RDM_TYPE} if rdm else {})}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        try:
            found[tag] = fake.handle(method, fake.api + path(), headers, data)
        except Dropped:
            found[tag] = "dropped"
        if tag == 1:
            record.append(str(json.loads(found[tag].body)["id"]))  # type: ignore[union-attr]

    r = lambda: f"/records/{record[0]}"  # noqa: E731
    files = lambda: f"{r()}/draft/files"  # noqa: E731
    two, three = [{"id": "cc-by-4.0"}, CUSTOM], [{"id": "cc-by-4.0"}, {"id": "mit"}, CUSTOM]
    send(1, "POST", lambda: "/records", draft(two), rdm=False)
    send(2, "GET", lambda: f"{r()}/draft")
    send(3, "POST", lambda: f"{r()}/draft/pids/doi")
    send(4, "POST", files, [{"key": "probe.txt"}])
    send(5, "PUT", lambda: f"{files()}/probe.txt/content", raw=b"p" * 2300)
    send(6, "POST", lambda: f"{files()}/probe.txt/commit")
    send(7, "GET", lambda: f"{r()}/draft")
    pids = json.loads(found[3].body)["pids"]  # type: ignore[union-attr]
    send(8, "PUT", lambda: f"{r()}/draft", {**draft([{"id": "cc-by-4.0"}, {"id": "LicenseRef-MIME-Example-1.0"}]),
                                            "pids": pids})
    send(9, "PUT", lambda: f"{r()}/draft", {**draft([{"id": "LicenseRef-MIME-Example-1.0", **CUSTOM}]), "pids": pids})
    send(10, "PUT", lambda: f"{r()}/draft", {**draft(three), "pids": pids})
    send(11, "PUT", lambda: f"{r()}/draft", {**draft([{"id": "cc-by-4.0"}], date=False), "pids": pids})
    send(12, "PUT", lambda: f"{r()}/draft", {**draft(three), "pids": pids})
    send(14, "GET", lambda: f"{r()}/draft", rdm=False)
    send(15, "POST", files, [{"key": "big.bin"}])
    fake.plant("cut", lambda method, path: path.endswith("/big.bin/content"))
    send(16, "PUT", lambda: f"{files()}/big.bin/content", raw=b"b" * 5000)
    send(17, "GET", lambda: f"{files()}/big.bin")
    send(18, "PUT", lambda: f"{files()}/big.bin/content", raw=b"b" * 4000)
    send(19, "GET", lambda: f"{files()}/big.bin")
    send(20, "GET", files)
    send(21, "POST", files, [{"key": "ctl.bin"}])
    send(22, "GET", lambda: f"{files()}/ctl.bin")
    send(23, "PUT", lambda: f"{files()}/ctl.bin/content", raw=b"c" * 4000)
    send(24, "GET", lambda: f"{files()}/ctl.bin")
    send(25, "PUT", lambda: f"{files()}/ctl.bin/content", raw=b"c" * 5000)
    send(26, "POST", lambda: f"{files()}/ctl.bin/commit")
    send(27, "GET", files)
    send(28, "POST", files, [{"key": "ctl.bin"}])
    send(29, "PUT", lambda: f"{files()}/ctl.bin/content", raw=b"c" * 5000)
    send(30, "POST", lambda: f"{files()}/ctl.bin/commit")
    send(31, "PUT", lambda: f"{files()}/probe.txt/content", raw=b"p" * 2300)
    send(32, "GET", files)
    send(47, "DELETE", lambda: f"{r()}/draft")
    send(49, "GET", lambda: f"{r()}/draft")
    send(50, "GET", lambda: f"/records/{int(record[0]) + 2}/draft")
    return found


def differences(found: dict[int, Answer | str], recorded: dict[str, dict[str, object]]) -> list[str]:
    problems: list[str] = []
    for tag, answer in sorted(found.items()):
        expected = comparable(recorded[str(tag)])
        made = shape_of(answer)
        if expected.get("status", "").__class__ is str:  # the cut: no answer on either side
            if made.get("status") != "dropped":
                problems.append(f"{tag}: the sandbox gave no answer, the fake {made.get('status')}")
            continue
        if expected.get("body") == "cut short in the log":
            expected = {key: expected[key] for key in ("status", "content_type", "rate_limit_headers")}
            made = {key: made.get(key) for key in expected}
        if expected.get("body") == "empty":
            expected = {key: value for key, value in expected.items() if key not in ("body", "content_type")}
            made = {key: value for key, value in made.items() if key not in ("body", "content_type")}
        for key in sorted(set(expected) | set(made)):
            if expected.get(key) != made.get(key):
                problems.append(f"{tag}: {key} is {expected.get(key)!r} on the sandbox and {made.get(key)!r} on the "
                                f"fake")
    return problems


def test_every_probe_request_is_replayed_or_named_as_not_replayable():
    replayed = set(replay(FakeRecords(TOKEN)))
    recorded = {int(tag) for tag in ANSWERS} | {44}
    assert replayed | NOT_REPLAYED == recorded and not replayed & NOT_REPLAYED


def test_the_fake_answers_each_replayed_probe_request_as_the_sandbox_did():
    assert differences(replay(FakeRecords(TOKEN)), ANSWERS) == []


@pytest.mark.parametrize("plant", ["the DOI a number", "a failed re-PUT without its errors", "the legacy shape kept"])
def test_control_a_change_in_one_answer_is_caught_by_the_comparison(plant: str):
    found = replay(FakeRecords(TOKEN))
    if plant == "the DOI a number":
        held = json.loads(found[3].body)  # type: ignore[union-attr]
        held["pids"]["doi"]["identifier"] = 10.5072
        found[3] = Answer(201, found[3].headers, json.dumps(held).encode())  # type: ignore[union-attr]
    elif plant == "a failed re-PUT without its errors":
        held = json.loads(found[31].body)  # type: ignore[union-attr]
        held.pop("errors")
        found[31] = Answer(200, found[31].headers, json.dumps(held).encode())  # type: ignore[union-attr]
    else:
        found[1] = found[2]
    assert differences(found, ANSWERS), plant


def test_the_fake_answers_a_draft_without_a_publisher_as_the_sandbox_lane_found():
    recorded = LANE["2026-11-01"]
    fake = FakeRecords(TOKEN)
    body = draft([{"id": "cc-by-4.0"}])
    body["metadata"].pop("publisher")  # type: ignore[union-attr]
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": RDM_TYPE}
    made = fake.handle("POST", fake.api + "/records", headers, json.dumps(body).encode())
    reserved = fake.handle("POST", fake.api + f"/records/{json.loads(made.body)['id']}/draft/pids/doi", headers, None)
    for answer in (made, reserved):
        assert recorded["status"] == "2xx" and 200 <= answer.status < 300
        assert json.loads(answer.body)["errors"] == recorded["errors"]


def test_the_fake_holds_a_draft_to_each_field_datacite_needs_for_its_doi():
    from mime.push._plan import DOI_FIELDS

    for field in DOI_FIELDS:
        fake = FakeRecords(TOKEN)
        body = draft([{"id": "cc-by-4.0"}])
        body["metadata"].pop(field)  # type: ignore[union-attr]
        made = fake.handle("POST", fake.api + "/records", {"Authorization": f"Bearer {TOKEN}", "Accept": RDM_TYPE},
                           json.dumps(body).encode())
        assert [each["field"] for each in json.loads(made.body)["errors"]] == [f"metadata.{field}"], field
