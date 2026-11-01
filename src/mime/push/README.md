# `mime.push`

**A saved experiment as a draft record on Zenodo: the plan shown first, the draft made, its DOI reserved, its files
uploaded and checked, and nothing published.** Publishing (the one irreversible step, behind a typed record number),
new versions and communities come with a later MIME (push's second piece, P2b).

## Commands

```
mime push <folder> --to zenodo|zenodo-sandbox            # the plan, then the draft and its files (asks first)
mime push <folder> --to <service> --init                 # write sharing.yaml, the record's credit
mime push <folder> --to <service> --status               # the draft as the service holds it
mime push <folder> --to <service> --discard              # delete the draft (only a draft can be deleted)
mime push <folder> --to <service> --overwrite-site-edits # write over metadata edited on the site
```

`--yes` goes ahead after the plan without asking; without a terminal and without `--yes`, nothing is sent
(`push.unconfirmed`). Every push, `--status` and `--discard` ends with a link on a line of its own: the draft's page
(once `--discard` has deleted it, the account's drafts, `<site>/me/uploads`). **`--to` is required**: `zenodo` and
`zenodo-sandbox` are two services, each fixed in code (its API, its site, its token, its records), never one flag
apart by accident. The token is the one `mime credentials set
<service>` stored, or `ZENODO_TOKEN` / `ZENODO_SANDBOX_TOKEN`; a draft needs the scope `deposit:write`.

## `sharing.yaml`

The record's credit and description, beside the saved form (`<folder>/sharing.yaml`) and never in it: Zenodo keeps a
record's metadata editable and its files immutable, so a misspelt name is a metadata edit, never a new version (design
#199). `--init` writes it from what the folder knows (the experiment's name as the title, its licence), with
each key explained; an existing one is never overwritten. It is read with YAML's safe loader against a closed schema:

| key | what |
|---|---|
| `title`, `description` | the record's title, and a paragraph shown above the card |
| `licence` | the experiment's licence, which must equal the one it declares (`exp.licence(...)`, bound to its files); `--init` pre-fills the Creative Commons Attribution licence (`mime.push.DEFAULT_LICENCE`) where it declares none, and push asks the experiment to declare it too |
| `creators` | each person (`family`, `given`) or `organisation`, with an ORCID iD (held to its form and check digit), an affiliation and its ROR identifier |
| `contributors` | the same, each with a DataCite contributor type (`DataCurator`, `Supervisor`, ...); the rights holder of each third-party file the record holds is added by push as `RightsHolder` |
| `keywords`, `related` | searchable words (push adds `mime-experiment`); related works by DOI or `https://` address with DataCite's relation |
| `uploader` | the person whose token pushes, and their institution: a declared permission's grantee must be one of them |

A key missing, unknown or of another type is refused by its path (`push.sharing`); no creator, or an ORCID iD whose
form or check digit is wrong, as `push.creators`. `pack` neither carries nor refuses the file, and `mime card` names
its creators.

## The plan

Every push begins with the plan, computed offline after the folder loads, with every refusal at once and no request
made: the record's files (the bundle `pack` makes, its NOTICE, and each asset of the lock that may be redistributed and
has no Zenodo location yet, from the cache, hashed in full), every file inside the bundle with its licence, whose it is
and its verdict, every file that travels as a reference with its origin and sha256 (never uploaded, whatever its
size), the experiment's licence and every licence on the record (Zenodo has no licence per file: the NOTICE maps
each), the creators, the contributors, the related works, each declared permission in #198's words ("You are
declaring publicly, under your name, that ... permitted you to redistribute ..."), each field of the metadata MIME
builds on a line of its own (the title, the description, the publisher, the date, the resource type, the access once
published, push's identifier of the draft), and the availability statement.

**The metadata** is InvenioRDM's: resource type `dataset`; the title, creators, contributors, keywords and related works
from `sharing.yaml`; the publisher `Zenodo` and the date, which DataCite needs to register the DOI; the card
(`mime card`'s) as the description, in HTML; every licence as a rights entry, an SPDX id as the vocabulary's id in
lower case and a `LicenseRef-` licence as a title, a link and what it allows with **no `id`** (the sandbox refuses a
`LicenseRef-` id); the MIME release it was saved with as `requires` and `iscompiledby`; and push's identifier of the
draft, a `urn:uuid` among its alternate identifiers (below). Each third-party file the record holds adds its rights
holder (its credit) as a `RightsHolder` contributor, as each declared permission adds its grantor; a file that travels
as a reference is not in the record, so its rights holder is credited beside it in the NOTICE and the card, and is no
contributor of the record. A field the service refuses, in any answer (a success that lists `errors`, or a refusal),
is said by who owns it (`mime.push.field_owner`): one the author writes (`sharing.yaml`, or the experiment's terms) is
theirs to correct; one MIME builds (the publisher, the date, the resource type, the description, MIME's keyword
`mime-experiment`, the rights holders it adds, its related identifiers) is MIME's defect, and the refusal says so, as
does one that names no field.

## The client, as the sandbox measured it

The sandbox probe before this piece chose InvenioRDM's records API, and the client codes against what it found:

- **every request asks for `Accept: application/vnd.inveniordm.v1+json`**: without it the same addresses answer in the
  legacy deposit shape, which holds one licence; a successful answer in another content type, or in that shape, is
  refused;
- **every successful answer is checked for `errors`**: a failed transfer, and a draft saved with a field missing, are
  both answered as successes;
- **no resume**: a file is sent whole into an entry started for it, never `PUT` twice (a second `PUT` to a completed
  file deletes it), never with `Content-Range` (ignored: a "resumed" tail is stored as a complete, truncated file);
  after each upload the service's md5 and size must equal MIME's (Zenodo states md5 only; MIME keeps the sha256);
- **rate limits**: requests half a second apart; an answer of too many requests (or of a service unavailable that states
  its `Retry-After`) waited out for its `Retry-After`, which the sandbox sends on ordinary successful answers too (never
  a throttle there); a spent window waited to its reset;
- **a gateway's bad-gateway or timeout answer** can come after the service acted, so a request that changes the
  service is never sent again on one: push reads the draft or its files again and goes on from what is there (a file
  started and uploaded once), and a draft whose making's answer was lost is said on the next run, which names the
  identifier it carries;
- **the reserved DOI as the service gives it** (under DataCite's test prefix on the sandbox); a production service that
  reserves one under the test prefix is refused;
- **the token** goes in the `Authorization` header alone, built inside the call that sends it, to the service's own
  host over `https`: a redirect is never followed; a service on this machine (the tests' fake, over `http`) is reached
  without the environment's proxy; and whatever stops a request inside the standard library (an error, a cut answer,
  an interrupt such as Ctrl-C) is caught at the transport and raised again, fresh, from a frame that holds no token,
  so a traceback captured with its locals shows none, through the real transport too (a test interrupts a request on
  a server on this machine). A `Service` made in Python pairs a real service's credential with that service's host
  alone (or a stand-in's on this machine or at a `.invalid` address), and the two services' stored tokens are never the
  same one (`push.token`).

## Drafts, and resuming

Each step is check-then-act against the service, read again on every run, so a push stopped at any request is resumed
by running it again, at the level of whole files: a file the draft holds with the plan's size and md5 was uploaded and
verified, and is not sent again; any other is deleted and sent whole. A file in the draft the plan does not hold is
deleted where push uploaded it earlier and refused where it did not (`push.draft-changed`). A push stopped while it
asked for the draft to be made, before the answer came, says on its next run that the service may hold a draft MIME
has no record of, and where to look.

**Push writes to, and deletes, only the draft it made.** Each draft carries an alternate identifier push writes into it,
`urn:uuid:...`, a UUID of the service, the experiment's name and a random mark kept in push's record; before it writes
to a draft or deletes one, push reads the draft and refuses one that does not carry it (`push.draft-foreign`: a record
naming another experiment's draft, edited, or copied from elsewhere, changes and deletes nothing).

**An edit made on the site is never written over unasked.** Push keeps what the draft's metadata was, as the service
held it, each time it wrote it (a hash of each field). The next push reads the draft first; a field that changed since,
and that push would now write otherwise, stops it (`push.metadata-edited`), naming the field with its value on the
site and the value from `sharing.yaml` (or from the experiment, or MIME's): copy the site's into `sharing.yaml` to keep
it, and the push goes on; `--overwrite-site-edits` writes over it (a field `sharing.yaml` has no key for is kept only by
not pushing that draft again). `--status` names the fields edited since.

**What push remembers** is in the user's state directory (`platformdirs`), under
`push/<service>/<experiment>-<hash>/state.json`, keyed by the service and the experiment's exact name (the hash is of
the name, so `Toy` and `toy` have two records on a file system that folds case; this changes design #171, which keyed
it by the bundle's hash): the draft's id, its concept, its reserved DOI, its page, push's mark, the hashes of the
metadata it last wrote, and each file it uploaded and verified with its size, md5 and sha256. Any service that is not
one of the two fixed in code (a test's fake, a dry run, whatever its name) has a namespace of its own
(`push/other/<name>-<hash>/`), so a record made against a stand-in is never read or written for a real service; a
record that names another service, or another experiment (case included), is refused (`push.state`), as is one written
before push marked its drafts (push's first record format), which says what to delete.

## Testing it

The tests run against an offline fake of the records API (`tests/sharing/zenodo_records.py`), matched to the probe's
answers and replayed against them request by request (`tests/sharing/test_push_fidelity.py`, which names what the
replay cannot show). CI never holds an upload token (design #171).

**The sandbox lane** is run by hand, on a machine where `mime credentials set zenodo-sandbox` stored a sandbox token:

```
uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" [--orcid <iD>] --demo toy --demo terms
uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" --demo toy --dry-run   # no network
```

It pushes each folder to the sandbox (a copy of a folder it is given, so it writes nothing into it), reads the draft,
pushes again (nothing is sent again, and nothing reads as edited on the site), runs the same against the fake and
compares the two answer by answer; it keeps each draft for a person to look at (`--discard` deletes them) and writes a
log of the answers' shapes, searched for the token before it is written, even when a push stops. A test CI does not
run is documentation, not a test, so each run's log is kept beside the probe's. A dry run needs no token (it holds one
of its own), compares the fake behind the real transport with the fake in this process and says it asked no sandbox,
writes a log only where `--log` names, and removes what else it wrote.

## Status

Push's draft piece, P2a (designs #198, #199), after terms and tiers (P1): `sharing.yaml` and `--init`, the client, the plan, drafts, `--status` and
`--discard`, the offline fake and the sandbox lane. Publishing, versions, the lock's `publication` and the confirmation
in words come with P2b.

Phases and releases are explained in the [package manifest](../README.md#status).
