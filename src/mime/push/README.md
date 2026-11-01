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
```

`--yes` goes ahead after the plan without asking; without a terminal and without `--yes`, nothing is sent
(`push.unconfirmed`). **`--to` is required**: `zenodo` and `zenodo-sandbox` are two services, each fixed in code (its
API, its site, its token, its records), never one flag apart by accident. The token is the one `mime credentials set
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
| `contributors` | the same, each with a DataCite contributor type (`DataCurator`, `Supervisor`, ...); each third-party file's rights holder is added by push as `RightsHolder` |
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
declaring publicly, under your name, that ... permitted you to redistribute ..."), and the availability statement.

**The metadata** is InvenioRDM's: resource type `dataset`; the title, creators, contributors, keywords and related works
from `sharing.yaml`; the publisher `Zenodo` and the date, which DataCite needs to register the DOI; the card
(`mime card`'s) as the description, in HTML; every licence as a rights entry, an SPDX id as the vocabulary's id in
lower case and a `LicenseRef-` licence as a title, a link and what it allows with **no `id`** (the sandbox refuses a
`LicenseRef-` id); and the MIME release it was saved with as `requires` and `iscompiledby`. A field the service
refuses is said by who owns it (`push._plan.FIELD_OWNERS`): one the author writes (`sharing.yaml`, or the experiment's
terms) is theirs to correct; one MIME builds is MIME's defect, and the refusal says so.

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
- **rate limits**: requests half a second apart; an answer of too many requests (or of a service unavailable) waited
  out for its `Retry-After`, which the sandbox sends on ordinary successful answers too (never a throttle there); a
  spent window waited to its reset;
- **the reserved DOI as the service gives it** (under DataCite's test prefix on the sandbox); a production service that
  reserves one under the test prefix is refused;
- **the token** goes in the `Authorization` header alone, built inside the call that sends it, to the service's own
  host over `https`: a redirect is never followed, and no frame holds the token's text under a name, so a traceback
  captured with its locals shows none.

## Drafts, and resuming

Each step is check-then-act against the service, read again on every run, so a push stopped at any request is resumed
by running it again, at the level of whole files: a file the draft holds with the plan's size and md5 was uploaded and
verified, and is not sent again; any other is deleted and sent whole. The draft's metadata is written again from
`sharing.yaml` on every run (an edit made on the site is overwritten); a file in the draft the plan does not hold is
deleted where push uploaded it earlier and refused where it did not (`push.draft-changed`). A push stopped while it
asked for the draft to be made, before the answer came, says on its next run that the service may hold a draft MIME
has no record of, and where to look.

**What push remembers** is in the user's state directory (`platformdirs`), under
`push/<service>/<experiment>/state.json`, keyed by the service and the experiment's name (this changes design #171,
which keyed it by the bundle's hash): the draft's id, its concept, its reserved DOI, its page, and each file it
uploaded and verified with its size, md5 and sha256. Any service that is not one of the two fixed in code (a test's
fake, a dry run, whatever its name) has a namespace of its own (`push/other/<name>-<hash>/`), so a record made against
a stand-in is never read or written for a real service; a record that names another service is refused
(`push.state`).

## Testing it

The tests run against an offline fake of the records API (`tests/sharing/zenodo_records.py`), matched to the probe's
answers and replayed against them request by request (`tests/sharing/test_push_fidelity.py`, which names what the
replay cannot show). CI never holds an upload token (design #171).

**The sandbox lane** is run by hand, on a machine where `mime credentials set zenodo-sandbox` stored a sandbox token:

```
uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" [--orcid <iD>] --demo toy --demo terms
uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" --demo toy --dry-run   # no network
```

It pushes each folder to the sandbox, reads the draft, pushes again (nothing is sent again), runs the same against the
fake and compares the two answer by answer; it keeps each draft for a person to look at (`--discard` deletes them) and
writes a log of the answers' shapes, searched for the token before it is written, even when a push stops. A test CI
does not run is documentation, not a test, so each run's log is kept beside the probe's.

## Status

Push's draft piece, P2a (designs #198, #199), after terms and tiers (P1): `sharing.yaml` and `--init`, the client, the plan, drafts, `--status` and
`--discard`, the offline fake and the sandbox lane. Publishing, versions, the lock's `publication` and the confirmation
in words come with P2b.

Phases and releases are explained in the [package manifest](../README.md#status).
