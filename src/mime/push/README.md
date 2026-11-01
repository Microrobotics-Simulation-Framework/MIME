# `mime.push`

**A saved experiment as a record on Zenodo: the plan shown first, a private draft made, its DOI reserved and its files
uploaded and checked; then, at `--publish`, the one irreversible step behind the confirmation in words, each declared
permission asked and the record number typed; afterwards the record checked as published, and new versions under one
concept DOI.**

## Commands

```
mime push <folder> --to zenodo|zenodo-sandbox            # the plan, then the draft and its files (no question)
mime push <folder> --to <service> --publish [--confirm <DOI>]   # publish the draft: cannot be undone
mime push <folder> --to <service> --init                 # write sharing.yaml, the record's credit
mime push <folder> --to <service> --status               # the draft as the service holds it
mime push <folder> --to <service> --discard              # delete the draft (only a draft can be deleted)
mime push <folder> --to <service> --full-plan            # the full plan: every file and every field
mime push <folder> --to <service> --keep-site-edits      # keep each edit made on the site, into sharing.yaml
mime push <folder> --to <service> --overwrite-site-edits # write over each edit made on the site
mime pull --from zenodo|zenodo-sandbox <record>          # a published record, by its number
```

A private draft is made without a question (the plan is shown first, `--discard` undoes it). `--yes` answers
`--discard`'s question only: it never publishes, never declares a permission and never settles an edit made on the
site. Every push, `--status`, `--discard` and `--publish` ends with a link: the draft's page, the record's once
published, the account's drafts once `--discard` has deleted it. **`--to` is required**: `zenodo` and
`zenodo-sandbox` are two services, each fixed in code (its API, its site, its token, its records), never one flag
apart by accident. The token is the one `mime credentials set <service>` stored, or `ZENODO_TOKEN` /
`ZENODO_SANDBOX_TOKEN`; a draft needs the scope `deposit:write`, publishing `deposit:actions`.

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
| `funding` | each funder (`funder`, its `ror` where known: Zenodo's funders are ROR's) and its award (`award`, its number; `title`; `url`); the award is sent as the record's own (title and number), never as an id of Zenodo's awards vocabulary, which MIME cannot check offline |
| `conference` | Zenodo's meeting field: `title`, `acronym`, `dates`, `place`, `url`, `session`, `part` |
| `fields` | any other field of the record, in the records API's own names and shapes, held to those shapes by path before anything is sent: `subjects`, `dates`, `languages`, `references`, `additional_titles`, `additional_descriptions`, and Zenodo's custom fields `journal:journal`, `imprint:imprint`, `thesis:university`, `thesis:type`; a field MIME builds, or one `sharing.yaml` has a key for, is refused here by name |

A key missing, unknown or of another type is refused by its path (`push.sharing`); no creator, or an ORCID iD whose
form or check digit is wrong, as `push.creators`. `pack` neither carries nor refuses the file, and `mime card` names
its creators.

## The plan

Every push begins with the plan, computed offline after the folder loads, with every refusal at once, and nothing is
written to the service before it is shown. **By default it is short**: what goes up (each record file and its size),
what travels as a reference and never goes up, the licences and creators, and each warning on a line of its own that
starts `!` (results that depend on a file the record does not hold, a declared permission, which becomes public when
published, a version that changes the title, the production service). **The full plan** (`--full-plan`) is every
record file with its licence and why it is there, every file inside the bundle with its licence, whose it is and its
verdict, every file that travels as a reference with its origin and sha256, the licences, the credit, each declared
permission in #198's words ("You are declaring publicly, under your name, that ... permitted you to redistribute
..."), each field of the metadata MIME builds on a line of its own, and the availability statement. **A push with
nothing to send** (the folder and `sharing.yaml` as push last sent them, the draft as push left it) reads the draft and
says so in one line, ending with its link; a published experiment unchanged since its last version says `unchanged
since <DOI>` in one line, with no request.

**The metadata** is InvenioRDM's: resource type `dataset`; the title, creators, contributors, keywords and related works
from `sharing.yaml`; the publisher `Zenodo` and the date, which DataCite needs to register the DOI; the card
(`mime card`'s) as the description, in HTML; every licence as a rights entry, an SPDX id as the vocabulary's id in
lower case and a `LicenseRef-` licence as a title, a link and what it allows with **no `id`** (the sandbox refuses a
`LicenseRef-` id); the MIME release it was saved with as `requires` and `iscompiledby`; and push's identifier of the
draft, a `urn:uuid` among its alternate identifiers (below). Each third-party file the record holds adds its rights
holder (its credit) as a `RightsHolder` contributor, as each declared permission adds its grantor; a file that travels
as a reference is not in the record, so its rights holder is credited beside it in the NOTICE and the card, and is no
contributor of the record. Funding, the conference and `fields` are sent as `sharing.yaml` holds them (`metadata.funding`, the custom fields
`meeting:meeting` and those `fields` names, and the metadata fields it names), and the version's number as
`metadata.version`. A field the service refuses, in any answer (a success that lists `errors`, or a refusal),
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

**An edit made on the site is never lost.** Push keeps what the draft's metadata was, as the service held it, each
time it wrote it (a hash of each field, custom fields included). The next push (and `--publish`) reads the draft
first; each field that changed since, and that push would now write otherwise, is shown with its value on the site
and the value from `sharing.yaml` (or MIME's), and settled field by field: **kept**, written back into `sharing.yaml`
under its key (that key's lines replaced, every other line and comment kept), the file read again and the payload made
from it held to the site's value, or put back as it was and the push stopped (`push.sharing`, with the lines to copy
in by hand); or **written over**. In a terminal push asks (`Write sharing.yaml's over the site's? [y/N]`, so the
default keeps the site's); `--keep-site-edits` keeps every edit `sharing.yaml` can hold, `--overwrite-site-edits`
writes over every one, and without either or a terminal an edit stops the push (`push.metadata-edited`). A field
`sharing.yaml` cannot hold (the licences, which are the files'; the publisher, the date, the resource type, the card,
push's identifier) is written over or stops the push. What the service adds to what it is sent (a person's display
name, a vocabulary entry's title) is not an edit: values are compared as the author writes them. Files stay MIME's: a
file added on the site is refused (`push.draft-changed`). `--status` names the fields edited since.

## Publishing

`--publish` publishes the draft an earlier push made, and only it: the folder packed again must make the files the
draft holds (`push.folder-changed` otherwise: push, look at the draft, then publish), and no draft is `push.no-draft`.
The metadata is written once more where it differs (an edit settled as a push settles it, the day's publication
date), the plan is shown as it is published, and then:

```
Publishing 'full_toy' on zenodo as 10.5281/zenodo.629525 (version 1 of a new concept).
This cannot be undone. The DOI is registered for good. Zenodo lets the record's owner delete it within 30 days, ...
  You are publishing, under your name, 4 files as your own work (or your lab's), licensed CC-BY-4.0.
  You state that everyone named as a creator or contributor agreed to be named.
  1. You are declaring publicly, under your name, that Maker Ltd permitted you to redistribute this file (...).
Declare permission 1 publicly, under your name? [y/N]
Type the record number to publish (629525):
```

**Each declared permission is asked on its own** and `y` declares it; anything else stops the publish. It is asked in
a terminal only, and never answered by `--yes` or `--confirm`: a publish without a terminal goes ahead only for the
declarations a person confirmed in a terminal for exactly these bytes (push's record keeps each confirmation by a hash
of the whole declaration, its files' sha256 among them), and is refused otherwise (`push.permission-unconfirmed`).
**The record number is typed** (never `y`, which a hand types by habit); a script passes `--confirm <DOI>`, which must
be the draft's reserved DOI (`push.not-confirmed`, before any request), and without either nothing is published
(`push.unconfirmed`).

**Never sent twice.** An answer lost is followed by reading the record: published, push goes on to the checks; not, it
stops, and the next run reads the record again first. A run that finds the draft already published (an earlier run's
answer lost, or the site's button) goes straight to the checks and publishes nothing.

**The checks afterwards**: the record read again as published, its DOI the reserved one, each file's size and md5 the
uploaded one's and no other file, the metadata push wrote, and the bundle downloaded and its sha256 the one packed
(`push.after-publish` otherwise, the folder left as it was; `--publish` again checks again).

**The staged lock.** A bundle cannot name the record it is uploaded to unless the record's DOI is known first, so the
draft is made and its DOI reserved before the bundle is packed with the staged lock: the folder's lock with a
`publication` key (the service, the version's record and DOI, its concept's record, the concept's DOI where the service
states it, the version's number) and the record's own location of each file it holds first (`doi:<DOI>/<file>` on
production; on the sandbox, whose test-prefix DOIs do not resolve through doi.org, the file's address in its records
API), its earlier locations kept as backups. Both are outside the input hash. After a production publish and its
checks, the folder's lock is replaced by the staged lock, when it is still the lock the bundle was packed from
(`push.lock-changed` otherwise), so **`mime pack <folder>` makes the published bundle again, byte for byte**, and the
folder names the record it is. A sandbox record is never written into the folder: the sandbox can be cleaned at any
time.

**Versions.** Push's record keeps the version last published. A push of a changed folder makes a new version of it
(`POST .../versions`, the concept's open one where there is one), settles the edits made on the published record's
page since, links the files of the version before into it (`files-import`) and uploads only what differs: an
unchanged file is linked, never sent again. A push of an unchanged folder says `unchanged since <DOI>` (and, where
`sharing.yaml` changed, that a published record's metadata is updated on its page until `mime push --metadata`
comes). `--discard` of a new version's draft keeps the published versions.

## Pulling what was published

`mime pull --from zenodo-sandbox <record>` (or `--from zenodo`) reads the published record by its number at the
service's records API, with no token, and pulls its one bundle by its address there, held to the md5 and size the
record states; it says which version it is and whether a newer one exists. A DOI under DataCite's test prefix (the
sandbox's) is never resolved through doi.org: `mime pull` of one is refused with the `--from` command
(`pull.sandbox-doi`), and an asset location under it is never tried there. A production DOI is pulled through doi.org
as before; a concept DOI resolves to the latest version, which pull names.

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
uv run python scripts/zenodo_sandbox_lane.py --creator "<family>, <given>" --demo toy --demo terms --publish
```

It pushes each folder to the sandbox (a copy of a folder it is given, so it writes nothing into it), reads the draft,
pushes again (nothing is sent again, and nothing reads as edited on the site), runs the same against the fake and
compares the two answer by answer; it keeps each draft for a person to look at (`--discard` deletes them) and writes a
log of the answers' shapes, searched for the token before it is written, even when a push stops. A test CI does not
run is documentation, not a test, so each run's log is kept beside the probe's. A dry run needs no token (it holds one
of its own), compares the fake behind the real transport with the fake in this process and says it asked no sandbox,
writes a log only where `--log` names, and removes what else it wrote. With `--publish` (the sandbox only), it
publishes each draft (a declared permission asked in the terminal), pulls each record by its number into a fresh folder
and verifies it, publishes the toy's next version, and says whether each file linked into that version is stored once
(its file id the published version's), which only the service can answer.

## Status

Push's draft piece, P2a (designs #198, #199), after terms and tiers (P1): `sharing.yaml` and `--init`, the client, the
plan, drafts, `--status` and `--discard`, the offline fake and the sandbox lane. Its publishing piece, P2b (#199, #202):
the quieter plan, `--publish` with its confirmation, the staged lock and the checks afterwards, versions, edits made on
the site kept or written over, funding, conferences and `fields`, and `mime pull --from`. Metadata updates of a
published record, communities, `--version-of` and asset records come with P3.

Phases and releases are explained in the [package manifest](../README.md#status).
