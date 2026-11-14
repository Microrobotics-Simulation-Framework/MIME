# `mime.push`

**A saved experiment as a record on Zenodo: the plan shown first, a private draft made, its DOI reserved and its files
uploaded and checked; then, at `--publish`, the one irreversible step behind the confirmation in words, each declared
permission asked and the record number typed; afterwards the record checked as published, and new versions under one
concept DOI.**

## Commands

```
mime push <folder> --to zenodo|zenodo-sandbox            # the plan, then the draft and its files (no question)
mime push <folder> --to <service> --publish [--confirm <DOI>[@<fingerprint>]]   # publish the draft: cannot be undone
mime push <folder> --to <service> --init                 # write mime.toml, the record's manifest
mime push <folder> --setup                               # make or edit mime.toml a field at a time, in a terminal
mime push --defaults                                     # your defaults, and where they live (= mime defaults)
mime check <folder> [--profile research]                 # mime.toml checked offline, on the profile workflow
mime defaults [show | edit]                              # your defaults: shown, or walked a field at a time
mime push <folder> --to <service> --status               # the draft as the service holds it
mime push <folder> --to <service> --discard              # delete the draft (only a draft can be deleted)
mime push <folder> --to <service> --full-plan            # the full plan: every file and every field
mime push <folder> --to <service> --keep-site-edits      # keep each edit made on the site, into [record]
mime push <folder> --to <service> --overwrite-site-edits # write over each edit made on the site
mime pull --from zenodo|zenodo-sandbox <record>          # a published record, by its number
mime pull --from <service> <record> --signed-in          # one shared with your account, with its stored token
mime pull '<site>/records/<record>?token=...'            # one shared by a link, with the key it carries
mime pull --link -                                       # the same, the link asked for with no echo
```

A private draft is made without a question (the plan is shown first, `--discard` undoes it). `--yes` answers
`--discard`'s question only: it never publishes, never declares a permission and never settles an edit made on the
site. Every push, `--status`, `--discard` and `--publish` ends with a link: the draft's page, the record's once
published, the account's drafts once `--discard` has deleted it. **`--to` is required**: `zenodo` and
`zenodo-sandbox` are two services, each fixed in code (its API, its site, its token, its records), never one flag
apart by accident. The token is the one `mime credentials set <service>` stored, or `ZENODO_TOKEN` /
`ZENODO_SANDBOX_TOKEN`; a draft needs the scope `deposit:write`, publishing `deposit:actions`.

## `mime.toml`

**One manifest for every kind of shared folder**, beside what it describes and never in it (`<folder>/mime.toml`):
its form (`format`, `mime.push.MANIFEST_FORMAT`), exactly one kind table and `[record]`. **Everything outside `[record]` defines the thing;
`[record]` is metadata**: Zenodo keeps a record's metadata editable and its files immutable, so a misspelt name, or a
paper's DOI added after publishing, is a metadata edit, never a new version.

```toml
format = "mime.toml/1"

[bundle]                 # a saved folder; [study] for a study folder
licence = "CC-BY-4.0"    # the experiment's own licence, which it declares and its files carry

[record]
title = "full_toy"
description = ""
creators = [
  { family = "Carberry", given = "Josiah", orcid = "0000-0002-1825-0097", affiliation = "Brown University" },
]
keywords = ["helices"]
also_cite = [{ identifier = "10.1234/paper", text = "A. Author et al., 2027" }]
```

| table | key | what |
|---|---|---|
| top | `format` | its form, `mime.push.MANIFEST_FORMAT`; a later one is `load.newer-form` |
| `[bundle]` | `licence` | a saved folder's: equal to the licence the experiment declares (`exp.licence(...)`, bound to its files); `--init` pre-fills it from your `[licences] content`, else the Creative Commons Attribution licence (`mime.push.DEFAULT_LICENCE`), where the experiment declares none, and push asks the experiment to declare it too |
| `[study]` | `licence` | a study folder's: equal to the licence its base experiment declares. The study folder's layout holds `mime.toml` (an older MIME refuses such a folder as a stray, `study.stray-file`); pushing a study comes with a later release |
| `[record]` | `title`, `description` | the record's title, and a paragraph shown above the card |
| | `creators` | each person (`family`, `given`) or `organisation`, with an ORCID iD (held to its form and check digit), an affiliation and its ROR identifier; one inline table a line |
| | `contributors` | the same, each with a DataCite contributor type (`DataCurator`, `Supervisor`, ...); the rights holder of each third-party file the record holds is added by push as `RightsHolder` |
| | `keywords`, `related` | searchable words (push adds `mime-experiment`); related works by DOI or `https://` address with DataCite's relation |
| | `also_cite` | works whoever uses the record is asked to cite as well: `identifier` (a DOI or an `https://` address) and `text` (how it is cited, optional). The card ends "If you use this, please also cite: ...", and push states each as a related identifier `IsSupplementTo` unless `related` names it |
| | `uploader` | the person whose token pushes, and their institution: a declared permission's grantee must be one of them |
| | `funding` | each funder (`funder`, its `ror` where known: Zenodo's funders are ROR's) and its award (`award`, its number; `title`; `url`); the award is sent as the record's own (title and number), never as an id of Zenodo's awards vocabulary, which MIME cannot check offline |
| | `conference` | Zenodo's meeting field: `title`, `acronym`, `dates`, `place`, `url`, `session`, `part` |
| | `access` | who may read the record and its files once published, where that is not everyone: `record`, `files` (`public` or `restricted`), and `embargo` (`until`, a day, quoted or a bare TOML date; `reason`). An access set on the draft's page is a site edit, kept here; left out, both are public |
| | `fields` | any other field of the record, in the records API's own names and shapes, held to those shapes by path before anything is sent: `subjects`, `dates`, `languages`, `references`, `additional_titles`, `additional_descriptions`, and Zenodo's custom fields `"journal:journal"`, `"imprint:imprint"`, `"thesis:university"`, `"thesis:type"`; a field MIME builds, or one `[record]` has a key for, is refused here by name |

A block's tables (`[block]`, `[requires]`, `[sockets]`, `[[assets]]`, `[licences.*]`, `[[permissions]]`, `[[validity]]`,
`[cases]`) are reserved for blocks and refused by name until this MIME reads one (`manifest.kind`), as are no kind table,
two, and a kind its folder is not. A key missing, unknown or of another type is refused by its table and path
(`manifest.key`); no creator, or an ORCID iD whose form or check digit is wrong, as `push.creators`; a file that is not
TOML that reads, or bytes that are no text, as `manifest.read`. An ORCID iD's digits are ASCII ones, and a ROR identifier's: a
full-width or another script's digit, which a terminal shows as a digit, is refused wherever either is read
(`mime.toml`, the defaults, `--setup`). `pack` neither carries nor refuses the file (the bundle's bytes are the same with
it and without it), and `mime card` names its creators.

**`sharing.yaml` is no longer read** (a development release keeps no reading period). A folder holding it alone is refused
(`manifest.missing`) with the route: `mime push <folder> --setup` writes `mime.toml` from it, a field at a time, each
of its values offered to keep (`--init` writes it without questions); MIME never deletes your file, so you delete
`sharing.yaml` once `mime.toml` holds what you want. A folder holding both is refused (`manifest.both`), naming the one to
delete: two files that may disagree are never merged.

**`--setup`** walks `[record]` and the kind table's licence a field at a time, in the order above, each with one line on
what it is for, in a terminal only (`push.setup` otherwise, naming `--init`), through the command line's own question
path. **Enter skips** a field; **a field that holds a value already** (what the file holds, what `sharing.yaml` held, or
a default where the file is new) is asked as `use <value>? [Y/n]`: Enter or `y` keeps it, `n` asks for another; **a value
that is not valid is said and asked again**, by the same checks (an ORCID iD by its check digit, a ROR identifier by its
form, DataCite's contributor types and relations, the experiment's own licence). `fields` is written by hand, and said so.
At the end a new file is written as `--init` writes one, each value in its place; an existing file is edited key by key,
and the keys it changed are said. It works on a study folder too (writing `[study]`), and then names `mime check`.

**Editing in place**, by `--setup` and by the write-back of a site edit (below): a key's lines are the fewest lines from
its `key =` line that TOML reads as that one key, so a bracket or a `#` inside a string is never taken for its end. They
are replaced, every other line kept byte for byte, comments and line endings included (CRLF stays CRLF); the comment
that ends the key's first line is kept on its new first line, and a comment inside its old value goes with it (those
lines are the key's own). The new lines keep the key's indentation and take the file's line ending (CRLF where any line
of the file is CRLF), and a file whose last line had no line feed has one once that line is replaced. A value
is written on one line, but for an array of tables, one table a line; a text of several lines is written escaped
(`"one\ntwo"`). The file is read again and must hold the edit and every other key as it was, or nothing is written. **What
an edit cannot replace safely is refused, never guessed** (`manifest.write`, the file left byte for byte): a key of
`[record]` written as dotted keys or as a table of its own (`fields.languages = ...`, `[record.fields]`), and `[record]`
written as an inline table, as dotted keys at the top (`record = { ... }`) or as an array of tables (`[[record]]`); write it as one key a line, `fields = {
... }`, and run again. A file that is a link is edited through its link, which stays a link, and a read-only one is
refused before anything is asked (`manifest.write`): it is never replaced. With standard output and standard error both
redirected to files, the questions go to the terminal (`/dev/tty`), never into the file. Defaults are offered for a new
file only. `--setup` takes no flag of a push or a publish (`--confirm`, `--yes`, `--full-plan`, the site-edit flags), nor
does `--defaults`. What push would refuse in it as it stands is said, not refused.

**`mime check <folder> [--profile exploratory|research]`** checks `mime.toml` offline, nothing built, packed or sent,
on the profile workflow (the one a build follows from exploratory to research): each item has a class, and the profile says what the class is, as the profile
rules do for a build (`mime.push._check.RULES`):

| class | what | exploratory | research |
|---|---|---|---|
| a defect | a key missing, unknown or of another type, an ORCID iD's check digit, the kind table's licence unlike the experiment's, a declared permission granted to another than the uploader, a path of this machine or a stored token in `[record]` | a finding | a finding |
| what push still needs | no creator named, no title, an experiment that declares no licence, a declared permission with no `uploader` named | a to-do | a finding |
| what a reviewer would ask | a person among the creators with no ORCID iD, no description, no keyword of the author's own | not shown | listed, never a finding |

The check runs push's own refusals on `[record]` (`push.permission`, the metadata scan) beside each key's check. It builds
and packs nothing, so what needs the bundle packed or a service chosen is not checked here: the folder's own files
(`mime pack`'s refusals), the record's size and file count, a file whose bytes may not travel, a cached asset's bytes and
a sandbox location pushed to production; a clean check says the manifest, not the folder, would pass.

No profile relaxes an input error; research refuses what exploratory waives and lists what a reviewer would ask; `strict`
is refused by name (`profile.reserved`), as a build is. The profile is the experiment's own (a study's base
experiment's) unless `--profile` gives one. Each item comes with the line to write and where it goes, as `mime pending`
gives the line to paste. A finding exits one, to-dos and listed items alone exit zero, and a folder that cannot be checked
(no `mime.toml`, `sharing.yaml`, TOML that does not read, a kind its folder is not, a newer format) is refused, exit two.

**Your defaults** (`defaults.toml`, in MIME's config directory, `platformdirs`, beside the
credentials file; never in a shared folder): `creators` at the top (a plugin's authors take them too); under
`[sharing]` the `contributors`, `keywords`, `uploader` and `funding` you repeat from record to record; under `[licences]`
the licence a new record's `content` takes where its experiment declares none and a scaffolded plugin's `code`; under
`[scaffold]` the `namespace` a scaffold prefills. Each is held to the checks `[record]` holds the same key to (an ORCID iD
by its check digit), a licence to SPDX's, a namespace to the registry's form; a problem is `defaults.file`, and a token
MIME holds is never saved in them, nor licence acceptances or asset mirrors, which keep their own files. `--init` fills
a new `mime.toml` from them, `--setup` offers each for a new one and asks at its end whether to save what it set
(`[y/N]`), **`mime defaults`** (or `mime defaults show`; `mime push --defaults` is its alias, the same bytes) shows them
and where they live, and **`mime defaults edit`** walks them a field at a time, in a terminal (`defaults.no-terminal`
otherwise), and writes them whole or not at all. **Moved once from `sharing-defaults.yaml`**: with `defaults.toml`
absent and the old file present, the first read writes `defaults.toml` from it (checked, whole or not at all) and says
so; the old file is kept, never read again, and `mime defaults` says it may be deleted. Where the config directory cannot
be written, the old file is read and the line says why.

## The plan

Every push begins with the plan, computed offline after the folder loads, with every refusal at once, and nothing is
written to the service before it is shown. **By default it is short**: what goes up (each record file and its size),
what travels as a reference and never goes up, the licences and creators, and each warning on a line of its own that
starts `!` (results that depend on a file the record does not hold, a declared permission, which becomes public when
published, a version that changes the title, access that is not open, a lock naming a record push has no record of,
the production service). **It always names the title and the contributors**, since the statement at a publish asks the
person to vouch for them. **The full plan** (`--full-plan`) is every
record file with its licence and why it is there, every file inside the bundle with its licence, whose it is and its
verdict, every file that travels as a reference with its origin and sha256, the licences, the credit, each declared
permission in its fixed words ("You are declaring publicly, under your name, that ... permitted you to redistribute
..."), each field of the metadata MIME builds on a line of its own, and the availability statement. **A push with
nothing to send** (the folder and `mime.toml` as push last sent them, the draft as push left it) reads the draft and
says so in one line, ending with its link; a published experiment unchanged since its last version says `unchanged
since <DOI>` in one line, with no request.

**The metadata** is InvenioRDM's: resource type `dataset`; the title, creators, contributors, keywords and related works
from `[record]`; the publisher `Zenodo` and the date, which DataCite needs to register the DOI; the card
(`mime card`'s, given the record it is published as: its version, DOI and how to pull and cite it, and its concept DOI
from the second version on, a first version's being minted when it is published) as the description, in HTML; every licence as a rights entry, an SPDX id as the vocabulary's id in
lower case and a `LicenseRef-` licence as a title, a link and what it allows with **no `id`** (the sandbox refuses a
`LicenseRef-` id); the MIME release it was saved with as `requires` and `iscompiledby`; and push's identifier of the
draft, a `urn:uuid` among its alternate identifiers (below). Each third-party file the record holds adds its rights
holder (its credit) as a `RightsHolder` contributor, as each declared permission adds its grantor; a file that travels
as a reference is not in the record, so its rights holder is credited beside it in the NOTICE and the card, and is no
contributor of the record. Funding, the conference and `fields` are sent as `[record]` holds them (`metadata.funding`, the custom fields
`meeting:meeting` and those `fields` names, and the metadata fields it names), and the version's number as
`metadata.version`. A field the service refuses, in any answer (a success that lists `errors`, or a refusal),
is said by who owns it (`mime.push.field_owner`): one the author writes (`[record]`, or the experiment's terms) is
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
time it wrote it (a hash of each field, custom fields and the record's access included; a field edited while the files
upload keeps its earlier hash, so the next push finds it). The next push (and `--publish`) reads the draft
first; each field that changed since, and that push would now write otherwise, is shown with its value on the site
and the value from `[record]` (or MIME's), and settled field by field: **kept**, written back into `[record]`
under its key (that key's lines replaced, every other line and comment outside them kept, its line endings too, a comment
inside the replaced lines going with them), the file read again and the payload made from it held to the site's value and every
other key as it was, or put back byte for byte and the push stopped (`manifest.write`, with the lines to copy in by
hand); or **written over**. In a terminal push asks (`Write mime.toml's over the site's? [y/N]`, so the
default keeps the site's); `--keep-site-edits` keeps every edit `[record]` can hold, `--overwrite-site-edits`
writes over every one, and without either or a terminal an edit stops the push (`push.metadata-edited`). A field
`[record]` cannot hold (the licences, which are the files'; the publisher, the date, the resource type, the card,
push's identifier) is written over or stops the push. A question shows a long value around the first place the two
differ, so its two lines never read the same. **An access setting is one of these edits**: files embargoed or restricted
on the draft's page are shown and settled like any other field (kept, into `[record]`'s `access`, so the plan says
so and the record is published so), and a preview chosen there survives a write. What the service adds to what it is sent (a person's display
name, a vocabulary entry's title) is not an edit: values are compared as the author writes them. Files stay MIME's: a
file added on the site is refused (`push.draft-changed`). `--status` names the fields edited since.

## Publishing

`--publish` publishes the draft an earlier push made, and only it: the folder packed again must make the files the
draft holds (`push.folder-changed` otherwise: push, look at the draft, then publish), and no draft is `push.no-draft`.
Each edit on the site is settled as a push settles it. **What is published is what was seen on the draft:** a script
(`--confirm`, or no terminal) publishes what the draft holds, the publication date aside, or nothing
(`push.metadata-changed`, naming each field with both values); in a terminal each field `[record]` changed since the
draft was written is shown by field, with its new value, and written to the draft. The plan is shown as it is
published, and then:

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
(`push.unconfirmed`). **A script's confirmation is bound to what it was shown**: every push prints,
beside the DOI, a short fingerprint of what the draft holds as it left it (the identity of its metadata, the
publication date aside, and each file's size, md5 and sha256), as `--confirm <DOI>@<fingerprint>`, and `--status`
prints it too. The DOI alone confirms the draft as the push that reserved that DOI, the first to print it, left it; once a push has changed the draft (the folder pushed again, a site edit kept), the DOI alone is refused
before any request (`push.confirm-stale`), as is a fingerprint the draft no longer holds, and the refusal names no
fingerprint a script could retry with unseen. A draft whose DOI was reserved before push kept a record of what it
showed (an earlier push's) is confirmed by `<DOI>@<fingerprint>` alone, changed or not: no later push stands for the one
that printed its DOI. The fingerprint is held to push's record of the draft, and the checks above hold the draft to
it: the folder's files (`push.folder-changed`) and `[record]` (`push.metadata-changed`) as push last wrote them.
**It is held once more immediately before the request**, computed from what the publish is about to publish (the files
the draft, read again, holds, and the metadata push writes as the publish read the folder; an edit made on the site is
held by its own flags): a push that lands while the publish reads the draft is refused there (`push.confirm-stale`).

**Read again immediately before the request.** The prompt can wait for hours, and the plan's own advice is to look at
the draft's page, so after the number, and before the one request that cannot be undone, the draft is read again: its
DOI the reserved one (read at the start of the run too), its metadata what push last wrote, exactly the plan's files.
Anything else publishes nothing (`push.draft-moved`, naming each). A draft published meanwhile is recorded, and no request
sent; one discarded is `push.no-draft`.

**A second concept is never made silently.** A folder whose lock names a published record push has no record of (a new
machine, a cleared state, a pulled folder) is told at every push that it makes a first version of a new concept; where
the record is on the same service, `--publish` refuses (`push.second-concept`) until `--version-of` comes, and saving the
experiment again, whose lock names no record, makes a new concept on purpose. A sandbox record's lock going to production
is told, and not refused.

**Never sent twice.** An answer lost is followed by reading the record: published, push goes on to the checks; not, it
stops, and the next run reads the record again first. A run that finds the draft already published (an earlier run's
answer lost, or the site's button) goes straight to the checks, once the record carries push's identifier
(`push.draft-foreign` otherwise), and publishes nothing. The request is sent again only after an answer of too many
requests, which refuses it unread; a service unavailable, whatever it states, is an answer lost. A folder that is read-only cannot have its lock written, and on
production that is refused before the request (`push.lock-unwritten`).

**The NOTICE as the record's preview.** The record holds the bundle's `NOTICE` as `NOTICE.txt`: Zenodo picks a
previewer by a file's extension (its plain text one takes `.txt`, its Markdown one `.md`), and a `NOTICE` with none was
shown no preview (the maintainer's sandbox session); the NOTICE is plain text, whose indented lines a Markdown renderer
would run together. The bundle's member keeps its name, so `mime pack` makes the bundle it always made. A publish makes
it the record's preview (`files.default_preview`), written with the publish's own write of the draft, since the records
API refuses a preview naming a file the draft does not hold yet (InvenioRDM's source reads so; not measured); a preview
chosen on the draft's page is kept. `mime pull` downloads the bundle alone, which carries the same NOTICE, held to its
hash; `NOTICE.txt` is not downloaded, and `mime card <folder> --notice` writes it again, byte for byte.

**The checks afterwards.** The version is recorded first, each file as the record's own listing holds it, never as the
folder holds it now: whatever the folder became, `unchanged since` is true of the record, and a re-saved folder's next
push makes its next version. Then the record's DOI is held to the reserved one, each file's size and md5 to the uploaded
one's with no other file, and the bundle, downloaded, to the sha256 packed; a difference stops here
(`push.after-publish`, the folder's lock left, and `--publish` again checks again and finishes). **The metadata never
holds the lock back**, which rests on the DOI and the bytes alone: each field is held to the draft as last read, by the
service's own answers, never to today's `mime.toml`. That the service keeps a field as the draft held it is a guess
the sandbox session measures, and a field that reads back otherwise is named (`push.metadata-read-back`) once the version
is recorded and the lock written; a later `--publish` says it again, with what was edited on the record's page since, and
refuses nothing. `mime push` of an unchanged folder says so where the lock is still owed.

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
(`POST .../versions`, the concept's open one where there is one), numbered as the service counts it where its draft says
so, and settles the edits made on the published record's page since, every field but those the service sets anew on a
version (`mime.push._draft.VERSION_RESETS`; which they are is the sandbox session's to measure): until they are settled,
push's record holds the published marks, so the same command stops at the same edit however often it runs. A paragraph
written above the card is kept above the new version's card. It links the files of the version before into it
(`files-import`, or those of the concept's open draft that are the published version's: one added on the site is refused,
`push.draft-changed`) and uploads only what differs: an unchanged file is linked, never sent again. A push of an unchanged
folder says `unchanged since <DOI>` (and, where
`[record]` changed, that a published record's metadata is updated on its page until `mime push --metadata`
comes). `--discard` of a new version's draft keeps the published versions.

## Pulling what was published

`mime pull --from zenodo-sandbox <record>` (or `--from zenodo`) reads the published record by its number at the
service's records API, with no token, and pulls its one bundle by its address there, held to the md5 and size the
record states; it says which version it is and whether a newer one exists. **A record not public** is read only when
asked: `--signed-in` sends the token `mime credentials` stored for that service (a record or files shared with that
account), and `mime pull '<site>/records/<record>?token=...'` the key a record's share link carries. Either travels as
push's token does: to the service's own API host alone, never on a redirect (a file's store is asked with nothing of
either), never through a proxy to a service on this machine, and never in a refusal, a traceback, a log, a printed
line or a file; a link is said by its record, and held as a `Token` from the moment pull takes it, so that no frame
below the caller holds it as text. **A link is read by its parts**, never its spelling: an address that carries a key
on a service's host, a name below it (`www.`) or the DOI resolver's, and is not a record's link in that service's own
scheme, host and port (`http://` for `https://`, another port or host name, a user part), is refused without its key
(`pull.source`); a written-out default port, an upper-case scheme or host and a trailing dot are the same address. An
address elsewhere that carries a `token` parameter is pulled as it was, its value hidden (`token=<hidden>`) in every
line, refusal and log line that says it. **`mime pull --link -`** asks for the link with no echo, and **`--link env`**
reads it from `MIME_SHARE_LINK`: a link given as an argument stays in the shell's history and, while pull runs, in its
arguments, which other accounts on the machine can read unless `/proc` is mounted with `hidepid`. **That the records
API takes a link's key as the parameter `token` is a guess** (InvenioRDM's, recalled) until a sandbox session makes a
link and pulls it. Without either, `pull.restricted`
names `--signed-in` where a token is stored, and how to store one where none is. An asset the record holds beside the
bundle is fetched as before, with no token. A DOI under DataCite's test prefix (the
sandbox's) is never resolved through doi.org, in any spelling of its address (`www.`, `dx.`, a port, an encoded slash):
`mime pull` of one is refused with the `--from` command (`pull.sandbox-doi`), and an asset location under it is never
tried there. A production DOI is pulled through doi.org
as before; a concept DOI resolves to the latest version, which pull names.

**What push remembers** is in the user's state directory (`platformdirs`), under
`push/<service>/<experiment>-<hash>/state.json`, keyed by the service and the experiment's exact name (the hash is of
the name, so `Toy` and `toy` have two records on a file system that folds case; the record is not keyed by the bundle's hash): the draft's id, its concept, its reserved DOI, its page, push's mark, the hashes of the
metadata it last wrote, and each file it uploaded and verified with its size, md5 and sha256. Any service that is not
one of the two fixed in code (a test's fake, a dry run, whatever its name) has a namespace of its own
(`push/other/<name>-<hash>/`), so a record made against a stand-in is never read or written for a real service; a
record that names another service, or another experiment (case included), is refused (`push.state`), as is one written
before push marked its drafts (push's first record format), which says what to delete.

## Testing it

The tests run against an offline fake of the records API (`tests/sharing/zenodo_records.py`), matched to the probe's
answers and replayed against them request by request (`tests/sharing/test_push_fidelity.py`, which names what the
replay cannot show). CI never holds an upload token.

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
publishes each draft (a declared permission asked in the terminal), says what reads back otherwise than the draft held
it and goes on, pulls each record by its number into a fresh folder and verifies it, publishes the toy's next version,
and says whether each file linked into that version is stored once (its file id the published version's), which only the
service can answer.

## Status

Push's draft side is built, after terms and tiers: the record's metadata file and `--init`, the client, the
plan, drafts, `--status` and `--discard`, the offline fake and the sandbox lane. Its publishing side comes
next: the quieter plan, `--publish` with its confirmation, the staged lock and the checks afterwards, versions, edits made on
the site kept or written over, funding, conferences and `fields`, and `mime pull --from`. Metadata updates of a
published record, communities, `--version-of` and asset records come later. In dev2 `mime.toml` replaced `sharing.yaml`
(for saved and study folders, blocks' tables reserved), with `defaults.toml`, `mime defaults` and `mime check`.

Releases are explained in the [package manifest](../README.md#status).
