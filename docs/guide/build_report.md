# Reading a build report

Every `.build()` returns a built experiment whose `report` says what the build did, in the experiment's own names. This page
goes through it by its parts. `built.report.text()` prints the summary; `built.report.text(details=True)` prints everything,
and `built.report.pending()` prints only what is pending ([profiles](../profiles.md)). The report is a frozen
`BuildReport`, so each part below is an attribute you can read in code, and the table is generated from the class:

<!-- mime-generate: build-report-fields -->

Each heading below is one of those fields, and a check holds the headings equal to the fields: a field added to the class
needs a heading here, and a heading for a field that is gone fails.

A report is a record of what was built, not a verdict that it is right. Where a part says *declared, not yet checked*, nothing
has tested it.

## `experiment`

The experiment's name, which every output carries. It opens the summary.

## `dt`

The base step in seconds. A node may run every n-th base step; `nodes` shows each node's rate. Without a base step the report
says so.

## `nodes`

Every node of the graph in the order it was added, with the entry that built it, the section and owner it belongs to, and its
rate. Read it to see which entry made which node. Only `text(details=True)` prints it; the summary lists the entries.

## `groups`

The coupling groups: sets of nodes the graph solves together, each with its tolerance and iteration cap, and who asked for it.
A group you did not expect has a `requested_by` that says which entry wanted it.

## `external_inputs`

The inputs the graph takes from outside, by name. Each must be fed; a controller declares the ones it feeds, and a run feeds
them ([install: feed every external input](install.md#feed-every-external-input)).

## `seeds`

For each entry that draws from a random stream, the seed derived for it from the experiment's seed. The same experiment gives
the same seeds, so a rerun draws the same numbers.

## `validity`

Each entry's own statement of where its model holds, printed as `validity of <entry> (declared, not yet checked)`. Read
these before trusting a result: they are what the entry says of itself. [Scope and limits](scope.md) lists them for MIME's
entries.

## `vocabulary`

The version of the vocabulary of each namespace the experiment's entries use, as the report was built under it.

## `refusals`

The refusals the build met, each a MIME error with a code that says what to change ([reading an error](../reading_errors.md)). The
summary prints `refusals: none` for a build that succeeded.

## `run`

How a run executes: as one compiled scan, or in compiled chunks with named things running on the host between them (a
controller called back, a hook, an observation outside the step).

## `warnings`

MADDENING's warnings from checking and compiling the graph, with where each came from and its text. Expect none; read any
that appear, they are the engine's.

## `derived_hash`

A hash of the built graph's canonical text. A saved experiment's lock holds it, and `mime verify` rebuilds the experiment and
compares: a changed hash means the graph changed.

## `values`

Every value an entry reads: its SI value and unit, where it came from (declared in the experiment, read from a record, derived
from parts), the reason, and which entries read it. This is the part to read for *what number did my experiment actually
use*. [Materials](../materials.md) says what each source means.

## `lags`

Every edge in the order added, with the step its value is read from: this step, the previous step, or the current iterate of a
coupling group. A previous-step read is *planned* when it carries a thing's published state or a chain segment's reaction, and
is refused otherwise unless a group closes the loop.

## `scene`

The scene's line: which way is up and the length unit the stage is written in ([the scene](../scene.md)).

## `orientation`

One line per thing: its kind, the stage prim it is, and each direction it declares, as the path it refers to beside the vector
it resolved to. Read it to check a direction is the one you meant.

## `stage_hash`

A hash of the USD stage written from the declarations, which a save writes to `scene/scene.usda` and records in the lock.

## `materials`

The conditions every record is read at; each record with the hash of its file and of its content; each read with
OpenMaterica's verdict and what the build did with it; and each body's parts. It prints only for an experiment that names a
record or a part.

## `descriptions`

Each chain segment's description as read (a URDF file, for an arm), listing every element not used or read past, with its
notes. An element MIME reads past is listed, and acknowledging one takes a reason
([the multibody note](../multibody.md)).

## `limits`

The limits each record states of its model, printed as `limits of <entry> (stated by its record)`.

## `controller`

The controller, with the mode it is placed in, its period and its latency, what it reads, and what it writes in place of a
drive. The line says when a tick reads the state and from which step its writes act.

## `masses`

Every mass mounted on a chain, with what carries it (a rigid body welded on a link, a spinner's rotor, nothing on a prescribed
motion), and a rotor's worst case across its axis and its kick on the joints of the segment that carries it.

## `profile`

What the experiment's profile made of it: the stamp every output carries, the items pending and listed, the defaults taken,
and the declared changes of role. Under the exploratory profile what you left unstated is waived and listed here; under
research it is refused ([profiles](../profiles.md)).

## `watches`

What the post-run watch compares after each run, for each entry it watches, and what is missing for it. A watch is computed
after the run and is a flag on the result, never a refusal and no part of the graph ([the results note](../results.md)).

## `assets`

Each asset or mesh a loaded experiment keeps at its pin while a source now holds another version of it. The pin wins:
the run reads the bytes the experiment was saved with, and the report names the version it did not take. To take the
other one, change the experiment's declaration and save it again ([assets in depth](assets.md)).

## `replaced`

Each declaration made in this session that replaced an earlier one, with what followed it: in a notebook or IPython, a
thing (an apparatus, body, medium or structure), a law, an observation or a value declared again and changed replaces the
earlier one, and the warning that says so can be removed by a warnings filter, so the experiment keeps it here, in the
order the replacements were made (`plan().replaced` holds the same lines). A thing's note names every law, observation,
value and other thing that holds its name and now finds the new one, and says when a declared value of the thing still
applies and overrides what the new declaration brings. It is empty in a script and for an experiment loaded from a folder.
The saved experiment holds the final declarations only ([reading errors](../reading_errors.md#a-declaration-made-twice)).

## `removed`

Each declaration removed in this session (`exp.remove(...)`), as a note: what was removed, and that nothing else named it
(a removal is refused while something does). Unlike a replacement, a removal raises no warning (it is your own call);
the experiment keeps the note (`plan().removed` holds the same lines). It is empty for an experiment nothing
was removed from. The saved experiment holds what was left ([reading errors](../reading_errors.md#taking-a-declaration-out)).

## `stale`

Set on a built experiment's `report` once the experiment has had a replacement or a removal since the build: what changed
and what to do (`this build predates the replacement of the apparatus 'field': build the experiment again`). The report's
first lines carry it as `STALE: ...`. It is `None` for a build that is current, and it is never part of the report's
summary for a client or of a run folder ([reading errors](../reading_errors.md#a-build-that-predates-a-change)).

## `checked`

Validity lines the build computed, by entry, such as a spinner's kick on its host's joints. Unlike `validity` they were checked.
They never refuse, and are the same in every profile.
