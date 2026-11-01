# Live sessions and `mime.live/1`

A live session runs a built experiment while you watch it: it steps the run in chunks of one fixed size, hands each
chunk's state to its subscribers as a frame, takes the controller's live inputs between ticks, and records the stage
on request. In a script or a notebook it is `built.live()`; in a process of its own, for a viewer such as
MICROROBOTICA, it is `mime serve`, which speaks the protocol below.

## In a script

```python
from mime import Experiment
from mime.units import ms

built = Experiment.load("a0_saved").build()
session = built.live(chunk=ms(16.5), pace=1.0, out="a0_live")   # 33 base steps a chunk, at real time
session.subscribe(lambda frame: print(frame.step, frame.values["umr.pose"][:3]))
session.step(10)              # ten chunks now, on this thread
session.start()               # or step on a thread of its own, paced, until pause() or stop()
session.pause()
session.record(True)          # the recording layer, from the state as it stands
session.step(60)
session.stop()                # the run folder: a0_live/
```

- **One chunk size for the session.** Each new length of a compiled run compiles again, so a session keeps one: about
  sixty chunks a simulated second when `chunk=` is not given, rounded to whole base steps, and a whole number of
  control periods when a controller is called back.
- **Frames.** A frame is the state after a chunk: its number in the session (from zero), the base steps taken, the
  time, and the fields its schema names (`select=`, `<node>.<field>`; by default every pose and vector the stage
  binding moves, every per-step output of an observation, and every flag). Frame zero is the state the session
  starts from. A frame's values are the run's own, in its own float type: a batch run of the same steps gives the
  same bits.
- **Pace.** `pace=` is a factor of real time (the default holds the loop to real time), and `pace=None` runs it as
  fast as it runs. A session that falls more than a chunk behind its pace (its first chunk compiles; a slow machine)
  takes the time it is at as its new start, rather than running ahead to make the time up. So pacing guarantees one
  thing on every machine, which a test holds: no frame is ever ahead of real time. Keeping up with real time is the
  machine's, measured by `scripts/live_pace_benchmark.py` and never a CI check: on a laptop's i7-12850HX (JAX on
  its CPU), timed from the first frame after the compile, A0 through `mime serve` ran at 3.59 times real time
  unpaced and at 1.005 paced at real time.
  A slow subscriber is its own concern: the session never waits on one.
- **Live inputs are the controller's only.** A controller declares them beside its reads and writes
  (`Controller.live`, each a `LiveInput` with its units, shape, meaning and `default`, the value at which it changes
  nothing the step does: zeros unless declared); in the callback mode, `session.set(label, value)` hands one to its
  step under its label from the next tick on. A session hands every live input to the step from its first tick, at its
  default until it is set, so that a step compiled once (a `jax.jit`, as `ArmTracking`'s is) compiles at the session's
  first chunk with every input it will read, each a float of its declared shape, and **no `set` compiles it again**.
  A session that sets nothing is a batch run (which calls the step with its reads alone) bit for bit, as a test holds
  for A0. A physical setting is never a live input: changing one is a rebuild, a new session from the saved folder.
  `ArmTracking` declares `offset`, which moves the point it holds the magnet above along its axis.
- **Every command is logged with its step** (`commands.jsonl` in the run folder), so a session replays:
  `mime run <saved folder> --commands <run folder>/commands.jsonl --out <dir>` steps the same chunks, sets the same
  inputs and records at the same steps, and writes the same run folder. The log is written as the session goes: its
  header when the session opens, each command before it acts, and after each chunk the step it reached
  (`{"reached": <step>}`, before that chunk's frame is sent), each line flushed. A session that dies (a crash, a
  kill) leaves a log that replays to the last step it reached, every frame a client received included. `--out` is
  refused when it is the folder the session or the replay reads, inside it, or holds it (`run.out-over-source`).
- **Every session and every run starts from the declared inputs.** A live input lasts as long as its session: it is
  cleared when a session opens and when it stops, and a batch run of the same built starts with none set.
- **Stop always stops**, in every state: ready, running, paused, or taking a step (on another thread, or in `mime
  serve`'s loop), within one chunk. It sends a last frame marked final and writes the run folder. `pause` ends a step
  in hand after its chunk, as it stops a running session.
- **A chunk in flight holds back only what changes the run.** A command that changes what a run does (`set`,
  `record`, `pace`, `start`, `step`, `pause`, `stop`) waits for the chunk running on another thread and acts at its
  end; `status()` and `hello()` answer at once, the status's `activity` saying what the session is doing:
  `compiling` while its first chunk runs (which compiles the run's programs, unless this process compiled them for an
  earlier session of the same built experiment), `stepping` while a later one runs, `idle` between chunks, and its
  `step` the one the chunk started at.

## The run folder

Every live session that is given `out=` writes a minimal run folder when it stops (`mime serve` always does):

| File | What it holds |
|---|---|
| `experiment/` | the saved folder the session was loaded from, copied as it is (or the built experiment saved) |
| `commands.jsonl` | its header (the protocol, the chunk, the experiment's input hash), then one line per command with its step and one per chunk with the step it reached, written as the session goes |
| `outputs.npz`, `outputs.json` | the state at the end and its manifest, as `RunResult.read` reads them |
| `run.json` | the session's record: steps, frames, wall time, the environment |
| `topology.json` | the built graph in MIME's names (`built.topology()`) |
| `report.json` | the build report's summary (`built.report_summary()`, below), as `RunResult.read` reads it |
| `recording.usdc` | the recording layer, when recording was on ([recordings](recordings.md)) |

`mime run`'s run folders (`run_to`) hold `report.json` too. A run folder is self-contained: it carries its own stage,
and nothing outside it. A folder saved by an earlier stage
writer (before chains' frames, visuals and vectors were drawn) loads and verifies as it was written, and a live session
of one whose stage lacks a prim the binding moves is refused (`live.stage-writer`): save it again with this MIME.

## `mime serve`

```sh
mime serve a0_saved                       # commands on tcp://127.0.0.1:5555, frames on tcp://127.0.0.1:5556
mime serve a0_saved --commands tcp://127.0.0.1:0 --frames tcp://127.0.0.1:0 --ready-file ready.json
mime serve a0_saved --start --unpaced --out runs/a0_live
```

It loads and builds the saved folder, opens its session and serves it until a `stop` command, SIGTERM or SIGINT, all
of which stop it cleanly and write its run folder. **It does not outlive the process that launched it**: when that
process is gone (a viewer that crashed, a test runner killed), the session stops as on SIGTERM, writes its run folder
and the server exits. On Linux the kernel signals it (`PR_SET_PDEATHSIG`); on every platform the loop also looks between
chunks at whether its launcher is alive (on POSIX its parent's pid changes when it is adopted; on Windows the
launcher's process handle is signalled). `--outlive-launcher` serves on regardless, for a server started to run
detached (with `nohup`, say). **It binds to localhost.** A session has no authentication of its
own: reach a remote one through an SSH tunnel, which authenticates and encrypts; `--remote` binds another address all
the same.

```sh
ssh -L 5555:localhost:5555 -L 5556:localhost:5556 <host>     # then connect to localhost as if it ran here
```

**It never goes silent.** Each chunk runs on a thread of its own while the server's loop moves bytes and answers, so a
client is answered while a chunk compiles or steps:

- `hello` and `status` are answered at once, with the status as the chunk runs (`activity` `compiling` or
  `stepping`, `step` the chunk's start);
- `stop` and `pause` are answered at once too, `ok` with the status as the chunk runs and the command in `pending`,
  and act at the chunk's end, as between chunks: a `stop`'s reply then names the folder it is writing, and the
  session's last frame, marked final, comes when it has stopped;
- every other command is answered at the chunk's end, where it acts.

The commands taken while a chunk runs act at its end in the order they came, so the command log holds them in that
order at the chunk's end step. Measured by `scripts/live_responsiveness.py` on a laptop's i7-12850HX (JAX on its CPU),
A0 with its controller called back, three runs: while its first chunk compiled (3.3 s), the server answered 252 to 265
`status` requests, the median in 0.4 ms and the slowest in 82 to 97 ms, so within 0.1 s; the step after its first
`set` took 3.5 to 3.9 ms, as any later step does. Before phase 7n-b the first `set` compiled the controller again: in a
script on the same laptop, A0's apparatus closed by `ArmTracking` took 1.9 s for the chunk after it. These are a
machine's measurements, never a CI check; a test holds the server answering `status`, `hello` and `pause` while a
first chunk is held open, and its control, a server stepping on its loop's own thread, answering none.

The transport is ZMQ: a REP socket for commands and a PUB socket for frames. MIME speaks ZMQ's wire protocol itself
(ZMTP with the NULL mechanism), so `mime serve` needs no extra package, and any ZMQ library connects to it as it would
to libzmq's own sockets: a REQ (or a DEALER) for commands, a SUB for frames. `mime.live.LiveClient` is a Python client.

## The protocol: `mime.live/1`

The messages are written independent of their transport: a WebSocket or a WebRTC data channel could carry them as
they are. Within `mime.live/1` fields may be added (a new schema id; a client ignores what it does not know); any other
change is a new major version.

### Requests and replies

A request is a JSON object, `{"command": "<name>", ...}` (with `"protocol": "mime.live/1"`, optional). A reply is a
JSON object with `protocol`, `ok` and `command`; on success the session's status (`state`, `step`, `time`, `frames`,
`pace`, `recording`, `non_finite`; `stepping`, the chunks a step in hand has still to take; `activity`, what the
session is doing, `compiling`, `stepping` or `idle`; and `pending`, the commands taken while a chunk runs that act at
its end, in order); on a refusal,
`errors`, each in the shape of [MIME's errors](errors.md): `code`, `class`, `message`, `why`, `action`, `where` and
`page`. A request a client gets wrong is refused with its code (an argument of the wrong type included: `live.step`,
`live.input-value`, `live.record`, `live.pace`), and the session goes on; an error of MIME's own inside a command is
answered in the same shape, the session going on. The server never stops for one peer's bytes: a peer that does not
speak ZMTP, does not finish its handshake in time, sends a message over its socket's limit, or sends requests without
reading their replies is closed, and the others are served on (the limits are `mime.live._zmtp`'s constants).

| Command | Arguments | What it does |
|---|---|---|
| `hello` | | describes the session (below) |
| `status` | | the session's status |
| `start` | | starts or resumes stepping, paced |
| `pause` | | stops stepping after the chunk in hand (taken while a chunk runs: answered at once, `pending`) |
| `step` | `chunks`, a whole number above zero | steps that many chunks now, from a session that is not running; answered when they are stepped, or when a `stop` or `pause` from another client ends the step (the server takes other requests between its chunks) |
| `stop` | | stops in every state, sends the last frame, writes the run folder (its path is in the reply; taken while a chunk runs: answered at once, `pending`) |
| `set` | `values`, live inputs by label | sets the controller's live inputs from its next tick |
| `record` | `on`, true or false | turns the recording layer on or off |
| `pace` | `rtf`, a factor or null | the real-time factor, or unpaced |

`hello` returns MIME's version, the experiment's name and its hashes, the base step and the chunk in base steps, the
controller (its entry, mode, period and latency) and its live inputs (each with its units, shape, meaning and
`default`), the build report's summary (`report`, below), the path of the stage the session serves
(`stage`), what the asset cache draws of its visuals (`visuals`: the overlay to add over the stage, and the visual sets
it lacks, and every set's notice, its credit and terms, whether its meshes are drawn or not: see
[recordings](recordings.md)), the stage binding (`binding`), the topology, the frame schema (`schema`), the commands,
and the status. `stage` is the path on the machine that serves the session: a client on another machine opens its own
copy of the saved folder (the one it gave `mime serve`, or a `mime pull` of it) and checks it is the same experiment
by `hashes`, the lock's.

### Frames

Each frame is one message on the PUB socket: binary by default, JSON with `--json-frames`. A binary frame is a header,
then each field of the schema in its order, its bytes little-endian in the run's own type:

```text
offset  size  field
0       4     magic "MIMF"
4       2     version, u16 (1)
6       2     flags, u16 (bit 0: the session's last frame)
8       4     schema id, u32 (the first four bytes of the sha256 of the schema's text)
12      4     frame number, u32 (from 0, one per frame sent: a gap is a dropped frame)
16      8     step, u64 (base steps taken)
24      8     time, f64 (seconds)
32      ...   the fields, in the schema's order
```

The schema (in `hello`) lists each field: `path` (`<node>.<field>`), `node`, `field`, `shape`, `dtype` as NumPy spells
it (`<f4`), `unit` where MIME knows one, and its byte `offset` after the header; with `frame_bytes`, the whole frame's
length. It is fixed for the session. A JSON frame holds the same keys as the header (`frame`, `step`, `time`,
`schema`, `final`) and `fields`, each as nested lists. A client checks each frame's number against the last
(`mime.live.FrameOrder`): a gap is a dropped frame, a number at or below one seen is one out of order. A ZMQ
subscriber that connects just before `start` may miss the first frames, as ZMQ's subscriptions take a moment to reach
the publisher (its "slow joiner"): connect the frames' socket first, then send `hello`, and read the first frame's
number rather than assume it is zero. The server moves bytes while a chunk runs, so a ZMQ heartbeat
(`ZMQ_HEARTBEAT_TIMEOUT`) is answered while a chunk compiles. A frame's header carries the step it is at: a client's
status line that follows the frames is current between its requests.

### The build report's summary

`report` in `hello`, and `report.json` in every run folder (`built.report_summary()`; `RunResult.read(folder).report`),
is the build report as a client shows its claims, plain JSON of format `mime.report/1`:

| Key | What it holds |
|---|---|
| `format`, `experiment` | `mime.report/1`, the experiment's name |
| `profile` | the profile (`exploratory`, `research`, `strict`), its stamp's text, each item pending and each listed, what was measured of them, and each default the build took |
| `validity` | each validity line, with its `owner` and its `kind`: `declared` (declared, not yet checked), `computed` (computed at the build) or `record` (a limit a capability record states) |
| `watches` | what each run is compared with after it (a flag on the result, never a refusal) |
| `flags` | each flag a run may raise: its `path` in a run's outputs and frames, its `owner` and `entry`, its `node`, its `meaning` (what it says when raised), and its `limits` where it is read against them |
| `settings` | each setting by `path`: its declared `role` (`design`, `physical`, `numerical`, `standard`, or null when its entry declares none), its `default`'s kind (`determined`, `convention`, `numerical`, `none`, `undeclared`), the default the build `taken` (its text, or null), its `source`, and the items of the profile's stamp about it, `pending` and `listed` |
| `summary` | the report's summary, its lines as `built.report.text()` prints them |

A run folder written before this has no `report.json`, and `RunResult.read` gives `report` as `None`; a report of
another format is refused (`result.format`).

### The stage binding

`binding` says which field moves which prim of the stage (`built.binding()`), so a client needs no knowledge of MIME's
state layout. Each entry has the prim, the node and field, the row of a stacked field (`index`), its `kind`, the thing
it belongs to, its public name and its units. Every bound prim is placed in the world (a chain's frame prims reset
their transform stack), so a `pose` (seven numbers: x, y, z, then the quaternion w, x, y, z) is the prim's translate
and orient as it stands, applied with no composition. A `vector` (a field a law reads at a body, three numbers in the
world's axes) is drawn at the position of the pose `at` names.

**Frames carry metres; the stage has its own units.** A pose's x, y and z are in metres (as its `unit` says) whatever
the stage is written in, and a stage's translate is in the stage's units. So a client reads the stage's
`metersPerUnit` once (`UsdGeom.GetStageMetersPerUnit(stage)`, the root layer's `metersPerUnit`) and writes
`translate = (x, y, z) / metersPerUnit`; the orientation is unitless and goes in as it is. In a stage in metres, as
A0's is, the two are the same numbers; in one in millimetres the translate is a thousand times the frame's numbers.
A vector's prim is translated the same way (the body's position divided by
`metersPerUnit`); its arrow is a metre long in the stage's own units already, so its uniform scale is `|v|` times the
prim's `mime:vector:metres_per_unit`, with no conversion. The recording layer is written so already
([recordings](recordings.md)).

## What this does not do yet

- **Pixel streaming** of the stage (WebRTC, Selkies) is MADDENING's next release: a session's frames are for local
  rendering, by a client with the stage.
- **A controller inside the compiled step** takes no live input: its state is in the graph. Declare it as a callback
  to steer it live.
- **Batch runs on Modal** play from their run folders (`mime modal`, then the folder), not live.
