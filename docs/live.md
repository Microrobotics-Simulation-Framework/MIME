# Live sessions and `mime.live/2`

A live session runs a built experiment while you watch it: it steps the run in chunks of one fixed size, hands each
chunk's state to its subscribers as a frame, takes the controller's live inputs between ticks, and records the stage
on request. In a script or a notebook it is `built.live()`; in a process of its own, for a viewer such as
MICROROBOTICA, it is `mime serve`, which speaks the protocol below.

## In a script

<!-- mime-fragment: the session API in a reader's script, shown and not run here; tests/live/ holds the session's behaviour -->
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
  binding moves, every per-step output of an observation, and every flag, but no surface's points or fields, which
  `select=` names: a large surface would be most of every frame). Frame zero is the state the session
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
  **The log is never replaced or written into**: a session opens it exclusively, before it writes anything else into
  its folder, so a folder that holds one already, another session's, is refused (`live.log-exists`), with `force` or
  without, and left as it is.
- **A recording has a budget**, the samples it may hold (one a chunk, over every stretch of it): `record_budget=` (or
  `mime serve --record-budget`), by default as many as fit in 512 MiB, which `hello` gives as `record_budget`. A rig's
  sample is its bound fields: A0's apparatus closed by `ArmTracking` takes 359,833 samples (100 minutes at 60 frames a
  second), a sheet of 40,000 vertices 670 (`scripts/live_message_sizes.py`). At the budget the recording stops by itself, keeping every sample it took (the
  status's `recording` turns false and its `recorded` reaches the budget), and a `record` that would start another
  stretch is refused (`live.record-budget`); `record` off is always taken. The budget is in the command log's header, so
  a replay stops where the session did. A budget, not a ring that drops the oldest samples: a recording is what the log
  says was recorded, from the step it began, and its end is a coded event a client sees, never a silent loss of its
  start.
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

Every live session that is given `out=` writes a minimal run folder when it stops (`mime serve` always does). The
folder takes `run_to`'s rule: a new folder, an empty one or MIME's own run folder (`mime serve`, as `mime run`, refuses
MIME's own too), and any other is refused (`run.out-not-empty`) unless `force=True` (`--force`), which writes into it,
replacing only the files of the names below and keeping every other file, `experiment/`'s included. **A name of those
below that is a symbolic link, or a folder of them with a link inside, is refused even with `force`** (`run.out-not-empty`,
naming the entry): a write through it would land where it points, outside the folder. `mime run --force` and
`run_to(force=True)` share this check.

**Python and the CLI differ on MIME's own run folder.** `mime serve` and `mime run` never write over one without
`--force`. `built.live(out=<a MIME run folder>)` follows `run_to`'s rule instead (so that a script that runs again into
its own folder keeps working): it replaces that folder's outputs without `force`, and refuses only a folder that
holds anything else, a command log included (`live.log-exists`).

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

<!-- mime-fragment: the serve command with its endpoints in comments, for a saved experiment the reader has -->
```sh
mime serve a0_saved                       # commands on tcp://127.0.0.1:5555, frames on tcp://127.0.0.1:5556
mime serve a0_saved --commands tcp://127.0.0.1:0 --frames tcp://127.0.0.1:0 --ready-file ready.json
mime serve a0_saved --start --unpaced --out runs/a0_live
mime serve a0_saved --idle-timeout 15min --max-wall 2h --watch umr.pose[0:3]
```

It loads and builds the saved folder, opens its session and serves it until a `stop` command, SIGTERM or SIGINT, all
of which stop it cleanly and write its run folder. **It does not outlive the process that launched it**: when that
process is gone (a viewer that crashed, a test runner killed), the session stops as on SIGTERM, writes its run folder
and the server exits. On Linux the kernel signals it (`PR_SET_PDEATHSIG`); on every platform the loop also looks between
chunks at whether its launcher is alive (on POSIX its parent's pid changes when it is adopted; on Windows the
launcher's process handle is signalled). `--outlive-launcher` serves on regardless, for a server started to run
detached (with `nohup`, say). **It binds to loopback.** A session has no authentication of its
own: reach a remote one through an SSH tunnel, which authenticates and encrypts. An address that is not loopback
(IPv4's loopback block or IPv6's loopback address, or `localhost` where it resolves to loopback only; a host name is
not resolved for this) is refused before anything binds, `live.commands-not-loopback` for the command socket and
`live.frames-not-loopback` for the frame socket. `--allow-remote-commands` binds it all the same and warns on
standard error, naming each such address: whoever reaches the command socket can start, stop and step the session and
set its live inputs. A port in use, by another `mime serve` say, is refused, naming it (`live.address-in-use`):
`tcp://127.0.0.1:0` takes a free port, which the ready line and the ready file below give. The server binds the address
it checked (`localhost` is resolved once), and a port outside 0 to 65535 is refused as a malformed endpoint
(`live.endpoint`). A listener sets `SO_REUSEADDR` on Linux alone, so that a server restarted on its port binds it while
the last one's connections wait out TIME_WAIT (Linux refuses a second listener on an overlapping address with it all the
same). On macOS and the BSDs it sets none: there `SO_REUSEADDR` would let a second server bind `127.0.0.1:P` while a
first, allowed off loopback, holds `0.0.0.0:P`, and take the connections to loopback; so a restart on the same port there
may be refused until TIME_WAIT ends, and asking for any free port is the way. On Windows it sets none either, as there the
option lets a second socket bind the very address held. The platforms workflow runs the live tests on macOS.

`--out` is refused before anything binds or loads unless it is new or empty (`run.out-not-empty`, as for `mime run`),
and `--force` serves into it all the same, warning what it holds; a folder that holds a command log is refused, with
`--force` too (`live.log-exists`). Without `--out` the session makes its run folder here,
`<experiment>-live-<UTC time>/`, and a second session in the same second takes `-2`, a third `-3`.

**Reaching it from another machine.** Once it listens, `mime serve` says the `ssh` command that reaches the session:
each port forwarded to the address it is bound to, the login it runs as on this machine's own name for itself (use the
name the other machine knows it by, if another). Run it on the other machine, and connect there to the same endpoints:

<!-- mime-fragment: the line mime serve prints, its ports, user and host this machine's own -->
```text
mime serve: from another machine, run there ssh -N -L 5555:127.0.0.1:5555 -L 5556:127.0.0.1:5556 you@this-host (with the name it knows this machine by, if another), and connect to the same endpoints
```

### The ready file

`--ready-file <file>` is how a program that starts `mime serve` learns where to connect, whatever the server prints: it is
written as one JSON object once the sockets listen. Its format is part of `mime.live/2` and follows the protocol's rule
below, that fields may be added and a reader ignores what it does not know.

| Key | What it holds |
|---|---|
| `protocol` | `mime.live/2` |
| `commands` | the endpoint the commands are sent to, `tcp://<host>:<port>` (the port that was bound, when any free one was asked for) |
| `frames` | the endpoint the frames are received on |
| `out` | the run folder the session writes when it stops |

**A key that is absent means what it meant before it existed.** The transport's security is the first: an absent
`mechanism` means NULL, the mechanism a session speaks today (no authentication; reach a remote one through an SSH
tunnel). Two keys are **reserved and not written yet**, for a later session that serves with CurveZMQ, where the
server proves who it is and a client is known by its key: `mechanism`, which is `curve` there, and `server_key`, the
server's long-term public key as Z85 text of 40 characters (as `zmq_curve_keypair` writes it and a libzmq client takes it),
present only when
`mechanism` is `curve`. A client that finds a `mechanism` it cannot speak refuses; it never falls back to
connecting without the security the server asked for. The key travels in this file, or in a client's own settings for an
address it was given by hand, and never in `hello`, which goes over the connection the key secures.

### What the terminal shows

On a terminal, `mime serve` draws a live view on standard error (anywhere else it prints what it always printed: the load
report, the line when it listens, the line with the `ssh` command that reaches it and the line when it stops, on
standard output, byte for byte; `--plain`, `NO_COLOR` and
`TERM=dumb` ask for the same). It is a strip of a few lines: the state in a word (`READY`, `COMPILING`, `RUNNING`,
`PAUSED`, `STOPPING`), the step, the simulated time against the wall time, the speed as a multiple of real time and the
frames a second (over the last few seconds), the clients, whether it is recording and the flags seen raised, and how to stop.
Above it the scrollback keeps an event for each client that connects or leaves and each command with what came of it (a
connection that is refused before its handshake is counted and said once with the others, not one by one). It is drawn about
four times a second whatever the frame rate: the loop that answers clients only stores into a plain record of the
session's state, and the view reads it. **A terminal that is not read never keeps the server from stopping**: serve writes to
it through a writer of its own that waits a quarter of a second for it once and then not at all, and drops what the terminal
does not take (the view says how many writes it dropped, if the terminal takes that), so `--max-wall`, `--idle-timeout`,
SIGTERM and Ctrl+C each write the run folder and exit whatever the terminal does (Ctrl+S, a stalled ssh session, a
launcher that never drains its pseudo-terminal). There are no keys; a client is how a session is steered (a command is logged and
replays like any other).

Under the strip is a table of values: by default each scalar output of an observation, each flag, and the position, the
first three numbers of the pose, of each body the stage binds; at most twelve rows, fewer in a short window, the rest said in
one line. `--watch <path>` (repeatable) names the rows instead: a field of the frames, with an index where it is not one
number, `<node>.<field>[<low>:<high>]`, one integer or slice for each leading dimension of the field; `all` adds every scalar and three-number field; `none` hides the
table. A name the frames do not carry is refused (`live.watch`) with the nearest name, and **`--watch` adds nothing to the
frames and changes none**: they are what `--select` makes them. A row of one number shows its unit as the schema states it
(or says it declares none), its value now, its **minimum and maximum over every frame the server published** (a frame is
the state at a chunk's end, so these are not the extremes over steps), and the last ten seconds as a sparkline of the values
drawn. A row of two or three numbers shows them and no extremes. At a narrow width the sparkline goes first, then the minimum
and maximum, then the unit, and no number is ever wrapped.

`--idle-timeout <duration>` stops the session when no client has been connected for that long (counted from when it
listens, and again from the last client to leave), and `--max-wall <duration>` when its wall time reaches it (counted from
when the session opened, as `wall_time` in the status is, so the loading and building before it are not). Each stops it as
SIGTERM does, writing the run folder, and says why on standard output (`mime serve: no client has been connected for
<seconds> s (--idle-timeout); stopping`). A duration is written as `mime run`'s `--duration` is, with a unit or as seconds;
one that is not above zero is refused (`live.timeout`). **Neither limit stops a session at the second**:
as SIGTERM does, it finishes the chunk in hand and stops at its end, so the stop comes at the end of the chunk in
hand, later than the limit by what is left of that chunk (the first chunk compiles, which can take many seconds, and a limit that
comes while it is in hand is passed by them). A session
that is not stepping (ready, paused) has no chunk in hand and stops at the limit. Both limits are for a session that
nobody is watching: a forgotten one, or one on a machine that is billed by the second.

**It never goes silent.** Each chunk runs on a thread of its own while the server's loop moves bytes and answers, so a
client is answered while a chunk compiles or steps:

- `hello` and `status` are answered at once, with the status as the chunk runs (`activity` `compiling` or
  `stepping`, `step` the chunk's start);
- `stop` and `pause` are answered at once too, `ok` with the status as the chunk runs and the command in `pending`,
  and act at the chunk's end, as between chunks. A `stop` taken this way is *pending*, not done: its reply has
  `taken: true`, a `message` that says "stop pending", the command in `pending`, and no `folder` (the run folder does
  not exist yet; the reply of a `stop` between chunks, which has taken effect, has it). The session's last frame,
  marked final, or the closed connection says it has stopped; a `status` sent after that is never answered, since the
  server has exited;
- every other command is answered at the chunk's end, where it acts.

The commands taken while a chunk runs act at its end in the order they came, so the command log holds them in that
order at the chunk's end step. Measured by `scripts/live_responsiveness.py` on a laptop's i7-12850HX (JAX on its CPU),
A0 with its controller called back, three runs: while its first chunk compiled (3.3 s), the server answered 252 to 265
`status` requests, the median in 0.4 ms and the slowest in 82 to 97 ms, so within 0.1 s; the step after its first
`set` took 3.5 to 3.9 ms, as any later step does. Earlier in development the first `set` compiled the controller again: in a
script on the same laptop, A0's apparatus closed by `ArmTracking` took 1.9 s for the chunk after it. These are a
machine's measurements, never a CI check; a test holds the server answering `status`, `hello` and `pause` while a
first chunk is held open, and its control, a server stepping on its loop's own thread, answering none.

The transport is ZMQ: a REP socket for commands and a PUB socket for frames. MIME speaks ZMQ's wire protocol itself
(ZMTP with the NULL mechanism), so `mime serve` needs no extra package, and any ZMQ library connects to it as it would
to libzmq's own sockets: a REQ (or a DEALER) for commands, a SUB for frames. `mime.live.LiveClient` is a Python client.

## The protocol: `mime.live/2`

The messages are written independent of their transport: a WebSocket or a WebRTC data channel could carry them as
they are. Within `mime.live/2` fields may be added (a new schema id; a client ignores what it does not know); any other
change is a new major version.

**`mime.live/2`** (this page) replaced `mime.live/1`, which the 0.3.0 development line carried until `mime serve`'s hardening: four values
changed meaning, so a client that kept `mime.live/1`'s reading would misread them without a sign. `hello`'s `stage` is a path
relative to the run folder (not the serving machine's absolute path), `visuals.overlay` is the overlay's id (not its
path), each `visuals.fetch` folder is relative to the run folder, and `stop`'s `folder` is the run folder's name. Frames
and their version are unchanged. A client that sends `"protocol": "mime.live/1"` is refused with `live.protocol` in
its first request; one that sends no `protocol` is answered as `mime.live/2`, so read `hello`'s `protocol`.

### Requests and replies

A request is a JSON object, `{"command": "<name>", ...}` with `"protocol": "mime.live/2"`, which `hello` requires (a
`hello` without it is refused, `live.protocol`, so a client of the earlier protocol cannot read its fields wrongly) and
the other commands take as optional. A reply is a
JSON object with `protocol`, `ok` and `command`; on success the session's status (`state`, `step`, `time`, `frames`,
`pace`, `recording`, `non_finite`; `stepping`, the chunks a step in hand has still to take; `activity`, what the
session is doing, `compiling`, `stepping` or `idle`; and `pending`, the commands taken while a chunk runs that act at
its end, in order; and three optional keys below); on a refusal,
`errors`, each in the shape of [MIME's errors](reading_errors.md): `code`, `class`, `message`, `why`, `action`, `where` and
`page`. A request a client gets wrong is refused with its code (an argument of the wrong type included: `live.step`,
`live.input-value`, `live.record`, `live.pace`), and the session goes on; an error of MIME's own inside a command is
answered in the same shape, the session going on. The server never stops for one peer's bytes: a peer that does not
speak ZMTP, does not finish its handshake in time, or sends requests without reading their replies is closed, and the
others are served on.

**Message sizes.** Every message received is held to its socket's limit, from the header of the frame that takes it
over, before that frame's bytes are read, its parts counted together. A request over the server's is answered with
`live.message-too-large` in its turn, its bytes dropped as they arrive, and the same client is served on; a subscription
or a command over its limit closes that peer; a client refuses a reply or a frame over its own limit by the same code,
dropping it unread. The limits, set from the largest messages measured through the real session on the repository's
rigs (`scripts/live_message_sizes.py`, sizes no machine changes), each with its margin:

| Socket receives | Limit | The largest measured | Margin |
|---|---|---|---|
| a request (the server's REP) | 64 KiB | 93 bytes: A0's `set` of its live input at the longest float text (24 characters) | 700 times: a `set` of some 2,500 numbers |
| a subscription (the server's PUB) | 64 KiB | 1 byte: MIME's client subscribes to every frame | |
| a reply (a client's REQ) | 64 MiB | 0.71 MB: the `hello` of a sheet of 40,000 vertices, its clusters in the binding (A0's is 59.4 kB) | 90 times |
| a frame (a client's SUB) | 256 MiB | 5.3 MB: that sheet's points and two fields selected, as JSON at the longest float text (0.80 MB binary; A0's 888 bytes) | 48 times |

Every status also says how many samples the recording holds, `recorded`, over every stretch of it.

**Three optional keys of the status**, which `mime serve` always sends and a client that does not read them is unaffected by:

| Key | What it holds |
|---|---|
| `wall_time` | the seconds since the session opened (the loading and building before it are not counted) |
| `clients` | `{"commands": n, "frames": n}`: the requesters and the subscribers connected now, whose handshake finished |
| `recent` | the last five commands the session took, oldest first, each `{"command", "arguments", "step"}`: its name, its arguments as the command log holds them and the step it took effect at |

They carry **no address of a peer and no text of a peer's but what the session accepted**: `recent` is read from the
session's command log, which holds only commands that were accepted, so a request that was refused is not in it, and a
connection is a count. A session that has no server (`built.live()` in a script) has no `clients`. A watcher that renders
the view from `status` and the frames shows what these keys let it, and no more than a server tells every client.

| Command | Arguments | What it does |
|---|---|---|
| `hello` | | describes the session (below) |
| `status` | | the session's status |
| `start` | | starts or resumes stepping, paced |
| `pause` | | stops stepping after the chunk in hand (taken while a chunk runs: answered at once, `pending`) |
| `step` | `chunks`, a whole number above zero and at most `hello`'s `max_step_chunks`, the chunks that cover one simulated hour (`live.step` beyond it) | steps that many chunks now, from a session that is not running; answered when they are stepped, or when a `stop` or `pause` from another client ends the step (the server takes other requests between its chunks) |
| `stop` | | stops in every state, sends the last frame, writes the run folder (between chunks its reply holds the run folder's name, `folder`, never its path; taken while a chunk runs it is answered at once as pending, and that reply has no `folder` at all, the key appears only in the reply of a session that has stopped: the final frame or the closed connection says it took effect) |
| `set` | `values`, live inputs by label | sets the controller's live inputs from its next tick |
| `record` | `on`, true or false | turns the recording layer on or off; on is refused once the recording holds its budget (`live.record-budget`) |
| `pace` | `rtf`, required: a finite factor above zero, or null | the real-time factor, or unpaced; a request without `rtf`, or with anything else, is refused (`live.pace`) and leaves the pace as it was |

`hello` returns MIME's version, the experiment's name and its hashes, the base step and the chunk in base steps, the
controller (its entry, mode, period and latency) and its live inputs (each with its units, shape, meaning and
`default`), the build report's summary (`report`, below), the stage the session serves (`stage`), what the asset
cache draws of its visuals (`visuals`: the overlay to add over the stage, the visual sets it lacks with the command that
fetches each, `fetch`, and every set's notice, its credit and terms, whether its meshes are drawn or not: see
[recordings](recordings.md)), the stage binding (`binding`), the topology, the frame schema (`schema`), the commands,
the recording's budget in samples (`record_budget`), the most chunks a `step` may ask for (`max_step_chunks`), and the status. **`hello` and `stop` name no path of the machine
that serves the session.** `stage` is relative to the run folder, `experiment/scene/scene.usda` (its `experiment/` is the
saved folder's copy; `null` for a session with no run folder); `visuals.overlay` is the overlay's id, its file name in the
asset cache's `overlays/` folder; each `fetch` command names its folder relative to the run folder, to be run there; and
`stop`'s `folder` is the run folder's name. A program that started `mime serve` has the run folder's path in the ready
file (`out`), on its own machine; a client on another machine opens its own copy of the saved folder (the one it gave
`mime serve`, or a `mime pull` of it) and checks it is the same experiment by `hashes`, the lock's. Nothing here serves
a file: a client that needs one fetches it as it fetched the saved folder.

### The stage binding

`binding` in `hello` says which field of the state moves which prim of the stage, so that a client needs no knowledge
of MIME's state layout: `{"format": "mime.binding/2", "entries": [...]}`, bodies first, then each chain segment's
frames, then field vectors, then each surface's points and fields. Each entry holds `prim`, `node`, `field`, `index`
(a row of a stacked field, or null), `kind`, `thing`, `name`, `unit` and `space`, and by its kind:

| `kind` | The value | Keys of its own |
|---|---|---|
| `pose` | seven numbers, the prim's world translation (metres) and orientation `(w, x, y, z)`, applied to its `translate` and `orient` | `space` `world` |
| `vector` | three numbers in the world's axes, a field at a body, drawn at the pose `at` names | `at` (`node`, `field`); `space` `world` |
| `points` | a surface's points, a point a vertex, metres in its thing's frame, the mesh's `points` | `space` `thing`; `vertices`, the mesh's; `clusters` where decimated |
| `field` | a surface's values, one a vertex, the mesh's primvar `primvars:mime:<surface_field>` | `space` `thing`; `vertices`; `surface_field`; `reduction`, `mean` or `max`; `clusters` where decimated |

A surface over the stage's triangle budget is drawn decimated ([recordings](recordings.md#surfaces-that-deform-and-carry-fields)):
its entries' `clusters` hold the drawn vertex of each of the node's vertices, and a client that moves the mesh from the
node's own values (`select=` names them) draws vertex `k` at the mean of the points whose cluster is `k`, and a field's
value there at their mean or maximum, as `reduction` says. The format's second version adds the kinds `points` and
`field`; its `pose` and `vector` entries are the first version's, unchanged.

### Frames

Each frame is one message on the PUB socket: binary by default, JSON with `--json-frames`. A binary frame is a header,
then each field of the schema in its order, its bytes little-endian in the run's own type:

<!-- mime-fragment: the layout of a wire frame, a table in plain text -->
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

A run folder written before this has no `report.json`, and `RunResult.read` gives `report` as `None`. A report of a
format this MIME does not read (a newer `mime.report/N`) does not stop the run being read: `RunResult.read` gives
`report` as `None` and `report_unreadable` as the format the file states (`"not JSON"` for one that cannot be parsed);
it is `None` when the report was read, or there was none.

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
- **Who sent each command** is not in the command log yet: it records the key behind each command once sessions
  authenticate clients by key (CurveZMQ, a later release).
