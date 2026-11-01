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
  fast as it runs.
  A slow subscriber is its own concern: the session never waits on one.
- **Live inputs are the controller's only.** A controller declares them beside its reads and writes
  (`Controller.live`, each a `LiveInput` with its units, shape and meaning); in the callback mode,
  `session.set(label, value)` hands one to its step under its label from the next tick on. Until one is set the step
  sees its reads alone, so a run that sets nothing is any other run. A physical setting is never a live input:
  changing one is a rebuild, a new session from the saved folder. `ArmTracking` declares `offset`, which moves the
  point it holds the magnet above along its axis.
- **Every command is logged with its step** (`commands.jsonl` in the run folder), so a session replays:
  `mime run <saved folder> --commands <run folder>/commands.jsonl --out <dir>` steps the same chunks, sets the same
  inputs and records at the same steps, and writes the same run folder.
- **Stop always stops**, in every state: ready, running, paused, or stepping on another thread (which finishes its
  chunk). It sends a last frame marked final and writes the run folder.

## The run folder

Every live session that is given `out=` writes a minimal run folder when it stops (`mime serve` always does):

| File | What it holds |
|---|---|
| `experiment/` | the saved folder the session was loaded from, copied as it is (or the built experiment saved) |
| `commands.jsonl` | its header (the protocol, the chunk, the experiment's input hash), then one line per command with its step |
| `outputs.npz`, `outputs.json` | the state at the end and its manifest, as `RunResult.read` reads them |
| `run.json` | the session's record: steps, frames, wall time, the environment |
| `topology.json` | the built graph in MIME's names (`built.topology()`) |
| `recording.usdc` | the recording layer, when recording was on ([recordings](recordings.md)) |

A run folder is self-contained: it carries its own stage, and nothing outside it.

## `mime serve`

```sh
mime serve a0_saved                       # commands on tcp://127.0.0.1:5555, frames on tcp://127.0.0.1:5556
mime serve a0_saved --commands tcp://127.0.0.1:0 --frames tcp://127.0.0.1:0 --ready-file ready.json
mime serve a0_saved --start --unpaced --out runs/a0_live
```

It loads and builds the saved folder, opens its session and serves it until a `stop` command, SIGTERM or SIGINT, all
of which stop it cleanly and write its run folder. **It binds to localhost.** A session has no authentication of its
own: reach a remote one through an SSH tunnel, which authenticates and encrypts; `--remote` binds another address all
the same.

```sh
ssh -L 5555:localhost:5555 -L 5556:localhost:5556 <host>     # then connect to localhost as if it ran here
```

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
`pace`, `recording`, `non_finite`); on a refusal, `errors`, each in the shape of [MIME's errors](errors.md): `code`,
`class`, `message`, `why`, `action`, `where` and `page`. A request a client gets wrong is refused, never dropped.

| Command | Arguments | What it does |
|---|---|---|
| `hello` | | describes the session (below) |
| `status` | | the session's status |
| `start` | | starts or resumes stepping, paced |
| `pause` | | stops stepping after the chunk in hand |
| `step` | `chunks` | steps that many chunks now, from a session that is not running |
| `stop` | | stops in every state, sends the last frame, writes the run folder (its path is in the reply) |
| `set` | `values`, live inputs by label | sets the controller's live inputs from its next tick |
| `record` | `on` | turns the recording layer on or off |
| `pace` | `rtf`, a factor or null | the real-time factor, or unpaced |

`hello` returns MIME's version, the experiment's name and its hashes, the base step and the chunk in base steps, the
controller (its entry, mode, period and latency) and its live inputs, the path of the stage the session serves
(`stage`), what the asset cache draws of its visuals (`visuals`: the overlay to add over the stage, and the visual sets
it lacks, see [recordings](recordings.md)), the stage binding (`binding`), the topology, the frame schema
(`schema`), the commands, and the status.

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
(`mime.live.FrameOrder`): a gap is a dropped frame, a number at or below one seen is one out of order.

### The stage binding

`binding` says which field moves which prim of the stage (`built.binding()`), so a client needs no knowledge of MIME's
state layout. Each entry has the prim, the node and field, the row of a stacked field (`index`), its `kind`, the thing
it belongs to, its public name and its units. Every bound prim is placed in the world (a chain's frame prims reset
their transform stack), so a `pose` (seven numbers: x, y, z, then the quaternion w, x, y, z) is the prim's translate
and orient as it stands, applied with no composition. A `vector` (a field a law reads at a body, three numbers in the
world's axes) is drawn at the position of the pose `at` names.

## What this does not do yet

- **Pixel streaming** of the stage (WebRTC, Selkies) is MADDENING's next release: a session's frames are for local
  rendering, by a client with the stage.
- **A controller inside the compiled step** takes no live input: its state is in the graph. Declare it as a callback
  to steer it live.
- **Batch runs on Modal** play from their run folders (`mime modal`, then the folder), not live.
