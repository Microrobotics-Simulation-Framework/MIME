# `mime.live`

**Live sessions: a built experiment run while a viewer watches it, and the protocol that serves one.**

## What goes here

- **`LiveSession`** (`built.live(chunk=..., pace=..., select=..., out=...)`): the run stepped in chunks of one fixed
  size (each new length compiles anew, so a session keeps one), a frame of the selected fields handed to every
  subscriber after each chunk, the controller's live inputs set between ticks (`set`), the recording layer turned on
  and off (`record`), paced to a factor of real time or unpaced, and stopped cleanly from every state, a step in hand
  included, within a chunk. Every command is logged with the step it took effect at, before it acts, and each chunk's
  step after it, the log flushed as it goes; `replay` runs the log again (a dying session's to the step it reached)
  and writes the same run folder, which every session given `out=` writes when it stops: the experiment, the command
  log, the final state and its manifest, the topology, the build report's summary (`report.json`), and the recording
  layer when recording was on. A live input lasts as long as its session, and the controller's step reads every one
  from the session's first tick, at its declared default until it is set, so that no `set` compiles the step again. A
  command that changes the run waits for the chunk in flight; `status` and `hello` do not, and say what the session is
  doing (`activity`). A request a client got wrong is refused by its code, and no request stops the session.
- **`mime.live/1`** (`_protocol.py`): the messages, independent of their transport. Requests and replies are JSON, a
  refusal in the coded shape of MIME's errors; a frame is binary (a header with its schema's id, its number, its step
  and its time, then each field's bytes in the run's own type) or JSON. `FrameOrder` sees a frame dropped or out of
  order.
- **`mime serve`** (`_serve.py`): a saved folder's session in a process of its own, over ZMQ (a REP for commands, a
  PUB for frames), bound to localhost unless asked otherwise, each chunk run on a thread of its own while the loop
  answers: `hello` and `status` at once, `stop` and `pause` at once and acting at the chunk's end (`pending`), the
  other commands at the chunk's end, in the order they came; `LiveClient` is its Python client.
- **ZMTP** (`_zmtp.py`): ZMQ's wire protocol with the NULL mechanism, spoken by MIME itself, so that serving needs no
  extra package; a libzmq peer connects as it would to libzmq's own sockets. A peer that does not speak it, finishes no
  handshake, oversteps a size limit or never reads its replies is closed, and the others are served on. Its tests
  replay the handshakes libzmq's own sockets sent, run libzmq itself (pyzmq, a development dependency) against `mime
  serve` both ways, and fuzz the handshake and the framing.

## Boundaries

- It imports `experiment` (for its types), `build`, `viz` and `units`; `BuiltExperiment.live` imports it.
- A live input is a controller's (`Controller.live`), in the callback mode: a physical setting is a rebuild.
- It speaks generic messages: nothing in it names a client.

## Status

Since dev1's phase 7n (design #197): the in-process session, `mime serve` over ZMQ, replay from the command log.
Since phase 7n-b: no recompile on a live input, a server that answers while a chunk runs, and the build report's
summary in `hello` and in every run folder.
Later: a WebSocket transport (the same messages), and pixel streaming of the stage, which is MADDENING's.

Phases and releases are explained in the [package manifest](../README.md#status).
