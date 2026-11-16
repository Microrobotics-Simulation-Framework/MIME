- **`mime serve` hardened** (beside loopback-only binding):
  - **No path of the serving machine in `hello` or `stop`.** `hello`'s `stage` is relative to the run folder
    (`experiment/scene/scene.usda`, the saved folder's copy; `null` for a session with no run folder), `visuals.overlay`
    is the overlay's id (its file name in the asset cache's `overlays/`), each `visuals.fetch` command names its folder
    relative to the run folder, and `stop`'s `folder` is the run folder's name. The ready file keeps the path, on the
    launcher's own machine.
  - **A bounded `record`.** A recording holds at most its budget of samples (`--record-budget`, `record_budget=`; by
    default as many as fit in 512 MiB, which `hello` gives), stops by itself there keeping what it took, and a `record`
    that would start another stretch is refused (`live.record-budget`, new). Every status says how many it holds
    (`recorded`); the budget is in the command log's header, so a replay stops where the session did.
  - **Message-size limits on every message received** (`mime.live._zmtp`), checked on each frame's header before its
    bytes are read and with a message's parts counted together: a request 64 KiB (was 1 MiB a frame), a subscription
    64 KiB, a reply 64 MiB, a frame 256 MiB, each a stated margin over the largest measured
    (`scripts/live_message_sizes.py`, `docs/live.md`). A request over the limit is answered with
    `live.message-too-large` (new) in its turn, its bytes dropped as they arrive, and the same client is served on; a
    client refuses a reply or a frame over its own limit by the same code.
  - **No second server on a port held, on macOS and the BSDs.** A listener sets `SO_REUSEADDR` on Linux alone: on BSD it
    let a second server bind `127.0.0.1:P` while a first held `0.0.0.0:P`. The platforms workflow runs `tests/live` on
    macOS.
  - **The transport checks for itself.** `mime.live._zmtp.Server` refuses an address off loopback unless allowed
    (`allow_remote=`), resolves `localhost` once and binds the address it resolved, and refuses a port outside 0 to
    65535 as a malformed endpoint (`live.endpoint`), closing the socket it opened first: before, `--frames
    tcp://127.0.0.1:99999` raised an uncoded `OverflowError` (exit 3) with the command socket left open.
  - **`mime.live/2`.** The four values above changed meaning, so the protocol is `mime.live/2`: the server announces it, MIME's
    client speaks it, and a request that says `"protocol": "mime.live/1"` is refused with `live.protocol`.
