# `mime.errors`

**The boundary of MIME's public calls.** Every error a public call gives a script (`Experiment.build`, a run, `save`,
`load`, `verify`, a result's accessors, the `mime` command) is a `MimeError` with a code, in one shape: what happened
in the experiment's names, why, what to do (pasteable where it can be), where (the script's line), and the code, which
names its generated page. How to read one, and how to reach its full traceback: [`docs/errors.md`](../../../docs/errors.md).

## What is here

- **The boundary** (`_boundary.py`). `public` makes a function a public call. An error raised inside one that is not
  a MIME error is translated into one, keeping it as `original`; the full chain (every frame from the script's first
  down to the raise, every chained error) is kept as `chain`. By default the error leaves with the script's own
  frames alone; with `MIME_TRACEBACK=full`, with every frame and the original as its cause. A public call made inside
  another is left to the outer one, and a defect of the translation never hides the error it translates.
- **The frames** (`_frames.py`). A frame is the library's when its module is MIME's, JAX's, MADDENING's or one of the
  libraries they compute with (`HIDDEN_PACKAGES`), or Python's own; every other frame is the script's.
- **The translations** (`_translate.py`). An error from below MIME recognises becomes its code: a device out of
  memory, `run.out-of-memory`. The errors of a controller's law are translated where MIME calls it: a traced value
  used as a Python value, `control.law-traced-branch` (`mime.build._in_graph`); a step that raises or returns other
  writes, `control.law-raised` and `control.law-writes` (`mime.control`). A state that is not finite at a run's end is
  `run.non-finite` (`mime.experiment`). The rest: an error MIME raised with no code is its defect (`error.uncoded`,
  internal); any other is unrecognised (`error.unrecognised`), shown at the script's line, never called MIME's defect.
- **The display** (`_display.py`). An uncaught MIME error is shown short: in a script through `sys.excepthook`, in
  IPython and Jupyter through the shell's handler of MIME's errors, in the CLI by the command itself (with no frames,
  and the exit code [`docs/errors.md`](../../../docs/errors.md) lists). The full chain is written to a file under
  `$MIME_TRACEBACK_DIR` or `tracebacks` in MIME's cache directory when it is shown, and only the newest `KEPT` are
  kept.

The classes themselves (`MimeError`, `BuildRefused`, `ArgumentRefused`, `RunFailed`, ...) and `Refusal` live in
`mime.build._refusals`, below everything that raises them; each that was a `ValueError`, `TypeError`, `KeyError` or
`RuntimeError` is still one.

## Status

Built in dev1, phase 7l-b. Phases and releases are explained in the [package manifest](../README.md#status).
