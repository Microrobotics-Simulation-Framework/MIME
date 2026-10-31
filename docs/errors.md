# Reading an error

Every error a public call of MIME gives (`Experiment.build`, a run, `save`, `load`, `verify`, a result's accessors,
the `mime` command) is a MIME error with a code. An error from below (JAX, MADDENING, NumPy, or your own code called
from inside a run) is translated into one, and the error it translates is kept on it.

## What an error says

```text
Traceback (most recent call last; MIME's, JAX's and MADDENING's frames left out):
  File "settle.py", line 18, in <module>
    exp.build()
BuildRefused: the build is refused:
[user] setting.value: interaction drag:ball (drag 0.3.0): spacing is None m, not a length above zero: ...
    why: ...
    do: correct the setting where interaction drag:ball is made, as the reason says: BEM(...)
    at: settle.py, line 14
    see: https://microrobotica.org/mime/pages/errors/setting.value
The full traceback is in ~/.cache/mime/tracebacks/...-setting.value.txt; MIME_TRACEBACK=full shows it here.
```

- **The frames** are your own: the lines of your script, its modules, a notebook's cells. MIME's, JAX's and
  MADDENING's frames, and those of the libraries they compute with, are left out of what is shown, never out of what
  is kept.
- **The first line of each reason** is its class in brackets, its code, and what happened, in the experiment's own
  names: its things and attached laws (`drag:ball`), never a graph's node.
- **`why`**, where the message does not already say it: one plain sentence.
- **`do`**: what to write instead, as code you can paste where it can be (`spacing=mm(...)`,
  `exp.declare(...)`, `` `mime assets fetch <folder>` ``). Every action names a MIME argument or command; `mime docs
  check` fails one that does not.
- **`at`**: the line of yours closer to the cause than the call, where MIME knows one (the line of a controller's
  step that raised, or that branches on a traced value).
- **`see`**: the code's page, which `mime docs build` writes for every code, with each place MIME gives it.

Several reasons are given at once: a build is refused with every reason it found.

## The four classes

- **user**: the script can change it. The error says what to write.
- **environment**: something the run needs is missing or different where it runs (a package, a file, a device, the
  network). The error names what to install, fetch or attach.
- **internal**: a check on MIME's own construction, which no supported script should reach, or an error MIME raised
  with no code (`error.uncoded`). It is MIME's defect: the error says so, with the address to report it at and the
  file to attach.
- **unrecognised**: an error from below that MIME has no translation for (`error.unrecognised`). It is shown at your
  own line, with the error's own type and message, and it is never called MIME's defect.

## Errors from below, translated

- A Python `if`, `int()` or `float()` on a traced value in a controller's step that runs inside the compiled step:
  `control.law-traced-branch`, at the line of the step that asks, with `jnp.where` (or `jax.lax.cond`) as the fix, or
  the callback mode, where the step may be any Python.
- A controller's step that raises while it runs as a callback, or returns writes of other labels or shapes:
  `control.law-raised` and `control.law-writes`, at the step's line.
- A state that is not finite numbers at the end of a run: `run.non-finite`, naming the first step, thing and field
  where the run kept every step's state (`history=True`), and how to find them where it did not. The run's result is
  kept on the error, as `error.result`.
- A device out of memory: `run.out-of-memory`, with what to keep less of.

## Reaching the full traceback

The full chain (every frame from your script's first down to the raise, and every error chained to it) is kept on
the error as `error.chain`, and written to a file when the error is shown: under `tracebacks` in MIME's cache
directory, or under `$MIME_TRACEBACK_DIR` when it is set. The file holds the error, the versions of Python, MIME,
MADDENING, JAX and NumPy, and the chain. Writing one deletes all but the newest, so the directory never grows; delete
it whenever you like.

- **`MIME_TRACEBACK=full`** shows every frame and every chained error in place of the short form, as Python prints
  them. MIME's test suite runs this way.
- **The CLI's `--traceback`**, anywhere on the line, does the same for one command.

```sh
MIME_TRACEBACK=full python settle.py
mime run <folder> --duration 1 --out run --traceback
```

- **In Jupyter and IPython** the short form is shown for a MIME error a cell does not catch, through the shell's own
  handler of MIME's errors (where the shell has no other); `MIME_TRACEBACK=full` gives IPython's own traceback.

## Catching an error in code

Every MIME error is a `mime.errors.MimeError`, with `codes()`, its `refusals` (each with its `code`, `category`,
`message`, `action`, `why` and `where`) and `original`, the error from below it translates. The classes are caught as
they were: `BuildRefused` and its subclasses (`LoadRefused`, `EnvRefused`); an error a run's arguments are refused
with is still a `ValueError` or a `TypeError`; `NotRecorded` and a result that does not hold an observation are
still a `KeyError`; a controller's `ControllerError` and a failed run's `RunFailed` are a `RuntimeError`. A caught
error keeps its short traceback, with your own frames, so `traceback.print_exc()` shows them; `write_traceback(error)`
writes its full chain to a file.

## The command's exit codes

A refused command prints why, as above but with no frames (you wrote no Python), and exits with the code that says
whose it is to act on:

```text
0  done
1  mime verify: an output moved
2  refused, for the user's or the environment's reason
3  a MIME defect, or an error MIME does not recognise
```
