# Reading an error

Every error a public call of MIME gives (`Experiment.build`, a run, `save`, `load`, `verify`, a result's accessors,
the `mime` command) is a MIME error with a code. An error from below (JAX, MADDENING, NumPy, or your own code called
from inside a run) is translated into one, and the error it translates is kept on it.

## What an error says

```text
Traceback (most recent call last; MIME's, JAX's and MADDENING's frames left out):
  File "settle.py", line 20, in <module>
    exp.build()
BuildRefused: the build is refused:
[user] setting.value: interaction drag:ball (drag 0.3.0): spacing is None m, not a length above zero
    why: the bem rung samples the body's surface with neighbouring samples about spacing apart
    do: write spacing=mm(...), where interaction drag:ball is declared (Drag(...))
    see: https://microrobotica.org/mime/0.3.0/pages/errors/setting.value
The full traceback is in ~/.cache/mime/tracebacks/mime-...-setting.value.txt; MIME_TRACEBACK=full shows it here.
```

- **The frames** are your own: the lines of your script, its modules, a notebook's cells. MIME's, JAX's and
  MADDENING's frames, and those of the libraries they compute with, are left out of what is shown, never out of what
  is kept.
- **The first line of each reason** is its class in brackets, its code, and what happened, in the experiment's own
  names: its things and attached laws (`drag:ball`), never a graph's node.
- **`why`**, where the message does not already say it: one plain sentence.
- **`do`**: what to write instead, as code you can paste where it can be (`spacing=mm(...)`,
  `exp.declare(...)`, `` `mime assets fetch <folder>` ``). Every action names a MIME argument or command; `mime docs
  check` fails one that does not. That check reads an action's shape, not its truth: it cannot tell a real argument
  from a misspelt one, or an action that would work from one that would not, which the tests of each refusal judge.
- **`at`**: the line of yours closer to the cause than the call, where MIME knows one (the line of a controller's
  step that raised, or that branches on a traced value).
- **`see`**: the code's page, which `mime docs build` writes for every code, with each place MIME gives it, in the
  documentation of the MIME version you run (`/mime/<version>/pages/errors/<code>`).

Several reasons are given at once: a build is refused with every reason it found.

## The four classes

- **user**: the script can change it. The error says what to write.
- **environment**: something the run needs is missing or different where it runs (a package, a file, a device, the
  network). The error names what to install, fetch or attach.
- **internal**: a check on MIME's own construction, which no supported script should reach. It is MIME's defect: the
  error says so, with the address to report it at and the file to attach. An entry's defect (`entry.defect`) is the
  defect of the package that defines the entry: one a lab registered is named as that package's, at the entry's own
  line that raised, and never called MIME's.
- An error MIME raised with no code (`error.uncoded`) is a gap in MIME's codes, not a fault of yours: its message is
  MIME's own and often says what to change, it keeps its own class (a `KeyError` is still one), and the footer asks
  you to report it so that it gets a code.
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
- A device out of memory, as XLA reports it: `run.out-of-memory`, with what to keep less of. The host's own
  `MemoryError` (an array of your script's too large for the machine) is not the device's: it stays unrecognised, and
  a `MemoryError`.
- A value given without its units, or in units of another dimension, where a call takes a quantity (a bare number
  for a run's duration): `call.units`, a `ValueError` and a `UnitError` as before, with the value written in its unit.
- A folder `save` cannot write into: `save.not-writable`, and still the `OSError` it was. A document that is not
  Unicode text: `load.document-encoding`. A reference outputs file that is truncated or corrupt: `load.reference`.

## Reaching the full traceback

The full chain (every frame from your script's first down to the raise, and every error chained to it) is kept on
the error as `error.chain`, and written to a file when the error is shown: under `tracebacks` in MIME's cache
directory, or under `$MIME_TRACEBACK_DIR` when it is set. The file holds the error, the versions of Python, MIME,
MADDENING, JAX and NumPy, and the chain. Writing one deletes all but the newest, so the directory never grows; delete
it whenever you like.

- **`MIME_TRACEBACK=full`** shows every frame and every chained error in place of the short form, as Python prints
  them. MIME's test suite runs this way.
- **The CLI's `--traceback`**, anywhere on the line, does the same for one command.
- **The file** is written readable by you alone, named `mime-<time>-<process>-<code>.txt`. Only files of that name
  are ever deleted, so `$MIME_TRACEBACK_DIR` may name a folder that holds your own files.

```sh
MIME_TRACEBACK=full python settle.py
mime run <folder> --duration 1 --out run --traceback
```

- **In Jupyter and IPython** the short form is shown for a MIME error a cell does not catch, through the shell's own
  handler of MIME's errors (where the shell has no other); `MIME_TRACEBACK=full` gives IPython's own traceback.
- **A hook of your own** (a crash reporter, an application's error dialog) set as `sys.excepthook` before
  `import mime` still receives MIME's errors, after their short form; one set after MIME's replaces it and receives
  every error itself, MIME's with their short traceback.
- **In a thread**, an uncaught MIME error is shown by Python's own thread handler, with its short traceback (your own
  frames) and no file written: the short form's text and its file are the main thread's.
- **A recursion** in your own code is shown with its repeated frame once and its count, as Python shows it.

## Catching an error in code

Every MIME error is a `mime.errors.MimeError`, with `codes()`, its `refusals` (each with its `code`, `category`,
`message`, `action`, `why` and `where`) and `original`, the error from below it translates. The classes are caught as
they were: `BuildRefused` and its subclasses (`LoadRefused`, `EnvRefused`); an error a run's arguments are refused
with is still a `ValueError` or a `TypeError`; `NotRecorded` and a result that does not hold an observation are
still a `KeyError`; a controller's `ControllerError` and a failed run's `RunFailed` are a `RuntimeError`.

An error from below that a public call translates is also of its own class, so the handler written for it still
catches it: your own exception raised in a hook or a controller's step (`except Done`), an `OSError` from `save`, a
`MemoryError`, a warning your filter raises as an error, and `pytest.raises(ValueError)` in your tests. Its attributes
are kept (`error.errno`, your exception's own); `error.original` is the exception itself. In the rare case where its
class cannot be a base of another (a class that forbids subclasses), the original leaves the call unchanged.

```python
try:
    built.run(s(1), hook=stop_when_settled, chunk=ms(10))
except Settled as done:            # the script's own exception, raised by its hook
    print("settled at", done.time)  # its own attribute
    raise done.original from None   # or the exception itself, with every frame
```

A caught error keeps its short traceback, with your own frames, so `traceback.print_exc()` and `logging.exception`
show them, not MIME's or JAX's. To see below them:

- **`error.chain`**: the full traceback as text, every frame and every chained error, as it was when the error left
  the call; `write_traceback(error)` writes it to a file.
- **`error.original`**: the error from below, whose own `__traceback__` holds every frame below the call:
  `traceback.print_exception(error.original)`, or `pdb.post_mortem(error.original.__traceback__)` to step into MIME's
  and JAX's frames.
- **`MIME_TRACEBACK=full`**, set before the run: the error leaves with every frame and the original chained as its
  cause, so `%debug` and `pdb.pm()` reach every frame.

## The command's exit codes

A refused command prints why, as above but with no frames (you wrote no Python), its first line naming the command
(`mime pull is refused:`, `mime run failed:`), and exits with the code that says whose it is to act on. A
`mime run` whose state is not finite at its end writes its outputs all the same (`outputs.json` and `run.json` say
`non_finite`), then says to run again with `--history` or with a smaller `dt` in the saved document:

```text
0  done
1  mime verify: an output moved; mime pending: something is pending
2  refused, for the user's or the environment's reason
3  a MIME defect, or an error MIME does not recognise
```
