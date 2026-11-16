# ${distribution}

Entries for [MIME](https://github.com/Microrobotics-Simulation-Framework/MIME) in the namespace `${namespace}`, made by
`mime scaffold plugin`. Every file in it is short and says what it is for; delete what you do not need, keep what you do.
Replace the example (a constant load on a body) with your physics.

## Use

Install this plugin, with its test tools; MIME comes with it:

```bash
pip install -e ".[test]"
```

Each block below that is tagged `bash ci` is what CI runs on every push (the `default` job reads this file and runs
them), so this section cannot go stale.

**1. See what it adds.** The plugin registers under the namespace `${namespace}`, and the example runs:

```bash ci
mime registry | grep ${namespace}
python examples/example.py
```

`mime registry --json` carries every entry's settings, roles, validity lines and gradient status, and is very large. To
read one namespace's:

```bash
mime registry --json | jq '.entries[] | select(.name | startswith("${namespace}."))'
```

**2. A benchmark folder.** `benchmarks/build_case.py` saves the example as a MIME experiment with its `mime.toml`
(`benchmarks/mime.toml`, the record's manifest); `mime check` lists what is still to write in it (your creator's ORCID, a
description) and `mime verify` loads it, rebuilds it and reruns its first steps bit for bit:

```bash ci
python benchmarks/build_case.py
mime check benchmarks/case
mime verify benchmarks/case
```

**3. Its records.** `src/${package}/records/` holds the plugin's records (the `mime.assets` entry point): the example is
open (CC-BY-4.0 unless you chose another licence), so its bytes ship in the wheel; `mime assets audit` on the built sdist
(CI does it) fails if a file that may not be redistributed is in it. `mime assets list` lists MIME's own records too, with no
filter by namespace.

**4. Cite it.** `CITATION.cff`.

## Names

The namespace `${namespace}` is filled in six places from one value: the `mime.entries` and `mime.assets` entry points in
`pyproject.toml`, `src/${package}/entries/_vocabulary.py`, the entry's name, the record's name and the instance in
`examples/`. Two separators:

- an entry, and a record, with a dot: `${namespace}.example_load`, `${namespace}.example_record` (a script names the
  record `asset("${namespace}.example_record")`; a record's earlier spelling with an underscore is still found in this
  release);
- an instance of an entry on a thing, with a colon: `${namespace}.example_load:probe`.

One name is one concept: a class or entry name that another layer already uses for something else is not reused (MIME's
scene `Cylinder` is bounded, while core's unbounded cylinder, which the engine evaluates, is to be named `CylinderShape`). An
error you raise is named for whose fault it is: `*Refused` where your entry declines an input, with a code the user can fix,
and `*Error` where something failed that is not the user's input.

## Layout

```
src/${package}/
  engine/        your law, pure JAX, no `mime` import (a test holds it)
  fit/           a second package that imports no `mime` (a fitting harness, say): empty, under the engine's rule
  entries/       the adapter: one module per entry, found through the `mime.entries` entry point
  records/       records.json, and the data it names in data/ (the `mime.assets` entry point)
tests/
  conftest.py test_registry.py test_nodes.py test_validity.py
  test_engine_imports.py   the engine and `fit` import no `mime`
  test_relative_imports.py the package imports itself relatively, so it can move as a directory
  test_durations.py        the duration guard's own test; duration_allowlist.txt lists the tests that may be slow
  engine/        properties (Hypothesis) and the planted error that shows one can fail
  verification/  a benchmark in the slow lane
  assert_types.py  types an editor completes from, held by pyright
benchmarks/      build_case.py and mime.toml
examples/        what a user runs
scripts/         check_durations.py (CI's duration guard); put your own scripts here: the sdist carries them, and so docs/
```

## Tests

```bash
JAX_PLATFORMS=cpu taskset -c 0 pytest          # the default lane: -m "not slow"
JAX_PLATFORMS=cpu taskset -c 0 pytest -m slow  # the slow lane: the benchmark in tests/verification/
```

**Pin to one core and the CPU** (`taskset -c 0`, `JAX_PLATFORMS=cpu`). An unpinned JAX run on a shared machine
oversubscribes the cores: one measurement took 51 s unpinned for what takes 11 s pinned.

CI (`.github/workflows/ci.yml`) has a `default` job, a `slow` job that runs on a pull request labelled `slow` and every
week, a weekly job against MIME's release branch (it tells you the week MIME moves under you), and a `typing` job. A
property test that flakes is read, not loosened: see the comments of `tests/engine/test_properties.py`.

**Both dtypes.** Run the suite in float32 and in float64 (`JAX_ENABLE_X64=1`) before you trust a number: the law itself
runs in whatever dtype the process computes in, and a test that passes in one only has not shown the other. The law's own path
is tested in both dtypes, always. The exception is a reference the engine is computed *against* (a closed form, a table of
numbers from another code, a reference run kept as float64 bits): that may be float64 alone, and the test that reads it says so
where it runs, under `with jax.experimental.enable_x64():` in that test (or under a guard your plugin declares for it), never
as a suite-wide switch such as an x64 line in `tests/conftest.py`: that would turn float64 on under every test and hide a law
that only works in one dtype. Never let the law's own path be float64-only because its reference is.

**The duration guard.** Keep the default lane short: `scripts/check_durations.py` (CI runs it on the `pytest
--junitxml=junit.xml` report) fails a test over 30 s unless `tests/duration_allowlist.txt` lists it with the reason it must
run on every push. A test that is slow because it is a benchmark goes behind `@pytest.mark.slow`; a physics assertion there
is documentation unless a lane runs it (the `slow` job does). Never shrink a Hypothesis test's examples to get under the
limit: list it with its measured time. The script is a small version of MIME's own gate (no cold-cache re-run, no ceiling on
a listed test); its docstring says what it leaves out.

## Editor support

`pyrightconfig.json` is strict over `src/${package}` and `tests/assert_types.py`, and names this folder's `.venv`, so that an
editor's language server finds MIME, MADDENING and JAX there (autocomplete is only as good as the types it reads).

**Strict, and what is relaxed.** jaxlib ships no type stubs, so JAX code written without annotations reads as unknown to a
strict checker (a real 3.5k-line engine gave 1,093 errors, 95 % of them unknown types). The packages that import no `mime`
(`engine/`, `fit/`) therefore have the unknown-type rules off (`reportUnknown...` and `reportMissingParameterType`, in
`executionEnvironments`) and every other strict rule on (unused variables, optional access, `None` defaults into typed
parameters); the adapter (`entries/`, `records/`) and `tests/assert_types.py` are strict whole. The route to strict for the
engine: annotate each function's parameters and results (`Array`, `Float[Array, "n"]`), starting with those the adapter calls
(an untyped return makes the adapter's call unknown), and delete its entry in `executionEnvironments` when it is clean. In
the adapter, call an untyped JAX function through `Any` once at the top of the module (`jnp: Any = jax.numpy`, as the example
entry does). What the `assert_type` file holds is what the checker reads from your entries: it fails if a type changes.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test,typing]"
```

If `.venv` does not exist, pyright says so on stderr and uses the environment it is given. CI creates `.venv` itself and runs
`pyright --pythonpath "$$(which python)"`. A `.venv` that exists but does not hold MIME overrides `--pythonpath` (measured),
so install into it or delete it.
