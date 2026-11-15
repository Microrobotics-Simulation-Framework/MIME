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
`examples/`. Three separators, as MIME names things today:

- an entry, with a dot: `${namespace}.example_load`;
- a record, with an underscore: `${namespace}_example_record`;
- an instance of an entry on a thing, with a colon: `${namespace}.example_load:probe`.

## Layout

```
src/${package}/
  engine/        your law, pure JAX, no `mime` import (a test holds it)
  entries/       the adapter: one module per entry, found through the `mime.entries` entry point
  records/       records.json, and the data it names in data/ (the `mime.assets` entry point)
tests/
  conftest.py test_registry.py test_nodes.py test_validity.py test_engine_imports.py
  engine/        properties (Hypothesis) and the planted error that shows one can fail
  verification/  a benchmark in the slow lane
  assert_types.py  types an editor completes from, held by pyright
benchmarks/      build_case.py and mime.toml
examples/        what a user runs
```

## Tests

```bash
JAX_PLATFORMS=cpu taskset -c 0 pytest          # the default lane: -m "not slow"
JAX_PLATFORMS=cpu taskset -c 0 pytest -m slow  # the slow lane: the benchmark in tests/verification/
```

**Pin to one core and the CPU** (`taskset -c 0`, `JAX_PLATFORMS=cpu`). An unpinned JAX run on a shared machine
oversubscribes the cores: one measurement took 51 s unpinned for what takes 11 s pinned.

CI (`.github/workflows/ci.yml`) has a `default` job, a `slow` job that runs on a pull request labelled `slow` and every
week, a weekly job against MIME's release branch (it tells you the week MIME moves under you), and a `typing` job. Run
the suite in float32 and in float64 (`JAX_ENABLE_X64=1`) before you trust a number. A property test that flakes is
read, not loosened: see the comments of `tests/engine/test_properties.py`.

## Editor support

`pyrightconfig.json` is strict over `src/${package}` and `tests/assert_types.py`, and names this folder's `.venv`, so that an
editor's language server finds MIME, MADDENING and JAX there (autocomplete is only as good as the types it reads):

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test,typing]"
```

If `.venv` does not exist, pyright says so on stderr and uses the environment it is given. CI creates `.venv` itself and runs
`pyright --pythonpath "$$(which python)"`. A `.venv` that exists but does not hold MIME overrides `--pythonpath` (measured),
so install into it or delete it.
