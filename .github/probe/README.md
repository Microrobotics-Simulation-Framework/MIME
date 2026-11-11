# Probes: heavy measurements on GitHub's runners

A probe is a measurement that is too heavy for a laptop's queue: a push to a branch `probe/<name>` runs the branch's
`probe/run.sh` on a standard GitHub runner (free for this public repository; the runner's size and the 6 h job limit are
GitHub's, see its documentation) and keeps what the script writes to `probe/out/` as a run artifact for 14 days.
`.github/workflows/probe.yml` is the workflow. **Probe branches are never merged and never opened as pull requests**: the
required check `no private name or path` fails a pull request whose head is `probe/*`.

The logs and artifacts of a probe are public, like any run's. The workflow's first job is the private-name check (the step
CI's required job runs), and the probe job waits for it; a probe job holds no secret. Write nothing to `probe/out/` or to
the log that you would not put in a commit.

## Make a probe

```
git fetch origin release/v0.3.0
git switch -c probe/<name> origin/release/v0.3.0
mkdir probe && $EDITOR probe/run.sh     # and any scripts it calls, under probe/
git add probe && git commit -m "probe: <what it measures>"
git push origin probe/<name>
```

The job sets up what CI's default lane does (Python 3.12, `uv sync --locked --extra verify`, the lab fixture, the unit tube
table), puts the environment's `bin` first on `PATH` (so `python` is the project's) and runs, from the repository root,
`timeout 320m bash probe/run.sh`. The environment has `JAX_PLATFORMS=cpu`. Write results to `probe/out/`; it is uploaded even
when the script fails or times out. Keep the script deterministic about its own settings: it gets whatever the runner's cores
and BLAS threading give, so pin threads in the script when the measurement needs it.

**Arms in parallel.** If the branch has `probe/arms.txt` (one name a line; letters, digits, `.`, `_`, `-`, at most 40, no
repeats), one job runs per arm with `PROBE_ARM` set to the name, all in parallel (500 jobs may run at once across the
organisation, so keep arms to what the measurement needs). Without the file one job runs and `PROBE_ARM` is empty. A newer push
to the same branch cancels the older run of that branch.

## Wait, download, clean up

```
gh run list --branch probe/<name> --workflow probe.yml --limit 1 --json databaseId     # your run's id
gh run watch <id> --exit-status
gh run download <id> --dir probe_results/        # one folder per artifact, named <branch>-<arm>, '/' written '-'
git push origin --delete probe/<name>            # the branch you pushed, and no other
```

Watch your own run by its id (another agent's pushes make runs of their own on their own branches). The artifact is
`probe-<name>-<arm>` (no arm: `probe-<name>`). A run that was cancelled by a newer push of the same branch has no result.

## Bit-for-bit between two commits: `bit_compare.py`

A claim that a change moved no bits is read between two programs of the same compiled model on one machine. Compare base
and head in the same job so the one-core-type rule (MIME-ANO-125) holds by construction:

```
python .github/probe/bit_compare.py --base origin/release/v0.3.0 --programs a b c \
    --command 'python {head}/probe/dump.py {program} {out}' --output probe/out/bit_compare.json
```

It makes a worktree of the base with its own virtual environment, then runs the command for each program in the base's tree
and in this checkout, each pinned to the same CPU set with `taskset` (default: one CPU) with one BLAS and XLA CPU thread. The
command must write an NPZ file at `{out}`; placeholders are `{program}`, `{out}`, `{head}` (this checkout, where the probe's
scripts live) and `{tree}` (the root the command runs in; braces of its own are doubled). The JSON holds, per program and
array, `numpy.array_equal` (`equal` also needs the same dtype and shape), the dtypes, the shapes and the largest absolute
difference, and `all_equal` overall; `--strict` makes a difference exit 1. It fetches the base from `origin` if the checkout
does not have it. `tests/test_probe_bit_compare.py` holds its controls.

## How often a control catches its plant: `detection_rate.py`

"A control must fail every run" is a rate, and a rate is measured. The target is a pytest node id or a script that reads an
environment variable and plants its error when it is set (a test that the oracle flags the planted error; a script that
runs the oracle over a planted draw and exits 1 when it flags it):

```
python .github/probe/detection_rate.py --node tests/<dir>/<file>.py::<test> --plant-var <NAME> --plant-value 1e-6 \
    --runs 40 --output probe/out/detection_rate.json
python .github/probe/detection_rate.py --script probe/oracle.py --plant-var <NAME> --runs 40 \
    --output probe/out/detection_rate.json -- --size 3
```

Each run is a fresh process with its own seed in `DETECTION_SEED` (`--seed-var`; run i gets `--seed-base` plus i; a node
also gets `--hypothesis-seed=<seed>` with `--hypothesis-seed`). The planted arm sets the variable; the clean arm, at the
same seeds, removes it from the environment, so it says how often the control fires with nothing planted. **Caught** is
exit 1 (and for a node, a test that failed), **not caught** is exit 0 with a test that passed, and everything else is
**inconclusive** and counts against `caught/N`: a crash, a timeout, a collection or setup error, a node that was skipped or
not collected (a test that did not run did not miss).

The JSON holds `planted.caught_of_runs`, `detection_rate`, `missed_seeds` (reproduce a miss with its seed),
`inconclusive_seeds`, `clean.flagged` and its seeds, the last lines of the output of each miss or false alarm, and
`lower_bound_95`: the exact one-sided 95 % lower bound on the rate. It is the number to quote with the count: 40 catches in
40 runs show a rate no lower than 0.93, not 1, and a control that misses one run in eight needs 23 runs to show a miss 95 %
of the time (`1 - (7/8)^N`). `every_run` is true when all planted runs were caught and the clean arm was quiet; `--strict`
exits 1 otherwise. Where the plant needs several runners, put the sizes or values in `probe/arms.txt` and read `PROBE_ARM`
in the script. `tests/test_probe_detection_rate.py` holds its controls.

## How much memory a probe needs, and where it goes: `mem_estimate.py`

A standard runner has 16 GB; 6 of 113 probe runs were killed by it. Before a probe runs, the estimate says which machine it
belongs on, from the compiled program and not from a guess:

| estimate (times the calibration factor) | route |
|---|---|
| up to 12 GiB | `actions`: a `probe/<name>` branch, this workflow |
| 12 to 48 GiB | `local`: the maintainer's machine, one at a time, under a 48 GiB cap, for memory-bound correctness only, never a timing claim |
| over 48 GiB | `modal`: only on the maintainer's word |

```
python .github/probe/mem_estimate.py --case module:function --path DIR --steps 10 --wrt node.leaf --output-json est.json
```

**A raw JAX program** (a jitted function, not an experiment) is named with `--raw module:function` instead of `--case`; the
factory returns `(function, arguments)` (arrays or `jax.ShapeDtypeStruct`s), and the estimate is XLA's compiled memory analysis
of the function on the arguments' shapes (arguments, outputs, temporaries, less what aliases), on top of the resident set after
the factory. The printed line says `estimate route: raw program`. Nothing is declared per arm beyond the line of
`probe/estimate.txt` that names the factory. Its factor is the `raw_program` family's (mode `program`) in `mem_calibration.json`,
calibrated on the cases the table holds for it, which are few (see the table's `scope`): a raw program unlike them is bounded
no better than that, and the printed line says how many cases the factor rests on.

For an experiment it builds the case, then asks XLA for the compiled memory analysis of a chunk of `--steps` steps of the graph and, with
`--wrt`, of the reverse-mode derivative of that chunk (`jax.jit(f).lower(...).compile().memory_analysis()` on abstract
shapes: nothing of the run is allocated; in reverse mode the larger of the checkpointed run and the plain chunk), adds the
host's resident set after the build, the nodes' own `memory_estimate()` where they have one, and the fixed part of a process
(about 0.7 GiB), and multiplies the largest by a calibration factor before reading the route. The factor is **per family of
nodes and per mode** (forward or reverse), from `mem_calibration.json`: the graph's nodes name their families (the MRS drag and
near field; the FVM), each family's factor is the largest measured/estimated ratio among its completed cases of that mode,
rounded up to 0.05 and never under 1.10, and a graph with a node of no calibrated family takes the table's stated conservative
factor for uncalibrated cases. The printed line says which, and over how many cases. A probe declares its cases in
`probe/estimate.txt`, one line each, `<arm or *>: <the arguments above>`; the workflow's step "Memory estimate and route" runs
the lines of its arm before the script and stops the job with the route if a case is beyond a runner. A probe with no
`probe/estimate.txt` runs, with a warning; **an arm with no line of its own and no `*` line is refused by name**, and a line
naming an arm that `probe/arms.txt` does not list is refused too (a typo would otherwise estimate as nothing).

**After the run.** The script runs under `mem_estimate.py --measure-peak`, which records the peak resident set of the largest
process in its tree (one process's peak, not the sum of parallel ones) in `probe/out/mem_peak.json`; the step "Memory against
the estimate" compares it with the bound the route was read from, writes `probe/out/mem_check.json` and **fails the arm if the
peak is over the bound**, naming it an underestimate. The check file holds a row ready for `mem_calibration.json`: the
calibration grows from real runs (add the row with its own name and `where`, and the family's factor follows the rule, which
`tests/test_probe_mem_estimate.py` checks against the table).

**What the calibration is evidence of.** `mem_calibration.json` holds the cases the estimator was checked against, each with
its family, mode, estimate, the peak measured on that case (with `measured_mib` where the source reported MiB) and their ratio,
and each family's factor. Two families are calibrated, the MRS drag (with its near field) and the FVM, each on a handful of
cases on a CPU, and the table says in its `scope` which of them were fitted to the same peaks and which reverse cases are lower
bounds from killed runs. A case of any other family is outside what was measured and takes the conservative factor. A measured
peak marked as a lower bound is a run that was killed: the estimator is not underestimating it so long as it reads at least
that. An estimate that jumps unexpectedly is a reason to look for the probe's own bug before sending the probe to a bigger
machine.

## What this does not tell

A probe measures what its script measures on a shared runner: timings are noisy (other jobs share the host) and are not a
substitute for the maintainer's timing runs on a quiet machine; a result is a measurement on one runner type, not a bound.
