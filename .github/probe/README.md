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

## What this does not tell

A probe measures what its script measures on a shared runner: timings are noisy (other jobs share the host) and are not a
substitute for the maintainer's timing runs on a quiet machine; a result is a measurement on one runner type, not a bound.
