# Contributing to MIME

## Development Setup

```bash
git clone git@github.com:Microrobotics-Simulation-Framework/MIME.git
cd MIME

# The locked environment, with the `verify` extra and the `dev` dependency group (pytest, Hypothesis, pyright,
# hatchling), which uv installs by default. uv takes Python 3.14 from `.python-version`; CI runs every supported
# Python (3.12 to 3.14) and type-checks against 3.12, the floor.
uv sync --locked --extra verify
uv pip install --no-deps ./lab_fixture
```

## Licensing of Contributions

MIME is licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). A contribution is
made under the same licence (inbound equals outbound, as section 5 of the licence states), with no separate
agreement to sign.

Each commit carries a [Developer Certificate of Origin](https://developercertificate.org/) sign-off, the line
`Signed-off-by: Your Name <you@example.org>` that `git commit -s` adds: by it you certify that you wrote the change,
or otherwise have the right to submit it under the project's licence. This is documented, not yet enforced: no CI
check reads the sign-off today.

## ID Prefix Convention

MIME uses the `MIME-` prefix for all compliance identifiers:

| Category | Prefix | Example |
|----------|--------|---------|
| Node algorithm IDs | `MIME-NODE-` | `MIME-NODE-001` |
| Anomaly IDs | `MIME-ANO-` | `MIME-ANO-001` |
| Verification benchmark IDs | `MIME-VER-` | `MIME-VER-001` |

Never use `MADD-` prefixed IDs for MIME artifacts. Cross-reference upstream MADDENING anomalies by their `MADD-ANO-*` ID in the `safety_relevance_rationale` field.

## Naming

A class is named for what it is, in a name that reads in its namespace (`mime.fields.PointDipole`,
`Drag.Surrogate`). Rules 2 and 6 are enforced by `tests/test_naming_rules.py`, and so is rule 5 for one name: the
method of regularised Stokeslets is the `MRS` rung, and its old generic name stays out of the repository, free for a
rung that uses boundary elements; the others are for review.

1. **Name a class for what it is,** not who published it or where it came from. Attribution goes in the docstring and
   the entry's `references`. A physical law or model universally known by a person's name (Stokeslet, Oseen, Faxén,
   Rotne-Prager) keeps it, because that name is the standard term.
2. **No word the namespace already carries.** `mime.fields.UniformRotating`, not `UniformRotatingField`; a family's
   rung is `Drag.Surrogate`, not `SurrogateDrag`. Implementation classes keep their type's suffix (`...Node`,
   `...State`), because there the suffix says which layer a class belongs to.
3. **One concept, one stem, across layers:** the scene body, the core shape, the node, its state, its registry name
   and its switch owners share one stem (`Screw`, `SCREW`, `screw`, `screw.seam`).
4. **No qualifier that is always true** in its context (`Robot` in a robotics library; `Solid` where every scene body
   is solid).
5. **A generic name only for the general concept.** A specific variant is a setting or a qualified name, and a
   generic name a later concept will need stays free (`Helix` for a helical filament, the geometry slender-body
   theory and many microswimmers use).
6. **A module never re-exports another layer's class** under a name it also defines; same-named classes in
   different layers are imported from their own module.
7. **Spelling:** classes `CapWords`; singletons and constants `UPPER_SNAKE`; registry names `lower_snake` equal to
   the class stem; switch owners `<registry name>.<switch>`.

## Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behaviour change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

## Running Tests

```bash
JAX_PLATFORMS=cpu uv run python -m pytest
uv run python scripts/typing_check.py    # pyright, strict over src/mime, at zero errors
```

When you name test files, name each directory's files together (`sorted` does it): pytest 9.1.1 drops a directory's
`conftest.py` fixtures when a run reaches the directory again after a file of the directory above it, which ran
`tests/saved` without its float64 mode. The run is refused, naming the fixture, when that happens
(`tests/_collection_guard.py`); a run with no path never meets it.

## CI: how many shards

CI runs each test lane (`default` on three Pythons, `maddening-main` and `openmaterica-main`) as several shard jobs,
split by test file (`tests/_sharding.py`), and the lane's own job, which branch protection requires for `default` on 3.12
and 3.14 (the other lanes: "CI: which lanes a pull request runs"), waits for them and
checks that together they ran every collected test, with none failed or skipped (`scripts/check_shards.py`). How many
shards is a setting, **twenty unless set**:

- the repository variable **`MIME_CI_SHARDS`** (Settings, Secrets and variables, Actions, Variables) sets it for every
  run; unset or empty means 20;
- a manual run's **`shards`** input (Actions, CI, Run workflow) sets it for that run and wins over the variable;
- it is a whole number from **1 to 32**. Anything else fails the `shard count` job by name, and the lanes with it, so
  no run starts on a count nobody chose.

The trade-off: every shard job costs about the same job-minutes whatever the count (its own setup is about 30 s), so
more shards shorten a run's wall time and put more jobs against the organisation's concurrent limit (60 on the Team
plan), and fewer shards do the opposite. Each lane is its shards, one completion-tests job and its own job: 22 jobs at
the default 20. A push's run is five lanes and 3 other jobs (`shard count`, `typecheck`, `no private name or path`): 113
jobs at the default 20, more than the 60 at once, so a push's last lanes wait for runners. A pull request's full run is two
lanes (`default` on Python 3.12 and 3.14) and those 3: 47 jobs, and more where its diff starts the other lanes. Lower the
count when several runs queue behind one another, raise it for a single run's speed. The compilation cache is kept per
shard count, so changing the count starts that count's caches cold.

**Why twenty, after sixteen** (sixteen was chosen on the table of 2026-11-09; predicted slowest shard job of a lane, the median of 40000 draws of the
runner noise measured on the ten-shard run, which predicts the 20.9 minutes that run's lanes took at the median): ten 20.6
minutes, twelve 17.3, fourteen 15.8, sixteen 14.1, eighteen 13.1, twenty 11.9, twenty-four 10.2; the first step of two shards
takes off 3.3 minutes and each one after it 1.0 to 1.7, while the jobs a pull request holds rise by four each. Beyond about eighteen a single test (`tests/oracles/test_closed_loop.py`, about 470 s) is the floor.
The table was rebuilt again on 2026-11-10, when the suite weighed 49586 s a lane against 28876 s the day before (the 16 shards of the
first table had run 10 to 25 minutes, two of them cancelled at their 30-minute limit): at sixteen the slowest shard of a lane is predicted
at 16.0 minutes on the median lane's seconds and 19.2 on the slowest lane's, from the five lanes' own seconds under the new deal.
The pushes of 2026-11-11 (runs 42939711722 and 42942952966) cancelled a shard of two lanes each at 30 minutes, with 5307 and
5547 s of tests in it against 3400 predicted: the suite had gained test files since the table (the scene tests of #232, the open
flow's of #235) and a file the table did not list went by a hash of its path, so shards 14 and 10 took 2000 to 2400 s of
unlisted files besides their own share. Two changes: the table now lists every file its reports hold and a file it does not
list is placed on the lightest shard at an assumed weight (below), and the count is twenty, because the rebuilt table at
sixteen predicts 23.8 minutes for the slowest shard on the slowest lane's seconds (18.2 on the median lane's), and at twenty
19.6 (14.1). The minutes are the lane's test seconds in the shard times 0.284 (the median over the 146 shard jobs of those two
runs that finished, shard 1 apart; 0.320 is their 90th percentile, and the highest 0.355).

**A file the table does not list** (one added since the table was built) is placed after all the listed ones, in the order
of its path, on the shard with the least time so far, at an assumed weight: the 90th percentile of the table's file weights
(268 s on the table of this writing; `tests/_sharding.py`, `new_file_weight`), a guess for a heavy file rather than a
typical one, and never on a shard the table gives to one file alone. It replaces the hash, which put a new heavy file on
whichever shard it fell on. A pull request that adds a file moves the shards of the files the table does not list that sort
after its path, so those restore a cache their base did not save; the listed files move nowhere. The next rebuild replaces
the guess with a measurement. The rebuild lists every file its reports hold (`--over` 0), so that "not listed" means "new".

The split works for any count: `tests/shard_weights.json` lists the files with their seconds, dealt longest first, and a
file stays whole on one shard, so its tests share the programs it compiles. A file heavier than a shard's mean would be
the floor of the slowest shard however many shards there are, so such a file is dealt **by test** instead (the table's
`split`, written by `scripts/shard_weights.py --split FILE`): its tests spread over the shards, and each shard that gets
some of them compiles the programs they share once more. **At the default no file is that heavy** (the heaviest,
`tests/saved/test_registry_walk.py`, weighs 2452 s against a mean of 2841), so the table splits only the lubrication oracle's file, and splitting
`test_registry_walk.py` would cost more than it saves; `tests/test_sharding.py` fails when a file passes the
mean at the default count, and `scripts/shard_weights.py` prints the files that pass it at the count it is given (from
twenty-four shards, `test_registry_walk.py` does). Rebuild the table from a push run's reports when the suite has changed
much (`gh run download <run> --pattern 'durations-*'`, then `scripts/shard_weights.py`, its docstring has the command).

## The development branch's name

The development branch is `release/v0.3.0`. The workflows that run on a push or a pull request (`ci.yml`, `platforms.yml`,
`ready.yml`, `stress.yml`) name `release/**` in their triggers, so they run on it and on the next release's branch alike. Two
steps still name `release/v0.3.0` itself and must be pointed at the next branch when this one closes: the docs check's
`--base` (`ci.yml`, `platforms.yml`) and the confidentiality scan's `--public` (`ci.yml`). It was named `v0.3`
until 2026-11-05. A clone
follows the rename in one of two ways, by whether it has a local branch `v0.3`:

- with one (`git branch --list v0.3` prints it):
  `git branch -m v0.3 release/v0.3.0 && git fetch origin && git branch -u origin/release/v0.3.0 release/v0.3.0 && git remote prune origin`;
- without one (a clone made after the rename, or one that only tracked `origin/v0.3`): `git fetch --prune origin`, then
  `git switch release/v0.3.0` to work on the branch.

The steps that read the branch (the docs check's `--base`, the confidentiality scan's `--public`) name
`origin/release/v0.3.0` and nothing else; the docs check's step fails when that branch cannot be read.

## CI: the `ready` label

A run of the whole matrix is 63 jobs against the organisation's 60 at once, so two pull requests in CI queue each other,
and an agent pushes two to four times a round. So `ci.yml` has two modes. The **full** matrix runs on a push to a branch
(the development branch's post-merge run included), on a manual run, and on a pull request that carries the label **`ready`**
(for a pull request, 27 jobs, or more when its diff starts the lanes of "CI: which lanes a pull request runs"). A pull
request without it runs a **reduced** set of 16 jobs: `typecheck`, `no private name or path`, the docs check, and the
`default` lane on Python 3.12 with all its shards.

A required check never passes by being skipped. In a reduced run the one required job it does not run for (`default (Python
3.14)`) still runs, and fails in its first step with "the full matrix has not run on this head: add the `ready` label". So a
pull request is mergeable only on a full run of its head: add `ready` when the head is final, and merge when every required
check is green on that run. Pushing while the label is on starts a full run of the new head.

A run reads the mode when it starts, from the pull request's labels as they are then (the `shards` job's mode step),
not from the event that started it. Neither `ci.yml` nor `platforms.yml` listens to `labeled`, so no label starts a run
of theirs. Adding `ready` starts `ready.yml`, whose one job re-runs the head's newest run of each (`scripts/ci_ready.py`):
a reduced run in flight is cancelled first, a run that has not read its mode yet is left to read it, and a full run in
flight or a full run that passed is left alone. The re-run is the pull request's own run, on the same commit and in the
same check suite, so branch protection counts its newest attempt; a newer push still cancels it. Re-running a run by
hand (Re-run all jobs) reads the label too, which is the way for a pull request from a fork, whose token cannot re-run
anything; Re-run failed jobs keeps the first attempt's mode, since the job that read it passed.

Why not a run per label, or a manual run: a workflow listening to `labeled` starts a run for every label, its jobs all
skipped for any label but `ready`, and that run, the newest of `ci.yml` on the head, left the pull request blocked with
every required check green (pull request #128 has the runs). And GitHub evaluates for a pull request only the checks of
runs started by `push`, `pull_request` and a few other events, not by a manual run (`workflow_dispatch`). Any other
label (`stress` among them) starts only `ready.yml`'s skipped job, in a group of its own, under no required name, and the
stress workflow's run. `platforms.yml` (not a required check) runs its mode job alone until `ready`.
`tests/test_ci_ready_label.py` reads the three workflows for all of this, and `tests/test_ci_ready.py` the script.

## CI: which lanes a pull request runs

From the merge of the change that moves the lanes, four checks are to be required on `release/v0.3.0`: **`default (Python
3.12)`**, **`default (Python 3.14)`**, **`typecheck (pyright, Python 3.12)`** and **`no private name or path`**. Until then
branch protection still requires seven (the three lanes below as well), so the maintainer edits it at the merge: remove
the three, then merge (or merge first, and every pull request after it waits on `default (Python 3.13)` until he does). 3.12 is the floor and 3.14 the newest Python, so a pull request
that passes both passes between them. Three lanes are not required, and run on a pull request only when its diff can
change by itself whether MIME works with them (a decision of 2026-11-08 the maintainer can reverse; it follows from the
count of jobs above and from the compilation cache below):

- `default` on **Python 3.13**, `maddening-main` (MADDENING's `main`) and `openmaterica-main` (OpenMaterica's `main`) run on a
  push to `main` or a `release/**` branch, on a manual run, and on the **nightly** run (`.github/workflows/nightly.yml`);
- on a pull request they run only in the full matrix (`ready`), and only when the diff touches (`scripts/ci_boundary.py`
  holds the paths, `tests/test_ci_boundary.py` pins them to the repository): the dependency pins (`pyproject.toml`,
  `uv.lock`) or CI itself (the workflows, the actions, the scripts they call), which start all three; or MIME's adapter to
  one upstream, which starts that upstream's lane alone. For MADDENING an adapter is a module that **calls into it**: that
  imports `GraphManager`, `EdgeSpec`, `BoundaryInputSpec`, `UnitMismatchWarning`, `resolve_dotted_name` or
  `maddening.testing`, outside `nodes/` (the graph's construction and compile, the build's checks, the experiment and
  control runs, the verification harness, `core/_node.py`, `core/ledger.py`), and `lab_fixture/`; the list is
  `MADDENING_ADAPTERS` in `scripts/ci_boundary.py` and `tests/test_ci_boundary.py` fails when a caller is missing from it.
  For OpenMaterica, the three modules of `src/mime/materials/` that import `materica`. A module that only describes a node
  (`NodeMeta`, `ParamSpec`, `StaticArray` and `BoundaryInputSpec` under `nodes/`) or names `SimulationNode` to annotate a
  type is no adapter: the push and nightly runs cover it;
- where a lane does not run, `default` on 3.13 has **no job**, and the two `-main` lanes' jobs are **skipped** (GitHub
  reports a job excluded by an `if` as skipped, and a skipped job passes a required check; so neither stays required).
  Counts of jobs in this file are jobs that run: a push's run lists one more, the skipped `usd-core-newest` entry.

The **nightly** run dispatches `ci.yml` on each `release/*` branch whose newest commit is under 45 days old. GitHub runs a
schedule from the default branch's file only, so it fires from the next fast-forward of `main` to the release branch (or
from a pull request of that one file into `main`), and can be run once by hand (Actions, Nightly, Run workflow) after that.
It asks for a warm compilation cache, which saves nothing. It then **waits for the runs it dispatched** and fails itself,
naming the branch and each lane that is red, missing or unfinished, so that GitHub's mail about a failed scheduled run
reaches the maintainer: the `-main` lanes' shard jobs are `continue-on-error`, so a red upstream `main` would otherwise
show nowhere. On a push and a manual run the `-main` lanes' own jobs (the ones under their names) turn the run red when
their shards fail; on a pull request they do not (neither lane is required there).

**The release gate: every lane, `default` on 3.13 and both `-main` lanes included, green on the release branch's head
before a release**, and it is enforced: `publish.yml`'s build job runs `scripts/release.py check-ci`, which refuses unless
the newest push or manual run of `ci.yml` on the tagged commit has all five lanes' jobs concluded `success`, naming each
that did not (a run's own conclusion is not read: it is `success` over a red `continue-on-error` job). To check before
tagging: `python3 scripts/release.py check-ci --repository <owner>/<repo> --commit <sha>`.

These three lanes keep **no compilation cache**: they restore none and save none. Each push saved about 13 GB of
compilation caches across the five lanes against GitHub's 10 GB per-repository limit, so each push evicted the last, and a
cold run of a lane that runs on a push or at night costs minutes nobody waits for. Their **per-test duration gate warns
instead of failing** (`scripts/report_test_durations.py --warn-only`: an unlisted test over its line is a warning annotation, and a
report that cannot be read still fails); the gate stays a failure on `default` for 3.12 and 3.14, which gate every pull
request. A `stress` run with `python=3.13` finds no compilation cache to restore and runs cold.

## CI: re-running one shard

A failed shard job can be re-run alone (Actions, the run, Re-run failed jobs, or `gh run rerun --failed`), on a named
cause (a runner, a flake whose test you have fixed or recorded), not to see green. Each shard job uploads its report as
`...-shard-<n>-attempt-<k>`, `k` the run attempt, and the lane's job (`default (Python 3.x)`, `maddening-main`,
`openmaterica-main`) keeps the newest attempt of each shard (`scripts/latest_attempt.py`) before it checks them, so the
re-run's report replaces the first attempt's and the lane clears when the shard passes. It does not hide a failure that
persists: the newest attempt's own outcome and JUnit report are what is read, and `default` also fails unless the run's
list of jobs, at their latest attempts, shows that Python's shard jobs all concluded success. The manual run's `plant_shard_failure` input (a shard number) fails that shard of the Python
3.12 default lane on the first attempt only, to show this on a run.

The shell-completion tests run in a job of their own per lane (`completion tests (...)`), which each lane's job under its own
name reads from the run's list of jobs at their latest attempts, as it reads its shards: the lanes whose jobs carry
`continue-on-error` cannot read it from `needs.<job>.result`. The manual run's `plant_completion_failure` input fails that
job in every lane; each lane's required check must then be red.

## CI: the duration gate

**The rule (decided 2026-11-09 on the maintainer's approval of a measured proposal; he can reverse it).** Each shard's
`Test durations` step (`scripts/report_test_durations.py`) judges every test by who answers for it:

- An unlisted test over **30 s** is re-run once, alone, with an empty compilation cache, and fails the job only if it is
  over 30 s again. **On a push, a manual run and the nightly run, that holds for every unlisted test, whatever the diff**:
  the push is the merged change, and its cache is more often warm than a pull request's. **On a pull request** it holds
  only for **the change's own** test: its file is in the change's diff, or its node id is new (the base does not collect
  it, which covers a case that a registry entry adds to a test file the change never opened).
- On a pull request, any **other** unlisted test fails only over a flat **90 s** (again after the re-run); **between 30 s
  and 90 s it warns, and the pull request passes.** That band is the price of the rule: a pull request that slows a test
  it did not touch (an un-jitted loop in `src`) can take a 3 s test to 89 s with green checks, and it is caught by the
  push after the merge (over 30 s, and for a listed test over its ceiling). Over 5 s every unlisted test warns, as before.
- A **listed** test has a **ceiling**: the larger of 3 times its median and 1.2 times its 95th percentile over the
  triage's window of cold runs, at least 60 s (`tests/duration_ceilings.json`, written by the triage below, never by
  hand). Over it, the step fails a push and warns a pull request, unless the pull request owns the test (the same two
  tests of "own"), where it fails. The re-run applies to a ceiling too. A listed test with no ceiling in the file (added
  since the last triage) has none.
- "The change" on a pull request is its diff against its base (`--own-against HEAD^1`: the merge commit against its first
  parent). If the diff or the base's collection cannot be read, or the base is a zero or an unfetchable revision, every
  test counts as the change's own (strict).
- The 3.13 and `-main` lanes apply the same rule and warn instead of failing (above).

**Why the ceiling is not 3 times the median.** On the 35 cold push reports of 6 to 9 November (729 listed tests, 694 with a
ceiling), 3 times the median with a 60 s floor left 23 of the 35 runs with a first-run exceedance and 9 with the same test
over in both lanes (a proxy for the re-run alone not clearing it): a red push most days. A spread term fixes most of it:

| ceiling | runs with an exceedance, of 35 | with the same test over in both lanes | rolling (ceiling from the previous 10 runs), of 34 |
|---|---:|---:|---:|
| 3 x median, at least 60 s | 23 | 9 | 17 and 8 |
| **max(3 x median, 1.2 x p95), at least 60 s** | **9** | **0** | **13 and 3** |
| max(3 x median, 1.5 x p90) | 11 | 2 | 13 and 4 |
| 4 x median, at least 60 s | 10 | 3 | 11 and 4 |

The in-sample columns are optimistic (the exceeding run is in the statistic); the rolling column is the fairer read. On the
last 10 of the 35 runs, 5 of the 694 listed tests pass the new ceiling (4 runs, none in both lanes), against 8 for the first.
Three tests changed regime inside the window (`test_closed_loop`'s 0.2.0 pair case, median 76 s, then 275 to 664 s in the
last 4 runs, with the 0.2.0 family-walk cases beside it; the two `driven` cases of `test_full_inertia_tensor`, 9 to 22 s,
then 36 to 87 s); no rule can be fitted to them, so the triage names each ceiling that **rose over 1.5 times** at the top
of its pull request body, with the test's median before and after (a ceiling fitted to a regression stops failing it, so a
rise is never folded in silently).

The numbers come from two days of pull request runs (28 push runs beside them): at 20 s the gate failed 20 shard jobs
(24 failing records) in 9 of 79 runs; 3 of its (run, test) pairs, in 2 runs, were slow in the real sense and the rest
cold-cache artefacts (a warm time of 0.1 to 17 s). At 30 s for the change's own tests and 90 s for the rest it fails 12
of those jobs (13 records), in 3 runs, one of them real (the other two are an own test's cold first run and a
cold-by-design run, which the re-run alone clears or does not). Both counts are upper bounds. What it lets through is slow
tests between 20 s and 30 s, and the triage is the check on those.

**The weekly triage** (`scripts/duration_triage.py`) reads a week of cold push and nightly reports (`durations-default-*`
of `ci.yml` on `release/v0.3.0`, Python 3.12 and 3.14: about 50 push runs and 2 GB of artifacts a week) and writes one
branch, made from the checkout's `HEAD`, so run it on a clean checkout of `release/v0.3.0`: the ceilings file,
additions to `tests/duration_allowlist.txt` by the rule below (only a test pytest collects on that checkout, and never
`tests/test_canary.py`, the planted tests), and removals (a listed test under 10 s in every report). Its pull request body
names, first, each ceiling that rose over 1.5 times; then the listed tests whose median moved over 2 times between the
window's earlier reports and its latest quarter, the listed tests whose largest report is over their ceiling, and the
unlisted tests whose cold median is over 20 s. **Run it as your own account**, never from a workflow: a pull request
opened with `GITHUB_TOKEN` starts no checks. No schedule is set; the maintainer sets one when he wants it.

```
python scripts/duration_triage.py --branch triage/durations-$(date +%F) --body-file body.md
git push -u origin HEAD
gh pr create --base release/v0.3.0 --title "ci: the weekly duration triage" --body-file body.md
```

An added entry's reason ends in a sentence a person writes (why the test must run on every push) before the pull request
merges; the allowlist's own test refuses an entry that does not start `kept:`, and `test_duration_rule.py` one that still
holds the placeholder.

**Borderline tests.** A push to `release/v0.3.0` runs cold (an empty compilation cache) and a shard draws whichever runner
class is free, so a test near the line fails the `Test durations` step on some runs and not others, with nothing wrong in
it. A test is listed in `tests/duration_allowlist.txt` as persistently borderline when, over the last cold push runs of the
CI workflow, it was over the line (30 s; 20 s before 2026-11-09, and the entries of that time stay) at least once, or over
85 % of it (25.5 s; 17 s before) in at least a quarter of them. Its entry says its largest cold duration and the runner,
and its typical warm duration. The list is a record of measurements: no test is made faster, slow-marked, shrunk or
derandomised to leave it.

## CI: stress runs

A property test, or the control that must fail every run, is shown stable by running it many times with fresh Hypothesis
seeds. CI does that in parallel (`.github/workflows/stress.yml`), so it does not hold a maintainer's machine for the
better part of an hour. It is not a required check.

**Ask for one** by adding the label **`stress`** to the pull request. The run starts when the label is added and again on
every push while the label stays (a newer push cancels the older run); another label starts nothing. By default it
runs every test file the pull request changes (`tests/**/test_*.py`). A line of the pull request's description adds to
that, or sets the options:

```
stress: tests/oracles/test_spinner.py tests/test_x.py::test_y[a] runs=30 python=3.14 seed=7 only
```

- a bare token is a test file or node id under `tests/` (no spaces in a node id), run in addition to the changed files;
  `only` leaves the changed files out;
- `runs=N` is 1 to 40 runs, **20** if absent; `python=3.12`, `3.13` or `3.14` is the Python, **3.12** if absent (3.14's
  shards have met flakes the others did not, so ask for it when that is the question);
- `seed=N` is the first run's Hypothesis seed; absent, a random one. Run *k* uses `N + k - 1`.

Editing the description does not start a run: remove the label and add it again. Anything the line cannot use, and a
selection that is empty, fails the `stress selection` job by name; a stress run that ran nothing is not a pass.

**Read it** on the run's page: the `stress summary` job lists, per test, how many of the runs it passed (tests that did
not pass every run first), then for each failure the run, its Hypothesis seed, the falsifying example and the
`@reproduce_failure` blob, and every run's seed. It fails if any run failed, or had no report, or passed no test. Each
run is a job (`stress run k of n`) set up as the default lane is, with the compilation caches the last push to `release/v0.3.0`
saved for the shards holding the selected files, restored and never saved (at most three shards' worth; the rest of the
files compile cold, so a first run can be slower). Each run's JUnit report is a run artifact, `stress-run-<k>`.

**Repeat a failing seed locally** on the same Python, with the tests the summary names:

```bash
JAX_PLATFORMS=cpu uv run python -m pytest tests/test_x.py::test_y --hypothesis-seed=1234567 -n 0
```

or paste the `@reproduce_failure(...)` line above the test for the exact example. A seed repeats the draws of a run, not
its timing; if it does not fail locally, run `seed=N` again on CI.

## Compliance Scripts

```bash
uv run --no-sync mime docs check            # the documentation's sources, the docs walk, the number rule, the citations, the pages built twice
uv run --no-sync mime docs build            # the pages, to docs/_build/ (untracked, never committed), for you to read
python scripts/check_anomalies.py
python scripts/check_confidential.py --key-file /path/to/confidential_scan.key --messages origin/release/v0.3.0..HEAD \
  --diffs origin/release/v0.3.0..HEAD
python scripts/check_confidential.py --key-file /path/to/confidential_scan.key --directory docs/_build   # the built pages
```

## Confidentiality

This repository is public. It never names the maintainer's private projects, and never carries a private path: an
absolute home directory, a session scratchpad, Claude Code's temporary directory, or the private planning directory
beside the repository, which only `CLAUDE.md` names. `scripts/check_confidential.py`, OpenMaterica's scan copied,
reads each tracked file's path and its content as the working tree holds it; with `--messages RANGE`, each commit's
message in the range; with `--diffs RANGE`, what each commit in the range adds against its first parent: the paths it
adds or changes, the lines it adds (with the line after each), and a binary content or a new file whole, so a name
added and removed again within a pull request is found; and with `--ref-name`, a branch's name. CI's `confidential`
job passes the pull request's or the push's range to both, and a pull request's branch name; a change into `main`
leaves out the development branch's commits (`--public origin/release/v0.3.0`), public already and scanned on their way
into it.

The names cannot be listed here, and plain hashes of them would not hide them, since hashing a word list recovers
short words. So `scripts/confidential_digests.txt` holds their HMAC-SHA256 under a secret key, 64 hexadecimal digits,
which CI reads from the repository secret `CONFIDENTIAL_SCAN_KEY`; the key is OpenMaterica's, so the digest file is
OpenMaterica's too. A key the digests were not made with is refused. A finding gives its file and line, and a term's
first letter and length, never the term.

**Without the key the scan fails**, so it cannot pass having checked nothing: on a push, on a pull request from one of
this repository's branches, and on your machine. The one exception is a pull request from a fork, which GitHub gives no
secrets: told apart by the event (its head repository is not this one), never by the key's absence alone, it checks the
paths, warns that the names were not checked, and passes when the paths are clean; the push after its merge scans with
the key.

A text file is read as UTF-8, a byte that is not read as a replacement, which keeps every ASCII word of a text in any
ASCII-compatible encoding. A binary file, or a text in UTF-16 or UTF-32, is read by its printable runs of four ASCII
characters or more, as bytes and as UTF-16 and UTF-32 of either byte order; its findings say `(binary)`, and their line
is the run's number. What the scan does not read: a run shorter than four characters; whatever a file holds
compressed, such as a PNG's compressed text chunks, a PDF's streams or an archive's members; an untracked file; a
commit outside the ranges it is given; and a line a commit only removes, which an earlier commit added (and the scan
read there, when it is in the range). Before committing a compressed file, strip its metadata, or look at it
decompressed.

- **Run it**, from the repository's root, with the key in a file or in `CONFIDENTIAL_SCAN_KEY`:
  ```sh
  python3 scripts/check_confidential.py --key-file /path/to/confidential_scan.key --messages origin/release/v0.3.0..HEAD \
    --diffs origin/release/v0.3.0..HEAD
  ```
  Without a key it still prints what it finds among the paths, and fails.
- **Add a term** (this needs the key): add it to the private list of terms, one a line, and write the digests again
  from the whole list. A term is found as a word, a pair of adjacent words or a hyphenated run, whatever the case and
  separators: `a-b` finds `A B`, `a_b` and `a-b`. The digest file is shared with OpenMaterica, so a term added is added
  there too.
  ```sh
  python3 scripts/check_confidential.py --key-file /path/to/confidential_scan.key --digest \
    < /path/to/confidential_terms.txt > scripts/confidential_digests.txt
  ```
- **A pre-push hook**, which would check what each push publishes before it does, is not here yet: CI's scan runs only
  once a push is public.

## MADDENING Version Update Policy

When MADDENING publishes a new version:

1. Review MADDENING's CHANGELOG.md — specifically Known Anomalies, Security, Verification sections
2. Review new/changed entries in MADDENING's known_anomalies.yaml
3. Run MIME's full test suite against the candidate version
4. Update the version pin in pyproject.toml
5. Update docs/validation/soup_package.md Section 8
6. Document the update in CHANGELOG.md
