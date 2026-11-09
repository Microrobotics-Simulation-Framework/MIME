#!/usr/bin/env python
"""The weekly duration triage: a week of cold CI runs in, one pull request's changes out (CONTRIBUTING.md, "CI: the duration
gate"). Standard library, and ``gh`` for the download; it loads ``scripts/report_test_durations.py`` for the report reader.

It reads the JUnit reports that the push runs and the nightly run of ``ci.yml`` upload (``durations-default-python-*``),
keeps the cold ones (a push starts with an empty compilation cache, the nightly run too), and writes

1. ``tests/duration_ceilings.json``, the file the gate reads: for each listed test seen in at least ``--min-runs``
   reports, the larger of 3 times its median and 1.2 times its 95th percentile, at least 60 s, and the medians it
   came from. The gate fails a push whose listed test passes its ceiling (once re-run alone), and warns on a pull
   request unless the pull request owns the test. This file is written here and nowhere else;
2. additions to ``tests/duration_allowlist.txt`` by CONTRIBUTING's borderline rule: an unlisted test that was over the
   line (30 s) at least once, or over 85 % of it in at least a quarter of its reports. The entry's reason is a template
   with the measured times, ending in what a person must write before the pull request merges;
3. removals from it: a listed test under 10 s in every one of its reports (at least ``--min-runs``).

It lists, in the pull request body and not in the files, what a person should look at, first of all **each ceiling
that rose more than 1.5 times** since the file it replaces, with the test's median before and after (a possible
regression: a ceiling fitted to a regression stops failing it, so a rise is never folded in silently), then the
listed tests whose median moved more than 2 times between the window's earlier reports and its latest quarter, the listed tests whose largest report is over
their ceiling, and the unlisted tests whose cold median is over 20 s (the real slow tests that the 30 s line passes).
An addition is proposed only for a test pytest collects on this checkout, and never for ``tests/test_canary.py``
(the planted tests of a manual run); run it on a checkout of ``release/v0.3.0``, because the branch is made from
``HEAD``. The download is every duration artifact of the window (about 37 MB a run, so some 2 GB for a week of pushes).

    python scripts/duration_triage.py --days 7 --branch triage/durations-2026-11-16 --body-file body.md
    git push -u origin triage/durations-2026-11-16
    gh pr create --base release/v0.3.0 --head triage/durations-2026-11-16 --title "ci: the weekly duration triage" --body-file body.md

**Run it as the maintainer's own account**, not from a workflow: a pull request opened with ``GITHUB_TOKEN`` starts no
checks (GitHub does not start a workflow from an event that token made), so its required checks would never run. The
schedule, if any, is the maintainer's to set. The script creates the branch and its commit and pushes nothing;
``--dry-run`` writes nothing and prints the body. ``--reports DIR`` reads runs already downloaded
(``DIR/<run id>/<artifact>/...``) instead of fetching.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import re
import statistics
import os
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tests" / "duration_allowlist.txt"
CEILINGS = ROOT / "tests" / "duration_ceilings.json"

#: The gate's line (``scripts/report_test_durations.py``) and the shares of it CONTRIBUTING's borderline rule uses.
LINE_S = 30.0
NEAR_SHARE = 0.85
QUARTER = 0.25
#: A listed test is ceilinged at the larger of ``CEILING_FACTOR`` times its median and ``SPREAD_FACTOR`` times its
#: 95th percentile, never under ``CEILING_FLOOR_S`` (the review of #218: on the 35 cold push reports this was red on
#: 9 raw and 0 persistent runs, against 23 and 9 for 3 times the median alone).
CEILING_FACTOR = 3.0
SPREAD_FACTOR = 1.2
CEILING_FLOOR_S = 60.0
#: A ceiling that rose over this share of the file it replaces is a possible regression, listed first in the body.
RISE_SHARE = 1.5
#: A listed test whose median in the window's latest quarter is this far from the rest of the window moved.
MOVED_SHARE = 2.0
#: The planted tests of a manual run (``plant_slow_tests``): never proposed.
CANARY_FILE = "tests/test_canary.py"
#: A listed test under this in every report is proposed for removal.
REMOVE_UNDER_S = 10.0
#: An unlisted test whose cold median is over this is named in the body, though the line passes it.
FLAG_MEDIAN_S = 20.0

ARTIFACT = re.compile(r"^durations-default-python-(?P<python>[\d.]+)-shard-(?P<shard>\d+)-attempt-(?P<attempt>\d+)$")
LANES = ("3.12", "3.14")


def _gate():
    spec = importlib.util.spec_from_file_location("_triage_report_test_durations", ROOT / "scripts" / "report_test_durations.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Observation(NamedTuple):
    run: int
    lane: str
    seconds: float


class Stats(NamedTuple):
    n: int
    median: float
    p90: float
    p95: float
    maximum: float
    max_run: int
    over_line: int
    near_line: int


class Proposal(NamedTuple):
    additions: dict[str, Stats]
    removals: dict[str, Stats]
    ceilings: dict[str, float]
    over_ceiling: dict[str, tuple[Stats, float]]
    flagged: dict[str, Stats]
    reports: int
    #: ``{test: (old ceiling, new ceiling, old median or None, new median)}`` for each ceiling that rose over RISE_SHARE.
    raised: dict[str, tuple[float, float, float | None, float]] = {}
    #: ``{test: (median of its earlier reports, of its latest quarter)}`` where they differ over MOVED_SHARE.
    moved: dict[str, tuple[float, float]] = {}
    #: Tests that would have been proposed but that pytest does not collect on this checkout.
    uncollected: tuple[str, ...] = ()


def _percentile(sorted_seconds: list[float], share: float) -> float:
    """Nearest rank."""
    return sorted_seconds[min(len(sorted_seconds) - 1, math.ceil(share * len(sorted_seconds)) - 1)]


def stats_of(observations: list[Observation], *, line: float = LINE_S) -> Stats:
    seconds = sorted(o.seconds for o in observations)
    worst = max(observations, key=lambda o: o.seconds)
    return Stats(len(seconds), statistics.median(seconds), _percentile(seconds, 0.9), _percentile(seconds, 0.95),
                 worst.seconds, worst.run, sum(s > line for s in seconds), sum(s > NEAR_SHARE * line for s in seconds))


def ceiling_of(median: float, p95: float = 0.0) -> float:
    return float(max(CEILING_FLOOR_S, math.ceil(max(CEILING_FACTOR * median, SPREAD_FACTOR * p95))))


def moved_of(observations: list[Observation]) -> tuple[float, float] | None:
    """The medians of a test's earlier reports and of its latest quarter (by run), if they differ over MOVED_SHARE: a
    regime change near the end of the window, which a median over the whole window hides."""
    ordered = [o.seconds for o in sorted(observations, key=lambda o: (o.run, o.lane))]
    if len(ordered) < 8:
        return None
    cut = len(ordered) - len(ordered) // 4
    early, late = statistics.median(ordered[:cut]), statistics.median(ordered[cut:])
    low, high = sorted((early, late))
    return (early, late) if low > 0 and high / low > MOVED_SHARE else None


def triage(observed: dict[str, list[Observation]], allow: dict[str, str], old_ceilings: dict[str, float], *,
           line: float = LINE_S, min_runs: int = 3, old_medians: dict[str, float] | None = None,
           collected: set[str] | None = None) -> Proposal:
    """What the week's cold observations propose. Pure: nothing is read or written here.

    ``collected`` (``None``: unknown) is the set of node ids pytest collects on this checkout; a test outside it is
    never proposed for the allowlist (it was renamed or removed since the reports). The planted canary never is.
    """
    old_medians = old_medians or {}
    observed = {n: o for n, o in observed.items() if not n.startswith(f"{CANARY_FILE}::")}
    stats = {nodeid: stats_of(obs, line=line) for nodeid, obs in observed.items()}
    candidates = {n: s for n, s in stats.items()
                  if n not in allow and (s.over_line >= 1 or s.near_line >= QUARTER * s.n)}
    additions = {n: s for n, s in candidates.items() if collected is None or n in collected}
    uncollected = tuple(sorted(set(candidates) - set(additions)))
    removals = {n: s for n, s in stats.items()
                if n in allow and s.n >= min_runs and s.maximum < REMOVE_UNDER_S}
    listed = (set(allow) | set(additions)) - set(removals)
    ceilings = {}
    for nodeid in sorted(listed):
        if nodeid in stats and stats[nodeid].n >= min_runs:
            ceilings[nodeid] = ceiling_of(stats[nodeid].median, stats[nodeid].p95)
        elif nodeid in old_ceilings:
            ceilings[nodeid] = old_ceilings[nodeid]      # not seen enough this week: its ceiling stands
    over = {n: (stats[n], c) for n, c in ceilings.items() if n in stats and stats[n].n >= min_runs and stats[n].maximum > c}
    flagged = {n: s for n, s in stats.items() if n not in allow and n not in additions and s.median > FLAG_MEDIAN_S}
    reports = len({(o.run, o.lane) for obs in observed.values() for o in obs})
    raised = {n: (old_ceilings[n], c, old_medians.get(n), stats[n].median) for n, c in ceilings.items()
              if n in old_ceilings and n in stats and stats[n].n >= min_runs and c > RISE_SHARE * old_ceilings[n]}
    moved = {n: m for n in ceilings if n in observed and (m := moved_of(observed[n]))}
    return Proposal(additions, removals, ceilings, over, flagged, reports, raised, moved, uncollected)


def addition_line(nodeid: str, s: Stats) -> str:
    return (f"{nodeid} # kept: borderline on cold push runs: its largest cold duration was {s.maximum:.1f} s (run "
            f"{s.max_run}; the runner is in that run's job log), over {LINE_S:g} s in {s.over_line} and over "
            f"{NEAR_SHARE * LINE_S:g} s in {s.near_line} of {s.n} cold reports, median {s.median:.1f} s; "
            f"WHY IT MUST RUN ON EVERY PUSH: to be written before this merges")


def apply_allowlist(text: str, proposal: Proposal) -> str:
    drop = set(proposal.removals)
    kept = [ln for ln in text.splitlines() if ln.partition(" # ")[0].strip() not in drop]
    kept += [addition_line(n, s) for n, s in sorted(proposal.additions.items())]
    return "\n".join(kept) + "\n"


def ceilings_json(proposal: Proposal, *, today: datetime.date, source: str, medians: dict[str, float] | None = None) -> str:
    """``medians`` (a test's median over the source) are kept so that the next triage can show a ceiling's rise."""
    medians = medians or {}
    return json.dumps({
        "written_by": "scripts/duration_triage.py: do not edit by hand",
        "generated": today.isoformat(),
        "source": source,
        "rule": (f"the larger of {CEILING_FACTOR:g} times the test's median and {SPREAD_FACTOR:g} times its 95th percentile "
                 f"over the cold reports of the source, at least {CEILING_FLOOR_S:g} s"),
        "ceilings": dict(sorted(proposal.ceilings.items())),
        "medians": {n: round(medians[n], 1) for n in sorted(proposal.ceilings) if n in medians},
    }, indent=1) + "\n"


def read_old(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """The ceilings and the medians of the file the triage replaces (none of either if it is absent)."""
    if not path.exists():
        return {}, {}
    table = json.loads(path.read_text(encoding="utf-8"))
    return ({n: float(c) for n, c in table["ceilings"].items()},
            {n: float(m) for n, m in table.get("medians", {}).items()})


def collect_ids(files: list[str]) -> set[str]:
    """The node ids pytest collects in ``files`` on this checkout (slow-marked ones too, as the allowlist's own test does)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "MADDENING_TEST_RELEASE_BETWEEN_FILES",
                        "PYTEST_ADDOPTS", "MIME_CI_PLANT_FAILURE", "MIME_CI_PLANT_SLOW_TESTS")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    files = [f for f in files if (ROOT / f).is_file()]      # a file the checkout no longer has collects nothing
    if not files:
        return set()
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                           "-m", "slow or not slow", *files], cwd=ROOT, env=env, capture_output=True, text=True)
    if done.returncode not in (0, 5):
        raise SystemExit(f"duration_triage: pytest could not collect {len(files)} files:\n{done.stdout[-2000:]}{done.stderr[-2000:]}")
    return {line.strip() for line in done.stdout.splitlines() if "::" in line}


# -- reading what CI uploaded ---------------------------------------------------------------------------------------

def read_run(directory: Path, run: int, gate, lanes: tuple[str, ...] = LANES) -> dict[str, list[Observation]]:
    """The cold, passing observations of one downloaded run: ``directory/<artifact>/...``, each shard's newest attempt."""
    newest: dict[tuple[str, str], tuple[int, Path]] = {}
    for child in sorted(directory.iterdir()):
        match = ARTIFACT.match(child.name) if child.is_dir() else None
        if match and match["python"] in lanes:
            key = (match["python"], match["shard"])
            if key not in newest or int(match["attempt"]) > newest[key][0]:
                newest[key] = (int(match["attempt"]), child)
    observed: dict[str, list[Observation]] = {}
    for (lane, _shard), (_attempt, artifact) in newest.items():
        modes = [m.read_text().strip() for m in artifact.glob("cache-mode-shard*.txt")]
        if modes != ["cold"]:                       # warm, or a mode nobody wrote: not what a ceiling is made of
            continue
        for report in artifact.glob("test-results-shard*.xml"):
            for t in gate.read_report(report):
                if t.outcome == "passed" and not t.collection_error:
                    observed.setdefault(t.nodeid, []).append(Observation(run, lane, t.seconds))
    return observed


def read_runs(root: Path, gate) -> tuple[dict[str, list[Observation]], list[int]]:
    observed: dict[str, list[Observation]] = {}
    runs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            runs.append(int(child.name))
            for nodeid, obs in read_run(child, int(child.name), gate).items():
                observed.setdefault(nodeid, []).extend(obs)
    return observed, runs


def _gh(*args: str) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} failed: {done.stderr.strip()}")
    return done.stdout


def fetch(repository: str, branch: str, since: datetime.date, into: Path) -> list[int]:
    """Download the duration reports of the push and manual runs of ``ci.yml`` on ``branch`` since ``since``."""
    runs: list[int] = []
    for event in ("push", "workflow_dispatch"):
        listing = json.loads(_gh("run", "list", "--repo", repository, "--workflow", "ci.yml", "--branch", branch,
                                 "--event", event, "--created", f">={since.isoformat()}", "--limit", "300",
                                 "--json", "databaseId,conclusion"))
        runs += [r["databaseId"] for r in listing if r["conclusion"] in ("success", "failure")]
    for run in runs:
        target = into / str(run)
        if not target.exists():
            args = ["run", "download", str(run), "--repo", repository, "--dir", str(target)]
            for lane in LANES:
                args += ["--pattern", f"durations-default-python-{lane}-shard-*"]
            subprocess.run(["gh", *args], capture_output=True, text=True)    # an expired run has nothing: no reports
            target.mkdir(exist_ok=True)
    return runs


# -- the pull request body ------------------------------------------------------------------------------------------

def _row(nodeid: str, s: Stats, extra: str = "") -> str:
    return (f"| `{nodeid}` | {s.n} | {s.median:.1f} | {s.p90:.1f} | {s.maximum:.1f} (run {s.max_run}) |"
            + (f" {extra} |" if extra else ""))


def body(proposal: Proposal, *, today: datetime.date, source: str, ceilings_only: bool) -> str:
    head = "| test | reports | median s | p90 s | max s |"
    rule = "|---|---:|---:|---:|---:|"
    out = [f"The weekly duration triage of {today.isoformat()}, from {source} ({proposal.reports} cold reports).", "",
           "Opened from the maintainer's own account, so that its checks run (a pull request opened with `GITHUB_TOKEN` "
           "starts none). Nothing here merges itself.", ""]
    if ceilings_only:
        out += ["This run wrote the ceilings file alone (`--ceilings-only`).", ""]
    out += [f"### Possible regressions: ceilings that rose more than {RISE_SHARE:g} times ({len(proposal.raised)})", "",
            "A ceiling fitted to a regression stops failing it, so each rise is looked at before this merges: find the "
            "commit that slowed the test, or say here why the new cost is the right one.", ""]
    out += ["| test | ceiling before s | ceiling after s | median before s | median after s |", "|---|---:|---:|---:|---:|",
            *[f"| `{n}` | {old:g} | {new:g} | {'not recorded' if before is None else f'{before:g}'} | {after:.1f} |"
              for n, (old, new, before, after) in sorted(proposal.raised.items())], ""] \
        if proposal.raised else ["None.", ""]
    out += [f"### Allowlist additions ({len(proposal.additions)})", "",
            f"Over {LINE_S:g} s at least once, or over {NEAR_SHARE * LINE_S:g} s in a quarter of the reports "
            "(CONTRIBUTING.md, borderline tests); only tests pytest collects on this checkout. **Each entry's reason ends "
            "with a sentence to write before merging** (`tests/test_duration_rule.py` refuses a merged placeholder), "
            "and the file's own check refuses an entry that does not start `kept:`.", ""]
    out += [head, rule, *[_row(n, s) for n, s in sorted(proposal.additions.items())], ""] if proposal.additions else ["None.", ""]
    if proposal.uncollected:
        out += ["Over the line but not collected on this checkout (renamed or removed since the reports; not proposed): "
                + ", ".join(f"`{n}`" for n in proposal.uncollected), ""]
    out += [f"### Allowlist removals ({len(proposal.removals)})", "",
            f"Under {REMOVE_UNDER_S:g} s in every cold report.", ""]
    out += [head, rule, *[_row(n, s) for n, s in sorted(proposal.removals.items())], ""] if proposal.removals else ["None.", ""]
    out += [f"### Ceilings ({len(proposal.ceilings)} listed tests)", "",
            f"`tests/duration_ceilings.json` is rewritten: the larger of {CEILING_FACTOR:g} times the median and "
            f"{SPREAD_FACTOR:g} times the 95th percentile, at least {CEILING_FLOOR_S:g} s.", ""]
    out += [f"Listed tests whose median moved more than {MOVED_SHARE:g} times within the window (the earlier "
            "reports against the latest quarter: a regime change, not noise; the ceiling is fitted to both):", "",
            "| test | earlier median s | latest quarter median s |", "|---|---:|---:|",
            *[f"| `{n}` | {early:.1f} | {late:.1f} |" for n, (early, late) in sorted(proposal.moved.items())], ""] \
        if proposal.moved else []
    out += ["Whose largest report is over their ceiling (the ceiling would have failed a push in the window; the ceiling is "
            "never under 1.2 times the 95th percentile, so only the top of the window can pass it):", ""]
    out += ["| test | reports | median s | p90 s | max s | ceiling s |", "|---|---:|---:|---:|---:|---:|",
            *[_row(n, s, f"{c:g}") for n, (s, c) in sorted(proposal.over_ceiling.items())], ""] \
        if proposal.over_ceiling else ["None.", ""]
    out += [f"### Real slow tests the {LINE_S:g} s line passes ({len(proposal.flagged)})", "",
            f"Unlisted, cold median over {FLAG_MEDIAN_S:g} s. A person decides whether each is listed, sped up or marked slow.", ""]
    out += [head, rule, *[_row(n, s) for n, s in sorted(proposal.flagged.items())], ""] if proposal.flagged else ["None.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reports", type=Path, help="a directory of downloaded runs, DIR/<run id>/<artifact>/ (default: fetch)")
    p.add_argument("--days", type=int, default=7, help="how many days back to fetch (default 7)")
    p.add_argument("--repository", default="Microrobotics-Simulation-Framework/MIME")
    p.add_argument("--source-branch", default="release/v0.3.0", help="the branch whose push and manual runs are read")
    p.add_argument("--min-runs", type=int, default=3, help="reports a test needs before it gets a ceiling or a removal")
    p.add_argument("--ceilings-only", action="store_true", help="write the ceilings file and propose no allowlist change")
    p.add_argument("--branch", help="create this branch from HEAD and commit the changes to it (nothing is pushed)")
    p.add_argument("--body-file", type=Path, help="write the pull request body here as well as printing it")
    p.add_argument("--dry-run", action="store_true", help="write nothing; print the body")
    args = p.parse_args(argv)

    gate = _gate()
    today = datetime.date.today()
    with tempfile.TemporaryDirectory(prefix="duration-triage-") as scratch:
        root = args.reports
        if root is None:
            root = Path(scratch)
            fetch(args.repository, args.source_branch, today - datetime.timedelta(days=args.days), root)
        observed, runs = read_runs(root, gate)
    if not observed:
        print(f"duration_triage: no cold report under {root}", file=sys.stderr)
        return 1
    allow = gate.read_allowlist(ALLOWLIST)
    old, old_medians = read_old(CEILINGS)
    proposal = triage(observed, allow, old, min_runs=args.min_runs, old_medians=old_medians)
    if proposal.additions and not args.ceilings_only:
        files = sorted({n.split("::", 1)[0] for n in proposal.additions})
        proposal = triage(observed, allow, old, min_runs=args.min_runs, old_medians=old_medians,
                          collected=collect_ids(files))
    if args.ceilings_only:
        proposal = proposal._replace(additions={}, removals={}, ceilings={
            n: c for n, c in proposal.ceilings.items() if n in allow})
    source = (f"{len(runs)} push and manual runs of ci.yml on {args.source_branch}, runs {min(runs)} to {max(runs)}, "
              f"Python {' and '.join(LANES)}, cold only")
    text = body(proposal, today=today, source=source, ceilings_only=args.ceilings_only)
    print(text)
    if args.body_file and not args.dry_run:
        args.body_file.write_text(text, encoding="utf-8")
    if args.dry_run:
        return 0
    if args.branch:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(ALLOWLIST), str(CEILINGS)], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            print(f"duration_triage: uncommitted changes to the files it writes:\n{dirty}", file=sys.stderr)
            return 1
        subprocess.run(["git", "switch", "-c", args.branch], cwd=ROOT, check=True)
    medians = {n: statistics.median(o.seconds for o in observed[n]) for n in proposal.ceilings if n in observed}
    medians |= {n: m for n, m in old_medians.items() if n in proposal.ceilings and n not in medians}
    CEILINGS.write_text(ceilings_json(proposal, today=today, source=source, medians=medians), encoding="utf-8")
    if not args.ceilings_only:
        ALLOWLIST.write_text(apply_allowlist(ALLOWLIST.read_text(encoding="utf-8"), proposal),
                             encoding="utf-8")
    if args.branch:
        subprocess.run(["git", "add", str(ALLOWLIST), str(CEILINGS)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"ci: the weekly duration triage of {today.isoformat()}"], cwd=ROOT, check=True)
        print(f"\ncommitted on {args.branch}; push it and open the pull request as yourself (see this file's docstring)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
