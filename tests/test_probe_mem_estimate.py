"""``.github/probe/mem_estimate.py`` says how much memory a probe needs and which machine it goes to.

Read here: the routes at their limits; the estimate of a compiled program against the bytes its shapes force (a planted
larger program must read larger, every run); a real case through the real estimator, forward and in reverse mode, with the
family it belongs to; the ``probe/estimate.txt`` reader (which lines an arm runs, a malformed line refused, an arm with no
line refused by name); the route a file of cases gives, with a case that is beyond a runner refused under ``--require
actions`` and a killed estimator child read as beyond a runner; the factor chosen per family and mode, and the conservative
one for a graph of no calibrated family; the peak measure and the comparison that flags an underestimate; and the calibration
table: every case in it states its family, mode, estimate and measured peak, each family's factor equals the rule applied to
its own cases (the largest ratio of a completed case, up to 0.05, never under 1.10), so that a table edited to hide an
underestimate or a factor edited by hand fails. What the table is evidence of is stated in its own ``scope``: a few cases of
two families on a CPU, not a bound for any other."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("mime_probe_mem_estimate", ROOT / ".github" / "probe" / "mem_estimate.py")
assert _SPEC is not None and _SPEC.loader is not None
mem_estimate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mem_estimate
_SPEC.loader.exec_module(mem_estimate)

SCRIPT = ROOT / ".github" / "probe" / "mem_estimate.py"


def run_script(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], cwd=cwd, text=True, capture_output=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", "HOME": str(cwd)})


@pytest.mark.parametrize(("gib", "route"), [(0.0, "actions"), (12.0, "actions"), (12.01, "local"), (48.0, "local"),
                                            (48.01, "modal"), (500.0, "modal")])
def test_the_routes_change_at_twelve_and_forty_eight_gib(gib, route):
    assert mem_estimate.route_of(gib) == route


def test_a_programs_bytes_are_what_its_shapes_force_and_a_larger_program_reads_larger():
    import jax
    import jax.numpy as jnp

    spec = jax.ShapeDtypeStruct((1000,), jnp.float32)
    small = mem_estimate.program_bytes(lambda x: x * 2.0, spec)
    assert (small["argument"], small["output"]) == (4000, 4000)  # exact: 1000 float32 in, 1000 out
    spec_big = jax.ShapeDtypeStruct((1000, 1000), jnp.float32)
    big = mem_estimate.program_bytes(lambda x: jnp.linalg.inv(x @ x.T + jnp.eye(1000)), spec_big)
    assert big["argument"] == 4_000_000 and big["total"] > 2 * big["argument"]  # temporaries beyond the arguments
    bigger = mem_estimate.program_bytes(lambda x: jnp.linalg.inv(x @ x.T + jnp.eye(2000)), jax.ShapeDtypeStruct((2000, 2000), jnp.float32))
    assert bigger["total"] > 3 * big["total"]  # a planted larger program: four times the elements, read at over three times


def test_the_estimate_file_runs_an_arms_lines_and_the_star_lines_and_refuses_a_malformed_one(tmp_path):
    path = tmp_path / "estimate.txt"
    path.write_text("# a comment\n\n*: --case a:b --arg n=1\nx: --case c:d --steps 3  # trailing\ny: --case e:f\n", encoding="utf-8")
    assert mem_estimate.lines_of(path, "x") == [["--case", "a:b", "--arg", "n=1"], ["--case", "c:d", "--steps", "3"]]
    assert mem_estimate.lines_of(path, "") == [["--case", "a:b", "--arg", "n=1"]]
    path.write_text("--case a:b\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mem_estimate.lines_of(path, "")


def test_a_real_case_forward_and_in_reverse_through_a_probes_estimate_file(tmp_path):
    (tmp_path / "probe").mkdir()
    line = (f"within: --case sphere_settling:sphere_settling --path {ROOT / 'examples'} --steps 4 --wrt drag:ball.viscosity "
            f"--output-json {tmp_path / 'r.json'}\n")
    (tmp_path / "probe" / "estimate.txt").write_text(line, encoding="utf-8")
    done = run_script("--from-file", "probe/estimate.txt", "--arm", "within", "--require", "actions", "--summary-json",
                      str(tmp_path / "s.json"), cwd=tmp_path)
    assert done.returncode == 0 and "route actions" in done.stdout, done.stdout[-800:] + done.stderr[-800:]
    found = json.loads((tmp_path / "r.json").read_text())
    assert set(found["program_bytes"]) == {"forward", "reverse", "reverse checkpointed", "reverse plain"}
    reverse = found["program_bytes"]["reverse"]
    assert reverse["total"] > found["program_bytes"]["forward"]["total"]  # the derivative holds more
    assert reverse["total"] == max(found["program_bytes"][k]["total"] for k in ("reverse checkpointed", "reverse plain"))  # the larger
    assert found["route"] == "actions" and found["estimate_gib"] > mem_estimate.FIXED_GIB * 0.5
    # The bound is the estimate times the factor of the case's family and mode, read from the table independently of `judge`.
    table = json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))
    assert found["families"] == ["mrs"] and found["nodes_of_no_family"] == [] and found["mode"] == "reverse"
    assert found["calibration_factor"] == table["families"]["mrs"]["reverse"]
    assert found["bound_gib"] == pytest.approx(found["estimate_gib"] * table["families"]["mrs"]["reverse"])
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["arm"] == "within" and summary["bound_gib"] == pytest.approx(found["bound_gib"]) and summary["families"] == ["mrs"]


def test_a_case_beyond_a_runner_is_refused_with_its_route(tmp_path):
    # The runner's limit planted far below the case: the estimator must refuse the case, naming its route (every run).
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['m'] = m; spec.loader.exec_module(m)\n"
        "m.ACTIONS_GIB = 0.1\n"  # the plant: a runner that holds 0.1 GiB
        "sys.exit(m.main(sys.argv[1:]))\n", encoding="utf-8")
    direct = subprocess.run([sys.executable, str(planted), "--case", "sphere_settling:sphere_settling", "--path",
                             str(ROOT / "examples"), "--steps", "2", "--require", "actions"], cwd=tmp_path, text=True,
                            capture_output=True, check=False, env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", "HOME": str(tmp_path)})
    assert direct.returncode == 1 and "route: local" in direct.stdout and "beyond actions" in direct.stderr


def rule_factor(table: dict, family: str, mode: str) -> float:
    """The factor the table's rule gives a family and mode from its own cases: the largest ratio of a completed (not killed)
    case, up to 0.05, and never under the minimum."""
    ratios = [c["ratio"] for c in table["cases"] if c["family"] == family and c["mode"] == mode
              and not c["measured_is_lower_bound"]]
    assert ratios, f"{family} {mode} has no completed case"
    return max(mem_estimate.MIN_FACTOR, math.ceil(max(ratios) * 20 - 1e-9) / 20)


def modes_of(family: str) -> tuple[str, ...]:
    """The modes a family holds a factor for: a node family's forward and reverse, the raw-program family's one."""
    return (mem_estimate.RAW_MODE,) if family == mem_estimate.RAW_FAMILY else ("forward", "reverse")


def table_problems(table: dict) -> list[str]:
    """What is wrong with a calibration table: a case without its figures, a ratio that is not measured over estimated, a
    unit mismatch (``measured_mib`` against ``measured_gib``), a factor that is not the rule applied to its own cases, a
    family or mode with no completed case, a case of a family the table does not declare, a missing conservative factor."""
    found = []
    if not table.get("scope") or len(table.get("cases", [])) < 4:
        found.append("the table states no scope or fewer than four cases")
    families = table.get("families", {})
    for case in table.get("cases", []):
        if not {"name", "family", "mode", "estimate_gib", "measured_gib", "ratio", "measured_is_lower_bound", "where"} <= set(case):
            found.append(f"case {case.get('name')}: a figure is missing")
        elif case["family"] not in families or case["mode"] not in modes_of(case["family"]):
            found.append(f"case {case['name']}: family {case['family']!r} or mode {case['mode']!r} is not declared")
        elif abs(case["ratio"] - case["measured_gib"] / case["estimate_gib"]) > 0.02 * case["ratio"]:
            found.append(f"case {case['name']}: the ratio is not measured over estimated")
        elif "measured_mib" in case and abs(case["measured_gib"] - case["measured_mib"] / 1024) > 0.002 * case["measured_gib"]:
            found.append(f"case {case['name']}: measured_gib is not measured_mib / 1024 (the unit)")
        elif not case["measured_is_lower_bound"] and case["ratio"] > families[case["family"]][case["mode"]] + 1e-9:
            found.append(f"case {case['name']}: underestimated by {case['ratio']:.2f}, over the {case['family']} {case['mode']} "
                         f"factor {families[case['family']][case['mode']]}")
    for family, entry in families.items():
        if {m for m in ("forward", "reverse", "program") if m in entry} != set(modes_of(family)):
            found.append(f"{family}: declares the factors of {sorted(m for m in ('forward', 'reverse', 'program') if m in entry)}, "
                         f"not of {list(modes_of(family))}")
            continue
        for mode in modes_of(family):
            try:
                if abs(entry[mode] - rule_factor(table, family, mode)) > 1e-9:
                    found.append(f"{family} {mode}: the factor {entry[mode]} is not the rule applied to its cases "
                                 f"({rule_factor(table, family, mode)})")
            except AssertionError as why:
                found.append(str(why))
    uncalibrated = table.get("uncalibrated", {})
    if not all(uncalibrated.get(mode, 0) > max([c["ratio"] for c in table.get("cases", [])] + [1.0]) for mode in ("forward", "reverse")):
        found.append("the conservative factor for uncalibrated cases is missing or not above every ratio measured")
    return found


def test_the_calibration_table_states_its_cases_and_every_factor_is_the_rule_applied_to_them():
    assert table_problems(json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))) == []


def test_a_table_with_a_hidden_underestimate_a_hand_edited_factor_or_a_wrong_unit_is_caught_every_time():
    table = json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))
    lowered = json.loads(json.dumps(table))
    lowered["families"]["fvm"]["forward"] = 1.0
    assert any("not the rule" in each or "underestimated" in each for each in table_problems(lowered))
    raised = json.loads(json.dumps(table))
    raised["families"]["mrs"]["forward"] = 3.0  # a factor nothing fitted
    assert any("not the rule" in each for each in table_problems(raised))
    worse = json.loads(json.dumps(table))
    case = next(c for c in worse["cases"] if c["family"] == "fvm" and not c["measured_is_lower_bound"])
    case["measured_gib"] = case["estimate_gib"] * 5.0
    case["ratio"] = 5.0
    case.pop("measured_mib", None)
    assert any("underestimated" in each for each in table_problems(worse))
    unit = json.loads(json.dumps(table))
    case = next(c for c in unit["cases"] if "measured_mib" in c)
    case["measured_gib"] = case["measured_mib"] / 1000  # MiB / 1000 labelled GiB, the table's old mistake
    case["ratio"] = case["measured_gib"] / case["estimate_gib"]
    assert any("unit" in each or "not the rule" in each or "underestimated" in each for each in table_problems(unit))
    undeclared = json.loads(json.dumps(table))
    undeclared["cases"][0]["family"] = "lbm"
    assert any("not declared" in each for each in table_problems(undeclared))
    thin = {**table, "cases": table["cases"][:2]}
    assert any("fewer than four" in each for each in table_problems(thin))
    no_default = {**table, "uncalibrated": {"forward": 1.0, "reverse": 1.0}}
    assert any("conservative" in each for each in table_problems(no_default))


def test_the_factor_is_chosen_per_family_and_mode_and_an_uncalibrated_graph_takes_the_conservative_one():
    table = {"families": {"a": {"node_modules": ["pkg.a"], "forward": 1.2, "reverse": 1.5},
                          "b": {"node_modules": ["pkg.b"], "forward": 1.7, "reverse": 1.9}},
             "uncalibrated": {"forward": 2.5, "reverse": 2.6}, "light_modules": ["pkg.light"], "cases": []}
    assert mem_estimate.classify_nodes({"x": "pkg.a.node", "y": "pkg.light.z"}, table) == (["a"], [])
    assert mem_estimate.classify_nodes({"x": "pkg.a.node", "y": "pkg.other"}, table) == (["a"], ["y"])
    assert mem_estimate.classify_nodes({"x": "pkg.light.z"}, table) == ([], [])
    assert mem_estimate.factor_for(["a"], [], "forward", table)[0] == 1.2
    assert mem_estimate.factor_for(["a"], [], "reverse", table)[0] == 1.5
    assert mem_estimate.factor_for(["a", "b"], [], "forward", table)[0] == 1.7  # the largest of the families present
    assert mem_estimate.factor_for(["a"], ["y"], "forward", table)[0] == 2.5  # a node of no family: not bounded by a's cases
    factor, basis = mem_estimate.factor_for([], [], "reverse", table)
    assert factor == 2.6 and "UNCALIBRATED" in basis and "no case behind it" in basis
    assert mem_estimate.factor_for(["a"], [], "forward", {**table, "families": {"a": {"forward": 1.01, "reverse": 1.0}}})[0] == mem_estimate.MIN_FACTOR


def test_the_bound_applies_the_tables_factor_to_the_estimate():
    # judge() on a found dict: the bound is the estimate times the factor the table gives, read here from the table itself
    # (a judge that ignored the table's factor read 1.0 and no test saw it).
    table = json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))
    for family in ("mrs", "fvm"):
        for mode in ("forward", "reverse"):
            found = mem_estimate.judge({"estimate_gib": 2.0, "families": [family], "nodes_of_no_family": [], "mode": mode})
            assert found["bound_gib"] == pytest.approx(2.0 * table["families"][family][mode]) and found["bound_gib"] > 2.0 * 1.05
    found = mem_estimate.judge({"estimate_gib": 2.0, "families": [], "nodes_of_no_family": ["n"], "mode": "forward"})
    assert found["bound_gib"] == pytest.approx(2.0 * table["uncalibrated"]["forward"])


def test_an_arm_with_no_estimate_line_is_refused_by_name_and_so_is_a_line_for_an_arm_the_probe_lacks(tmp_path):
    path = tmp_path / "estimate.txt"
    path.write_text("a_1: --case a:b\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="a1"):
        mem_estimate.lines_of(path, "a1")  # the typo: the arm is 'a1', the line says 'a_1'
    path.write_text("a1: --case a:b\n*: --case c:d\n", encoding="utf-8")
    assert mem_estimate.lines_of(path, "a1", ["a1", "a2"]) == [["--case", "a:b"], ["--case", "c:d"]]
    assert mem_estimate.lines_of(path, "a2", ["a1", "a2"]) == [["--case", "c:d"]]  # the star line serves every arm
    path.write_text("a_1: --case a:b\n*: --case c:d\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="a_1"):
        mem_estimate.lines_of(path, "a1", ["a1"])  # the line names an arm the probe does not have
    path.write_text("a1: --case a:b --arg note=\"x#y\"  # a comment\n", encoding="utf-8")
    assert mem_estimate.lines_of(path, "a1") == [["--case", "a:b", "--arg", "note=x#y"]]  # a # inside the arguments stays


def test_the_script_refuses_an_arm_with_no_line_with_its_name_and_exit_2(tmp_path):
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "estimate.txt").write_text("tpyo: --case sphere_settling:sphere_settling --history 100000000000\n",
                                                     encoding="utf-8")
    done = run_script("--from-file", "probe/estimate.txt", "--arm", "typo", "--require", "actions", cwd=tmp_path)
    assert done.returncode == 2 and "route actions" not in done.stdout  # a refusal is 2; 1 means a route beyond --require
    assert "typo" in done.stderr and "no line for arm" in done.stderr


def test_an_estimator_child_killed_by_the_system_is_read_as_beyond_a_runner(tmp_path):
    # The route a killed child gives (the review's mutant read it as 'modal', and no test ran the path).
    module = tmp_path / "killme.py"
    module.write_text("import os, signal\n\ndef make():\n    os.kill(os.getpid(), signal.SIGKILL)\n", encoding="utf-8")
    (tmp_path / "estimate.txt").write_text(f"*: --case killme:make --path {tmp_path}\n", encoding="utf-8")
    done = run_script("--from-file", "estimate.txt", "--arm", "", "--require", "actions", cwd=tmp_path)
    assert done.returncode == 1 and "route local" in done.stdout and "ran out of memory" in done.stderr, done.stdout + done.stderr


def test_the_peak_measure_records_the_largest_process_and_passes_the_exit_status_through(tmp_path):
    big = "import numpy; a = numpy.ones(60_000_000); print(a.sum())"  # 480 MB
    done = run_script("--measure-peak", str(tmp_path / "p.json"), "--", sys.executable, "-c", big, cwd=tmp_path)
    assert done.returncode == 0
    peak = json.loads((tmp_path / "p.json").read_text())
    assert 0.45 < peak["peak_gib"] < 1.2 and peak["exit"] == 0
    small = run_script("--measure-peak", str(tmp_path / "q.json"), "--", sys.executable, "-c", "raise SystemExit(7)", cwd=tmp_path)
    assert small.returncode == 7 and json.loads((tmp_path / "q.json").read_text())["peak_gib"] < peak["peak_gib"]


def test_a_run_whose_peak_exceeds_its_bound_is_flagged_and_recorded_and_one_within_it_is_not(tmp_path):
    estimate = {"arm": "a1", "estimate_gib": 1.0, "bound_gib": 1.5, "families": ["fvm"], "uncalibrated": False, "mode": "forward"}
    (tmp_path / "e.json").write_text(json.dumps(estimate), encoding="utf-8")
    for peak, flagged, over_raw in [(0.9, False, False), (1.2, False, True), (1.5, False, True), (1.51, True, True), (3.0, True, True)]:
        (tmp_path / "p.json").write_text(json.dumps({"peak_gib": peak}), encoding="utf-8")
        done = run_script("--compare", str(tmp_path / "e.json"), str(tmp_path / "p.json"), "--check-json", str(tmp_path / "c.json"),
                          cwd=tmp_path)
        check = json.loads((tmp_path / "c.json").read_text())
        assert (done.returncode == 1) == flagged and check["exceeds_bound"] is flagged and check["exceeds_raw"] is over_raw, peak
        assert ("UNDERESTIMATE" in done.stdout) == flagged
        assert check["calibration_row"]["family"] == "fvm" and check["calibration_row"]["measured_gib"] == pytest.approx(peak, abs=1e-3)


# -- the raw-program route (--raw): a jitted function, estimated from XLA's compiled memory analysis ----------------------

RAW_FACTORY = '''\
import jax
import jax.numpy as jnp


def program(n=500, wide=False):
    """A square product and a solve, so that temporaries come beyond the arguments."""
    def run(a, b):
        return jnp.linalg.solve(a @ a.T + jnp.eye(a.shape[0], dtype=a.dtype), b)
    columns = 2 * n if wide else 8
    return jax.jit(run), (jax.ShapeDtypeStruct((n, n), jnp.float64), jax.ShapeDtypeStruct((n, columns), jnp.float64))


def not_a_pair():
    return 3
'''


def raw_estimate(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "rawfactory.py").write_text(RAW_FACTORY, encoding="utf-8")
    return run_script("--raw", "rawfactory:program", "--path", str(tmp_path), "--output-json", str(tmp_path / "raw.json"),
                      *arguments, cwd=tmp_path)


def test_a_raw_program_is_estimated_from_xlas_memory_analysis_and_says_which_route_it_took(tmp_path):
    done = raw_estimate(tmp_path, "--arg", "n=500")
    assert done.returncode == 0, done.stdout[-800:] + done.stderr[-800:]
    assert "estimate route: raw program" in done.stdout and "route: actions" in done.stdout
    found = json.loads((tmp_path / "raw.json").read_text())
    program = found["program_bytes"]["raw program"]
    assert program["argument"] == (500 * 500 + 500 * 8) * 8 and program["output"] == 500 * 8 * 8  # exact: float64 shapes
    assert program["temp"] > 500 * 500 * 8  # the product a @ a.T is held beyond the arguments
    assert found["estimate_route"] == "raw program" and found["mode"] == "program" and found["nodes_of_no_family"] == []
    assert found["estimate_gib"] >= program["total"] / mem_estimate.GIB  # the resident set after the factory comes on top


def test_a_larger_raw_program_reads_larger_every_run(tmp_path):
    """The control: the route must follow the program's size; a raw route that returned a constant would pass the exact-shape
    check above for one program and fail here."""
    totals = []
    for size in (200, 400, 800):
        assert raw_estimate(tmp_path, "--arg", f"n={size}").returncode == 0
        totals.append(json.loads((tmp_path / "raw.json").read_text())["program_bytes"]["raw program"]["total"])
    assert totals[0] * 2.5 < totals[1] and totals[1] * 2.5 < totals[2], totals  # four times the elements each step


def test_the_raw_route_takes_the_raw_program_factor_of_the_table_and_the_bound_follows_it(tmp_path):
    table = json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))
    assert mem_estimate.RAW_FAMILY in table["families"], "the raw route has no calibrated family: it would take the conservative one"
    assert raw_estimate(tmp_path).returncode == 0
    found = json.loads((tmp_path / "raw.json").read_text())
    assert found["families"] == [mem_estimate.RAW_FAMILY]
    factor = table["families"][mem_estimate.RAW_FAMILY][mem_estimate.RAW_MODE]
    assert found["calibration_factor"] == factor and found["bound_gib"] == pytest.approx(found["estimate_gib"] * factor)
    assert "raw_program, program" in found["calibration_basis"]


def test_a_raw_factory_that_does_not_return_a_function_and_its_arguments_is_refused(tmp_path):
    (tmp_path / "rawfactory.py").write_text(RAW_FACTORY, encoding="utf-8")
    done = run_script("--raw", "rawfactory:not_a_pair", "--path", str(tmp_path), cwd=tmp_path)
    assert done.returncode == 2 and "must return (function, arguments)" in done.stderr


def test_the_raw_and_the_experiment_routes_are_exclusive(tmp_path):
    both = run_script("--raw", "a:b", "--case", "c:d", cwd=tmp_path)
    neither = run_script(cwd=tmp_path)
    assert both.returncode == 2 and neither.returncode == 2


def test_a_raw_line_of_an_estimate_file_is_run_and_summarised_as_a_raw_program(tmp_path):
    (tmp_path / "rawfactory.py").write_text(RAW_FACTORY, encoding="utf-8")
    (tmp_path / "estimate.txt").write_text(f"*: --raw rawfactory:program --path {tmp_path} --arg n=300\n", encoding="utf-8")
    done = run_script("--from-file", "estimate.txt", "--arm", "", "--require", "actions", "--summary-json", str(tmp_path / "s.json"),
                      cwd=tmp_path)
    assert done.returncode == 0 and "estimate route: raw program" in done.stdout, done.stdout[-600:] + done.stderr[-600:]
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["estimate_routes"] == ["raw program"] and summary["mode"] == mem_estimate.RAW_MODE
    assert summary["families"] == [mem_estimate.RAW_FAMILY] and summary["uncalibrated"] is False
    # and the comparison after the run records a row for the raw family, ready for the table
    (tmp_path / "p.json").write_text(json.dumps({"peak_gib": summary["bound_gib"] * 1.01}), encoding="utf-8")
    flagged = run_script("--compare", str(tmp_path / "s.json"), str(tmp_path / "p.json"), "--check-json", str(tmp_path / "c.json"),
                         cwd=tmp_path)
    check = json.loads((tmp_path / "c.json").read_text())
    assert flagged.returncode == 1 and "UNDERESTIMATE" in flagged.stdout
    assert check["calibration_row"]["family"] == mem_estimate.RAW_FAMILY and check["calibration_row"]["mode"] == mem_estimate.RAW_MODE
