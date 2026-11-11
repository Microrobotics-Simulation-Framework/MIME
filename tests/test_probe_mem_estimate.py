"""``.github/probe/mem_estimate.py`` says how much memory a probe needs and which machine it goes to.

Read here: the routes at their limits; the estimate of a compiled program against the bytes its shapes force (a planted
larger program must read larger, every run); a real case through the real estimator, forward and in reverse mode; the
``probe/estimate.txt`` reader (which lines an arm runs, a malformed line refused); the route a file of cases gives, with a
case that is beyond a runner refused under ``--require actions``; and the calibration table: every case in it states its
estimate and its measured peak, and the factor the estimator multiplies by is never under 1 and is at least the ratio of
every case it underestimated, so that a table edited to hide an underestimate fails. What the table is evidence of is
stated in its own ``scope``: a few cases of one family on a CPU, not a bound for any other."""

import importlib.util
import json
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
    done = run_script("--from-file", "probe/estimate.txt", "--arm", "within", "--require", "actions", cwd=tmp_path)
    assert done.returncode == 0 and "route actions" in done.stdout, done.stdout[-800:] + done.stderr[-800:]
    found = json.loads((tmp_path / "r.json").read_text())
    assert set(found["program_bytes"]) == {"forward", "reverse"}
    assert found["program_bytes"]["reverse"]["total"] > found["program_bytes"]["forward"]["total"]  # the derivative holds more
    assert found["route"] == "actions" and found["estimate_gib"] > mem_estimate.FIXED_GIB * 0.5
    assert found["bound_gib"] == pytest.approx(found["estimate_gib"] * found["calibration_factor"])


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


def table_problems(table: dict) -> list[str]:
    """What is wrong with a calibration table: a case without its figures, a ratio that is not measured over estimated, a
    factor under 1 or under the ratio of a case the estimator underestimated."""
    found = []
    if not table.get("scope") or len(table.get("cases", [])) < 4:
        found.append("the table states no scope or fewer than four cases")
    for case in table.get("cases", []):
        if not {"name", "estimate_gib", "measured_gib", "ratio", "measured_is_lower_bound", "where"} <= set(case):
            found.append(f"case {case.get('name')}: a figure is missing")
        elif abs(case["ratio"] - case["measured_gib"] / case["estimate_gib"]) > 0.02 * case["ratio"]:
            found.append(f"case {case['name']}: the ratio is not measured over estimated")
        elif case["ratio"] > table["factor"] + 1e-9:
            found.append(f"case {case['name']}: underestimated by {case['ratio']:.2f}, over the factor {table['factor']}")
    if table.get("factor", 0) < 1.0:
        found.append("the factor is under 1")
    return found


def test_the_calibration_table_states_its_cases_and_its_factor_covers_every_underestimate():
    assert table_problems(json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))) == []


def test_a_table_with_a_hidden_underestimate_or_a_lowered_factor_is_caught_every_time():
    table = json.loads(mem_estimate.CALIBRATION.read_text(encoding="utf-8"))
    lowered = {**table, "factor": 0.5}
    assert any("under 1" in each for each in table_problems(lowered))
    worse = json.loads(json.dumps(table))
    case = worse["cases"][0]
    case["measured_gib"] = case["estimate_gib"] * (table["factor"] + 0.5)
    case["ratio"] = table["factor"] + 0.5
    assert any("underestimated" in each for each in table_problems(worse))
    thin = {**table, "cases": table["cases"][:2]}
    assert any("fewer than four" in each for each in table_problems(thin))
