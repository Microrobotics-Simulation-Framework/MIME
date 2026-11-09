"""The FVM's declaration at the build: the build report's line for the box (its cells, spacing and resolution, and
the run's memory estimate, the cells times the measured bytes a cell, against the memory a run can have), the memory
refusal (gate 13): a box whose run will not fit is refused when planned, with its route; what a CPU run can have (the
host's available memory, or its control group's limit where that is less); and the flow named as unmetered in a run's
energy account with the energy ledger attached."""

from __future__ import annotations

import pathlib

import pytest
from test_fvm_experiment import experiment

import mime.dynamics.flow._fvm as fvm
from mime.build import BuildRefused
from mime.build._maddening import available_memory, cgroup_limit, device_memory
from mime.dynamics import FVM
from mime.observations import EnergyLedger
from mime.units import mm, ms


def _box_line(built: object) -> str:
    lines = [line for held in built.report.checked.values() for line in held]  # type: ignore[attr-defined]
    return next(line for line in lines if line.startswith("the box:"))


def _cells(built: object) -> int:
    c = built.graph.get_node("water.flow").constants()  # type: ignore[attr-defined]
    return int(c["cells_x"] * c["cells_y"] * c["cells_z"])


def test_the_build_reports_the_box_and_its_memory_in_the_runs_dtype(dtype):
    built = experiment().build()
    cells = _cells(built)
    c = built.graph.get_node("water.flow").constants()
    line = _box_line(built)
    assert f"{c['cells_x']} x {c['cells_y']} x {c['cells_z']} = {cells} cells" in line
    per_cell = fvm.BYTES_PER_CELL[(fvm.StageKind.MONOLITHIC_PROJECTED_CG, dtype)]
    assert f"memory about {cells * per_cell / 2 ** 20:.4g} MiB" in line
    assert "16 across the lumen" in line


@pytest.mark.parametrize("spacing", [None, mm(0.03)], ids=["cells_across", "spacing"])
def test_a_box_whose_run_will_not_fit_is_refused_with_its_route(dtype, spacing):
    """The default box against a planted memory a byte under its estimate is refused (``flow.memory``), and a byte
    over it plans (the control); the route names the setting the grid was given by."""
    settings = {} if spacing is None else {"spacing": spacing}
    needed = fvm.memory_estimate("steady", _cells(experiment(**settings).build()))
    for room, fits in ((int(needed) - 1, False), (int(needed) + 1, True)):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fvm, "device_memory", lambda room=room: ("a planted memory", room))
            if fits:
                experiment(**settings).plan()
                continue
            with pytest.raises(BuildRefused) as refused:
                experiment(**settings).plan()
        found = [each for each in refused.value.refusals if each.code == "flow.memory"]
        assert len(found) == 1 and "a planted memory" in found[0].message
        assert ("FVM(spacing=" if spacing is not None else "FVM(cells_across=") in found[0].action


def test_a_box_of_billions_of_cells_is_refused_against_this_machines_memory(float64):
    with pytest.raises(BuildRefused) as refused:
        experiment(cells_across=2000, length=mm(1.0)).plan()
    assert "flow.memory" in [each.code for each in refused.value.refusals]


def _host(root: pathlib.Path, available_kib: int, lines: str, limits: dict[str, str]) -> pathlib.Path:
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/meminfo").write_text(f"MemTotal: {64 * 2 ** 20} kB\nMemFree: 1000 kB\n"
                                       f"MemAvailable: {available_kib} kB\n")
    (root / "proc/self/cgroup").write_text(lines)
    for path, value in limits.items():
        (root / "sys/fs/cgroup" / path).parent.mkdir(parents=True, exist_ok=True)
        (root / "sys/fs/cgroup" / path).write_text(value + "\n")
    return root


@pytest.mark.parametrize(("lines", "limits", "expected"), [
    ("0::/job/step\n", {"job/memory.max": "1073741824", "job/step/memory.max": "max"}, 2 ** 30),
    ("0::/job/step\n", {"job/step/memory.max": "max"}, None),
    ("0::/\n", {}, None),
    ("4:memory:/slurm/job\n2:cpu:/x\n", {"memory/slurm/job/memory.limit_in_bytes": "2147483648",
                                          "memory/memory.limit_in_bytes": "9223372036854771712"}, 2 ** 31),
], ids=["v2-parent", "v2-unlimited", "v2-root", "v1"])
def test_a_cpu_runs_room_is_the_hosts_available_memory_or_its_control_groups_limit_where_less(tmp_path, lines,
                                                                                             limits, expected):
    """Never the host's total memory: MemTotal is 64 GiB here, and what a run can have is MemAvailable (4 GiB) or the
    least memory limit of the control group and those above it, where that is less."""
    root = _host(tmp_path, 4 * 2 ** 20, lines, limits)
    assert available_memory(root) == 4 * 2 ** 30
    assert cgroup_limit(root) == expected
    where, room = device_memory(root)
    if expected is not None and expected < 4 * 2 ** 30:
        assert (where, room) == ("the control group's memory limit", expected)
    else:
        assert (where, room) == ("the host's available memory", 4 * 2 ** 30)


@pytest.mark.parametrize("ledger", [True, False], ids=["ledger", "control-no-ledger"])
def test_the_run_names_the_flow_unmetered_with_its_reason_in_its_account_and_its_text(float64, ledger):
    """With the energy ledger attached the run's account is not empty: it names the flow, which the build does not yet
    meter, with its reason, and so does the run's text; the build report lists the drive once and says why it is not
    metered. Without the ledger there is no account and no such line."""
    exp = experiment()
    if ledger:
        exp.observe(EnergyLedger())
    built = exp.build()
    result = built.run(ms(40.0))
    named = f"water.flow is not metered, so not in the account: {FVM.unmetered}"
    if not ledger:
        assert result.energy is None and "not metered" not in result.text()
        return
    account = result.energy
    assert account is not None and dict(account.unmetered) == {"water.flow": FVM.unmetered}
    assert f"water.flow: {FVM.unmetered}" in account.text() and "no receiver is metered" in account.text()
    assert f"energy ledger: {named}" in result.text().splitlines()
    lines = built.report.energy
    assert any(line.startswith("water.flow: receives deliveries (body_force:water) and is not metered")
               for line in lines), lines
    assert any(line.startswith("body_force:water: declares no energy class") and "not metered: " in line
               for line in lines), lines
