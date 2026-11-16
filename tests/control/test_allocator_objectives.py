"""``ctx.allocator``'s objectives, offset and regularisation, against SciPy's BVLS on rows the host builds itself.

The allocator reads its rows from the believed model's affine coefficients and its offset at zero command. This
module checks that wiring (the rows' order, their scales, the offset, the regularisation) in two steps, each with a
tolerance derived from rounding:

- **The model's rows and offset are the host's** (``actuation``, float64, with the torque's ``m x B`` and the force's
  ``m . G`` written here): each element within the first-order bound of the rounding on the model's own path
  (:mod:`model_rounding`), ``u`` times the sum over every rounding of how far the element moves with it. It is a worst
  case: the error measured at 72 points of the region is 0.02 to 0.29 of it in float32 and at most 0.11 in float64,
  so it is not the line the allocation is judged to. The control (``AP3_PLANT=coefficient``) puts one source of each
  coil off by 5e-5 of its coefficient in the model, ten times the least such error the bound flags on every draw.
- **The allocation is BVLS's on the model's own rows and offset**, a :class:`allocation_draws.Problem` judged by
  :func:`allocation_draws.verdict`, whose lines are derived from the law's rounding (its perturbation bound, its tie
  windows). Taking the rows from the model, not the host, keeps the model's own rounding out of the law's line.

``test_allocator.py`` holds the force-only closed loop the allocator is built from; here every objective the model
gives is allocated at a drawn point, in both dtypes:

- **Every objective** (``flux_density``, ``gradient``, both, ``force``, ``torque``, ``force`` with ``torque``), with a
  nonzero background field (the model's offset at zero command) where the objective reads the field.
- **The offset is guarded**: with the allocator reading the offset as zero the commands move off BVLS's past the line on
  every case below (asserted on each, not a rate).
- **``effort``** (towards zero) and **``toward="last"``**: the regularisation's rows ``(u - reference) / effort`` are
  stacked under the objective's in the oracle; with the effort ignored the commands differ from the oracle's on every
  case below.

What it bounds: the allocator agrees with a bounded least squares on its model's rows, and the model's rows with the
host's field. It grounds neither sign nor frame against hardware."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping
from functools import cache
from typing import Any

import jax
import model_rounding
import numpy as np
import pytest
from allocation_draws import Problem, bvls, region, two_sided_window, verdict
from believed_cases import LIMIT, OWNER, believed_sphere
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.control import _models
from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.registry import ForwardModel
from mime.units import A, N, N_m, T, T_per_m, m, mT

jnp: Any = jax.numpy
PROPERTY = settings(deadline=None, print_blob=True, max_examples=25)
#: The plant a control is measured with (``AP3_PLANT``): ``offset``, the model's offset at zero command taken as zero;
#: ``coefficient``, the model's coefficients with ``PLANTED_SOURCES``' off by ``PLANTED_PART`` of themselves.
PLANT = os.environ.get("AP3_PLANT", "")
PLANTED_SOURCES, PLANTED_PART = ((0, 0), (1, 0), (2, 0)), 5e-5
SYSTEM = "navion_2"
#: What each quantity's rows are over: the scale of a unit miss, SI.
SCALES = {"flux_density": 1e-2, "gradient": 1.0, "force": 1e-6, "torque": 1e-8}
UNITS = {"flux_density": T, "gradient": T_per_m, "force": N, "torque": N_m}
#: Backgrounds (T) the truth's field has with every current at zero: a model's offset that is not zero.
BACKGROUNDS = ((0.0, 0.0, 0.0), (1e-3, -2e-3, 1.5e-3), (-3e-3, 1e-3, 4e-3))
OBJECTIVES = {"flux_density": ("flux_density",), "gradient": ("gradient",), "flux_and_gradient": ("flux_density", "gradient"),
              "force": ("force",), "torque": ("torque",), "force_and_torque": ("force", "torque")}
ROWS = {"flux_density": 3, "gradient": 5, "force": 3, "torque": 3}


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


@cache
def plans(background: tuple[float, float, float]) -> Any:
    truth = CalibratedMultipole(calibration=asset(SYSTEM), background=T(background))
    return believed_sphere(truth=truth).plan()


def make_allocator(background: tuple[float, float, float], objective: tuple[str, ...], **given: Any) -> Any:
    held = {quantity: UNITS[quantity](SCALES[quantity]) for quantity in objective}
    return _models.ControlModels(plans(background)).allocator("coils", commands=("currents",), objective=held,
                                                              owner=OWNER, **given)


def five(gradient: np.ndarray) -> np.ndarray:
    return np.stack([gradient[0, 0], gradient[0, 1], gradient[0, 2], gradient[1, 1], gradient[1, 2]])


def host_rows(point: np.ndarray, moment: np.ndarray, background: tuple[float, float, float],
              objective: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The oracle's rows, offsets and scales for ``objective`` (its quantities in alphabetical order, as the allocator
    stacks them), from the host's ``actuation`` at ``point`` in float64: the matrix (R, K) per ampere, its value at zero
    command, and the rows' scales."""
    actuation = CalibratedMultipole(calibration=asset(SYSTEM), background=T(background)).actuation(m(tuple(point)))
    field, gradient = actuation.field, actuation.gradient  # (3, K), (3, 3, K)
    cross = np.asarray([[0.0, -moment[2], moment[1]], [moment[2], 0.0, -moment[0]], [-moment[1], moment[0], 0.0]])
    parts = {"flux_density": (field, actuation.background), "gradient": (five(gradient), np.zeros(5)),
             "force": (np.einsum("i,ijk->jk", moment, gradient), np.zeros(3)),
             "torque": (cross @ field, cross @ actuation.background)}
    ordered = sorted(objective)
    matrix, offset = (np.concatenate([parts[q][i] for q in ordered]) for i in range(2))
    scales = np.concatenate([np.full(ROWS[q], SCALES[q]) for q in ordered])
    return matrix, offset, scales


def as_quantities(rows: np.ndarray, objective: tuple[str, ...]) -> dict[str, Any]:
    """A vector over ``objective``'s rows as the wanted values the allocator takes (a gradient as its symmetric 3 x 3
    from its five independent entries, ``G_zz`` making it trace-free)."""
    wanted: dict[str, Any] = {}
    at = 0
    for quantity in sorted(objective):
        part = rows[at:at + ROWS[quantity]]
        at += ROWS[quantity]
        if quantity == "gradient":
            g = np.zeros((3, 3))
            g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2] = part
            g[1, 0], g[2, 0], g[2, 1] = g[0, 1], g[0, 2], g[1, 2]
            g[2, 2] = -g[0, 0] - g[1, 1]
            wanted[quantity] = jnp.asarray(g)
        else:
            wanted[quantity] = jnp.asarray(part)
    return wanted


@st.composite
def cases(draw: st.DrawFn, objectives: tuple[str, ...] = tuple(OBJECTIVES)) -> dict[str, Any]:
    """An objective, a point in the central half of the region, a moment, a background, a wanted value of size 0.1 to
    1 of what every coil at its limit reaches, in the rows' own scales, and a last command."""
    name = draw(st.sampled_from(objectives))
    centre, half = region(SYSTEM)
    point = centre + np.asarray([draw(st.floats(-1.0, 1.0)) for _ in range(3)]) * half
    part = st.one_of(st.just(0.0), st.builds(lambda size, sign: sign * size, st.floats(0.1, 1.0), st.sampled_from((-1.0, 1.0))))
    moment = np.asarray([draw(part) for _ in range(3)])  # a body's: no component whose product underflows float32
    if not moment.any():  # a moment is never zero: a force or torque of zero is its own case
        moment[2] = 1.0
    moment = moment / np.linalg.norm(moment) * draw(st.floats(1e-4, 1e-3))
    background = draw(st.sampled_from(BACKGROUNDS))
    objective = OBJECTIVES[name]
    matrix, offset, scales = host_rows(point, moment, background, objective)
    scaled = matrix / scales[:, None]
    direction = np.asarray([draw(st.floats(-1.0, 1.0)) for _ in range(len(matrix))])
    if np.linalg.norm(direction) < 1e-3:
        direction[0] = 1.0
    reach = float(np.linalg.norm(scaled, axis=0).sum()) * LIMIT
    size = float(np.exp(draw(st.floats(np.log(0.1), np.log(1.0)))))
    wanted = direction / np.linalg.norm(direction) * reach * size * scales + offset
    last = np.asarray([draw(st.floats(-LIMIT, LIMIT)) for _ in range(3)])
    return {"name": name, "objective": objective, "point": point, "moment": moment, "background": background,
            "matrix": matrix, "offset": offset, "scales": scales, "wanted": wanted, "last": last}


def allocated(case: Mapping[str, Any], dtype: str, **given: Any) -> np.ndarray:
    with in_dtype(dtype):
        solve = make_allocator(case["background"], case["objective"], **given)
        commands, report = solve(as_quantities(case["wanted"], case["objective"]),
                                 {"point": jnp.asarray(case["point"]), "moment": jnp.asarray(case["moment"])},
                                 {"currents": jnp.asarray(case["last"])})
    assert set(report.made) == set(case["objective"])
    return np.asarray(commands["currents"], dtype=np.float64)


def planted_coefficients(coefficients: Any) -> np.ndarray:
    """The control's coefficients: ``PLANTED_SOURCES``' off by ``PLANTED_PART`` of themselves."""
    held = np.reshape(np.array(coefficients, dtype=np.float64), calibration_of(asset(SYSTEM)).coefficients.shape)
    for k, s in PLANTED_SOURCES:
        held[k, s] *= 1.0 + PLANTED_PART
    return held.ravel()


def model_rows(case: Mapping[str, Any], dtype: str, *, planted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The believed model's rows per ampere and its value at zero command at the case's point, in ``dtype`` (as float64),
    in the objective's alphabetical order, each quantity's rows written here (a gradient's five independent entries).
    ``planted``: the model with the control's coefficients."""
    with in_dtype(dtype):
        model = make_allocator(case["background"], case["objective"]).model
        if planted:
            model = model.bound({"coefficients": planted_coefficients(model.theta["coefficients"])})
        coefficients, offset = model.affine({"point": jnp.asarray(case["point"]), "moment": jnp.asarray(case["moment"])},
                                            commands=("currents",), dtype=jnp.zeros((), dtype=dtype).dtype)
    rows, zero = [], []
    for quantity in sorted(case["objective"]):
        held, base = np.asarray(coefficients[quantity]["currents"], dtype=np.float64), np.asarray(offset[quantity], dtype=np.float64)
        if quantity == "gradient":
            rows.append(five(held))
            zero.append(five(base))
        else:
            rows.append(held)
            zero.append(base)
    return np.concatenate(rows), np.concatenate(zero)


def rounding_bound(case: Mapping[str, Any], dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Each element's bound on the model's rows and offset against the host's, stacked as :func:`model_rows` stacks
    them (:func:`model_rounding.bound`): the gradient's and the force's offsets are exact zeros in both."""
    found = model_rounding.bound(SYSTEM, case["point"], case["moment"], np.asarray(case["background"]), dtype)
    rows, zero = [], []
    for quantity in sorted(case["objective"]):
        rows.append(five(found[quantity]) if quantity == "gradient" else found[quantity])
        zero.append(found.get(f"{quantity}_offset", np.zeros(ROWS[quantity])))
    return np.concatenate(rows), np.concatenate(zero)


def oracle_problem(case: Mapping[str, Any], dtype: str, *, effort: float | None = None,
                   reference: np.ndarray | None = None) -> Problem:
    """``case`` as BVLS's problem on the model's own rows: the target is what is wanted less the model's offset; a
    regularisation adds the identity's rows over ``effort`` towards ``reference`` (zero by default)."""
    matrix, offset = model_rows(case, dtype)
    target, scales = case["wanted"] - offset, case["scales"]
    if effort is not None:
        near = np.zeros(matrix.shape[1]) if reference is None else reference
        matrix = np.concatenate([matrix, np.eye(matrix.shape[1])])
        target = np.concatenate([target, near])
        scales = np.concatenate([scales, np.full(matrix.shape[1], effort)])
    return Problem(SYSTEM, case["name"], matrix, target, scales, LIMIT)


# -- every objective ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
@PROPERTY
@given(case=cases())
def test_the_models_rows_and_offset_are_the_hosts(dtype, case):
    """The model's rows per ampere and its offset at zero command, in ``dtype``, against the host's (float64), each
    element within the first-order bound of the rounding on the model's own path (:func:`rounding_bound`)."""
    held, zero = model_rows(case, dtype, planted=PLANT == "coefficient")
    rows, offset = rounding_bound(case, dtype)
    assert np.all(np.abs(held - case["matrix"]) <= rows), case["name"]
    assert np.all(np.abs(zero - case["offset"]) <= offset), case["name"]


def test_the_force_row_ci_found_past_ten_eps_over_c_is_within_the_derived_bound():
    """CI's stress run (Hypothesis seed 2038640930) drew a force objective at (0.045, 0.078, 0.0434625) m with the
    moment along z: the float32 force ``F_z`` of coil 0, ``m_z G_zz``, was off by 11.1 ``eps / c`` of its row's
    largest term, past the earlier line of 10 (``c = 0.0153``, the field's cancellation there). Coil 0's ``G_zz`` is
    1/37 of its ``G_zy``, and its error is set by how its opposed sources' terms move with each rounding, not by
    its own size. At that draw the error is within the derived bound, and the control's coefficient is past it."""
    point, moment = np.asarray([0.045, 0.078, 0.0434625]), np.asarray([0.0, 0.0, 6.0718e-4])
    matrix, offset, scales = host_rows(point, moment, BACKGROUNDS[0], ("force",))
    case = {"name": "force", "objective": ("force",), "point": point, "moment": moment, "background": BACKGROUNDS[0],
            "matrix": matrix, "offset": offset, "scales": scales}
    held, _ = model_rows(case, "float32")
    rows, _ = rounding_bound(case, "float32")
    assert np.all(np.abs(held - matrix) <= rows)
    planted, _ = model_rows(case, "float32", planted=True)
    assert np.any(np.abs(planted - matrix) > rows)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@PROPERTY
@given(case=cases())
def test_the_allocator_agrees_with_bvls_on_every_objective(dtype, case):
    """Each objective the coil model gives, at a drawn point and background, in both dtypes: the commands within the
    allocation oracle's line (:func:`allocation_draws.verdict`) of BVLS on the model's own rows and offset."""
    commands = allocated(case, dtype)
    assert verdict(oracle_problem(case, dtype), commands, dtype) == [], (case["name"], dtype)


def test_the_gradient_and_the_flux_density_and_gradient_objectives_are_the_older_allocate_in_float64():
    """The same wanted field and gradient through ``allocate()`` (the host's allocation of the same rows, float64): the
    commands agree to rounding, for the gradient alone and with the flux density."""
    from mime.fields._coil_array import allocate

    point = np.asarray([0.001, -0.0005, 0.0007])
    field_wanted = np.asarray([0.01, 0.005, -0.004])
    gradient_wanted = np.asarray([[0.3, 0.1, 0.0], [0.1, -0.2, 0.05], [0.0, 0.05, 0.1]])
    gradient_wanted[2, 2] = -gradient_wanted[0, 0] - gradient_wanted[1, 1]
    background = BACKGROUNDS[1]
    actuation = CalibratedMultipole(calibration=asset(SYSTEM), background=T(background)).actuation(m(tuple(point)))
    for objective in (("gradient",), ("flux_density", "gradient")):
        with in_dtype("float64"):
            wanted = {"gradient": jnp.asarray(gradient_wanted), "flux_density": jnp.asarray(field_wanted)}
            solve = make_allocator(background, objective)
            commands, _ = solve({q: wanted[q] for q in objective}, {"point": jnp.asarray(point)})
        older = allocate(actuation, limits=A(LIMIT), field=T(tuple(field_wanted)) if "flux_density" in objective else None,
                         gradient=T_per_m(tuple(gradient_wanted.ravel())),
                         field_scale=T(SCALES["flux_density"]) if "flux_density" in objective else None,
                         gradient_scale=T_per_m(SCALES["gradient"]) if "flux_density" in objective else None)
        found = np.asarray(commands["currents"])
        expected = np.asarray(older.currents.value)
        assert np.max(np.abs(found - expected)) <= 64 * float(np.finfo(np.float64).eps) * LIMIT, (objective, found, expected)


# -- the offset ---------------------------------------------------------------------------------------------------

def zeroed_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plant: the model's affine form with its offset at zero command taken as zero."""
    held = ForwardModel.affine

    def without(self: ForwardModel, x: Any, **kwargs: Any) -> Any:
        coefficients, offset = held(self, x, **kwargs)
        return coefficients, {q: jnp.zeros_like(v) for q, v in offset.items()}

    monkeypatch.setattr(ForwardModel, "affine", without)


#: A point in the region's middle with a wanted flux density of 1/4 of what the coils reach there, and a background of
#: (-3, 1, 4) mT: the offset moves the commands by amperes.
FIXED = {"name": "flux_density", "objective": ("flux_density",), "background": BACKGROUNDS[2],
         "point": region(SYSTEM)[0] + 0.5 * region(SYSTEM)[1], "moment": np.asarray([1e-4, -2e-4, 3e-4]),
         "last": np.zeros(3)}


def fixed_case(objective: tuple[str, ...]) -> dict[str, Any]:
    matrix, offset, scales = host_rows(FIXED["point"], FIXED["moment"], FIXED["background"], objective)
    direction = np.linspace(1.0, -0.5, len(matrix))
    reach = float(np.linalg.norm(matrix / scales[:, None], axis=0).sum()) * LIMIT
    wanted = direction / np.linalg.norm(direction) * 0.25 * reach * scales + offset
    return {**FIXED, "name": "_".join(objective), "objective": objective, "matrix": matrix, "offset": offset,
            "scales": scales, "wanted": wanted}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("objective", [("flux_density",), ("flux_density", "gradient"), ("torque",), ("force", "torque")],
                         ids=["flux_density", "flux_density_and_gradient", "torque", "force_and_torque"])
@pytest.mark.parametrize("planted", [False, True], ids=["as_built", "control_the_offset_zeroed"])
def test_the_allocation_reads_the_models_offset_at_zero_command(objective, dtype, planted, monkeypatch):
    """With a background field (a torque's offset is ``m x B``), the commands are BVLS's on the target less the model's
    offset. The control, the allocator reading the offset as zero (the oracle's problem made before the plant), moves
    them off BVLS's past the line: asserted on every case here."""
    case = fixed_case(objective)
    problem = oracle_problem(case, dtype)
    if planted or PLANT == "offset":
        zeroed_offset(monkeypatch)
    found = verdict(problem, allocated(case, dtype), dtype)
    assert bool(found) is planted, (objective, dtype, found)


# -- the regularisation -------------------------------------------------------------------------------------------

EFFORT = 3.0
#: The controls' effort: small enough that the regularisation's rows dominate the objective's, so that ignoring them or their
#: reference changes the residual by far more than the law's tie window (at 3 A the torque rows of a force-and-torque
#: objective outweigh them: their rows are 10 to 20 per ampere over the scale).
CONTROL_EFFORT = 0.03


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("toward", ["zero", "last"])
@PROPERTY
@given(case=cases(("flux_density", "flux_and_gradient", "torque", "force_and_torque")))
def test_the_regularisation_is_the_identity_rows_over_the_effort_towards_zero_or_the_last_command(toward, dtype, case):
    """``effort`` a current: the oracle stacks ``(u - reference) / effort`` under the objective's rows, the reference
    zero for ``toward="zero"`` and the last command for ``toward="last"``."""
    commands = allocated(case, dtype, effort=A(EFFORT), toward=toward)
    reference = case["last"] if toward == "last" else None
    problem = oracle_problem(case, dtype, effort=EFFORT, reference=reference)
    assert verdict(problem, commands, dtype) == [], (toward, dtype)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("toward", ["zero", "last"])
@pytest.mark.parametrize("objective", [("flux_density",), ("force", "torque")], ids=["flux_density", "force_and_torque"])
def test_ignoring_the_effort_or_the_reference_moves_the_commands_off_the_regularised_oracle(objective, toward, dtype):
    """The controls: the same case allocated without ``effort`` (nothing regularised), and, towards the last command,
    with the reference taken as zero, each misses the regularised oracle's commands by more than its line (asserted
    here on every case: the as-built arm matches, each control does not)."""
    case = {**fixed_case(objective), "last": np.asarray([3.0, -2.0, 1.0])}
    problem = oracle_problem(case, dtype, effort=CONTROL_EFFORT, reference=case["last"] if toward == "last" else None)
    assert verdict(problem, allocated(case, dtype, effort=A(CONTROL_EFFORT), toward=toward), dtype) == []
    assert verdict(problem, allocated(case, dtype, toward=toward), dtype), "the effort ignored"
    if toward == "last":
        assert verdict(problem, allocated(case, dtype, effort=A(CONTROL_EFFORT), toward="zero"), dtype), "the reference ignored"


# -- a call missing what the objective needs ---------------------------------------------------------------------

@pytest.mark.parametrize("quantity", ["force", "torque"])
def test_an_objective_of_a_force_or_torque_called_without_a_moment_says_so_where_the_flux_density_does_not_need_one(quantity):
    """The force and torque are on a moment the call states in ``at``. Without it the allocator says which input to
    give (it was a ``KeyError`` naming the quantity); the control, with the moment, allocates; the flux density needs
    none."""
    case = fixed_case((quantity,))
    with in_dtype("float64"):
        solve = make_allocator(case["background"], (quantity,))
        wanted = as_quantities(case["wanted"], (quantity,))
        point = {"point": jnp.asarray(case["point"])}
        with pytest.raises(ValueError, match=rf"asked for {quantity}.*only where moment.*'moment'"):
            solve(wanted, point)
        solve(wanted, {**point, "moment": jnp.asarray(case["moment"])})
        flux = fixed_case(("flux_density",))
        make_allocator(flux["background"], ("flux_density",))(as_quantities(flux["wanted"], ("flux_density",)), point)


def test_the_flake_ci_found_is_the_laws_documented_tie_not_a_rounding_error():
    """CI's probe found, in float32, a force-and-torque objective (a moment along y, the torque asked for beyond the box)
    whose commands were 2.2 A off BVLS's at a 25 A bound: the residual 665.518 against BVLS's 665.489, inside the law's tie
    window of 0.015 each side. Held here as that case's numbers: the verdict, which holds a unique minimiser to BVLS's commands
    only past the law's tie windows, accepts it, and a command off by the same distance with a residual far above the
    windows is refused."""
    point = np.asarray([-0.011, 0.022, 0.013775])
    moment = np.asarray([0.0, 4.5557e-4, 0.0])
    matrix, offset, scales = host_rows(point, moment, BACKGROUNDS[0], ("force", "torque"))
    wanted = np.asarray([0.0, 0.0, -6.61324859e-4, 0.0, 0.0, -8.81766478e-6])
    case = {"name": "force_and_torque", "objective": ("force", "torque"), "point": point, "moment": moment,
            "background": BACKGROUNDS[0], "matrix": matrix, "offset": offset, "scales": scales, "wanted": wanted,
            "last": np.asarray([-3.0, 0.0, 0.0])}
    problem = oracle_problem(case, "float32", effort=EFFORT, reference=case["last"])
    reference = bvls(problem, "float32")
    nearby = np.asarray([0.64944434, -17.13034248, 25.0])
    a, b = problem.scaled("float32")
    window = two_sided_window(problem, reference, nearby, "float32")
    gap = float(np.linalg.norm(a @ nearby - b) - np.linalg.norm(a @ reference - b))
    assert 0.0 <= gap <= window < 1.0 and np.max(np.abs(nearby - reference)) > 1.0
    assert verdict(problem, nearby, "float32") == []
    far = reference + np.asarray([0.0, 0.0, 2.2]) * 0.0 + np.asarray([6.0, 6.0, 0.0])
    assert verdict(problem, far, "float32")
