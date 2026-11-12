"""Shared set-ups for the one elimination over a flow's exchange partners (F3's second PR): a flow node holding point
bodies (and, open, its openings), its step fed as a body and its other laws would feed it, and the monolithic
reference the return-path gate judges a step against: the whole step, the field's every face, the openings' pressures,
each body's force, couple and twist, as one dense linear system solved by one LU, independent of the unit fields, their
warm starts, the elimination's small system and the conjugate gradient. Each plant is switched by
``MIME_PLANT_<NAME>`` (``point_cases.plant``), each draw seeded by ``DETECTION_SEED``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from mime.core import run_dtype
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode
from mime.nodes.flow.fvm._partners import DELIVERIES, delivered, read, stencils
from mime.nodes.flow.fvm._open import OpenMACGrid
from mime.nodes.flow.fvm._ports import RowKind

WATER_MU, WATER_RHO = 1e-3, 1e3
#: The monolithic step's matrix is formed this many columns at a time (the step's linearisation applied to unit
#: vectors), so that its memory is the matrix's own and not the batched step's over every column at once.
COLUMNS = 256


@dataclass(frozen=True)
class Body:
    """A point body as the gate feeds its flow: its radius and point (m, in the world), its twist (m/s; rad/s), its
    mass and moment of inertia, its kernel's width (cells) and the loads of its other laws (N; N m)."""

    radius: float
    point: tuple[float, float, float]
    twist: tuple[float, ...]
    mass: float
    moment: float
    width: float = 1.0
    loads: tuple[float, ...] = (0.0,) * 6


def body_inputs(body: Body, b: int, dt: float, dtype: Any) -> dict[str, Any]:
    """What body ``b``'s dynamics and other laws feed the flow: its pose, its own step's row (its stepping mass ``diag(m
    I, J I)``, an isotropic body's, no gyroscopic term, beside its free load ``M V / dt``) and its other loads."""
    mass = np.diag([body.mass] * 3 + [body.moment] * 3)
    twist = np.asarray(body.twist)
    return {f"partner_{b}_pose": jnp.asarray([*body.point, 1.0, 0.0, 0.0, 0.0], dtype=dtype),
            f"partner_{b}_step_row": jnp.asarray(np.concatenate([mass, (mass @ twist / dt)[:, None]], axis=1),
                                                 dtype=dtype),
            f"partner_{b}_loads": jnp.asarray(body.loads, dtype=dtype),
            f"partner_{b}_resistances": jnp.zeros((6, 6), dtype=dtype)}


def _flat(velocity: Sequence[Any]) -> Any:
    return jnp.concatenate([jnp.reshape(c, (-1,)) for c in velocity])


def _unflat(z: Any, shapes: Sequence[tuple[int, ...]]) -> tuple[Any, Any, Any]:
    out = []
    start = 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(jnp.reshape(z[start:start + size], shape))
        start += size
    return (out[0], out[1], out[2])


@dataclass(frozen=True)
class Monolithic:
    """The monolithic step: the field at the solution, the openings' pressures (absolute) and fluxes, and each body's
    force and couple, twist and reading, a row a body."""

    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]
    pressures: np.ndarray
    fluxes: np.ndarray
    forces: np.ndarray
    twists: np.ndarray
    readings: np.ndarray
    system: System


@dataclass(frozen=True)
class System:
    """The monolithic step as one linear system ``matrix @ z = right`` (float64) and its solution, refined once in
    float64 (Wilkinson's refinement: its residual's componentwise backward error then ``gamma_(n+1)``): the unknowns the
    field's faces (``faces`` of them, each weighed ``weights`` in the grid's inner product), then each opening's
    pressure relative to the reference and the level, then each body's force, couple and twist. ``pressures`` and
    ``fluxes`` are the openings' absolute pressures and fluxes as rows over ``z``."""

    matrix: np.ndarray
    right: np.ndarray
    solution: np.ndarray
    faces: int
    weights: np.ndarray
    pressures: np.ndarray
    fluxes: np.ndarray


def monolithic_step(node: ImplicitFlowNode, p: Mapping[str, Any], state: Mapping[str, Any],
                    inputs: Mapping[str, Any], dt: float) -> Monolithic:
    """One step of ``node`` (IMEX Euler or its steady limit) from ``state`` as one dense system in float64, the
    unknowns the field's every face, each opening's pressure relative to the reference with the level, and each body's
    force, couple and twist: the field's projected momentum ``P A u + (I - P) u = P (rhs + openings' forces + bodies'
    forces)``, each opening's row, each body's law ``(M0 - M_inf) f + G_F u' = V'`` and own row ``S V' + f = r``.
    Its matrix by the step's own operators applied to unit vectors (its linearisation, :data:`COLUMNS` columns at a
    time), solved by one LU."""
    with jax.enable_x64(True):
        p = {k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, (np.ndarray, jax.Array, float)) else v)
             for k, v in p.items()}
        state = {k: jnp.asarray(v, dtype=jnp.float64) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v
                 for k, v in state.items()}
        inputs = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
        grid = node.grid(p)
        fields = node.fields(p, grid)
        sigma = node.resistivity(p, fields)
        scheme = node.scheme
        shift = 1.0 / dt if scheme.stores_inertia else 0.0
        rho = jnp.asarray(p["density"])
        before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        drive = node.forcing(fields, inputs, scheme, jnp.float64)(0, 0.0)
        convection = grid.convection(before) if node.constants()["convection"] else grid.zeros(jnp.float64)
        rhs = tuple(shift * before[k] + drive[k] + convection[k] for k in range(3))
        operator = node.operator_of(p, grid, sigma)(shift)
        bodies = node.body_partners(p, inputs, dt, jnp.float64) if node.body_count else []
        held = [stencils(grid, body.point, body.width) for body in bodies]
        shapes = [grid.shape(k) for k in range(3)]
        faces = sum(int(np.prod(s)) for s in shapes)
        opened = isinstance(node, OpenFlowNode)
        count = node.opening_count() if opened else 0
        if opened:
            assert isinstance(node, OpenFlowNode)
            rows = node.rows()
            reference = node.reference()
            others = node.others()
            a, blocks = node.law_rows(jnp.asarray(p["opening_parameters"]), jnp.asarray(state["opening_state"]),
                                      (jnp.asarray(state["steps"]) + 1) * dt, dt)
            coupling = node.coupling()
        n = faces + count + 2 * DELIVERIES * len(bodies)

        def residual(z: Any) -> Any:
            u = _unflat(z[:faces], shapes)
            start = faces
            forcing = rhs
            out = []
            if opened:
                relative = jnp.zeros(count, dtype=jnp.float64)
                for k, j in enumerate(others):
                    relative = relative.at[j].set(z[start + k])
                level = z[start + count - 1]
                start += count
                port = grid.port_force(relative, rho)
                forcing = tuple(forcing[k] + port[k] for k in range(3))
            twists, forces = [], []
            for b, body in enumerate(bodies):
                force = z[start + 12 * b:start + 12 * b + 6]
                twist = z[start + 12 * b + 6:start + 12 * b + 12]
                forces.append(force)
                twists.append(twist)
                density = delivered(grid, held[b], force, body.delivery)
                forcing = tuple(forcing[k] + density[k] / rho for k in range(3))
            applied = operator(u)
            projected = grid.project(tuple(applied[k] - forcing[k] for k in range(3)))
            complement = tuple(u[k] - grid.project(u)[k] for k in range(3))
            out.append(_flat(tuple(projected[k] + complement[k] for k in range(3))))
            if opened:
                q = grid.opening_fluxes(u)
                for i in range(count):
                    if rows[i] is RowKind.FLUX:
                        out.append(jnp.reshape(q[i] - a[i], (1,)))
                    else:
                        pushed = sum((blocks[i, j] * q[j] for j in range(count) if coupling[i][j]),
                                     start=jnp.zeros((), dtype=jnp.float64))
                        out.append(jnp.reshape(level + relative[i] - pushed - a[i], (1,)))
            for b, body in enumerate(bodies):
                out.append(body.law * forces[b] + read(grid, held[b], u, body.reading) - twists[b])
                out.append(body.system @ twists[b] + forces[b] - body.right)
            return jnp.concatenate(out)

        zero = jnp.zeros(n, dtype=jnp.float64)
        offset, linear = jax.linearize(residual, zero)
        columns = jax.jit(jax.vmap(linear))
        chunk = min(n, COLUMNS)
        transposed = []
        for start in range(0, n, chunk):
            basis = jnp.zeros((chunk, n), dtype=jnp.float64).at[jnp.arange(chunk), jnp.minimum(
                start + jnp.arange(chunk), n - 1)].set(1.0)
            transposed.append(np.asarray(columns(basis))[:min(chunk, n - start)])
        matrix = np.concatenate(transposed, axis=0).T
        right = -np.asarray(offset)
        factors = scipy.linalg.lu_factor(matrix)
        solution = scipy.linalg.lu_solve(factors, right)
        solution = solution + scipy.linalg.lu_solve(factors, right - matrix @ solution)
        u = tuple(np.asarray(c) for c in _unflat(jnp.asarray(solution[:faces]), shapes))
        start = faces
        pressures = fluxes = np.zeros(0)
        if opened:
            relative = np.zeros(count)
            for k, j in enumerate(others):
                relative[j] = solution[start + k]
            pressures = relative + solution[start + count - 1]
            fluxes = np.asarray(grid.opening_fluxes(tuple(jnp.asarray(c) for c in u)))
            start += count
        forces = np.stack([solution[start + 12 * b:start + 12 * b + 6] for b in range(len(bodies))]) if bodies \
            else np.zeros((0, 6))
        twists = np.stack([solution[start + 12 * b + 6:start + 12 * b + 12] for b in range(len(bodies))]) if bodies \
            else np.zeros((0, 6))
        readings = np.stack([np.asarray(read(grid, held[b], tuple(jnp.asarray(c) for c in u), body.reading))
                             for b, body in enumerate(bodies)]) if bodies else np.zeros((0, 6))
        weights = grid.weights(jnp.float64) if isinstance(grid, OpenMACGrid) else grid.zeros(jnp.float64)
        if not isinstance(grid, OpenMACGrid):
            weights = tuple(w + 1.0 for w in weights)
        pressure_rows = np.zeros((count, n))
        flux_rows = np.zeros((count, n))
        if opened:
            for k, j in enumerate(others):
                pressure_rows[j, faces + k] = 1.0
            pressure_rows[:, faces + count - 1] = 1.0
            flux_rows[:, :faces] = np.asarray(jax.jacrev(lambda f: grid.opening_fluxes(_unflat(f, shapes)))(
                jnp.zeros(faces, dtype=jnp.float64)))
        system = System(matrix, right, solution, faces, float(grid.cell_volume) * np.asarray(_flat(weights)),
                        pressure_rows, flux_rows)
        return Monolithic((u[0], u[1], u[2]), pressures, fluxes, forces, twists, readings, system)


#: The float rounding the derived line carries, each a count of the unit roundoff ``u`` (half the dtype's epsilon) per
#: operation, from the arithmetic that forms it (Higham's ``gamma_k = k u``, first order):
#: - the field's rows (the conjugate gradient's true residual ``P b - P A P x``, recomputed in the run's dtype): the
#:   operator's nine terms a face (the shift, the Laplacian's seven, the resistivity), the projector's two fast
#:   transforms (``log2`` of the cells each) and its difference stencils, the subtraction: ``FIELD_ROUNDING`` plus
#:   ``2 log2(cells)``, normwise in the grid's inner product on ``|A| |x| + |P b|`` (the solve's own floor,
#:   ``mime.nodes.flow.fvm._solve.rounding_floor``'s form);
#: - the partners' rows, by the elimination's two stages (``mime.core.elimination``): the deliveries ``x = x0 + X V``
#:   from every partner's rows with each body's twist free (a dense LU of ``n1`` deliveries, ``3 n1``), then the twists
#:   from each body's own row with ``x`` put in (``3 n2`` for ``n2`` twists, ``n1`` more for ``X`` put in), each row's
#:   rounding componentwise on the stage's own terms in magnitude (``|x0|``, ``|X| |V|``, not ``|x|``: a cancellation
#:   between them is the stage's, Oettli and Prager's scale), the readings of the unit fields a sum over the kernel's
#:   faces (``log2`` of their count) on the unit fields in magnitude, and ``ROW_ROUNDING`` more;
#: - the field's sum of its base and each unit field times its delivery, one rounding a term, componentwise.
FIELD_ROUNDING = 13
ROW_ROUNDING = 2


@dataclass(frozen=True)
class Lines:
    """The derived line of every quantity gate 1 judges, each in its own units: the field on each face (m/s), each
    body's force and couple (N; N m) and twist (m/s; rad/s), the openings' pressures (Pa) and fluxes (m^3/s); and the
    sizes of the terms forming them (``parts``: the field's residual from the solves' achieved residual and from their
    rounding, both in the grid's norm, and the reported residual), for the record."""

    field: np.ndarray
    forces: np.ndarray
    twists: np.ndarray
    pressures: np.ndarray
    fluxes: np.ndarray
    parts: dict[str, float]


def float_lines(node: ImplicitFlowNode, p: Mapping[str, Any], state: Mapping[str, Any], inputs: Mapping[str, Any],
                dt: float, reference: Monolithic, ended: Mapping[str, Any]) -> Lines:
    """Gate 1's line, derived per draw from the step's own solve and the dtype's rounding, carried through the
    monolithic step's inverse (the field's conditioning and the small system's together, first order).

    The step's answer ``z`` (its field, each opening's pressure, each body's force, couple and twist) solves the
    monolithic system ``K z = c`` up to a residual ``d``, and its error is ``K^-1 d``. Its field rows' residual is the
    solves' combined: the base's and each unit field's times its delivery ``x_s``, each solve's true residual at most
    its reported relative residual ``rho`` (the step's worst, ``solve_report``) times its right-hand side, in the grid's
    norm, plus the residual's own rounding (:data:`FIELD_ROUNDING`); a field-row error enters a quantity ``k`` through
    row ``k`` of ``K^-1`` on the field's columns in the dual norm (Cauchy and Schwarz). The partners' rows (each body's
    law and own row, each opening's) are solved to their rounding, componentwise (:data:`ROW_ROUNDING`), through
    ``|K^-1|``; the field's assembly adds its own rounding. The reference's own error (float64, refined once) enters as
    the partners' does at float64's ``u`` over every row (``gamma_(n+1)``). Nothing in the line reads the elimination's
    answer: its deliveries are the reference's, and the solves' unit fields enter only in magnitude, so a fault in the
    elimination is judged against the solves' accuracy and never absorbed by it (the base field's magnitude is the
    reference's field less the unit fields at the reference's deliveries)."""
    system = reference.system
    dtype = np.dtype(str(ended["velocity_x"].dtype))
    u = float(np.finfo(dtype).eps) / 2.0
    u64 = float(np.finfo(np.float64).eps) / 2.0
    with jax.enable_x64(True):
        p = {k: (jnp.asarray(v, dtype=jnp.float64) if isinstance(v, (np.ndarray, jax.Array, float)) else v)
             for k, v in p.items()}
        before64 = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in state.items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)}
        inputs = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()}
        grid = node.grid(p)
        fields = node.fields(p, grid)
        sigma = node.resistivity(p, fields)
        scheme = node.scheme
        shift = 1.0 / dt if scheme.stores_inertia else 0.0
        rho = jnp.asarray(p["density"])
        before = (before64["velocity_x"], before64["velocity_y"], before64["velocity_z"])
        drive = node.forcing(fields, inputs, scheme, jnp.float64)(0, 0.0)
        convection = grid.convection(before) if node.constants()["convection"] else grid.zeros(jnp.float64)
        rhs = tuple(shift * before[k] + drive[k] + convection[k] for k in range(3))
        magnitude = node.magnitude_of(p, grid, sigma)(shift)
        bodies = node.body_partners(p, inputs, dt, jnp.float64) if node.body_count else []
        held = [stencils(grid, body.point, body.width) for body in bodies]
        faces = system.faces
        solves: list[tuple[float, Any, Any]] = []
        base_rhs = rhs
        if isinstance(node, OpenFlowNode):
            count = node.opening_count()
            predicted = np.asarray(before64["opening_pressure_relative"])
            for k, j in enumerate(node.others()):
                moved = float(system.solution[faces + k]) - float(predicted[j])
                unit = tuple(jnp.asarray(np.asarray(ended[f"unit_{x}"][k], dtype=np.float64)) for x in "xyz")
                solves.append((moved, grid.port_force(jnp.zeros(count, dtype=jnp.float64).at[j].set(1.0), rho), unit))
            port = grid.port_force(jnp.asarray(predicted), rho)
            base_rhs = tuple(rhs[k] + port[k] for k in range(3))
        for b, body in enumerate(bodies):
            for k in range(DELIVERIES):
                force = delivered(grid, held[b], jnp.zeros(DELIVERIES, dtype=jnp.float64).at[k].set(1.0),
                                  body.delivery)
                unit = tuple(jnp.asarray(np.asarray(ended[f"partner_units_{x}"][DELIVERIES * b + k], dtype=np.float64))
                             for x in "xyz")
                solves.append((float(reference.forces[b, k]), tuple(f / rho for f in force), unit))
        base = tuple(jnp.asarray(v) for v in reference.velocity)
        for c, _, unit in solves:
            base = tuple(base[k] - c * unit[k] for k in range(3))
        solves.insert(0, (1.0, base_rhs, base))

        def norm(v: Any) -> float:
            return float(jnp.sqrt(grid.inner(v, v)))

        achieved = float(np.asarray(ended["solve_report"])[1])
        cells = int(np.prod(grid.cells))
        field_gamma = (FIELD_ROUNDING + 2 * math.ceil(math.log2(cells))) * u
        solved_part = 0.0
        floor = grid.zeros(jnp.float64)
        magnitudes = grid.zeros(jnp.float64)
        for c, b, unit in solves:
            projected = grid.project(b)
            solved_part += abs(c) * achieved * norm(projected)
            size = magnitude(tuple(jnp.abs(x) for x in unit))
            floor = tuple(floor[k] + abs(c) * (size[k] + jnp.abs(projected[k])) for k in range(3))
            magnitudes = tuple(magnitudes[k] + abs(c) * jnp.abs(unit[k]) for k in range(3))
        rounding_part = field_gamma * norm(floor)
        residual = solved_part + rounding_part
        in_magnitude = np.asarray(_flat(magnitudes))
        assembly = (len(solves) + 1 + 2 * math.ceil(math.log2(cells))) * u * in_magnitude
        matrix, z = system.matrix, system.solution
        inverse = np.linalg.inv(matrix)
        n = matrix.shape[0]
        shapes = [grid.shape(k) for k in range(3)]
        weights = jnp.asarray(system.weights)

        def dual_norm(row: Any) -> Any:
            """The size of a quantity's row on the field's columns against a divergence-free residual of unit norm:
            ``|| P W^-1 g ||_W``, the solves' residuals being projected (``K`` is the identity on the gradients)."""
            projected = _flat(grid.project(_unflat(row / weights, shapes)))
            return jnp.sqrt(jnp.sum(weights * projected * projected))

        def dual_norms(rows: np.ndarray) -> np.ndarray:
            normed = jax.jit(jax.vmap(dual_norm))
            return np.concatenate([np.asarray(normed(jnp.asarray(rows[i:i + COLUMNS])))
                                   for i in range(0, rows.shape[0], COLUMNS)]) if rows.shape[0] else np.zeros(0)

        dual = dual_norms(inverse[:, :faces])
        flux_dual = dual_norms(system.fluxes @ inverse[:, :faces])
    rest = np.arange(faces, n)
    count = len(reference.pressures)
    twists = np.concatenate([faces + count + 12 * b + 6 + np.arange(6) for b in range(len(bodies))]).astype(int) \
        if bodies else np.zeros(0, dtype=int)
    deliveries = np.setdiff1d(rest, twists)
    # The field taken out (its Schur complement, float64): the partners' rows over their deliveries and the twists.
    eliminated = scipy.linalg.lu_solve(scipy.linalg.lu_factor(matrix[:faces, :faces]),
                                       np.concatenate([matrix[:faces, faces:], system.right[:faces, None]], axis=1))
    schur = matrix[faces:, faces:] - matrix[faces:, :faces] @ eliminated[:, :-1]
    reduced = system.right[faces:] - matrix[faces:, :faces] @ eliminated[:, -1]
    d, w = deliveries - faces, twists - faces
    first = schur[np.ix_(d, d)]
    solved = np.linalg.solve(first, np.concatenate([reduced[d, None], -schur[np.ix_(d, w)]], axis=1))
    at_rest, response = solved[:, 0], solved[:, 1:]
    twist = z[twists]
    terms = np.zeros(n - faces)
    terms[d] = np.abs(first) @ np.abs(at_rest) + np.abs(reduced[d]) + (
        (np.abs(first) @ np.abs(response) + np.abs(schur[np.ix_(d, w)])) @ np.abs(twist))
    if len(w):
        closing = schur[np.ix_(w, w)] + schur[np.ix_(w, d)] @ response
        terms[w] = np.abs(closing) @ np.abs(twist) + np.abs(reduced[w]) + np.abs(schur[np.ix_(w, d)]) @ (
            np.abs(at_rest) + np.abs(response) @ np.abs(twist))
    reach = np.count_nonzero(matrix[faces:, :faces], axis=1)
    reads = np.ceil(np.log2(np.maximum(reach, 2))) * u * (np.abs(matrix[faces:, :faces]) @ in_magnitude)
    gammas = np.full(n - faces, (3 * len(d) + ROW_ROUNDING) * u)
    gammas[w] = (3 * len(w) + len(d) + ROW_ROUNDING) * u
    rows_bound = gammas * terms + reads
    reference_terms = np.abs(matrix) @ np.abs(z) + np.abs(system.right)
    lines = dual * residual + np.abs(inverse[:, faces:]) @ rows_bound + \
        np.abs(inverse) @ ((n + 1) * u64 * reference_terms)
    lines[:faces] += assembly
    pressure_lines = np.abs(system.pressures) @ lines if count else np.zeros(0)
    flux_lines = (flux_dual * residual + np.abs(system.fluxes @ inverse[:, faces:]) @ rows_bound +
                  np.abs(system.fluxes[:, :faces]) @ assembly) if count else np.zeros(0)
    start = faces + count
    rows = np.reshape(lines[start:], (len(bodies), 2, DELIVERIES))
    return Lines(lines[:faces], rows[:, 0], rows[:, 1], pressure_lines, flux_lines,
                 {"achieved": achieved, "solved": solved_part, "rounding": rounding_part})


def sphere_mass(radius: float, ratio: float) -> tuple[float, float]:
    """A uniform sphere's mass and moment of inertia at ``ratio`` times water's density."""
    mass = ratio * WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    return mass, 0.4 * mass * radius ** 2


__all__ = ["FIELD_ROUNDING", "ROW_ROUNDING", "WATER_MU", "WATER_RHO", "Body", "Lines", "Monolithic", "System",
           "body_inputs", "float_lines", "monolithic_step", "sphere_mass"]
