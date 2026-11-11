"""How a confining wall reaches the method of regularised Stokeslets: as the MRS's correction for a wall kind, a registered
function the MRS's node evaluates, keyed by the wall kind's name and version (``mime.registry.WallKind`` is the
declaration every solver shares: the geometry it matches, the parameters it reads, its validity and gradient status).

A correction takes the body's samples in the world frame (``N x 3``), its parameters by name (each a leaf of the
node's ``wall_<name>``), the viscosity and the drag's regularisation ``eps_squared`` (the blob's ``eps^2``, m^2, the one
the free part is made with, passed to every wall kind, whether it uses it or not: ``WallCorrection.regularised``), and
returns the ``3N x 3N`` block matrix added to the free-space matrix, the velocity at each sample per unit force at each
that the wall's image system adds, in m/(N s), the matrix's own units; and whether every read it made of its data was
accepted (a table read inside its readable range), which the node publishes. ``prepare`` turns the parameters the
wall kind reads from the geometry kind, in SI, into those node constants at build: a frame across the tube's axis, a
table loaded and compiled for the run's dtype. The node knows Stokes kernels and quadrature, and nothing about tubes or
planes.

Free space is the trivial wall, with no correction. The straight circular tube's (``_tube``) is its image system read
from a table, singular (no blob in it); the plane's (``_plane``) is the regularised image of the drag's own blob, in
closed form."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, cast

from jax import Array

from mime.build._refusals import ArgumentRefused, Refusal, entry_defect
from mime.core import jnp
from mime.nodes.hydro.mrs._quadrature import Chunks, cell_integrated

#: The free-space wall kind's name: a medium with no wall, unbounded.
FREE_SPACE = "free_space"


class Corrected(NamedTuple):
    """A wall's correction at a body's samples: the ``3N x 3N`` ``matrix``, and whether every read it made was
    ``accepted``."""

    matrix: Array
    accepted: Array


class CorrectionAt(Protocol):
    """A wall's correction at a body's samples (``WallCorrection.correction``): ``points`` (world frame, m), the
    correction's ``parameters``, the ``viscosity`` and the drag's ``eps_squared`` (m^2)."""

    def __call__(self, points: Array, parameters: Mapping[str, Any], viscosity: Any, eps_squared: Any) -> Corrected: ...


class ImageBetween(Protocol):
    """A wall's image between two point sets (``WallCorrection.image``): at each of ``targets`` per unit force at each
    of ``sources`` (world frame, m), with the correction's ``parameters``, the ``viscosity`` and the drag's
    ``eps_squared`` (m^2: one value, or one per source)."""

    def __call__(self, targets: Array, sources: Array, parameters: Mapping[str, Any], viscosity: Any,
                 eps_squared: Any) -> Corrected: ...


def _nothing_to_prepare(parameters: Mapping[str, Any], dtype: Any) -> dict[str, object]:
    return {}


def _any_reach(parameters: Mapping[str, Any], reach: float) -> list[tuple[str, str]]:
    return []


@dataclass(frozen=True)
class WallCorrection:
    """The MRS's correction for the wall kind ``name`` ``version``: the ``parameters`` it reads (the node's
    ``wall_<name>`` constants, in SI), each one's ``units``, those ``held`` as built (not a study's leaf: a frame, a
    table), the function, and ``prepare``, which makes the constants from what the wall kind reads of the geometry,
    for a run in a dtype; and ``beyond`` says, when the experiment is planned, why a body that reaches ``reach`` (m)
    from its centre would be read outside the correction's data wherever it is (a table's axial reach), from what the
    wall kind reads, each as a message and an action. ``assets`` names the shipped asset references ``prepare`` reads
    (the tube's table): the build records each in a saved experiment's lock, so that ``mime assets fetch <folder>`` and
    a run on Modal fetch them, and a derivation through this wall names them as its input ``wall:<name>``.
    ``adds_nothing`` declares that the correction is zero at every pose (free space's), so that a node on a separated
    quadrature holds the body's whole resistance at build and solves nothing per step; a correction that does not
    declare it is evaluated and solved with at every step. ``distance`` gives each of a body's points' distance from
    the wall (world frame, m: above zero in the medium), from the same constants: the node on a separated quadrature
    reports the body's least distance from it each step, against its force spacing (MIME-ANO-130); a correction
    without one has its body's distance unreported. ``image`` is the wall's image as a kernel between any two point sets:
    at each of ``targets`` per unit force at each of ``sources`` (world frame, m), ``(T, S, 3, 3)`` blocks in the
    matrix's units, with whether every read was accepted; ``correction`` is its value with the targets the sources. A
    node integrates it over a body's cells (:func:`cell_image`), and the same kernel serves any targets and any source
    cells: one body's force points against another's cells. A correction without one is taken at the force points
    only. ``regularised`` declares whether the correction and the image use the drag's regularisation they are passed
    (``eps_squared``, the blob's): the plane's image is regularised by it, so that it is the exact no-slip partner of
    the free part's regularised Stokeslet; the tube's table image is singular and free space adds nothing, so both
    take it and ignore it. The build report says which each wall uses. ``cost_text`` and ``near_wall_text`` are a wall
    kind's own figures for the build report where it has them: what a step costs (a wall with no table to read has no
    reads to count, and the tube's step and memory measurements are not its), and what was measured of the resistance
    near it; left empty, the report quotes the tube's, which are the table walls'."""

    name: str
    version: str
    parameters: tuple[str, ...]
    correction: CorrectionAt
    prepare: Callable[[Mapping[str, Any], Any], dict[str, object]] = _nothing_to_prepare
    units: Mapping[str, str] = field(default_factory=dict[str, str])
    held: frozenset[str] = frozenset()
    beyond: Callable[[Mapping[str, Any], float], list[tuple[str, str]]] = _any_reach
    assets: tuple[str, ...] = ()
    adds_nothing: bool = False
    distance: Callable[[Array, Mapping[str, Any]], Array] | None = None
    image: ImageBetween | None = None
    #: The floating values one pair of ``image`` holds while a chunk of a cell's quadrature is read: a chunk is cut to
    #: hold :data:`mime.nodes.hydro.mrs._quadrature.CHUNK_BYTES` of them against every target.
    pair_values: int = 16
    regularised: bool = False
    cost_text: str = ""
    near_wall_text: str = ""
    #: How the near-contact hand-over opens a near contact with this wall (``mime.nodes.hydro.mrs._handover``): the
    #: constants with the wall moved ``delta`` (m) away from the medium along its normal everywhere (a plane
    #: translated, a tube dilated), a JAX function of both; ``None`` for a wall that cannot be opened, with which the
    #: drag takes no hand-over.
    opened: Callable[[Mapping[str, Any], Any], dict[str, Any]] | None = None
    #: The region the medium fills, from the constants: the registered shape's name (``mime.core.shape_named``) and
    #: its parameters, which a near contact's search and the wall's curvature read; ``None`` with ``opened``.
    region: Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]] | None = None


def cell_image(wall: WallCorrection, targets: Array, points: Array, weights: Array, chunks: Chunks,
               parameters: Mapping[str, Any], viscosity: Any, eps_squared: Any) -> Corrected:
    """``wall``'s image at ``targets`` (world frame, m) per unit traction on each source cell, ``3T x 3C``: ``sum over q
    in cell c of image(x_t, y_q) w_q``, the cells' quadrature ``points`` (world frame) and ``weights`` taken in
    ``chunks`` (``mime.nodes.hydro.mrs._quadrature.cell_integrated``), with the drag's regularisation ``eps_squared``
    (m^2, the source body's); and whether every read was accepted. The kernel between arbitrary targets and source
    cells: a body's force points against its own cells, or against another body's."""
    image = wall.image
    if image is None:
        raise ValueError(f"the wall correction {wall.name} {wall.version} gives no image between point sets")

    def kernel(at: Array, sources: Array) -> tuple[Array, Array]:
        found = image(at, sources, parameters, viscosity, eps_squared=eps_squared)
        return found.matrix, found.accepted

    found, accepted = cell_integrated(kernel, targets, points, weights, chunks)
    return Corrected(found, accepted)


_corrections: dict[tuple[str, str], WallCorrection] = {}


#: The calls the node makes of a wall's functions, by name: the positional arguments, then ``eps_squared=`` (the drag
#: passes its regularisation to every wall kind).
_CALLS = {"correction": ("points", "parameters", "viscosity"), "image": ("targets", "sources", "parameters", "viscosity")}


def register_wall_correction(wall: WallCorrection) -> WallCorrection:
    """Register ``wall`` for the wall kind it names. Refused (``entry.defect``, naming the function's package) where its
    ``correction`` or ``image`` does not take the call the node makes, ``correction(points, parameters, viscosity,
    eps_squared=...)`` and ``image(targets, sources, parameters, viscosity, eps_squared=...)``: found here, not at the
    first trace as a bare ``TypeError``."""
    refusals: list[Refusal] = []
    for label, arguments in _CALLS.items():
        function = cast("Callable[..., Any] | None", getattr(wall, label))
        if function is None:
            continue
        try:
            signature = inspect.signature(function)
        except ValueError:  # no signature to read (a builtin): the first trace judges it
            continue
        try:
            signature.bind(*arguments, eps_squared=None)
        except TypeError as error:
            refusals.append(entry_defect(
                cast(type, function), f"its {label}, {function.__qualname__}{signature}, does not take the call the node makes, {label}("
                f"{', '.join(arguments)}, eps_squared=...): {error}. The drag passes its regularisation to every wall "
                f"kind, so the function takes it (a default of None serves one that ignores it)",
                label=f"wall correction {wall.name} {wall.version}"))
    if refusals:
        raise ArgumentRefused(refusals)
    held = _corrections.get((wall.name, wall.version))
    if held is not None and held is not wall:
        raise ValueError(f"a wall correction {wall.name} {wall.version} is already registered")
    _corrections[(wall.name, wall.version)] = wall
    return wall


def wall_correction_named(name: str, version: str) -> WallCorrection:
    if (name, version) not in _corrections:
        raise KeyError(f"no wall correction {name} {version} is registered for the method of regularised Stokeslets; those "
                       f"registered are {wall_corrections()}")
    return _corrections[(name, version)]


def wall_corrections() -> list[str]:
    """The registered corrections, as ``name version``."""
    return sorted(f"{name} {version}" for name, version in _corrections)


def _unbounded(points: Array, parameters: Mapping[str, Any], viscosity: Any, eps_squared: Any = None) -> Corrected:
    """No wall: nothing to add, and no read to refuse."""
    count = 3 * points.shape[0]
    return Corrected(jnp.zeros((count, count), dtype=points.dtype), jnp.asarray(True))


def _no_image(targets: Array, sources: Array, parameters: Mapping[str, Any], viscosity: Any,
              eps_squared: Any = None) -> Corrected:
    """No wall: no image between any two point sets."""
    return Corrected(jnp.zeros((targets.shape[0], sources.shape[0], 3, 3), dtype=targets.dtype), jnp.asarray(True))


#: Free space's correction, which adds nothing.
UNBOUNDED = register_wall_correction(WallCorrection(FREE_SPACE, "0.1.0", (), _unbounded, adds_nothing=True,
                                                    image=_no_image))


__all__ = ["FREE_SPACE", "UNBOUNDED", "Corrected", "CorrectionAt", "ImageBetween", "WallCorrection", "cell_image",
           "register_wall_correction", "wall_correction_named", "wall_corrections"]
