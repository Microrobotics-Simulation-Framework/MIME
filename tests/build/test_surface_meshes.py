"""A surface an entry declares (``mime.registry.SurfaceMesh``), from the declaration to the stage binding: the triangle
budget and the decimation at rest, each field's reduction over a cluster, the binding's ``mime.binding/2`` record, the
near-origin check at build, the ghosted comparison's refusal of members whose surfaces differ, and the live frames,
which leave surfaces out unless named.

**What this bounds.** That a surface over ``SURFACE_TRIANGLES`` is drawn within it; that its clusters do not fold
against the declared surface under a crater (a dimple deeper than its width, its material pulled towards its centre),
judged against the declared surface's own normals at every sample; that a field declared ``max`` keeps a damage front
exactly where the mean smears it; that the binding's record reads back as the entries it came from. **What it does
not.** That vertex clustering is a good decimation of every surface: a feature thinner than a cube collapses, and a
section folded on itself (a thin wall's two sides within a cube) is merged; the crater is the case the brief named.

**Controls.** Each fails on every run (the inputs are deterministic): a cluster map mirrored across the sheet folds
every drawn triangle; the mean in place of the declared maximum loses the front; a surface past the origin line is
refused, and one just inside it plans; members of equal vertex counts are compared, and of unequal counts refused; a
surface named in ``select=`` is carried."""

import dataclasses
import json
import warnings

import numpy as np
import pytest

import mime.testing.toys as toys
from mime.build import ArgumentRefused, BuildRefused
from mime.build._binding import FIELD, POINTS, Bound, StageBinding
from mime.build._surfaces import SURFACE_TRIANGLES, clustered, reduce_clusters
from mime.experiment import Experiment
from mime.live._session import frame_schema
from mime.registry import SURFACE_ORIGIN_EXTENTS, MeshField, Port, SurfaceMesh
from mime.results._stage import StageRecord
from mime.scene import Scene, Z
from mime.things import Body, State
from mime.units import m, m_per_s2, s
from mime.viz._compare import compare_layer

#: The budget the docs state (docs/recordings.md), written here as a number so that a change to it is a change here.
DECLARED_SURFACE_TRIANGLES = 16_000
#: A sheet over the budget: 129 x 129 vertices, 32,768 triangles.
FINE = 129
SIDE = 1e-3


def sheet_experiment(**declared) -> Experiment:
    experiment = Experiment("sheet", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(0.01))
    experiment.add_body(Body("sheet", dynamics=toys.ToyIndentedSheet(**declared),
                             initial=State(position=m((0.0, 0.0, 0.0)))))
    return experiment


def built_quietly(experiment: Experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.build()


# --- the budget and the decimation ------------------------------------------------------------------------------------


def test_the_budget_is_the_declared_one_and_a_surface_within_it_is_drawn_whole():
    assert SURFACE_TRIANGLES == DECLARED_SURFACE_TRIANGLES
    rest, faces = toys.sheet(SIDE, 33)
    drawn = clustered(rest, faces)
    assert drawn.clusters is None and np.array_equal(drawn.rest, rest) and np.array_equal(drawn.faces, faces)


def test_a_surface_over_the_budget_is_drawn_within_it():
    rest, faces = toys.sheet(SIDE, FINE)
    assert len(faces) > SURFACE_TRIANGLES
    drawn = clustered(rest, faces)
    assert drawn.clusters is not None and len(drawn.clusters) == len(rest)
    assert 0.5 * SURFACE_TRIANGLES < len(drawn.faces) <= SURFACE_TRIANGLES, "within the budget, and not far under it"
    assert np.array_equal(drawn.rest, reduce_clusters(rest, drawn.clusters, len(drawn.rest), points=True))
    assert drawn.faces.min() >= 0 and drawn.faces.max() < len(drawn.rest)


def crater(rest: np.ndarray, depth: float, pull: float, width: float) -> np.ndarray:
    """The sheet under a crater at its centre: sunk by ``depth`` at the centre and drawn towards it by ``pull`` of each
    point's distance, both falling off as a Gaussian of standard deviation ``width``. The radial map ``r (1 - pull g)``
    increases with ``r`` for ``pull < 1``, so the declared surface does not fold."""
    g = np.exp(-(rest[:, 0] ** 2 + rest[:, 1] ** 2) / (2.0 * width * width))
    moved = rest.copy()
    moved[:, :2] *= (1.0 - pull * g)[:, None]
    moved[:, 2] -= depth * g
    return moved


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Each vertex's normal, the sum of its triangles' area-weighted normals."""
    a, b, c = (points[faces[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    found = np.zeros_like(points)
    for k in range(3):
        np.add.at(found, faces[:, k], face)
    return found


def folded(rest: np.ndarray, faces: np.ndarray, clusters: np.ndarray, drawn_faces: np.ndarray,
           points: np.ndarray) -> int:
    """The drawn triangles whose normal points away from the declared surface's: against the sum of the declared
    surface's vertex normals over the members of its three clusters, at the same sample."""
    count = int(clusters.max()) + 1
    drawn = reduce_clusters(points, clusters, count, points=True)
    reference = np.zeros((count, 3))
    np.add.at(reference, clusters, vertex_normals(points, faces))
    a, b, c = (drawn[drawn_faces[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    around = reference[drawn_faces].sum(axis=1)
    return int(np.sum(np.einsum("ij,ij->i", normal, around) <= 0.0))


#: The crater's samples: to ten times deeper than its width, its material pulled a third and three fifths of the way
#: in at its centre. Measured beyond them (``tests/viz/predictions_usd_surfaces.json``'s outcomes): at a pull of 0.9, the
#: crater's mouth narrowed to a tenth of its width (under a cluster's cube), two of 15,842 drawn triangles at ten widths
#: deep lean against the wall's normals there, though none turns over in plan; a feature under a cube is not resolved.
WIDTH = 0.08 * SIDE
DEPTHS = np.linspace(0.0, 10.0 * WIDTH, 11)
PULLS = (0.35, 0.6)
PULL = PULLS[0]


@pytest.mark.parametrize("pull", PULLS)
def test_the_clusters_do_not_fold_under_a_crater(pull):
    rest, faces = toys.sheet(SIDE, FINE)
    drawn = clustered(rest, faces)
    assert drawn.clusters is not None
    cube = SIDE / np.sqrt(len(drawn.rest))
    assert WIDTH < 8 * cube, "the crater spans a few clusters, so a fold there would show"
    for depth in DEPTHS:
        points = crater(rest, depth, pull, WIDTH)
        assert folded(rest, faces, np.arange(len(rest)), faces, points) == 0, "the declared surface does not fold"
        assert folded(rest, faces, drawn.clusters, drawn.faces, points) == 0, depth


def test_control_a_cluster_map_mirrored_across_the_sheet_folds_every_drawn_triangle():
    rest, faces = toys.sheet(SIDE, FINE)
    drawn = clustered(rest, faces)
    assert drawn.clusters is not None
    mirrored = drawn.clusters.reshape(FINE, FINE).T.reshape(-1)
    points = crater(rest, DEPTHS[-1], PULL, WIDTH)
    assert folded(rest, faces, mirrored, drawn.faces, points) == len(drawn.faces)


def test_a_field_declared_max_keeps_a_damage_front_and_the_mean_smears_it():
    rest, faces = toys.sheet(SIDE, FINE)
    drawn = clustered(rest, faces)
    assert drawn.clusters is not None
    count = len(drawn.rest)
    radius = np.hypot(rest[:, 0], rest[:, 1])
    damage = (radius < 0.23 * SIDE).astype(np.float64)
    reached = np.zeros(count, dtype=bool)
    reached[drawn.clusters[damage > 0]] = True
    kept = reduce_clusters(damage, drawn.clusters, count, "max")
    assert np.array_equal(kept, reached.astype(np.float64)), "every cluster the front reaches reads it, whole"
    smeared = reduce_clusters(damage, drawn.clusters, count, "mean")
    assert np.any((smeared > 0.0) & (smeared < 1.0)), "the control: the mean reads a cluster across the front as less"
    assert np.any(smeared[reached] < kept[reached])


def test_the_reductions_of_samples_are_each_samples_own():
    rest, faces = toys.sheet(SIDE, FINE)
    drawn = clustered(rest, faces)
    assert drawn.clusters is not None
    count = len(drawn.rest)
    samples = np.stack([crater(rest, depth, PULL, WIDTH) for depth in DEPTHS])
    together = reduce_clusters(samples, drawn.clusters, count, points=True)
    assert together.shape == (len(DEPTHS), count, 3)
    for k, points in enumerate(samples):
        assert np.array_equal(together[k], reduce_clusters(points, drawn.clusters, count, points=True))


# --- the binding ----------------------------------------------------------------------------------------------------


def prims_of(text: str) -> list[str]:
    from pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    return [str(prim.GetPath()) for prim in stage.Traverse()]


@pytest.mark.parametrize("count", [9, FINE], ids=["whole", "decimated"])
def test_the_binding_names_the_surface_and_reads_back_from_its_record(count):
    built = built_quietly(sheet_experiment(count=count, peak=True))
    binding = built.binding()
    record = json.loads(json.dumps(binding.record()))
    assert record["format"] == "mime.binding/2"
    surface = [entry for entry in binding.entries if entry.kind in (POINTS, FIELD)]
    assert [entry.kind for entry in surface] == [POINTS, FIELD, FIELD]
    assert [entry.reduction for entry in surface] == [None, "max", "mean"]
    assert {entry.source_count for entry in surface} == {count * count}
    held = [entry for entry in record["entries"] if entry["kind"] in (POINTS, FIELD)]
    assert {entry["space"] for entry in held} == {"thing"} and {entry["vertices"] for entry in held} == {surface[0].count}
    assert all(("clusters" in entry) == (count == FINE) for entry in held)
    assert [Bound.of(entry) for entry in record["entries"]] == list(binding.entries)
    assert binding.problems(prims_of(built.stage_text), built.current_state()) == []


def test_a_surface_field_of_the_wrong_vertex_count_is_caught():
    built = built_quietly(sheet_experiment(count=9))
    binding = built.binding()
    (entry,) = [e for e in binding.entries if e.kind == FIELD and e.surface_field == "depth"]
    wrong = StageBinding(tuple(dataclasses.replace(e, count=80) if e is entry else e for e in binding.entries))
    found = wrong.problems(prims_of(built.stage_text), built.current_state())
    assert len(found) == 1 and "has shape (81,), not (80,)" in found[0]


# --- the near-origin check --------------------------------------------------------------------------------------------


def offset_by(distance: float):
    """The toy's surface moved by ``distance`` along x, as an entry that declared it in the world would."""
    own = toys.ToyIndentedSheet.surface_meshes

    def moved(self, settings):
        (surface,) = own(self, settings)
        return (dataclasses.replace(surface, rest=surface.rest + np.array([distance, 0.0, 0.0])),)

    return moved


@pytest.mark.parametrize("factor", [1.001, 0.999], ids=["beyond", "within"])
def test_a_surface_far_from_its_things_origin_is_refused_at_build(monkeypatch, factor):
    monkeypatch.setattr(toys.ToyIndentedSheet, "surface_meshes", offset_by(factor * SURFACE_ORIGIN_EXTENTS * SIDE))
    experiment = sheet_experiment(count=5)
    if factor > 1.0:
        with pytest.raises(BuildRefused) as refused:
            built_quietly(experiment)
        assert "toy.indented_sheet" in str(refused.value) and "times its extent" in str(refused.value)
    else:
        built_quietly(experiment)  # the control: just inside the line, it plans and builds


def test_the_declarations_problems_are_each_named():
    rest, faces = toys.sheet(SIDE, 3)
    good = SurfaceMesh("s", rest, faces, Port("points"), (MeshField("d", "m", Port("d")),))
    assert good.problems() == []
    planted = {
        "reduction": dataclasses.replace(good, fields=(MeshField("d", "m", Port("d"), reduction="median"),)),
        "display": dataclasses.replace(good, display="rotated"),
        "range": dataclasses.replace(good, fields=(MeshField("d", "m", Port("d"), range=(1.0, 0.0)),)),
        "coincide": dataclasses.replace(good, rest=np.zeros_like(rest)),
        "faces": dataclasses.replace(good, faces=faces + 1),
    }
    said = {"reduction": "has reduction 'median'", "display": "display 'rotated'", "range": "low not below high",
            "coincide": "rest points all coincide", "faces": "names a vertex outside"}
    for name, surface in planted.items():
        assert any(said[name] in line for line in surface.problems()), name


# --- the ghosted comparison -------------------------------------------------------------------------------------------


class _Member:
    def __init__(self, stage: StageRecord) -> None:
        self.stage = stage


class _Study:
    """A study's members as ``compare_layer`` reads them, each from a toy sheet's built binding and state."""

    def __init__(self, counts: list[int]) -> None:
        self.records = [self._record(count) for count in counts]
        self.points = list(range(len(counts)))
        self.groups = list(range(len(counts)))
        self.folder = None
        self.axes = ()
        self.experiment = "sheet"

    @staticmethod
    def _record(count: int) -> StageRecord:
        built = built_quietly(sheet_experiment(count=count))
        state = built.current_state()
        fields = {key: np.stack([np.asarray(state[key[0]][key[1]])] * 3) for key in built.binding().fields()}
        meta = {"binding": built.binding().record()["entries"], "up": [0.0, 0.0, 1.0], "meters_per_unit": 1.0,
                "dt": 0.01, "experiment": "sheet", "input_hash": "-"}
        return StageRecord(np.array([0, 1, 2]), fields, meta)

    def __len__(self) -> int:
        return len(self.records)

    def member(self, index: int) -> _Member:
        return _Member(self.records[index])


def test_the_comparison_refuses_members_whose_surfaces_differ_in_vertex_count():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ArgumentRefused) as refused:
            compare_layer(_Study([5, 6]))  # pyright: ignore[reportArgumentType]
    assert refused.value.codes() == ["compare.scene"]
    assert "m1 has a surface of another vertex count than the reference's, m0" in str(refused.value)


def test_control_members_whose_surfaces_match_are_compared():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = compare_layer(_Study([5, 5]))  # pyright: ignore[reportArgumentType]
    assert layer.GetPrimAtPath("/Compare/m1/sheet/sheet") is not None


# --- live frames ------------------------------------------------------------------------------------------------------


def test_live_frames_leave_surfaces_out_unless_named():
    built = built_quietly(sheet_experiment(count=5))
    surface = {f"{entry.node}.{entry.field}" for entry in built.binding().entries if entry.kind in (POINTS, FIELD)}
    assert len(surface) == 3, "the premise: the binding moves the surface"
    by_default = {field.path for field in frame_schema(built).fields}
    assert not by_default & surface
    named = {field.path for field in frame_schema(built, sorted(surface)).fields}
    assert named == surface, "the control: named, they are carried"
