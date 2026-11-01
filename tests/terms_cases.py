"""Saved folders with third-party files, for the tests of terms and tiers: a toy (a body on a spring, the cheapest
experiment that saves) with mesh layers referenced into its scene, each a real USD layer of a chosen size (a synthetic
stand-in for a robot's visual meshes), each with its terms or none, the experiment's licence and its permissions as
given.

The licences are real SPDX ids, drawn by tier from the packaging standards' list as ``packaging`` ships it, and the
expected verdicts are this file's own table, written apart from ``mime.terms``, so that a test can hold the code to it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from packaging.licenses._spdx import LICENSES  # the list packaging ships, read only to draw real ids

from mime.experiment import Experiment, _experiment
from mime.scene import asset
from mime.terms import CustomLicence, Permission, Terms
from mime.testing._graph import declare_in_order
from mime.testing.toys._experiments import ADD_PROBE, ATTACH_SPRING, toy_experiment

#: A folder MIME wrote at version 3 of the saved form, before licences: the full toy with a mesh layer
#: (``scene/layers/arm_visual.usda``), written at the commit before version 4.
V3_FIXTURE = Path(__file__).parent / "saved" / "previous" / "layered_toy"

#: The toy's declarations: a body and a spring, since nothing here depends on the physics.
SPRUNG = (ADD_PROBE, ATTACH_SPRING)

#: The licences that let a file travel in a bundle as it is, by this file's reading of #198 (written apart from
#: ``mime.terms.OPEN``).
OPEN_IDS = ("CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-2.5", "CC-BY-2.0", "CC-BY-1.0", "MIT", "Apache-2.0",
            "BSD-2-Clause", "BSD-3-Clause")
_NC, _SA, _ND = "non-commercial", "share-alike", "no-derivatives"
#: Those that let it travel on conditions, each with them: the Creative Commons licences with a non-commercial,
#: share-alike or no-derivatives element, at every version the list holds, written out one by one.
CONDITIONAL_IDS = {
    "CC-BY-NC-1.0": (_NC,), "CC-BY-NC-2.0": (_NC,), "CC-BY-NC-2.5": (_NC,), "CC-BY-NC-3.0": (_NC,),
    "CC-BY-NC-4.0": (_NC,),
    "CC-BY-SA-1.0": (_SA,), "CC-BY-SA-2.0": (_SA,), "CC-BY-SA-2.5": (_SA,), "CC-BY-SA-3.0": (_SA,),
    "CC-BY-SA-4.0": (_SA,),
    "CC-BY-ND-1.0": (_ND,), "CC-BY-ND-2.0": (_ND,), "CC-BY-ND-2.5": (_ND,), "CC-BY-ND-3.0": (_ND,),
    "CC-BY-ND-4.0": (_ND,),
    "CC-BY-NC-SA-1.0": (_NC, _SA), "CC-BY-NC-SA-2.0": (_NC, _SA), "CC-BY-NC-SA-2.5": (_NC, _SA),
    "CC-BY-NC-SA-3.0": (_NC, _SA), "CC-BY-NC-SA-4.0": (_NC, _SA),
    "CC-BY-NC-ND-1.0": (_NC, _ND), "CC-BY-NC-ND-2.0": (_NC, _ND), "CC-BY-NC-ND-2.5": (_NC, _ND),
    "CC-BY-NC-ND-3.0": (_NC, _ND), "CC-BY-NC-ND-4.0": (_NC, _ND)}
#: Every id of the SPDX list the installed packaging ships, in order: the enumeration's range.
EVERY_ID = tuple(sorted(held["id"] for held in LICENSES.values()))
#: Every other real id of the list: MIME judges none of them, so each is unknown, and travels as a reference.
UNKNOWN_IDS = tuple(each for each in EVERY_ID if each not in OPEN_IDS and each not in CONDITIONAL_IDS)
#: The Creative Commons licences ported to a jurisdiction (CC-BY-3.0-DE, CC-BY-NC-SA-2.0-UK, ...): on neither list
#: (design D2: kept referenced until a person decides them), so each is unknown.
JURISDICTION_PORTS = tuple(each for each in UNKNOWN_IDS if each.startswith("CC-BY-"))
#: A licence of a rights holder's own that forbids redistribution, as an AR4-like mesh's is.
NO_REDISTRIBUTION = CustomLicence("LicenseRef-Robot-Maker-NonCommercial", "Robot Maker Non-Commercial Licence 1.1",
                                  "https://example.org/robot-maker/LICENSE.txt", "not_permitted")


def mesh_bytes(name: str, size: int) -> bytes:
    """A USD layer whose default prim is a mesh of one triangle, padded with a comment-free string attribute to about
    ``size`` bytes (at least its few hundred bytes of structure)."""
    head = (f'#usda 1.0\n(\n    defaultPrim = "{name}"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
            f'def Xform "{name}"\n{{\n    def Mesh "Link"\n    {{\n        int[] faceVertexCounts = [3]\n'
            f'        int[] faceVertexIndices = [0, 1, 2]\n'
            f'        point3f[] points = [(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)]\n'
            f'        custom string pad = "')
    tail = '"\n    }\n}\n'
    padding = max(0, size - len(head) - len(tail))
    return (head + "p" * padding + tail).encode("utf-8")


def write_mesh(folder: Path, name: str, size: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.usda"
    path.write_bytes(mesh_bytes(name, size))
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layered(name: str, layers: Sequence[tuple[Path, Terms | None]], *, licence: str | CustomLicence | None = None,
            permissions: Sequence[Permission] = ()) -> Experiment:
    """The toy, named ``name``, with each layer referenced at ``/World/<its file's stem>`` and declared with its
    terms (none: the author's own)."""
    experiment = declare_in_order(lambda: toy_experiment(name), SPRUNG, None)
    scene = experiment._settings.scene
    for path, terms in layers:
        scene.reference(asset(path, terms=terms), at=f"/World/{path.stem}")
    if licence is not None:
        experiment.licence(licence)
    for permission in permissions:
        experiment.permission(permission)
    return experiment


_reference_runs: dict[int, object] = {}


@contextmanager
def same_reference_run() -> Iterator[None]:
    """Every save's reference outputs, the toy's first steps, run once and reused: they are a run of the toy's graph,
    which no layer, licence or terms changes, and the tests of terms read none of them. A save is otherwise real."""
    real = _experiment.BuiltExperiment._fresh_history

    def cached(self: _experiment.BuiltExperiment, steps: int) -> object:
        if steps not in _reference_runs:
            _reference_runs[steps] = real(self, steps)
        return _reference_runs[steps]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_experiment.BuiltExperiment, "_fresh_history", cached)
        yield
