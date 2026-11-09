"""A wall correction is checked when it is registered. The drag passes its regularisation to every wall kind, so
``correction(points, parameters, viscosity, eps_squared=...)`` and ``image(targets, sources, parameters, viscosity,
eps_squared=...)`` are the calls the node makes; a lab's correction written before that (three arguments) was found
only at the first trace, as a bare ``TypeError``. Now it is refused where the lab wrote it, ``entry.defect``, naming the
function, its signature and the argument."""

import jax.numpy as jnp
import pytest

from mime.build import ArgumentRefused
from mime.nodes.hydro.mrs import _walls
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection, register_wall_correction, wall_corrections


def three_arguments(points, parameters, viscosity):
    return Corrected(jnp.zeros((3 * points.shape[0],) * 2), jnp.asarray(True))


def four_arguments(points, parameters, viscosity, eps_squared):
    return three_arguments(points, parameters, viscosity)


def defaulted(points, parameters, viscosity, eps_squared=None):
    return three_arguments(points, parameters, viscosity)


def image_without_it(targets, sources, parameters, viscosity):
    return Corrected(jnp.zeros((targets.shape[0], sources.shape[0], 3, 3)), jnp.asarray(True))


@pytest.fixture(autouse=True)
def private_registry(monkeypatch):
    monkeypatch.setattr(_walls, "_corrections", dict(_walls._corrections))


def test_a_correction_that_does_not_take_the_regularisation_is_refused_at_registration():
    with pytest.raises(ArgumentRefused) as refused:
        register_wall_correction(WallCorrection("labx_old", "0.1.0", (), three_arguments))
    assert refused.value.codes() == ["entry.defect"]
    text = str(refused.value)
    assert "wall correction labx_old 0.1.0" in text and "three_arguments" in text and "eps_squared" in text, text
    assert "labx_old 0.1.0" not in wall_corrections()


def test_an_image_that_does_not_take_it_is_refused_with_its_own_name():
    with pytest.raises(ArgumentRefused) as refused:
        register_wall_correction(WallCorrection("labx_old", "0.1.0", (), four_arguments, image=image_without_it))
    assert refused.value.codes() == ["entry.defect"]
    assert "its image, image_without_it(" in str(refused.value), str(refused.value)


def test_control_a_correction_that_takes_it_registers_with_or_without_a_default():
    """The check can pass: the same wall with the argument, positional or defaulted, and the repository's own kinds."""
    register_wall_correction(WallCorrection("labx_new", "0.1.0", (), four_arguments))
    register_wall_correction(WallCorrection("labx_default", "0.1.0", (), defaulted))
    assert {"labx_new 0.1.0", "labx_default 0.1.0", "plane 0.1.0", "tube 0.1.0"} <= set(wall_corrections())
