"""G27, a mesh changed after a build never gives a stale image: each way a built camera's mesh can be changed in process
either refuses by name or makes the image again from the change, and none leaves the image of the mesh it was built with.

The camera builds its bounding volume hierarchy once, from the meshes it was built with, and holds the image of the static
structure made from it. The review found three ways to change the mesh after that, each of which gave an image identical
to the unchanged control (difference 0.0), where a fresh build of the same change differs by 236:

- **a write to the array the node holds** (``constants()["mesh_triangles"].array[...] = x``): refused by NumPy, the
  array is a read-only copy over immutable bytes (``ValueError``), and its flags cannot be set writable;
- **a record replaced** (``node.params = {..., "mesh_triangles": Frozen(other)}``): refused, ``camera.cache-stale``, where
  the node's constants are read; an equal mesh in another object is taken for the same;
- **a leaf written** (``graph.params["nodes"]["camera:"]["mesh_v0"] = x`` and the other tree arrays): the held structure
  is made from the leaves it was made at, compared among the numbers it holds (:func:`mime.sensing._mesh.signature`), so a
  written leaf is not the structure's own, and the structure is made again from it: the image is the fresh build's of the
  same change, never the control's.

**The mesh's arrays are not in the node's parameters** where nothing reads them: ``mesh_triangles`` and ``mesh_owners``
stay a constructor argument (the saved form and the derived graph's hash), and are no parameter leaf."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import ballistic
from mesh_cases import VESSEL, rotation, tube, with_vessel

from mime.build import ArgumentRefused
from mime.sensing._mesh import KEYS, Frozen
from mime.units import ms

#: The turn that lays a tube drawn along z along x.
ALONG_X = rotation((2 ** -0.5, 0.0, 2 ** -0.5, 0.0))
POINTS, FACES = tube(60e-6, 70e-6, 480e-6, 32)
#: The change: every number of the mesh scaled, which moves its silhouette and its chords by a large share.
SCALE = 1.3


def scene(points: np.ndarray):
    return with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),)), points @ ALONG_X.T, FACES)


def image(built) -> np.ndarray:
    return np.asarray(built.current_state()["camera:"]["expected"], dtype=np.float64).copy()


def run_twice(built) -> np.ndarray:
    built.run(ms(3))
    built.run(ms(6))
    return image(built)


_CONTROLS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


@pytest.fixture
def control(dtype):
    """The image of the unchanged mesh, and of a fresh build of the changed one, once a dtype."""
    if dtype not in _CONTROLS:
        _CONTROLS[dtype] = run_twice(scene(POINTS).build()), run_twice(scene(POINTS * SCALE).build())
    return _CONTROLS[dtype]


def test_the_two_images_differ_so_that_a_stale_one_can_be_told(control):
    unchanged, changed = control
    assert np.abs(changed - unchanged).max() > 10.0


def test_a_write_to_the_array_the_node_holds_is_refused_by_numpy(dtype, control):
    built = scene(POINTS).build()
    built.run(ms(3))
    node = built.graph.get_node("camera:")
    held = node.constants()["mesh_triangles"]
    assert isinstance(held, Frozen)
    with pytest.raises(ValueError, match="read-only"):
        held.array[...] = held.array * SCALE
    with pytest.raises(ValueError):
        held.array.flags.writeable = True
    built.run(ms(6))
    assert np.array_equal(image(built), control[0])


def test_a_record_replaced_with_another_mesh_is_refused_by_name(dtype, control):
    built = scene(POINTS).build()
    built.run(ms(3))
    node = built.graph.get_node("camera:")
    node.params = {**node.params, "mesh_triangles": Frozen(np.asarray(node.constants()["mesh_triangles"]) * SCALE)}
    with pytest.raises(ArgumentRefused) as refused:
        built.run(ms(6))
    assert refused.value.codes() == ["camera.cache-stale"]
    assert "mesh_triangles" in refused.value.refusals[0].message and "build" in refused.value.refusals[0].action


def test_a_record_replaced_with_the_same_mesh_in_another_object_is_not_refused(dtype, control):
    built = scene(POINTS).build()
    built.run(ms(3))
    node = built.graph.get_node("camera:")
    node.params = {**node.params, "mesh_triangles": Frozen(np.asarray(node.constants()["mesh_triangles"]).copy())}
    built.run(ms(6))
    assert np.array_equal(image(built), control[0])


def test_a_leaf_written_makes_the_image_again_from_it_never_the_unchanged_ones(dtype, control):
    unchanged, changed = control
    built = scene(POINTS).build()
    built.run(ms(3))
    leaves = built.graph.params["nodes"]["camera:"]
    assert set(KEYS) <= set(leaves) and not {"mesh_triangles", "mesh_owners"} & set(leaves)
    for key in KEYS[:-1]:
        leaves[key] = leaves[key] * SCALE
    built.run(ms(6))
    made = image(built)
    assert np.abs(made - unchanged).max() > 10.0, "the image of the mesh as built: the structure was read stale"
    assert np.abs(made - changed).max() <= 1e-4 * np.abs(changed - unchanged).max()


def test_the_held_structure_is_not_held_at_a_tree_written_after_it_was_made(dtype):
    built = scene(POINTS).build()
    built.run(ms(3))
    node = built.graph.get_node("camera:")
    p = dict(built.graph.params["nodes"]["camera:"])
    intrinsics = node._intrinsics(p, np.asarray(p["fx"]).dtype)  # noqa: SLF001
    assert bool(node.holds_at(p, node._structure_numbers(intrinsics, p)))  # noqa: SLF001
    for key in KEYS:
        largest = np.unravel_index(int(np.argmax(np.abs(np.asarray(p[key])))), np.shape(p[key]))
        written = dict(p, **{key: jnp.asarray(p[key]).at[largest].set(jnp.nextafter(p[key][largest], jnp.inf))})
        assert not bool(node.holds_at(written, node._structure_numbers(intrinsics, written))), key  # noqa: SLF001
    swapped = dict(p, mesh_v0=p["mesh_v0"][::-1])
    assert not bool(node.holds_at(swapped, node._structure_numbers(intrinsics, swapped)))  # noqa: SLF001


def test_a_built_graph_holds_the_meshes_as_a_constructor_argument_and_not_as_leaves(dtype):
    node = scene(POINTS).build().graph.get_node("camera:")
    assert {"mesh_triangles", "mesh_owners"} <= set(node.params) and not {"mesh_triangles", "mesh_owners"} & set(
        node.params_pytree())
    assert all(not each.flags.writeable for each in node._tree.arrays.values())  # noqa: SLF001
