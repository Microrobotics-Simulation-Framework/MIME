# `mime.viz._gltf`: USD to glTF, derived from usd2gltf

**Origin.** usd2gltf 0.3.5 by Mike Lyndon (<https://github.com/mikelyndon/usd2gltf>), its only release, from the
PyPI sdist `usd2gltf-0.3.5.tar.gz`, whose sha256 (equal to PyPI's) is
`6bc4d9db4c4960a5ee4e13dfa376585b7ebc9d488f9fd8af79d6fe1d8cc05291`, under the MIT licence. `LICENSE` here is that
sdist's `LICENSE`, byte for byte, with its copyright line (its sha256 held by `tests/viz/test_gltf_licence.py`). The
git commit of the release was not matched: the sdist is the origin.

**What was taken**, each file with its header naming the origin and what MIME changed, and `SPDX-License-Identifier:
MIT`: `converter.py` as `_converter.py` (the traversal, the node hierarchy, the animation's bookkeeping),
`converters/usd_mesh.py` as `_mesh.py` (the fan triangulation with its handedness, the first-sample topology, normals
and `displayColor`), `converters/usd_xform.py` as `_xform.py` (a node's local transform and its time samples) and
`common.py` as `_common.py` (a static value, the colour clamp, the quaternion order). MIME's modifications to these
files are offered under the same MIT licence.

**What was not taken, and why**: `converters/usd_material.py` (MIME's stages hold `displayColor` and `displayOpacity`,
no material; a material is made from them), `converters/usd_skel.py` (no skeletons), `converters/usd_lux.py` and
`converters/usd_camera.py` (the viewer lights and frames the scene), `cli.py` (MIME's command is `mime visuals web`),
the tests (none useful), and the dependency on `gltflib`, replaced by `_writer.py`.

**What MIME changed**: a transform read from its xform ops (translate, orient and scale taken exactly, quaternions
made continuous in sign) in place of `XformCommonAPI`'s Euler angles; USD's analytic shapes tessellated
(`mime.viz._primitives`); a material from `displayColor` and `displayOpacity`; one buffer and one animation; a root
node turning the stage's up axis to +Y and scaling its units; typed for strict pyright, and every `pxr` module reached
through `mime.scene._pxr`; the `PointInstancer` gathering, which raised a `NameError` in 0.3.5, dropped with USDZ,
materials, skins, lights and cameras.

**MIME's own files here**, under MIME's own licence, as their SPDX lines say: `__init__.py` and `_writer.py`, the GLB container,
which writes glTF's binary form in place of gltflib.

**Updating.** There is no upstream merge: compare a newer usd2gltf file by file against the files named above, and
carry over what applies by hand, with the header updated.
