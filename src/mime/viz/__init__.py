"""Recording as a time-sampled layer over the scene, the meshes the asset cache holds for a stage's visuals with the
overlay that draws them, and a view of a stage or a recording that may be shared: a glTF file and a web page; with
scene, the only importer of pxr."""

from mime.viz._export import GltfResult, gltf
from mime.viz._meshes import Decimated, Mesh, TriangleBudget, decimate, mesh_distance, read_stl, write_mesh
from mime.viz._recording import (
    RECORDING_FILE,
    RECORDING_FORMAT,
    RUN_STAGE,
    RecordingSettings,
    StageSamples,
    crate_version,
    recording_layer,
    write_recording,
)
from mime.viz._tiers import SHAREABLE_TIERS, refuse_unshareable, unshareable
from mime.viz._visuals import (
    CONVERTER,
    CONVERTER_VERSION,
    INDEX_FORMAT,
    LOCK_VISUALS,
    Baked,
    Overlay,
    bake,
    baked_key,
    baked_layers,
    fetch,
    lock_manifests,
    manifests_for,
    open_stage,
    overlay,
    view,
    visuals_dir,
)
from mime.viz._web import WebPage, display, page, web

__all__ = ["CONVERTER", "CONVERTER_VERSION", "INDEX_FORMAT", "LOCK_VISUALS", "RECORDING_FILE", "RECORDING_FORMAT",
           "RUN_STAGE", "SHAREABLE_TIERS", "Baked", "Decimated", "GltfResult", "Mesh", "Overlay", "RecordingSettings",
           "StageSamples", "TriangleBudget", "WebPage", "bake", "baked_key", "baked_layers", "crate_version", "decimate",
           "display", "fetch", "gltf", "lock_manifests", "manifests_for", "mesh_distance", "open_stage", "overlay",
           "page", "read_stl", "recording_layer", "refuse_unshareable", "unshareable", "view", "visuals_dir",
           "web", "write_mesh", "write_recording"]
