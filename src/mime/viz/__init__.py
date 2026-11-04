"""Recording as a time-sampled layer over the scene, and the meshes the asset cache holds for a stage's visuals with
the overlay that draws them; with scene, the only importer of pxr."""

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

__all__ = ["CONVERTER", "CONVERTER_VERSION", "INDEX_FORMAT", "LOCK_VISUALS", "RECORDING_FILE", "RECORDING_FORMAT",
           "RUN_STAGE", "Baked", "Decimated", "Mesh", "Overlay", "RecordingSettings", "StageSamples", "TriangleBudget",
           "bake", "baked_key", "baked_layers", "crate_version", "decimate", "fetch", "lock_manifests",
           "manifests_for", "mesh_distance", "open_stage", "overlay", "read_stl",
           "recording_layer", "view", "visuals_dir", "write_mesh", "write_recording"]
