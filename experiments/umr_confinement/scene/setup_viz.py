"""Hook: scene_setup — materials, ground plane, lighting.

Creates UsdPreviewSurface materials for the glass vessel, robot body,
and desk surface, then binds them to the appropriate prims.
"""

from __future__ import annotations


def setup_scene(bridge, params, actor_config, env_config):
    """Set up scene dressing for the UMR confinement experiment.

    Parameters
    ----------
    bridge : StageBridge
    params : dict
    actor_config : dict
        From experiment.yaml scene.actors.
    env_config : dict
        From experiment.yaml scene.environment.
    """
    # Glass vessel
    glass = bridge.create_material(
        "Glass",
        diffuse_color=(0.85, 0.92, 1.0),
        opacity=0.15,
        roughness=0.15,
        ior=1.3,
        specular_color=(0.4, 0.4, 0.4),
    )

    # Robot body (light blue metallic)
    robot_mat = bridge.create_material(
        "Robot",
        diffuse_color=(0.45, 0.72, 0.82),
        roughness=0.3,
        metallic=0.4,
        specular_color=(0.6, 0.6, 0.6),
    )

    # Desk surface (light matte)
    desk_mat = bridge.create_material(
        "Desk",
        diffuse_color=(0.82, 0.79, 0.75),
        roughness=0.9,
    )

    # Bind materials to prims
    for env_spec in env_config.values():
        prim_path = env_spec.get("prim_path", "")
        if prim_path and glass:
            bridge.bind_material(prim_path, glass)

    for name, spec in actor_config.items():
        if "position" in spec.get("state_fields", []):
            prim_path = spec.get("prim_path", "")
            if prim_path and robot_mat:
                bridge.bind_material(prim_path, robot_mat)

    # Ground plane (parallel to vessel axis)
    bridge.add_ground_plane(offset=-0.006, size=0.03)
    if desk_mat:
        bridge.bind_material("/World/Environment/Ground", desk_mat)
