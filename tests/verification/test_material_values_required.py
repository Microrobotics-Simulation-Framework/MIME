"""A material-valued constructor argument has no default in the ported nodes.

Spike 1 (MIME 0.3.0, "where it hurt" item 5): the nodes defaulted viscosity to
1e-3 Pa.s, the value the experiments typed, so a seam that fed no node built
the same graph, bit for bit, as one that fed every node.  An omitted material
value now refuses to build.  RigidBodyNode reads its three material arguments
only in some modes: it refuses an omitted one where its mode reads it, and an
omitted one is no parameter at all where its mode does not (spike 1, item 3:
never-read copies).
"""

import pytest

from mime.experiments.dejongh import default_mlp_weights_path
from mime.nodes.environment.gravity_node import GravityNode
from mime.nodes.environment.stokeslet.lubrication_node import LubricationCorrectionNode
from mime.nodes.environment.stokeslet.mlp_resistance_node import MLPResistanceNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode

DT = 5e-4


def test_mlp_drag_refuses_without_viscosity():
    with pytest.raises(TypeError, match="mu_Pa_s"):
        MLPResistanceNode("mlp", DT, mlp_weights_path=str(default_mlp_weights_path()),
                          nu=2.33, L_UMR_mm=7.47)


def test_lubrication_refuses_without_viscosity():
    with pytest.raises(TypeError, match="mu_Pa_s"):
        LubricationCorrectionNode("lub", DT, R_ves_mm=3.175)


def test_gravity_refuses_without_density_contrast():
    with pytest.raises(TypeError, match="delta_rho_kg_m3"):
        GravityNode("gravity", DT, volume_m3=6e-8)


def test_magnet_refuses_without_moment():
    with pytest.raises(TypeError, match="m_single"):
        PermanentMagnetResponseNode("magnet", DT, n_magnets=2)


def test_rigid_body_refuses_viscosity_where_its_mode_reads_it():
    with pytest.raises(TypeError, match="fluid_viscosity_pa_s"):
        RigidBodyNode("body", DT)                      # overdamped, analytical drag


def test_rigid_body_refuses_density_where_its_mode_reads_it():
    with pytest.raises(TypeError, match="density_kg_m3"):
        RigidBodyNode("body", DT, use_analytical_drag=False, use_inertial=True, I_eff=1e-10)


@pytest.mark.parametrize("mode", [
    dict(kinematic_mode=True),
    dict(use_analytical_drag=False, use_inertial=True, I_eff=1e-10, m_eff=8.5e-5),
])
def test_rigid_body_omitted_unread_values_are_not_parameters(mode):
    node = RigidBodyNode("body", DT, **mode)
    leaves = node.params_pytree()
    assert not {"density_kg_m3", "fluid_viscosity_pa_s", "fluid_density_kg_m3"} & set(leaves)
