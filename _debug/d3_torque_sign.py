"""D3: is the frame-aware BEM drag TORQUE opposing omega? Compare z-config (identity)
vs x-config (body-z->world-x). Drag torque must oppose the angular velocity."""
import numpy as np, jax.numpy as jnp, dataclasses
from mime.nodes.environment.stokeslet.dejongh_geometry import dejongh_fl_mesh
from mime.nodes.environment.stokeslet.fluid_node import StokesletFluidNode
from mime.nodes.environment.stokeslet.interface import create_interface_mesh
from mime.nodes.environment.stokeslet.cylinder_wall_table import load_wall_table
from mime.core.quaternion import rotate_vector, rotate_vector_inverse

# verify rotate_vector convention with q = +90 about y (body-z -> world-x)
q=jnp.array([0.70710678,0,0.70710678,0])
print("rotate_vector(q, body-z=(0,0,1)) =", np.round(np.asarray(rotate_vector(q,jnp.array([0.,0,1]))),3), "(expect world-x (1,0,0) if body->world)")
print("rotate_vector_inverse(q, world-x=(1,0,0)) =", np.round(np.asarray(rotate_vector_inverse(q,jnp.array([1.,0,0]))),3), "(expect body-z (0,0,1))")

m=dejongh_fl_mesh(9,n_theta=20,n_zeta=28); P=(np.asarray(m.points)-np.asarray(m.points).mean(0))*1e-3
mesh=dataclasses.replace(m,points=P,weights=np.asarray(m.weights)*1e-6)
table=load_wall_table('data/dejongh_benchmark/wall_tables/wall_R2.035.npz')
bem=StokesletFluidNode('bem',5e-4,mu=1e-3,body_mesh=mesh,interface_mesh=create_interface_mesh(radius=1.8,n_refine=1),
    wall_table=table,R_cyl=float(table.R_cyl),length_scale=1.56e-3)
nb=mesh.n_points
def dragT(om,q):
    st=bem.initial_state()
    ns=bem.update(st,{'body_velocity':jnp.zeros(3),'body_angular_velocity':jnp.asarray(om,float),
        'body_orientation':jnp.asarray(q,float),'background_flow':jnp.zeros((nb,3))},5e-4)
    return np.asarray(ns['drag_torque'])
# z-config: spin about body-z=world-z (identity). Drag torque should oppose (-z).
Tz=dragT([0,0,1.0],[1,0,0,0])
# x-config: body rotated body-z->world-x, spin about world-x. Drag torque should oppose (-x).
Tx=dragT([1.0,0,0],[0.70710678,0,0.70710678,0])
print("z-config: omega=+z -> drag_torque", np.round(Tz*1e9,3),"nN·m (T_z should be NEGATIVE = opposing)")
print("x-config: omega=+x -> drag_torque", np.round(Tx*1e9,3),"nN·m (T_x should be NEGATIVE = opposing)")
print("z opposes:", Tz[2]<0, "| x opposes:", Tx[0]<0)
