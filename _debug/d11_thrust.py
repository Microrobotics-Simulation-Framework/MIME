"""D11: does the COUPLED experiment generate axial corkscrew thrust when the screw is
ROTATED (prescribing the locked rotation, sidestepping the sync problem)? Held body,
prescribe Omega about x at the drive rate, V=0, measure axial drag_force_x = thrust.
If nonzero, the swim mechanism works; only the magnetic SYNC remains."""
import numpy as np, jax.numpy as jnp, importlib.util
from mime.experiments import schwarz_vessel_helix as S
p={'N_THETA':20,'N_ZETA':28,'SWIM_MODE':'held','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':False,'DELTA_RHO':0.0}
ref=S.screw_points(p)
gm,_=S.build_experiment(p).build()
S._seed_body_orientation(gm,p)   # body-z -> world-x
# prescribe rotation about world-x (the pipe axis) at several rates; V=0
for f in [0.0, 3.0, 10.0]:
    st=None
    for i in range(40):
        ext=S.default_external_inputs(p,body_points_ref=ref,state=st)
        ext['body']={'external_velocity':jnp.zeros(3),'external_angular_velocity':jnp.array([2*np.pi*f,0,0])}
        st=gm.step(ext)
    F=np.asarray(st['bem']['drag_force'])   # force on fluid (reaction); body thrust = -F
    print('rotate %4.1f Hz about x: drag_force (N) =%s | axial thrust F_x=%.3e N (=> would swim if !=0)'%(f, np.round(F*1e9,3), -F[0]))
