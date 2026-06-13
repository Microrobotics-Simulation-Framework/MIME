"""D7: with the coupling group, sweep MAGNET STRENGTH (dipole moment) — the field is
reach-limited at ~0.15m, so weaken the magnet itself. Find the lock+swim regime."""
import numpy as np, jax.numpy as jnp, importlib.util
from mime.experiments import schwarz_vessel_helix as S
_orig=S.build_experiment
def patched(params=None):
    exp=_orig(params); exp.add_coupling_group(["body","ext_magnet","magnet"],max_iterations=8,tolerance=1e-8); return exp
S.build_experiment=patched
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
for dip in [18.89, 3.0, 0.5, 0.1]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(int(dip*10)))
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_DIPOLE':dip}
    gm=S.build_graph(p); st=None; wh=[]; xs=[]
    for i in range(200): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
    wl=np.array(wh[-60:])/6.283; swim=(xs[-1]-xs[0])*1e3
    print('dipole=%5.2f A.m2: w_x mean=%6.2f std=%6.2f Hz (lock=std<<|mean|~3) | x-swim=%.4f mm'%(dip,wl.mean(),wl.std(),swim))
