"""D4: hypothesis — the magnetic over-spin is a one-step orientation-lag negative
damping (no coupling group). Test: wrap body+magnet+response in a coupling group and
see if the spin locks to the 3 Hz drive."""
import numpy as np, jax.numpy as jnp
from mime.experiments import schwarz_vessel_helix as S
# monkeypatch build_experiment to add a magnetic coupling group
_orig=S.build_experiment
def patched(params=None):
    exp=_orig(params)
    exp.add_coupling_group(["body","ext_magnet","magnet"], max_iterations=8, tolerance=1e-8)
    return exp
S.build_experiment=patched

import importlib.util
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c')
p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_STANDOFF_M':0.20}
gm=S.build_graph(p); st=None; wh=[]
for i in range(120): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0]))
b=gm.get_node_state('body'); pos=np.asarray(b['position'])
print("WITH magnetic coupling group: w_x trace (Hz):", [round(w/6.283,2) for w in wh[::20]])
print("  final w_x=%.2f Hz (drive 3) | x-swim=%.4f mm"%(wh[-1]/6.283, pos[0]*1e3))
