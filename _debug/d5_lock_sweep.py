"""D5: with the magnetic coupling group, sweep field strength (standoff) to find the
lock regime where the screw synchronises to the 3 Hz drive and swims along x."""
import numpy as np, jax.numpy as jnp, importlib.util
from mime.experiments import schwarz_vessel_helix as S
_orig=S.build_experiment
def patched(params=None):
    exp=_orig(params)
    exp.add_coupling_group(["body","ext_magnet","magnet"], max_iterations=8, tolerance=1e-8)
    return exp
S.build_experiment=patched
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

for stand in [0.30, 0.50, 0.80, 1.2]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(int(stand*100)))
    B=1e-7*18.89/stand**3*1e3
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,
       'DELTA_RHO':0.0,'MAG_STANDOFF_M':stand,'CONTROL_STANDOFF_M':stand}
    try:
        gm=S.build_graph(p); st=None; wh=[]; xs=[]
        for i in range(150): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
        # lock metric: std of w over last 50 steps (low = locked)
        wl=np.array(wh[-50:])/6.283; swim=(xs[-1]-xs[0])*1e3
        print("stand=%.2f B~%.3fmT: w_x mean=%.2f std=%.2f Hz (locked if std<<mean, near 3) | x-swim=%.4f mm"%(stand,B,wl.mean(),wl.std(),swim))
    except Exception as e:
        print("stand=%.2f: ERROR %s"%(stand,str(e)[:60]))
