"""D8: OVERDAMPED body (Stokes microswimmer) — does the screw lock + swim cleanly?
Compare overdamped vs inertial, and with the magnetic coupling group."""
import numpy as np, importlib.util
from mime.experiments import schwarz_vessel_helix as S
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def trial(tag, model, group, dip=18.89, n=200, drho=0.0):
    import mime.experiments.schwarz_vessel_helix as SS
    if group:
        _orig=SS.build_experiment
        SS.build_experiment=lambda params=None,_o=_orig:(lambda e:(e.add_coupling_group(["body","ext_magnet","magnet"],max_iterations=8,tolerance=1e-8),e)[1])(_o(params))
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py',tag)
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':drho,'MAG_DIPOLE':dip,'BODY_MODEL':model}
    try:
        gm=SS.build_graph(p); st=None; wh=[]; xs=[]
        for i in range(n): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
        wl=np.array(wh[-60:])/6.283; swim=(xs[-1]-xs[0])*1e3
        print('%-28s: w_x mean=%6.2f std=%6.2f Hz | x-swim=%.4f mm (%.2f mm/s)'%(tag,wl.mean(),wl.std(),swim,swim/(n*5e-4)))
    except Exception as e:
        print('%-28s: ERROR %s'%(tag,str(e)[:70]))
    if group: SS.build_experiment=_orig
trial('overdamped_nogroup','overdamped',False)
trial('overdamped_group','overdamped',True)
trial('overdamped_group_d3','overdamped',True,dip=3.0)
