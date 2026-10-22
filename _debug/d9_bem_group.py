"""D9: overdamped body needs the BEM (drag source) in its coupling group to resolve
the force balance self-consistently (else step-0 zero-drag blowup). Try [body, bem,
ext_magnet, magnet] (bem also lives in the two-scale [fvm,bem] group)."""
import numpy as np, importlib.util
import mime.experiments.schwarz_vessel_helix as SS
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def trial(tag, members, dip=18.89, n=200):
    _orig=SS.build_experiment
    SS.build_experiment=lambda params=None,_o=_orig,mm=members:(lambda e:(e.add_coupling_group(mm,max_iterations=10,tolerance=1e-9),e)[1])(_o(params))
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py',tag)
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_DIPOLE':dip,'BODY_MODEL':'overdamped'}
    try:
        gm=SS.build_graph(p); st=None; wh=[]; xs=[]
        for i in range(n): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
        wl=np.array(wh[-60:])/6.283; swim=(xs[-1]-xs[0])*1e3
        print('%-30s: w_x mean=%6.2f std=%6.2f Hz | x-swim=%.4f mm (%.2f mm/s)'%(tag,wl.mean(),wl.std(),swim,swim/(n*5e-4)))
    except Exception as e:
        print('%-30s: ERROR %s'%(tag,str(e)[:80]))
    SS.build_experiment=_orig
trial('body+bem+magnet', ["body","bem","ext_magnet","magnet"])
trial('body+bem+magnet+fvm', ["body","bem","ext_magnet","magnet","fvm"])
