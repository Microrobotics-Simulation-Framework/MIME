"""D10: overdamped body + the big implicit group [fvm,bem,body,ext_magnet,magnet].
Does the screw lock to the drive and swim through the pipe?"""
import numpy as np, importlib.util
from mime.experiments import schwarz_vessel_helix as S
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
for dip in [18.89, 3.0]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(int(dip*10)))
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_DIPOLE':dip}
    try:
        gm=S.build_graph(p); print('built; nodes',sorted(gm.node_names)); st=None; wh=[]; xs=[]
        for i in range(200): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
        wl=np.array(wh[-60:])/6.283; swim=(xs[-1]-xs[0])*1e3
        print('dipole=%5.2f: w_x mean=%6.2f std=%5.2f Hz (lock~3) | x-swim=%.4f mm (%.3f mm/s) | trace %s'%(
            dip,wl.mean(),wl.std(),swim,swim/(200*5e-4),[round(x*1e3,4) for x in xs[::40]]))
    except Exception as e:
        import traceback; print('dipole=%.1f ERROR'%dip, str(e)[:100])
