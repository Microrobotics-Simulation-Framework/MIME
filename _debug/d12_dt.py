"""D12: is the rotational libration an explicit-integration artifact? Sweep DT — a
smaller step should resolve the stiff magnetic-rotational mode and let it lock."""
import numpy as np, importlib.util
from mime.experiments import schwarz_vessel_helix as S
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
for dt in [5e-4, 1e-4, 2e-5]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(dt))
    nsteps=int(0.1/dt)   # fixed 0.1s physical time
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_DIPOLE':3.0,'DT':dt}
    gm=S.build_graph(p); st=None; wh=[]; xs=[]
    for i in range(nsteps): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0])); xs.append(float(gm.get_node_state('body')['position'][0]))
    wl=np.array(wh[-nsteps//3:])/6.283; swim=(xs[-1]-xs[0])*1e3
    print('dt=%.0e (%4d steps): w_x mean=%6.2f std=%5.2f Hz (lock~3) | x-swim=%.4f mm'%(dt,nsteps,wl.mean(),wl.std(),swim))
