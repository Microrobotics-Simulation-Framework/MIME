import numpy as np, importlib.util
from mime.experiments import schwarz_vessel_helix as S
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
res=[]
for dt in [5e-4, 1.5e-4]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(dt))
    nsteps=int(0.03/dt)
    p={'N_THETA':14,'N_ZETA':20,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_DIPOLE':3.0,'DT':dt}
    gm=S.build_graph(p); st=None; wh=[]
    for i in range(nsteps): st=gm.step(ctrl.get_external_inputs(p,i,st)); wh.append(float(gm.get_node_state('body')['angular_velocity'][0]))
    wl=np.array(wh[-nsteps//2:])/6.283
    res.append((dt,nsteps,wl.mean(),wl.std()))
    print('dt=%.1e (%3d steps): w_x mean=%6.2f std=%5.2f Hz'%(dt,nsteps,wl.mean(),wl.std()), flush=True)
