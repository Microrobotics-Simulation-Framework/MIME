"""D6: in the REAL coupled config (no monkeypatch), does the magnetic torque on the
body vary with field? And is it significant vs the hydro? Two standoffs, 16x field."""
import numpy as np, importlib.util
from mime.experiments import schwarz_vessel_helix as S
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
for stand in [0.20, 0.80]:
    ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c'+str(int(stand*100)))
    p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,
       'DELTA_RHO':0.0,'MAG_STANDOFF_M':stand,'CONTROL_STANDOFF_M':stand}
    gm=S.build_graph(p); st=None
    for i in range(20): st=gm.step(ctrl.get_external_inputs(p,i,st))
    # inspect: rotor pose, field at body, magnetic torque, drag torque, body w + magnet target
    rot=np.asarray(gm.get_node_state('motor')['rotor_pose_world'])
    fv=np.asarray(gm.get_node_state('ext_magnet').get('field_vector',[0,0,0]))
    mt=np.asarray(gm.get_node_state('magnet').get('magnetic_torque',[0,0,0]))
    dt=np.asarray(gm.get_node_state('bem').get('drag_torque',[0,0,0]))
    b=gm.get_node_state('body')
    print('stand=%.2f: rotor pos %s | field=%.3e T | mag_torque=%.3e | drag_torque(reaction)=%.3e | w_x=%.2f Hz'%(
        stand, np.round(rot[:3],3), np.linalg.norm(fv), np.linalg.norm(mt), np.linalg.norm(dt), b['angular_velocity'][0]/6.283))
