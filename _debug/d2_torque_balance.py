"""D2: in the coupled x-config, is the BEM drag torque DAMPING or DRIVING the spin?
Print magnetic_torque vs bem drag_torque (x-component = the spin axis) each step."""
import numpy as np, importlib.util
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
from mime.experiments import schwarz_vessel_helix as S
p={'N_THETA':16,'N_ZETA':24,'SWIM_MODE':'free','FLOW_PROFILE':'poiseuille','INCLUDE_ARM':True,'DELTA_RHO':0.0,'MAG_STANDOFF_M':0.30}
gm=S.build_graph(p)
ctrl=load('experiments/schwarz_vessel_helix/control/controller.py','c')
print("step |  w_x(Hz) | mag_torque_x | bem_dragT_x | (drag opposes spin if opposite sign to w_x)")
st=None
for i in range(40):
    st=gm.step(ctrl.get_external_inputs(p,i,st))
    if i%5==0 or i<3:
        b=gm.get_node_state('body'); wx=float(b['angular_velocity'][0])
        # magnetic torque on body, bem drag torque on body
        mt=np.asarray(gm.get_node_state('magnet').get('magnetic_torque',[0,0,0]))
        dt=np.asarray(gm.get_node_state('bem').get('drag_torque',[0,0,0]))
        print("%4d | %8.2f | %+.3e | %+.3e | drag*w=%+.2e"%(i, wx/6.283, mt[0], dt[0], dt[0]*wx))
