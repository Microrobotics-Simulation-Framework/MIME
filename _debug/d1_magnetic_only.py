"""D1: magnetic drive ONLY (no hydro) — does the rotating field sync a body whose
long axis is rotated body-z -> world-x? Isolates the magnetic chain."""
import numpy as np, jax.numpy as jnp
from mime.effects.experiment import Experiment, Body, Medium
from mime.effects.magnetic import MagneticModel
from mime.nodes.actuation.motor import MotorNode
from mime.nodes.actuation.permanent_magnet import PermanentMagnetNode
from mime.nodes.robot.permanent_magnet_response import PermanentMagnetResponseNode
from mime.nodes.robot.rigid_body import RigidBodyNode

DT=5e-4
def run(q_init, motor_axis, mom_axis, standoff_pos, drive_hz=3.0, n=200):
    motor=MotorNode("motor",DT,inertia_kg_m2=1e-5,kt_n_m_per_a=0.05,r_ohm=1.0,l_henry=1e-3,
                    damping_n_m_s=1e-4,axis_in_parent_frame=motor_axis,
                    tool_offset_in_rotor_frame=(0,0,0,1,0,0,0))
    magnet=PermanentMagnetNode("ext_magnet",DT,dipole_moment_a_m2=18.89,
            magnetization_axis_in_body=(1,0,0),magnet_radius_m=17.5e-3,magnet_length_m=20e-3,
            field_model="point_dipole",earth_field_world_t=(0,0,0))
    resp=PermanentMagnetResponseNode("magnet",DT,n_magnets=2,m_single=8.4e-4,moment_axis=mom_axis)
    body=RigidBodyNode("body",DT,use_inertial=True,I_eff=7e-11,m_eff=5.7e-5,
            semi_major_axis_m=1.56e-3,semi_minor_axis_m=0.78e-3,density_kg_m3=1000.,
            fluid_viscosity_pa_s=1e-3,fluid_density_kg_m3=1000.)
    exp=Experiment("d1",mime_version_min="0.1.0")
    exp.set_body(Body("body",node=body,properties={"magnetic":{}}))
    exp.set_medium(Medium({"density":1000.,"viscosity":1e-3}))
    exp.attach(MagneticModel.PointDipole(magnet,resp,pose_source=motor))
    exp.add_external_input("motor","commanded_velocity",())
    exp.add_external_input("motor","parent_pose_world",(7,))
    gm,_=exp.build()
    st=dict(gm.get_node_state("body")); st["orientation"]=jnp.asarray(q_init); gm.set_node_state("body",st)
    ext={"motor":{"commanded_velocity":jnp.asarray(2*np.pi*drive_hz),
                  "parent_pose_world":jnp.asarray(standoff_pos)}}
    whist=[]
    for i in range(n):
        s=gm.step(ext); whist.append(np.asarray(gm.get_node_state("body")["angular_velocity"]))
    w=whist[-1]
    # field + torque diagnostics
    fv=np.asarray(gm.get_node_state("ext_magnet").get("field_vector",[0,0,0]))
    mt=np.asarray(gm.get_node_state("magnet").get("magnetic_torque",[0,0,0]))
    return w, fv, mt

print("=== z-config (legacy): body identity, motor about z, magnet above at z ===")
w,fv,mt=run([1,0,0,0],(0,0,1),(1,0,0),[0,0,0.15,1,0,0,0])
print(" body w (Hz): [%.2f,%.2f,%.2f] |drive 3| field=%.2emT torque=%.2e"%(w[0]/6.28,w[1]/6.28,w[2]/6.28,np.linalg.norm(fv)*1e3,np.linalg.norm(mt)))

print("=== x-config (ar4): body-z->world-x, motor about x (EE-z=x), magnet above at z ===")
# motor parent oriented so motor-z -> world-x: +90 about y
w,fv,mt=run([0.7071,0,0.7071,0],(0,0,1),(1,0,0),[0,0,0.15,0.7071,0,0.7071,0])
print(" body w (Hz): [%.2f,%.2f,%.2f] |drive 3| field=%.2emT torque=%.2e"%(w[0]/6.28,w[1]/6.28,w[2]/6.28,np.linalg.norm(fv)*1e3,np.linalg.norm(mt)))
