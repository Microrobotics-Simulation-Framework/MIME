"""MIME control layer — policies, primitives, sequences, and the PolicyRunner.

The control layer sits outside MADDENING's GraphManager. It produces
external_inputs dicts that GraphManager.step() consumes. The design is
transport-agnostic: control commands can come from a local ControlPolicy,
a remote ZMQ subscriber, a ROS bridge, or any callable.
"""
