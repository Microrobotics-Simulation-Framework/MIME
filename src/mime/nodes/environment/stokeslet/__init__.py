"""Regularised Stokeslet BEM fluid solver for confined microrobot flows.

Provides StokesletFluidNode — a quasi-static Stokes flow solver using
the method of regularised Stokeslets (Cortez, Fauci & Medovikov 2005)
with Nyström BEM discretization. Computes drag force and torque on a
rigid body via a 6×6 resistance matrix, optionally including vessel
wall confinement via explicit wall surface discretization.

No Mach number constraint — operates at any rotation frequency.
"""
