# MIME — Intended Use and Regulatory Positioning

## Platform Positioning Statement

MIME (MIcrorobotics Multiphysics Engine) is a domain-specific physics engine for microrobot simulation, built on the MADDENING framework. It is open-source research software distributed under LGPL-3.0-or-later.

**MIME is not a medical device** as defined by EU MDR (EU 2017/745) Article 2(1). It does not have a medical purpose, does not make clinical predictions, and does not provide diagnostic, therapeutic, or monitoring functionality. MIME is a computational tool for simulating microrobot physics — analogous to a domain-specific extension of a finite element library. Under the qualification criteria of MDCG 2019-11, software without a medical purpose is not a medical device.

MIME provides microrobotics-specific physics models (magnetic actuation, rigid body dynamics in viscous flow, drug release kinetics) and structured metadata (anatomical operating regimes, biocompatibility descriptors, benchmark results) that downstream tools may use. When incorporated into a regulated medical device, MIME is classified as SOUP (Software of Unknown Provenance) under IEC 62304. MADDENING, on which MIME depends, is MIME's own SOUP dependency (SOUP-of-SOUP).

The downstream commercial manufacturer is solely responsible for assessing MIME's (and MADDENING's) suitability for their specific context of use and for performing all required regulatory activities.

## Cybersecurity Boundary Statement (MDCG 2019-16)

MIME is a Python physics library that assumes trusted inputs. It performs no input sanitisation, authentication, authorisation, or network security functions. All simulation parameters, graph topologies, external inputs, and boundary conditions are assumed to be provided by a trusted caller.

When MIME is incorporated into a regulated product, the commercial integration layer is solely responsible for: validating and sanitising all simulation parameters before they reach MIME/MADDENING; ensuring graph topologies are well-formed; and preventing injection of malicious parameters through any user-facing interface.

## LGPL Replaceability Statement

MIME is licensed under LGPL-3.0-or-later. The LGPL "replaceability" obligation is satisfied via Python's standard module system: the end user can replace the `mime` package by installing a modified version into the same Python environment. No special linking, build steps, or binary compatibility mechanisms are required.
