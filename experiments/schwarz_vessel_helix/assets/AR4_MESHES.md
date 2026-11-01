# The AR4's meshes

The AR4 (mk5) visual meshes are Annin Robotics' design files. Annin's licence (the `ar4-hmi` repository's
`LICENSE.txt`, version 1.1, section 2) allows educational, research and personal non-commercial use and does not allow
redistributing them, so this repository does not carry them, baked or raw. Build them locally:

```
.venv/bin/python experiments/ar4_helical_drive/assets/_fetch_meshes.py
.venv/bin/python experiments/ar4_helical_drive/scene/_bake_meshes.py
```

The first downloads the STLs from `Annin-Robotics/ar4_ros_driver` at a pinned commit; the second bakes and decimates
them into `assets/ar4_meshes.usdc`, which `.gitignore` keeps out of commits. This experiment reads its own copy: copy `experiments/ar4_helical_drive/assets/ar4_meshes.usdc` here after baking. Without the file the scene still
opens and the simulation runs unchanged: the arm's link prims are there, drawn empty, and USD reports the unresolved
reference.

Meshes: Annin Robotics, https://www.anninrobotics.com, non-commercial use.
