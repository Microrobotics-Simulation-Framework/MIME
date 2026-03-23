# Docker Image Update Plan

## Current state

`ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest` uses:
- Base image: `nvidia/cuda:12.2.2-runtime-ubuntu22.04`
- JAX install: `jax[cuda12]>=0.4,<0.6` (resolves to JAX 0.5.3 with cuDNN for CUDA 12.2)
- Python: 3.10 (system default on Ubuntu 22.04)

**Problem**: RunPod A100 SXM hosts now run NVIDIA driver 570 (CUDA 12.8). The
cuDNN bundled with JAX 0.5.3's CUDA 12.2 build is incompatible with this driver,
producing `CUDNN_STATUS_INTERNAL_ERROR` on any GPU kernel launch. Discovered
during the T2.6 cloud rehearsal (2026-03-23).

**Current workaround**: The job config setup script includes
`pip3 install --upgrade 'jax[cuda12]'` as the first line, which upgrades JAX
to 0.6.2 with CUDA 12.9 cuDNN at runtime. This adds ~30 seconds to setup
but works reliably.

## Proposed change

Update the Docker image to use a CUDA 12.8+ base and a compatible JAX version:

1. Base image: `nvidia/cuda:12.8.0-runtime-ubuntu22.04` (or latest available 12.x)
2. JAX install: `jax[cuda12]>=0.5.0` (confirmed working at JAX 0.6.2 in rehearsal)
3. Tag: `latest` (overwrite — single user, no dependents)

## Steps to implement (deferred)

1. Update `MADDENING/docker/Dockerfile.cloud`:
   - Change `FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04` to `FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04`
   - Change `jax[cuda12]>=0.4,<0.6` to `jax[cuda12]>=0.5.0`
2. Build locally:
   ```bash
   cd /path/to/MADDENING
   docker build -f docker/Dockerfile.cloud -t ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest .
   ```
3. Test locally:
   ```bash
   docker run --gpus all ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest \
     python3 -c "import jax; print(f'JAX {jax.__version__}: {jax.default_backend()}')"
   ```
4. Push:
   ```bash
   docker push ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest
   ```
5. Remove `pip3 install --upgrade 'jax[cuda12]'` from all job config setup scripts
6. Test with a short rehearsal run to confirm end-to-end

## When to implement

After the T2.6 production sweep completes and results are reviewed. The setup
script workaround is sufficient for the production run. Updating the image
during active development risks breaking the production config if the build fails.

## Risk assessment

- **Single user, latest tag, no dependents** — no risk of breaking other users
- **Build failure**: leaves existing image untouched until push succeeds
- **Dockerfile is single-stage**: base image update is a one-line change
- **MADDENING version pin**: `jax>=0.5.0` is compatible with MADDENING's
  `jax>=0.4,<0.6` constraint. JAX 0.6.2 (installed in rehearsal) technically
  violates MADDENING's `<0.6` upper bound — this constraint should be relaxed
  to `<0.7` in MADDENING's pyproject.toml as part of this update.
