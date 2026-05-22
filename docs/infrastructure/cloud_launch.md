# Launching cloud jobs

MIME runs its long sweeps (the confinement sweep, GNN data generation) on
cloud GPUs. A run is described by a **job spec** — a YAML file in `jobs/` —
and launched with `scripts/launch_job.py`.

## Job specs

A job spec is a MADDENING `JobConfig` YAML: a `provider`, a `gpu_type`, cost
guards, and `setup` / `run` shell blocks. `jobs/` ships:

| Spec | Provider |
|---|---|
| `production_h100.yaml` | RunPod — H100-SXM, on-demand |
| `production_h100_aws.yaml` | AWS — template |
| `production_h100_gcp.yaml` | GCP — template |
| `rehearsal_a100.yaml` | RunPod — A100, rehearsal |

The AWS and GCP specs are **templates**: `gpu_type`, `region` and the cost
guards must be checked against the provider before a real launch —
`sky show-gpus --cloud aws|gcp` for accelerator names, and the provider's
current on-demand / spot pricing for the cost guards.

## Launching

```bash
# dry run — resolve provider, instance and cost without provisioning
python scripts/launch_job.py --job jobs/production_h100_aws.yaml --dry-run

# real launch
python scripts/launch_job.py --job jobs/production_h100_aws.yaml
```

`launch_job.py` is provider-agnostic — it reads the spec's `provider:` field
and dispatches through MADDENING's `CloudLauncher`, which supports `runpod`,
`aws`, `gcp` and `lambda_labs`.

## Credentials

`CloudLauncher` reads credentials from `~/.maddening/cloud_credentials.yaml`
(one block per provider). Each provider also has a native credential file it
expects — `~/.runpod/config.toml`, `~/.aws/credentials`,
`~/.config/gcloud/application_default_credentials.json` — see MADDENING's
`cloud/providers.py` for the exact formats.

## Spot instances and resume

The AWS/GCP templates set `use_spot: true`. Spot is safe for the confinement
sweep because it is **resumable** — `ResumableSweep` (see
[preempt/resume](../preempt_resume.md)) checkpoints after every combo. Point
`SWEEP_SNAPSHOT_DIR` at durable storage (a mounted S3/GCS bucket) so the
checkpoint survives a preemption; a relaunch then resumes instead of
restarting from zero. `spot_fallback: true` additionally falls back to
on-demand if spot capacity is unavailable.
