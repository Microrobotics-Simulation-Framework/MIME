#!/usr/bin/env python3
"""Launch a cloud job via MADDENING CloudLauncher.

General-purpose launcher for rehearsal, production, and any other cloud
job. The job-specific logic lives in the YAML config and the script it
runs — the launcher handles provisioning, polling, retrieval, teardown.

Architecture:
  - SkyPilot handles: provisioning, Docker image, workdir sync, setup
  - SSH polling: waits for job completion (file-existence + process check)
  - scp: retrieves HDF5 output after job completes
  - CloudLauncher: credentials, cost guards, explicit teardown

Usage:
    python3 scripts/launch_job.py --job jobs/rehearsal_a100.yaml --output data/rehearsal_192.h5
    python3 scripts/launch_job.py --job jobs/production_h100.yaml --output data/umr_training_v1.h5
    python3 scripts/launch_job.py --job jobs/production_h100.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _extract_script_name(yaml_path: Path) -> str:
    """Extract the Python script name from the YAML run: field.

    Parses the run command to find 'python3 scripts/foo.py' and returns 'foo.py'.
    Falls back to 'python3' if extraction fails.
    """
    try:
        import yaml
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        run_cmd = config.get("run", "")
        # Match 'python3 scripts/something.py' or 'python3 something.py'
        m = re.search(r"python3?\s+(?:scripts/)?(\S+\.py)", run_cmd)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "python3"


def main():
    parser = argparse.ArgumentParser(description="Launch cloud job via CloudLauncher")
    parser.add_argument("--job", required=True, help="Path to job config YAML")
    parser.add_argument("--creds", default=None, help="Path to credentials YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no provisioning")
    parser.add_argument("--output", default="data/output.h5",
                        help="Local path to save retrieved HDF5 file")
    parser.add_argument("--remote-output", default=None,
                        help="Remote path on cloud instance (default: ~/sky_workdir/data/<output basename>)")
    parser.add_argument("--skip-retrieve", action="store_true",
                        help="Skip HDF5 retrieval")
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job config not found: {job_path}")
        sys.exit(1)

    # Derive remote output path from local output path
    output_basename = Path(args.output).name
    remote_path = args.remote_output or f"~/sky_workdir/data/{output_basename}"

    # Extract script name for process polling
    script_name = _extract_script_name(job_path)

    try:
        from maddening.cloud.launcher import (
            CloudLauncher, JobConfig,
            CostLimitError, CredentialError, LaunchError,
        )
    except ImportError:
        print("CloudLauncher requires SkyPilot. Install with:\n"
              "  pip install 'skypilot[runpod]>=0.11'")
        sys.exit(1)

    try:
        launcher = CloudLauncher(credentials_path=args.creds)

        # ── Validate ──────────────────────────────────────────────
        print("Validating configuration...")
        result = launcher.validate(job_path)
        print(f"  Provider: {result['provider']}")
        print(f"  GPU: {result['gpu_type']}")
        print(f"  Instance: {result['instance_type']}")
        print(f"  Hourly cost: ${result['hourly_cost']:.2f}")
        print(f"  Budget used: ${result['budget_used']:.2f}")
        print(f"  Budget remaining: ${result['budget_remaining']:.2f}")

        if args.dry_run:
            print("\nDry run complete — no resources provisioned.")
            return

        # ── Write git hash for provenance ─────────────────────────
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=5,
            ).strip()
            Path(".mime_git_hash").write_text(git_hash + "\n")
            print(f"  Git hash: {git_hash}")
        except Exception:
            print("  WARNING: Could not capture git hash")

        # ── Launch (provisions + setup + run) ─────────────────────
        print(f"\nLaunching instance (provision + setup)...")
        print("(SkyPilot streams provisioning and setup logs)\n")

        job = launcher.launch(job_path)

        print(f"\nInstance ready.")
        print(f"  Cluster: {job.cluster_name}")
        print(f"  VM IP: {job.vm_ip}")
        print(f"  SSH port: {job.ssh_port}")

        # ── Wait for job completion ───────────────────────────────
        # Polls every 30s via SSH for a .done completion marker file.
        # The HDF5 file exists from schema creation (before runs start),
        # so it cannot be used as a completion signal. The sweep script
        # writes <output>.done only after all runs finish.
        # max_wait: 2 hours (covers 90-min production + setup margin).
        done_marker = remote_path.rsplit(".", 1)[0] + ".done"
        print(f"\nWaiting for job to complete on instance...")
        print(f"  Polling: every 30s, script={script_name}")
        print(f"  Completion marker: {done_marker}")

        max_wait = 7200   # 2 hours
        poll_interval = 30  # 30 seconds between polls
        t_start = time.time()

        while time.time() - t_start < max_wait:
            # Primary check: completion marker file exists
            check = job.ssh_run(
                f"test -f {done_marker} && echo EXISTS || echo MISSING",
                capture=True, check=False, timeout=30,
            )
            if check.returncode == 0 and "EXISTS" in check.stdout:
                print(f"\n  Completion marker detected after {time.time() - t_start:.0f}s")
                break

            # Secondary check: job process still running
            proc_check = job.ssh_run(
                f"pgrep -f '{script_name}' > /dev/null 2>&1 && echo RUNNING || echo DONE",
                capture=True, check=False, timeout=30,
            )
            if proc_check.returncode == 0:
                status = proc_check.stdout.strip()
                elapsed = time.time() - t_start
                print(f"  [{elapsed:.0f}s] Job: {status}", flush=True)

                if "DONE" in status:
                    time.sleep(5)  # flush delay
                    check2 = job.ssh_run(
                        f"test -f {done_marker} && echo EXISTS || echo MISSING",
                        capture=True, check=False, timeout=30,
                    )
                    if check2.returncode == 0 and "EXISTS" in check2.stdout:
                        print(f"\n  Completion marker found after job completed")
                    else:
                        print(f"\n  WARNING: Job finished but completion marker not found")
                        log_check = job.ssh_run(
                            "for f in /tmp/ray_skypilot/session_*/logs/worker-*.out; do "
                            "  size=$(wc -c < \"$f\" 2>/dev/null); "
                            "  if [ \"$size\" -gt 100 ] 2>/dev/null; then tail -50 \"$f\"; break; fi; "
                            "done 2>/dev/null || echo 'No job logs found'",
                            capture=True, check=False, timeout=30,
                        )
                        if log_check.returncode == 0:
                            print(log_check.stdout)
                    break

            time.sleep(poll_interval)
        else:
            print(f"\n  WARNING: Timed out after {max_wait}s waiting for job completion")

        # ── Retrieve job logs ─────────────────────────────────────
        print(f"\nRetrieving job logs...")
        log_result = job.ssh_run(
            "for f in /tmp/ray_skypilot/session_*/logs/worker-*.out; do "
            "  size=$(wc -c < \"$f\" 2>/dev/null); "
            "  if [ \"$size\" -gt 100 ] 2>/dev/null; then cat \"$f\"; break; fi; "
            "done 2>/dev/null || "
            "cat /tmp/sky_logs/*/run.log 2>/dev/null || "
            "echo 'No job logs found'",
            capture=True, check=False, timeout=60,
        )
        if log_result.returncode == 0 and log_result.stdout.strip():
            print("\n" + "=" * 60)
            print("JOB OUTPUT")
            print("=" * 60)
            print(log_result.stdout)

        # ── Retrieve HDF5 output ──────────────────────────────────
        if not args.skip_retrieve:
            print(f"\nRetrieving output: {remote_path} -> {args.output}")
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

            scp_result = subprocess.run(
                [
                    "scp",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-P", str(job.ssh_port),
                    f"root@{job.vm_ip}:{remote_path}",
                    str(args.output),
                ],
                capture_output=True, text=True,
            )

            if scp_result.returncode == 0:
                file_size = os.path.getsize(args.output)
                print(f"  Retrieved: {args.output} ({file_size} bytes)")
            else:
                print(f"  WARNING: scp failed (exit {scp_result.returncode})")
                print(f"  stderr: {scp_result.stderr.strip()}")

        # ── Teardown ──────────────────────────────────────────────
        print(f"\nTearing down instance...")
        job.teardown()
        print("Instance terminated.")

        # ── Cost report ───────────────────────────────────────────
        try:
            cost = job.cost_so_far()
            print(f"\nEstimated cost: ${cost:.2f}")
        except Exception:
            pass

        print("\nDone.")

    except CredentialError as e:
        print(f"Credential error: {e}")
        sys.exit(1)
    except CostLimitError as e:
        print(f"Cost limit exceeded: {e}")
        sys.exit(1)
    except LaunchError as e:
        print(f"Launch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
