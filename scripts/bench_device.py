"""Benchmark training throughput: CPU vs GPU, per sampling scheme.

Runs short real training jobs (no eval, fixed steps) sequentially — one at a
time, so there is no contention — and prints a steps/sec table plus ratios.
Use it to decide where to run a sweep on your hardware:

    python scripts/bench_device.py                       # cpu + cuda if visible
    python scripts/bench_device.py --devices cpu         # cpu only
    python scripts/bench_device.py --steps 10000 --schemes uniform,per,precond

Interpretation: single-env small-net SAC is latency-bound, so CPU frequently
beats a big GPU here (per-step kernel-launch + host<->device sync overhead).
Multiply single-run sps by your parallel worker count for sweep throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench_one(env_id, scheme, device, steps, out_dir, torch_threads):
    name = f"bench__{device}__{scheme}"
    cmd = [
        sys.executable, "-m", "rl_buffer.sac",
        "--env-id", env_id, "--scheme", scheme,
        "--total-steps", str(steps), "--learning-starts", str(min(500, steps // 4)),
        "--eval-frequency", str(10 ** 9),          # no eval: pure train throughput
        "--buffer-size", str(max(steps, 20000)),
        "--torch-threads", str(torch_threads),
        "--device", device, "--track", "0", "--verbose", "0",
        "--out-dir", out_dir, "--exp-name", name,
    ]
    env_vars = dict(os.environ, PYTHONPATH=REPO,
                    OMP_NUM_THREADS=str(torch_threads), MKL_NUM_THREADS=str(torch_threads))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env_vars, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-400:]
    with open(os.path.join(out_dir, name + ".json")) as f:
        d = json.load(f)
    return steps / d["total_wall_s"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="HalfCheetah-v4")
    ap.add_argument("--schemes", default="uniform,per,precond")
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    import torch
    devices = [d for d in args.devices.split(",") if d.strip()]
    if "cuda" in devices and not torch.cuda.is_available():
        print("note: cuda requested but not available on this machine -> cpu only")
        devices = [d for d in devices if d != "cuda"]
    schemes = [s for s in args.schemes.split(",") if s.strip()]

    print(f"env={args.env_id} steps={args.steps} torch_threads={args.torch_threads} "
          f"(sequential runs, no contention)\n")
    results = {}
    with tempfile.TemporaryDirectory() as out_dir:
        for device in devices:
            for scheme in schemes:
                sps, err = bench_one(args.env_id, scheme, device, args.steps,
                                     out_dir, args.torch_threads)
                if sps is None:
                    print(f"{device:5s} {scheme:9s} FAILED\n{err}")
                else:
                    results[(device, scheme)] = sps
                    print(f"{device:5s} {scheme:9s} {sps:6.0f} steps/s", flush=True)

    # comparison table
    if len(devices) == 2 and all((d, s) in results for d in devices for s in schemes):
        print(f"\n{'scheme':9s} {'cpu':>8s} {'cuda':>8s} {'cuda/cpu':>9s}")
        for s in schemes:
            c, g = results[("cpu", s)], results[("cuda", s)]
            print(f"{s:9s} {c:8.0f} {g:8.0f} {g / c:9.2f}x")
        print("\ncuda/cpu < 1 means the GPU is slower for this workload "
              "(latency-bound single-env SAC) -> run the sweep on CPU workers.")


if __name__ == "__main__":
    main()
