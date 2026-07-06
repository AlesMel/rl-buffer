"""Parallel, idempotent sweep runner.

Runs a grid of (env, scheme, priority_mode, seed) SAC jobs with a fixed
concurrency.  Each job is a single-thread subprocess so N jobs saturate N cores
without thread oversubscription.  Jobs whose output JSON already exists are
skipped, so re-running extends a sweep (e.g. adds seeds) without recomputation.

Usage:
    python scripts/run_sweep.py --preset pilot   --out-dir results/pilot
    python scripts/run_sweep.py --preset full     --out-dir results/full   # cluster
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    # minimal real run sized for a 4-core CPU session (this environment)
    "mini": dict(
        envs=["HalfCheetah-v4"],
        schemes=["uniform", "per", "euclid", "precond", "precond2"],
        priority_mode="lazy",
        seeds=[1, 2, 3],
        total_steps=60_000,
        learning_starts=5000,
    ),
    # small, real, underpowered pilot that fits a 4-core CPU box
    "pilot": dict(
        envs=["HalfCheetah-v4"],
        schemes=["uniform", "per", "euclid", "precond", "precond2"],
        priority_mode="lazy",
        seeds=[1, 2, 3],
        total_steps=100_000,
        learning_starts=5000,
    ),
    # the task's protocol: 4 envs, >=10 seeds, 1M steps (3M Humanoid). Cluster only.
    "full": dict(
        envs=["HalfCheetah-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"],
        schemes=["uniform", "per", "euclid", "precond", "precond2"],
        priority_mode="lazy",
        seeds=list(range(1, 11)),
        total_steps=1_000_000,
        learning_starts=5000,
    ),
    # compute-matched two-stage comparison (fresh priorities, ~pool_mult x cost)
    "twostage": dict(
        envs=["HalfCheetah-v4"],
        schemes=["euclid", "precond", "precond2"],
        priority_mode="two_stage",
        seeds=[1, 2, 3],
        total_steps=100_000,
        learning_starts=5000,
    ),
}


def humanoid_steps(env, steps):
    return 3_000_000 if env == "Humanoid-v4" and steps >= 1_000_000 else steps


def run_one(job):
    env, scheme, pmode, seed, steps, lstart, out_dir, gpu = job
    steps = humanoid_steps(env, steps)
    name = f"{env}__{scheme}__{pmode}__seed{seed}"
    out_path = os.path.join(out_dir, name + ".json")
    if os.path.exists(out_path):
        return f"skip  {name}"
    device = "cuda" if gpu is not None else "auto"
    cmd = [
        sys.executable, "-m", "rl_buffer.sac",
        "--env-id", env, "--scheme", scheme, "--priority-mode", pmode,
        "--seed", str(seed), "--total-steps", str(steps),
        "--learning-starts", str(lstart), "--eval-frequency", "10000",
        "--eval-episodes", "10", "--torch-threads", "1", "--out-dir", out_dir,
        "--device", device,
    ]
    env_vars = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if gpu is not None:
        # pin this run to a single GPU; inside the subprocess it is cuda:0
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env_vars, capture_output=True, text=True)
    dt = time.time() - t0
    tag = f"gpu{gpu}" if gpu is not None else "cpu"
    if r.returncode != 0:
        return f"FAIL  {name} [{tag}]  ({dt:.0f}s)\n{r.stderr[-800:]}"
    return f"done  {name} [{tag}]  ({dt:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="pilot")
    ap.add_argument("--out-dir", default="results/pilot")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent runs (total). CPU: set ~= core count. GPU: split across --gpus")
    ap.add_argument("--gpus", type=str, default="",
                    help="comma-separated GPU ids to round-robin across, e.g. '0,1'. "
                         "Empty => CPU/auto. Runs are pinned one-GPU-each; pack several per GPU "
                         "by setting --workers above the GPU count (small nets share a card well).")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    os.makedirs(args.out_dir, exist_ok=True)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    triples = list(itertools.product(cfg["envs"], cfg["schemes"], cfg["seeds"]))
    jobs = [
        (env, scheme, cfg["priority_mode"], seed, cfg["total_steps"], cfg["learning_starts"],
         args.out_dir, (gpus[i % len(gpus)] if gpus else None))
        for i, (env, scheme, seed) in enumerate(triples)
    ]
    where = f"gpus={gpus} ({args.workers//max(1,len(gpus))}/gpu)" if gpus else "cpu"
    print(f"preset={args.preset} jobs={len(jobs)} workers={args.workers} {where} out={args.out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(run_one, jobs):
            print(msg, flush=True)
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
