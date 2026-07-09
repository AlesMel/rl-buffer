"""Run the sampling-scheme sweep for per-env experiment folders.

Each ``experiments/<Env>/`` folder is self-contained:

    experiments/<Env>/config.json   env id + kwargs, steps, seeds, schemes
    experiments/<Env>/results/      one JSON per run (written by rl_buffer.sac)
    experiments/<Env>/logs/         TensorBoard event dirs, one per run

Usage:
    python scripts/run_experiments.py --env HalfCheetah-v4                 # one env
    python scripts/run_experiments.py --env all --workers auto --nice 10   # everything
    python scripts/run_experiments.py --env LunarLander-v3 --seeds 1,2,3 --steps 100000

Jobs whose result JSON already exists are skipped, so re-running a folder
extends it (add seeds / recover from an interrupt) without recomputation.
Analyze one folder or several together:
    python analysis/rliable_analysis.py --results-dir experiments/<Env>/results ...
    python analysis/rliable_analysis.py --results-dir experiments/*/results ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_ROOT = os.path.join(REPO, "experiments")


def load_experiments(which: str):
    if which == "all":
        names = sorted(d for d in os.listdir(EXP_ROOT)
                       if os.path.isfile(os.path.join(EXP_ROOT, d, "config.json")))
    else:
        names = [which]
    out = []
    for name in names:
        path = os.path.join(EXP_ROOT, name, "config.json")
        if not os.path.isfile(path):
            raise SystemExit(f"no config.json in experiments/{name} "
                             f"(available: {os.listdir(EXP_ROOT)})")
        with open(path) as f:
            out.append((name, json.load(f)))
    return out


def run_one(job):
    (name, cfg, scheme, seed, steps, seeds_note, gpu, nice) = job
    exp_dir = os.path.join(EXP_ROOT, name)
    results_dir = os.path.join(exp_dir, "results")
    logs_dir = os.path.join(exp_dir, "logs")
    run_name = f"{cfg['env_id']}__{scheme}__{cfg.get('priority_mode','lazy')}__seed{seed}"
    out_path = os.path.join(results_dir, run_name + ".json")
    if os.path.exists(out_path):
        return f"skip  {name}/{run_name}"
    device = "cuda" if gpu is not None else "auto"
    cmd = [
        sys.executable, "-m", "rl_buffer.sac",
        "--env-id", cfg["env_id"],
        "--env-kwargs", json.dumps(cfg.get("env_kwargs", {})),
        "--scheme", scheme,
        "--priority-mode", cfg.get("priority_mode", "lazy"),
        "--seed", str(seed),
        "--total-steps", str(steps),
        "--learning-starts", str(cfg.get("learning_starts", 5000)),
        "--eval-frequency", str(cfg.get("eval_frequency", 10000)),
        "--eval-episodes", str(cfg.get("eval_episodes", 10)),
        "--torch-threads", "1",
        "--out-dir", results_dir,
        "--log-dir", logs_dir,
        "--device", device,
        "--verbose", "0",   # no per-step tqdm in children; eval lines stream below
    ]
    if nice:
        cmd = ["nice", "-n", str(nice)] + cmd
    env_vars = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if gpu is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tag = f"gpu{gpu}" if gpu is not None else "cpu"

    # Stream the child's output: eval progress lines ("[<run>] step=... eval=...")
    # go straight to the terminal as they happen (the live heartbeat), and the
    # full output is teed to logs/<run>.out for tail -f / postmortem.
    out_log = os.path.join(logs_dir, run_name + ".out")
    print(f"start {name}/{run_name} [{tag}]  (log: {out_log})", flush=True)
    t0 = time.time()
    with open(out_log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env_vars, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if line.startswith("["):          # sac.py progress/eval lines
                print(f"  {line.rstrip()}", flush=True)
        proc.wait()
    dt = time.time() - t0
    if proc.returncode != 0:
        with open(out_log) as lf:
            tail = "".join(lf.readlines()[-15:])
        return f"FAIL  {name}/{run_name} [{tag}]  ({dt:.0f}s)\n{tail}"
    return f"done  {name}/{run_name} [{tag}]  ({dt:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True,
                    help="experiment folder name under experiments/, or 'all'")
    ap.add_argument("--workers", type=str, default="auto",
                    help="concurrent runs, or 'auto' = cpu_count-2")
    ap.add_argument("--gpus", type=str, default="",
                    help="comma-separated GPU ids to round-robin (usually leave empty: CPU wins here)")
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="",
                    help="override seeds, e.g. '1,2,3' (default: config.json)")
    ap.add_argument("--steps", type=int, default=0,
                    help="override total_steps for every env (default: config.json)")
    ap.add_argument("--schemes", type=str, default="",
                    help="override schemes, e.g. 'uniform,precond'")
    args = ap.parse_args()

    workers = max(1, (os.cpu_count() or 4) - 2) if args.workers == "auto" else int(args.workers)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    seed_override = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    scheme_override = [s for s in args.schemes.split(",") if s.strip() != ""]

    jobs = []
    for name, cfg in load_experiments(args.env):
        os.makedirs(os.path.join(EXP_ROOT, name, "results"), exist_ok=True)
        os.makedirs(os.path.join(EXP_ROOT, name, "logs"), exist_ok=True)
        seeds = seed_override or cfg.get("seeds", list(range(1, 11)))
        schemes = scheme_override or cfg.get("schemes",
                    ["uniform", "per", "euclid", "precond", "precond2"])
        steps = args.steps or cfg["total_steps"]
        for scheme, seed in itertools.product(schemes, seeds):
            jobs.append((name, cfg, scheme, seed, steps, None, None, args.nice))

    # round-robin GPUs across the flat job list
    if gpus:
        jobs = [j[:6] + (gpus[i % len(gpus)], j[7]) for i, j in enumerate(jobs)]

    where = f"gpus={gpus}" if gpus else "cpu"
    print(f"env={args.env} jobs={len(jobs)} workers={workers} {where} "
          f"nice={args.nice} cores={os.cpu_count()}", flush=True)

    # sweep-level progress bar: one tick per finished run (per-run tqdm is
    # disabled in the children -- their output is captured; watch individual
    # runs live with tensorboard --logdir experiments/<Env>/logs)
    pbar = None
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(jobs), desc=f"experiments[{args.env}]", unit="run",
                    dynamic_ncols=True, smoothing=0.0)
    except ImportError:
        pass

    n_done = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):        # report in completion order
            msg = fut.result()
            n_done += msg.startswith("done")
            n_skip += msg.startswith("skip")
            n_fail += msg.startswith("FAIL")
            if pbar is not None:
                pbar.write(msg)
                pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail, refresh=False)
                pbar.update(1)
            else:
                print(msg, flush=True)
    if pbar is not None:
        pbar.close()
    print(f"experiments complete: {n_done} done, {n_skip} skipped, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
