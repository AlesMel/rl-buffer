# Experiments

One folder per environment; each is self-contained:

```
experiments/<Env>/
  config.json    env id + gym.make kwargs, steps, seeds, schemes, priority mode
  results/       one JSON per run (curve, args, compute counters)  [git-ignored]
  logs/          TensorBoard event dirs, one per run               [git-ignored]
  figs/          rliable figures + summary.json (after analysis)
```

| folder          | env                          | steps | note |
|-----------------|------------------------------|-------|------|
| Pendulum-v1     | Pendulum-v1                  | 100k  | cheap sanity / pipeline smoke |
| LunarLander-v3  | LunarLander-v3 (continuous)  | 500k  | Box2D; needs `gymnasium[box2d]` |
| Hopper-v4       | Hopper-v4                    | 1M    | unstable, early termination |
| HalfCheetah-v4  | HalfCheetah-v4               | 1M    | no early termination; pilot env |
| Walker2d-v4     | Walker2d-v4                  | 1M    | mid difficulty |
| Ant-v4          | Ant-v4                       | 1M    | high-dim, contact-rich |
| Humanoid-v4     | Humanoid-v4                  | 3M    | hardest; 3M per the protocol |

## Run

```bash
# one env (all 5 schemes x seeds from its config.json)
python scripts/run_experiments.py --env HalfCheetah-v4 --workers auto --nice 10

# everything
python scripts/run_experiments.py --env all --workers auto --nice 10

# quick partial pass (overrides config)
python scripts/run_experiments.py --env LunarLander-v3 --seeds 1,2,3 --steps 100000
```

Runs are idempotent (existing result JSONs are skipped), so re-running a folder
adds missing seeds / recovers from interrupts. Watch live:
`tensorboard --logdir experiments/<Env>/logs` (or `--logdir experiments` for all).

## Analyze

```bash
# per-env
python analysis/rliable_analysis.py \
  --results-dir experiments/HalfCheetah-v4/results \
  --fig-dir     experiments/HalfCheetah-v4/figs

# aggregate across all envs (per-env-normalized IQM, stratified bootstrap)
python analysis/rliable_analysis.py \
  --results-dir experiments/*/results \
  --fig-dir     experiments/figs_all
```

## Notes

- SAC needs continuous actions, so LunarLander runs in continuous mode via
  `"env_kwargs": {"continuous": true}` (any `gym.make` kwarg works there).
- All SAC hyperparameters stay at the fixed known-good config across variants;
  the folders vary only env + budget. Do not tune per variant.
- To add an environment: `mkdir experiments/<Env>` and drop in a `config.json`
  (copy one of these); the runner picks it up under `--env <Env>` / `--env all`.
