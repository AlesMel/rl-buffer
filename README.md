# rl-buffer — preconditioner-metric importance sampling for the SAC critic

Does sampling replay transitions in proportion to the per-sample critic
gradient norm **in Adam's preconditioner metric**, `p_i ∝ ‖g_i‖_{D⁻¹}`, improve
sample efficiency over (i) uniform replay, (ii) PER (`|δ|`), and (iii) Euclidean
gradient-norm sampling (`‖g_i‖₂`)?

This ports a supervised-learning result — that for a preconditioned optimizer
(Adam) the variance-optimal minibatch distribution weights examples by **one**
power of the inverse preconditioner `D⁻¹`, not `D⁰` (Euclidean) and not `D⁻²`
(double-counted) — to the off-policy RL setting, where the SAC critic minimises a
squared soft-Bellman residual with Adam over a replay buffer and prioritized
replay is already "non-uniform buffer sampling by a priority".

## The five sampling schemes (one switch: `--scheme`)

For the squared Bellman residual the per-sample gradient factorises (target `y_i`
is stop-gradient): `g_i = δ_i · ∇_θ Q(s_i,a_i)`, so every rule is `|δ_i|` times a
Jacobian-norm factor in some metric `M`:

| scheme     | priority `p_i ∝`                    | metric `M` | optimal for |
|------------|-------------------------------------|-----------|-------------|
| `uniform`  | `1`                                 | —         | — |
| `per`      | `|δ_i|`                             | (Jacobian dropped) | TD-error proxy (standard PER) |
| `euclid`   | `|δ_i|·‖∇Q_i‖₂`                     | `D⁰ = I`  | SGD (un-preconditioned) variance |
| `precond`  | `|δ_i|·‖∇Q_i‖_{D⁻¹}` **(ours)**     | `D⁻¹`     | **Adam (preconditioned) variance** |
| `precond2` | `|δ_i|·‖∇Q_i‖_{D⁻²}`                | `D⁻²`     | (metric sanity — should lose) |

`D = diag(√v̂ + ε)` is Adam's diagonal preconditioner (bias-corrected second
moment), read live from the optimizer state each update. PER is the special case
where the Jacobian norm is assumed constant across transitions; the gradient-norm
rules restore it in the appropriate metric.

## Correctness (non-negotiable)

- **Same IS weight on every loss.** Draw `i ∼ p`, set `w_i = 1/(N p_i)`, and apply
  the *identical* `w_i` to the critic, actor, **and** temperature per-sample
  losses. All three stay unbiased estimators of the uniform-buffer objectives, so
  prioritization changes only the critic estimator's **variance** — a clean test.
- **Shape safety.** `w` and every per-sample loss are 1-D `(B,)` tensors, asserted
  at every update (`assert w.shape == L_critic.shape == L_actor.shape ==
  L_temp.shape == (B,)`). This kills the `(B,)×(B,1)→(B,B)` broadcast bug that
  silently turns IS weighting into a global rescale.
- **Per-sample gradient norms without materialising per-sample grads.** Computed
  via the *weighted ghost-norm* trick (a diagonal-metric generalisation of
  Goodfellow 2015): for a linear layer `z = Wa+b` with backprop signal `s=∂Q/∂z`,
  `‖∇_W Q_i‖²_M = (s² · ((a²) @ R^T)).sum(1)` with `R = D^{-power}`. Validated
  against a `torch.func.vmap(grad)` ground truth for `M ∈ {D⁰, D⁻¹, D⁻²}`.
- **Weights.** `w_i = 1/(N p_i)`. PER's `w_i/max_j w_j` stabilisation (with `β→1`)
  is a rescale — **not** exactly unbiased — and is flagged as such.

## Priority maintenance

- **`--priority-mode lazy`** (default): sum-tree priorities per buffer entry,
  refreshed only when a transition is sampled; new entries get max priority.
  Scales `O(log N)`; priorities are **stale** (last-visit params) — the same
  approximation PER makes.
- **`--priority-mode two_stage`**: draw a uniform candidate pool (`pool_mult·B`),
  compute fresh priorities at current params, subsample `B` by priority
  (LaBER-style unbiased weighting, Lahire et al. 2022). Fresh priorities at
  `~pool_mult×` critic-gradient cost.

## Layout

```
rl_buffer/
  networks.py     squashed-Gaussian actor, twin soft-Q critics (CleanRL-faithful)
  buffers.py      sum-tree, uniform + prioritized ring buffer, lazy + two-stage
  priorities.py   Adam preconditioner readout + weighted ghost-norm + vmap truth
  sac.py          SAC training loop; the only knob is the sampling scheme
tests/            ghost-norm vs autograd, unbiasedness (critic+actor), shapes, sum-tree
analysis/         rliable-style IQM + stratified-bootstrap CIs + performance profiles
scripts/          run_sweep.py (parallel, idempotent), profile_priorities.py
logs/             TensorBoard event files, one dir per experiment (git-ignored)
WRITEUP.md        methodology, results with CIs, effect-size + caveats
```

## Quickstart

```bash
pip install -r requirements.txt

# unit tests (correctness gates)
PYTHONPATH=. python tests/test_priorities.py
PYTHONPATH=. python tests/test_unbiasedness.py
PYTHONPATH=. python tests/test_buffers.py

# single run (TensorBoard on by default -> logs/<experiment_name>)
PYTHONPATH=. python -m rl_buffer.sac --env-id HalfCheetah-v4 --scheme precond --seed 1
tensorboard --logdir logs        # losses, eval/return, and IS-weight diagnostics

# force CPU on a GPU box (small-net single-env SAC is usually faster on CPU)
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python -m rl_buffer.sac --scheme precond --seed 1

# reduced pilot on this box, then the full cluster protocol
python scripts/run_sweep.py --preset mini  --out-dir results/pilot   # 4-core CPU
python scripts/run_sweep.py --preset full  --out-dir results/full    # 4 envs × 10 seeds × 1M

# RECOMMENDED for a many-core box: CPU-only, one run per core, machine-friendly.
# The sweep is embarrassingly parallel across runs and CPU-bound (env sim is CPU),
# so cores beat GPUs here. 'auto' = cpu_count-2; --nice keeps the box responsive.
python scripts/run_sweep.py --preset full --workers auto --nice 10 --out-dir results/full

# (optional) pack runs across GPUs instead — usually NOT faster for this workload:
python scripts/run_sweep.py --preset full --gpus 0,1 --workers 16 --out-dir results/full

# aggregate + figures
PYTHONPATH=. python analysis/rliable_analysis.py --results-dir results/pilot --fig-dir results/pilot/figs
```

## Base and hyperparameters

Faithful fork of CleanRL's `sac_continuous_action.py` (2×256 ReLU MLPs, twin
critics, autotuned temperature). **Every** SAC hyperparameter is held at its
CleanRL default across all variants; the sampling scheme is the only thing that
changes. See `WRITEUP.md` for the effect-size result, the `D⁻¹`-vs-Euclidean/PER
comparison, and the staleness/compute caveats.
