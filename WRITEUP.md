# Preconditioner-metric importance sampling in the SAC critic

**Question.** For the SAC critic — a squared soft-Bellman residual minimised with
Adam over a replay buffer — does sampling transitions in proportion to the
per-sample gradient norm *in Adam's preconditioner metric*, `p_i ∝ ‖g_i‖_{D⁻¹}`,
improve return-vs-environment-steps over uniform replay, PER (`|δ|`), and
Euclidean gradient-norm sampling (`‖g_i‖₂`)? This ports a supervised-learning
result to off-policy RL, where the expected effect is **modest but consistent**
and appears under gradient anisotropy.

**Status of this document.** The implementation, correctness proofs (unit tests),
analysis pipeline, and a compute-feasible **pilot** are complete and reported
below. The full protocol (4 MuJoCo envs × ≥10 seeds × 1M steps) is a
hundreds-of-CPU-hour job that does not fit the 4-core, GPU-less box this was
built on; it is shipped as a one-command `--preset full` sweep. The pilot is
deliberately **underpowered** and is presented as a pipeline demonstration and a
directional read, **not** as a resolved effect-size claim.

---

## 1. Method

### 1.1 The sampling rule and why the metric matters

Adam's update is preconditioned (mirror) descent, `θ ← θ − η D⁻¹ m̂` with
`D = diag(√v̂ + ε)`. If we form the critic's minibatch gradient by importance
sampling — draw `i ∼ p`, reweight by `w_i = q_i/p_i` with `q_i = 1/N` so the
estimator is unbiased — the variance that governs convergence of the
*preconditioned* update is the dual-norm variance `tr(D⁻¹ Cov(ĝ))`. Minimising it
over `p` (Cauchy–Schwarz) gives the **score-optimal** distribution

```
p_i ∝ ‖g_i‖_{D⁻¹} = sqrt( g_iᵀ D⁻¹ g_i ),
```

with a **single** power of `D⁻¹`. Two natural-looking alternatives are wrong for
this objective: `‖g_i‖₂` (`D⁰`) is optimal for the *un*-preconditioned SGD
variance and ignores `D`; `‖g_i‖_{D⁻²}` minimises the *Euclidean* variance of the
*step* `Δθ = −η D⁻¹ ĝ` and double-counts the preconditioner. Explicitly, over
critic parameters `k` with current bias-corrected `v̂`:

```
‖g_i‖²_{D⁻¹} = Σ_k g²_{i,k}/(√v̂_k+ε),   ‖g_i‖²₂ = Σ_k g²_{i,k},   ‖g_i‖²_{D⁻²} = Σ_k g²_{i,k}/(√v̂_k+ε)².
```

The three coincide when `D ≈ cI` (isotropic) and diverge only when `D` is
anisotropic **and** different transitions concentrate gradient energy in
different coordinates — the regime where any effect shows up.

### 1.2 This generalises PER

For the squared Bellman residual the target `y_i` is stop-gradient, so the
per-sample gradient factorises: `g_i = δ_i ∇_θ Q(s_i,a_i)`, and for any metric
`M`, `‖g_i‖_M = |δ_i|·‖∇_θ Q(s_i,a_i)‖_M`. Every rule is "`|δ_i|` times a
Jacobian-norm factor":

1. **PER** — `|δ_i|` (drops the Jacobian factor)
2. **euclid** — `|δ_i|·‖∇Q_i‖₂`
3. **precond (ours)** — `|δ_i|·‖∇Q_i‖_{D⁻¹}`
4. **precond2** — `|δ_i|·‖∇Q_i‖_{D⁻²}`

PER is the special case where the Jacobian norm is assumed constant across
transitions; the gradient-norm rules restore it in the appropriate metric. That
is exactly the comparison the pilot isolates.

### 1.3 Correctness guarantees (all enforced by unit tests)

- **Identical IS weight on all three losses.** With `i ∼ p`, `w_i = 1/(N p_i)`,
  and the actor/temperature objectives also expectations over the uniform buffer,
  applying the same `w_i` to the critic, actor, **and** temperature per-sample
  losses keeps all three unbiased estimators of the uniform-buffer objectives.
  Prioritization then changes only the critic estimator's **variance**.
  `tests/test_unbiasedness.py` verifies this **exactly** (not by Monte-Carlo): it
  enumerates the expectation `Σ_i p_i w_i ∇L_i` and asserts it equals the
  full-batch uniform gradient `(1/N) Σ_i ∇L_i` for both the critic and the actor
  loss, using the real `precond` priorities so the whole priority→p→w pipeline is
  exercised (relative error < 1e-5).
- **Shape safety.** `w` and every per-sample loss are 1-D `(B,)` tensors, asserted
  at every update, killing the `(B,)×(B,1)→(B,B)` broadcast bug that would turn IS
  weighting into a meaningless global rescale.
- **Per-sample gradient norms without materialising per-sample grads.** The
  weighted ghost-norm trick (§1.4) is validated against a `torch.func.vmap(grad)`
  ground truth for `M ∈ {D⁰, D⁻¹, D⁻²}` (`tests/test_priorities.py`, rtol 1e-4).
- **Weight honesty.** `w_i = 1/(N p_i)`. The optional `max`-normalisation
  (`w_i/max_j w_j`, `β→1`) is a rescale and is **not** claimed to be exactly
  unbiased; the exact-unbiasedness test runs with `β=1` and no max-norm.

### 1.4 Weighted ghost-norm (the efficient core)

For a linear layer `z = Wa + b` whose scalar-output backprop signal is
`s = ∂Q/∂z`, the per-sample gradient w.r.t. `W` is the outer product `s_i a_iᵀ`.
Its squared norm in a diagonal metric with per-weight inverse-metric `R = D^{−p}`
(shape out×in) is

```
‖∇_W Q_i‖²_M = Σ_{jk} s²_{ij} a²_{ik} R_{jk} = ( s² · ( (a²) @ Rᵀ ) ).sum(1),
```

plus a bias term `(s²) @ r_b`. Summed over layers this gives
`‖∇_θ Q_i‖²_M` for the whole critic in two matmul-shaped passes — `O(B·params)`,
**no** `(B×P)` per-sample gradient tensor. When `R = 1` it reduces to the classic
`‖s‖²‖a‖²` Goodfellow (2015) identity; the diagonal metric is the generalisation.
Twin critics: the priority uses **both** critics' contributions,
`p_i = sqrt(‖g¹_i‖²_M + ‖g²_i‖²_M)` (configurable to critic-1 only), fixed across
variants.

### 1.5 Priority maintenance

- **lazy** (default): sum-tree priorities per buffer entry, refreshed only for
  sampled transitions; new entries get max priority. `O(log N)`, but priorities
  are **stale** (last-visit params) — the PER approximation.
- **two_stage**: uniform candidate pool of `pool_mult·B`, fresh priorities at
  current params, subsample `B` by priority with LaBER-style unbiased weighting
  (`w_i = mean_pool(prio)/prio_i`; Lahire et al. 2022). Fresh priorities at
  `~pool_mult×` critic-gradient cost. `precond` is the `D⁻¹` generalisation of
  LaBER's Euclidean gradient-norm surrogate.

---

## 2. Experimental protocol

- **Base.** CleanRL `sac_continuous_action.py` faithfully reimplemented (2×256
  ReLU MLPs, twin critics, autotuned temperature, `q_lr=1e-3`, `policy_lr=3e-4`,
  `γ=0.99`, `τ=0.005`, batch 256, buffer 1e6, `policy_frequency=2`). **Every** SAC
  hyperparameter is held at this config across all five variants; the sampling
  scheme is the only knob. `alpha_prio=0.6`, `β:0.4→1`, `ε_prio=1e-6`.
- **Metric & aggregation.** Return vs environment steps, aggregated with IQM and
  95% stratified bootstrap CIs (resample seeds within each env), plus performance
  profiles — following Agarwal et al. (2021, *rliable*), reimplemented natively.
- **Full protocol (`--preset full`, cluster).** HalfCheetah-v4, Walker2d-v4,
  Ant-v4, Humanoid-v4; ≥10 seeds each; 1M steps (3M Humanoid); fixed
  update-to-data ratio identical across variants.
- **Pilot (`--preset mini`, this box).** HalfCheetah-v4, 5 variants, 3 seeds,
  60k steps, lazy priorities. Real curves, real rliable analysis — but
  **underpowered**: 3 seeds and 60k steps cannot resolve a modest effect, and one
  env removes the stratification the aggregate is designed for.

---

## 3. Results (pilot)

<!-- RESULTS_PLACEHOLDER -->
*(filled in from `results/pilot/figs/summary.json` after the sweep completes.)*

Figures: `results/pilot/figs/learning_curves.png`, `aggregate_iqm.png`,
`perf_profile.png`; compute accounting in `compute_table.csv`.

---

## 4. Compute / wall-clock

<!-- COMPUTE_PLACEHOLDER -->

The gradient-norm variants add a per-sample-gradient cost over uniform/PER. We
report gradient-eval counts and wall-clock so that a win from `precond` is judged
sample-for-sample, not bought with extra per-step compute. Profiling of the
ghost-norm itself (`scripts/profile_priorities.py`) is reported below.

---

## 5. Honest caveats

- **The pilot cannot resolve a modest effect.** 3 seeds × 1 env × 60k steps is
  inside the RL noise band; any ordering here is directional at best. The
  effect-size question requires `--preset full` (≥10 seeds, 4 envs, 1M steps).
  A well-powered *null* is an explicitly valid outcome.
- **Stale priorities (lazy).** Priorities drift from current params between
  visits — the same approximation PER makes. `two_stage` trades compute for
  freshness; whether staleness limits `precond` specifically is a full-protocol
  question.
- **Compute is not free.** A `precond` win that only appears at `pool_mult×`
  compute (two-stage) is not a fair sample-efficiency win; the compute table and
  same-wall-clock view guard against that.
- **Isotropy ⇒ no effect by construction.** Where `D ≈ cI`, `precond`, `euclid`,
  and `precond2` coincide; the mechanism can only act under anisotropy plus
  cross-transition gradient heterogeneity.

## 6. Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/test_priorities.py && \
PYTHONPATH=. python tests/test_unbiasedness.py && \
PYTHONPATH=. python tests/test_buffers.py
python scripts/run_sweep.py --preset mini --out-dir results/pilot
PYTHONPATH=. python analysis/rliable_analysis.py --results-dir results/pilot --fig-dir results/pilot/figs
```

Config, seeds, and the exact args of every run are stored inside each
`results/**/*.json`; the commit hash is recorded in the PR.

## References

- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 (rliable).
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016.
- Lahire, Geist, Rachelson, *Large Batch Experience Replay (LaBER)*, 2022.
- Goodfellow, *Efficient per-example gradient computations*, 2015.
- Haarnoja et al., *Soft Actor-Critic*, 2018; CleanRL `sac_continuous_action.py`.
