"""Aggregate SAC sweep results with rliable-style robust statistics.

Point estimator: interquartile mean (IQM).  Uncertainty: stratified bootstrap
(resample seeds within each environment, 95% percentile CIs) following Agarwal
et al., "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021).  The
stratified bootstrap and performance profiles are implemented natively in numpy
so the analysis has no fragile heavy dependencies; ``rliable.metrics.aggregate_iqm``
is used where available and matched by the native IQM otherwise.

Outputs (into ``--fig-dir``):
    learning_curves.png   per-env IQM return vs env steps, 95% CI shading
    aggregate_iqm.png     per-env-normalized aggregate IQM bar, 95% CI
    perf_profile.png      performance profiles (run-score distributions)
    compute_table.csv     wall-clock / gradient-eval accounting per scheme
    summary.json          IQM + CI per scheme, pairwise precond-vs-baseline deltas
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

SCHEME_ORDER = ["uniform", "per", "euclid", "precond", "precond2"]
SCHEME_LABEL = {
    "uniform": "uniform", "per": "PER |δ|", "euclid": "euclid ‖g‖₂",
    "precond": "precond ‖g‖_{D⁻¹} (ours)", "precond2": "precond2 ‖g‖_{D⁻²}",
}
SCHEME_COLOR = {
    "uniform": "#7f7f7f", "per": "#1f77b4", "euclid": "#2ca02c",
    "precond": "#d62728", "precond2": "#9467bd",
}


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    core = x[lo:hi]
    return float(core.mean()) if core.size else float(x.mean())


def stratified_bootstrap(scores: np.ndarray, stat_fn, n_boot=5000, seed=0):
    """scores: (num_seeds, num_tasks). Resample seeds within each task."""
    rng = np.random.default_rng(seed)
    num_seeds, num_tasks = scores.shape
    point = stat_fn(scores.ravel())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        res = np.empty_like(scores)
        for t in range(num_tasks):
            idx = rng.integers(0, num_seeds, num_seeds)
            res[:, t] = scores[idx, t]
        boot[b] = stat_fn(res.ravel())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if not r.get("eval_curve"):
            continue
        runs.append(r)
    return runs


def index_runs(runs):
    """-> data[env][scheme] = list of run dicts (one per seed)."""
    data = defaultdict(lambda: defaultdict(list))
    for r in runs:
        env = r["args"]["env_id"]
        scheme = r["args"]["scheme"]
        data[env][scheme].append(r)
    return data


def final_score_matrix(data, scheme):
    """Return (num_seeds, num_tasks) matrix of final eval returns for a scheme.

    Uses the common minimum seed count across envs so the matrix is rectangular.
    """
    envs = sorted(data.keys())
    per_env = []
    for env in envs:
        finals = [r["eval_curve"][-1]["eval_return"] for r in data[env].get(scheme, [])]
        per_env.append(finals)
    if not per_env or min(len(x) for x in per_env) == 0:
        return None, envs
    k = min(len(x) for x in per_env)
    mat = np.array([sorted(x)[:k] if False else x[:k] for x in per_env]).T  # (k, num_tasks)
    return mat, envs


def normalize_per_env(data):
    """Per-env min-max scaling constants over ALL runs' final scores."""
    norm = {}
    for env, schemes in data.items():
        vals = [r["eval_curve"][-1]["eval_return"]
                for s in schemes.values() for r in s]
        lo, hi = min(vals), max(vals)
        norm[env] = (lo, hi if hi > lo else lo + 1.0)
    return norm


# --------------------------------------------------------------------------- #
# Aggregate stats
# --------------------------------------------------------------------------- #
def compute_summary(data, n_boot=5000):
    envs = sorted(data.keys())
    norm = normalize_per_env(data)
    schemes = [s for s in SCHEME_ORDER if any(s in data[e] for e in envs)]

    agg = {}
    norm_mats = {}
    for scheme in schemes:
        per_env = []
        for env in envs:
            runs = data[env].get(scheme, [])
            lo, hi = norm[env]
            per_env.append([(r["eval_curve"][-1]["eval_return"] - lo) / (hi - lo) for r in runs])
        k = min((len(x) for x in per_env), default=0)
        if k == 0:
            continue
        mat = np.array([x[:k] for x in per_env]).T  # (k seeds, num_tasks)
        norm_mats[scheme] = mat
        point, clo, chi = stratified_bootstrap(mat, iqm, n_boot=n_boot)
        agg[scheme] = dict(iqm=point, ci_low=clo, ci_high=chi, n_seeds=int(k), n_envs=len(envs))

    # pairwise precond vs baselines: bootstrap the IQM difference
    pairwise = {}
    if "precond" in norm_mats:
        for base in ["uniform", "per", "euclid", "precond2"]:
            if base not in norm_mats:
                continue
            a, b = norm_mats["precond"], norm_mats[base]
            k = min(a.shape[0], b.shape[0])
            a, b = a[:k], b[:k]
            rng = np.random.default_rng(0)
            diffs = np.empty(n_boot)
            for i in range(n_boot):
                ra = np.empty_like(a); rb = np.empty_like(b)
                for t in range(a.shape[1]):
                    ia = rng.integers(0, k, k); ib = rng.integers(0, k, k)
                    ra[:, t] = a[ia, t]; rb[:, t] = b[ib, t]
                diffs[i] = iqm(ra.ravel()) - iqm(rb.ravel())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            pairwise[f"precond_minus_{base}"] = dict(
                delta_iqm=float(iqm(a.ravel()) - iqm(b.ravel())),
                ci_low=float(lo), ci_high=float(hi),
                prob_improve=float((diffs > 0).mean()),
            )
    return dict(envs=envs, aggregate=agg, pairwise=pairwise, norm=norm)


def compute_table(data):
    rows = []
    for scheme in SCHEME_ORDER:
        gevals, walls, sps = [], [], []
        for env in data:
            for r in data[env].get(scheme, []):
                gevals.append(r.get("total_grad_evals", np.nan))
                walls.append(r.get("total_wall_s", np.nan))
                if r["eval_curve"]:
                    sps.append(r["eval_curve"][-1].get("sps", np.nan))
        if gevals:
            rows.append(dict(scheme=scheme,
                             mean_grad_evals=float(np.nanmean(gevals)),
                             mean_wall_s=float(np.nanmean(walls)),
                             mean_sps=float(np.nanmean(sps))))
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_learning_curves(data, fig_dir):
    if not HAVE_MPL:
        return
    envs = sorted(data.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.2), squeeze=False)
    for j, env in enumerate(envs):
        ax = axes[0][j]
        for scheme in SCHEME_ORDER:
            runs = data[env].get(scheme, [])
            if not runs:
                continue
            steps = [e["step"] for e in runs[0]["eval_curve"]]
            L = min(len(r["eval_curve"]) for r in runs)
            steps = steps[:L]
            curves = np.array([[e["eval_return"] for e in r["eval_curve"][:L]] for r in runs])
            mid = np.array([iqm(curves[:, i]) for i in range(L)])
            # bootstrap CI over seeds at each step
            los, his = [], []
            rng = np.random.default_rng(0)
            for i in range(L):
                col = curves[:, i]
                bs = [iqm(col[rng.integers(0, len(col), len(col))]) for _ in range(500)]
                lo, hi = np.percentile(bs, [2.5, 97.5]); los.append(lo); his.append(hi)
            ax.plot(steps, mid, color=SCHEME_COLOR[scheme], label=SCHEME_LABEL[scheme], lw=1.8)
            ax.fill_between(steps, los, his, color=SCHEME_COLOR[scheme], alpha=0.15)
        ax.set_title(env); ax.set_xlabel("environment steps"); ax.set_ylabel("IQM eval return")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=130)
    plt.close(fig)


def plot_aggregate(summary, fig_dir):
    if not HAVE_MPL:
        return
    agg = summary["aggregate"]
    schemes = [s for s in SCHEME_ORDER if s in agg]
    y = np.arange(len(schemes))
    pts = [agg[s]["iqm"] for s in schemes]
    los = [agg[s]["iqm"] - agg[s]["ci_low"] for s in schemes]
    his = [agg[s]["ci_high"] - agg[s]["iqm"] for s in schemes]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(y, pts, xerr=[los, his], color=[SCHEME_COLOR[s] for s in schemes],
            alpha=0.85, capsize=4)
    ax.set_yticks(y); ax.set_yticklabels([SCHEME_LABEL[s] for s in schemes])
    ax.set_xlabel("per-env-normalized IQM (95% stratified bootstrap CI)")
    ax.grid(alpha=0.3, axis="x"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "aggregate_iqm.png"), dpi=130)
    plt.close(fig)


def plot_perf_profile(data, summary, fig_dir):
    if not HAVE_MPL:
        return
    envs = summary["envs"]; norm = summary["norm"]
    taus = np.linspace(0, 1.0, 60)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for scheme in SCHEME_ORDER:
        vals = []
        for env in envs:
            lo, hi = norm[env]
            for r in data[env].get(scheme, []):
                vals.append((r["eval_curve"][-1]["eval_return"] - lo) / (hi - lo))
        if not vals:
            continue
        vals = np.array(vals)
        frac = [(vals > t).mean() for t in taus]
        ax.plot(taus, frac, color=SCHEME_COLOR[scheme], label=SCHEME_LABEL[scheme], lw=1.8)
    ax.set_xlabel("normalized score τ"); ax.set_ylabel("fraction of runs > τ")
    ax.set_title("performance profiles"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "perf_profile.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/pilot")
    ap.add_argument("--fig-dir", default="results/pilot/figs")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    runs = load_runs(args.results_dir)
    if not runs:
        print(f"no runs found in {args.results_dir}")
        return
    data = index_runs(runs)
    summary = compute_summary(data, n_boot=args.n_boot)
    table = compute_table(data)
    summary["compute_table"] = table

    with open(os.path.join(args.fig_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # compute table csv
    import csv
    with open(os.path.join(args.fig_dir, "compute_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "mean_grad_evals", "mean_wall_s", "mean_sps"])
        w.writeheader()
        for row in table:
            w.writerow(row)

    plot_learning_curves(data, args.fig_dir)
    plot_aggregate(summary, args.fig_dir)
    plot_perf_profile(data, summary, args.fig_dir)

    # console summary
    print(f"\n=== runs loaded: {len(runs)} | envs: {summary['envs']} ===")
    print("\nAggregate per-env-normalized IQM (95% CI):")
    for s in SCHEME_ORDER:
        if s in summary["aggregate"]:
            a = summary["aggregate"][s]
            print(f"  {s:9s} IQM={a['iqm']:.3f}  CI=[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
                  f"  (n_seeds={a['n_seeds']}, n_envs={a['n_envs']})")
    print("\nPairwise precond - baseline (ΔIQM, 95% CI, P(improve)):")
    for k, v in summary["pairwise"].items():
        print(f"  {k:22s} Δ={v['delta_iqm']:+.3f}  CI=[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
              f"  P={v['prob_improve']:.2f}")
    print("\nCompute table:")
    for row in table:
        print(f"  {row['scheme']:9s} grad_evals={row['mean_grad_evals']:.0f}"
              f"  wall_s={row['mean_wall_s']:.0f}  sps={row['mean_sps']:.0f}")
    print(f"\nfigures + summary.json written to {args.fig_dir}")


if __name__ == "__main__":
    main()
