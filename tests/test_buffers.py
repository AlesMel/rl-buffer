"""Sum-tree sampling distribution and buffer bookkeeping."""
import numpy as np

from rl_buffer.buffers import ReplayBuffer, SamplingConfig, SumTree


def test_sumtree_sampling_distribution():
    cap = 1000
    tree = SumTree(cap)
    rng = np.random.default_rng(0)
    prios = rng.uniform(0.1, 5.0, size=cap)
    for i, v in enumerate(prios):
        tree.set(i, float(v))
    assert abs(tree.total() - prios.sum()) < 1e-6

    # empirical sampling frequency should track normalized priorities
    counts = np.zeros(cap)
    n = 200_000
    total = tree.total()
    for _ in range(n):
        counts[tree.find(rng.random() * total)] += 1
    emp = counts / n
    target = prios / prios.sum()
    # correlation close to 1; a handful of bins may be noisy
    assert np.corrcoef(emp, target)[0, 1] > 0.99


def test_uniform_buffer_weights_are_one():
    cfg = SamplingConfig(scheme="uniform")
    buf = ReplayBuffer(500, 4, 2, cfg, seed=0)
    for _ in range(300):
        buf.add(np.zeros(4), np.zeros(2), 0.0, np.zeros(4), 0.0)
    idxs, batch, w = buf.sample_uniform(64)
    assert w.shape == (64,)
    assert np.allclose(w.numpy(), 1.0)
    assert batch["obs"].shape == (64, 4)


def test_two_stage_laber_weight_unbiased_mean():
    """LaBER weight w_i = mean(prio)/prio_i makes E[w_i * f_i] over the pool draw
    equal the pool mean of f_i for any f (here f = a linear score)."""
    cfg = SamplingConfig(scheme="euclid", priority_mode="two_stage", alpha=1.0, eps_priority=0.0)
    buf = ReplayBuffer(1000, 3, 1, cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        buf.add(rng.normal(size=3), rng.normal(size=1), 0.0, np.zeros(3), 0.0)

    pool_idxs = np.arange(200)
    prio = rng.uniform(0.2, 3.0, size=200)
    f = rng.normal(size=200)                       # arbitrary per-sample scalar
    pool_mean = f.mean()

    est = 0.0
    draws = 40000
    for _ in range(draws):
        _, _, w, sel = buf.subsample_from_pool(pool_idxs, prio, batch_size=1)
        est += (w.numpy()[0] * f[sel[0]])
    est /= draws
    assert abs(est - pool_mean) < 0.05, (est, pool_mean)


def test_global_norm_is_batch_independent():
    """Global normalisation divides by a per-dataset constant, so a given
    transition gets the SAME weight in every batch (a pure rescale => unbiased up
    to scale). Batch-max normalisation divides by a sample-correlated quantity, so
    the same transition gets DIFFERENT weights across batches (the bias source)."""
    from collections import defaultdict
    for mode, expect_constant in [("clip", True), ("global", True), ("batch", False)]:
        cfg = SamplingConfig(scheme="per", alpha=1.0, beta0=1.0, beta1=1.0,
                             normalize_mode=mode)
        buf = ReplayBuffer(8, 3, 1, cfg, seed=0)
        for _ in range(8):
            buf.add(np.zeros(3), np.zeros(1), 0.0, np.zeros(3), 0.0)
        # distinct priorities so the per-batch max varies across draws
        buf.update_priorities(np.arange(8), np.arange(1.0, 9.0))
        wm = defaultdict(list)
        for _ in range(300):
            idxs, _, w = buf.sample_lazy(4, step=10_000)   # beta==1 here
            for i, wi in zip(idxs, w.numpy()):
                wm[int(i)].append(float(wi))
        max_std = max(np.std(v) for v in wm.values() if len(v) > 3)
        if expect_constant:
            assert max_std < 1e-6, f"global: per-index weight should be constant, std={max_std:.2e}"
        else:
            assert max_std > 1e-3, f"batch: per-index weight should vary, std={max_std:.2e}"


if __name__ == "__main__":
    test_sumtree_sampling_distribution()
    test_uniform_buffer_weights_are_one()
    test_two_stage_laber_weight_unbiased_mean()
    test_global_norm_is_batch_independent()
    print("test_buffers OK")
