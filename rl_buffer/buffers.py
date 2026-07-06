"""Replay buffers and sampling schemes.

Storage is a flat ring buffer of transitions.  The *only* thing that varies
across experimental variants is how a minibatch of indices is drawn and what
importance weight is attached to each draw.

Sampling schemes (``SamplingConfig.scheme``):
    uniform  : p_i propto 1
    per      : p_i propto |delta_i|^alpha           (TD-error proxy; PER)
    euclid   : p_i propto ||g_i||_2^alpha           (D^0 gradient norm)
    precond  : p_i propto ||g_i||_{D^-1}^alpha       (ours)
    precond2 : p_i propto ||g_i||_{D^-2}^alpha       (metric sanity)

Priority maintenance:
    lazy      : sum-tree priorities per entry, refreshed only for sampled
                transitions (stale, PER-style, scalable).
    two_stage : draw a uniform candidate pool of size ``pool_mult * B``, compute
                fresh priorities on the pool, subsample B by priority.  Fresh
                priorities at ~pool_mult x critic-gradient cost.  LaBER-style
                weighting (Lahire et al. 2022).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Sum tree (O(log N) prioritized sampling), used by the lazy scheme.
# --------------------------------------------------------------------------- #
class SumTree:
    """Fixed-capacity binary sum tree over per-leaf priorities."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)  # 1-indexed heap

    def total(self) -> float:
        return float(self.tree[1])

    def set(self, idx: int, value: float):
        i = idx + self.capacity
        delta = value - self.tree[i]
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def set_batch(self, idxs: np.ndarray, values: np.ndarray):
        for i, v in zip(idxs, values):
            self.set(int(i), float(v))

    def get(self, idx: int) -> float:
        return float(self.tree[idx + self.capacity])

    def find(self, prefix: float) -> int:
        """Return leaf index whose cumulative range contains ``prefix``."""
        i = 1
        while i < self.capacity:
            left = 2 * i
            if prefix <= self.tree[left]:
                i = left
            else:
                prefix -= self.tree[left]
                i = left + 1
        return i - self.capacity

    def max_leaf(self, size: int) -> float:
        if size == 0:
            return 1.0
        m = float(self.tree[self.capacity:self.capacity + size].max())
        return m if m > 0 else 1.0


class MinTree:
    """Fixed-capacity binary min tree; tracks the minimum leaf priority in O(log N).

    Used to compute the *global* smallest sampling probability p_min, hence the
    buffer-wide maximum IS weight, so weights can be normalised by a per-dataset
    constant (unbiased up to scale) rather than the per-batch max (biased).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.full(2 * capacity, np.inf, dtype=np.float64)

    def set(self, idx: int, value: float):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def set_batch(self, idxs: np.ndarray, values: np.ndarray):
        for i, v in zip(idxs, values):
            self.set(int(i), float(v))

    def min(self) -> float:
        return float(self.tree[1])


@dataclass
class SamplingConfig:
    scheme: str = "uniform"          # uniform | per | euclid | precond | precond2
    alpha: float = 0.6               # priority exponent (PER default)
    beta0: float = 0.4               # IS-weight exponent start (lazy scheme)
    beta1: float = 1.0               # IS-weight exponent end (anneal to 1)
    total_anneal_steps: int = 1_000_000
    eps_priority: float = 1e-6       # floor so p keeps full support
    priority_mode: str = "lazy"      # lazy | two_stage
    pool_mult: int = 4               # candidate-pool multiplier for two_stage
    # IS-weight normalisation for the lazy scheme:
    #   global : divide by the buffer-wide max weight (per-dataset constant) ->
    #            a pure learning-rate rescale, UNBIASED up to scale (recommended)
    #   batch  : divide by the per-batch max (PER's classic trick) -> BIASED,
    #            because the normaliser is correlated with the sampled batch
    #   none   : raw w_i = (N p_i)^{-beta} -> exactly unbiased at beta=1, higher variance
    normalize_mode: str = "global"   # global | batch | none
    metric_power: int = field(init=False, default=0)

    def __post_init__(self):
        self.metric_power = {"euclid": 0, "precond": 1, "precond2": 2}.get(self.scheme, 0)

    @property
    def is_gradient_scheme(self) -> bool:
        return self.scheme in ("euclid", "precond", "precond2")

    @property
    def is_prioritized(self) -> bool:
        return self.scheme != "uniform"


class ReplayBuffer:
    """Flat ring buffer with pluggable sampling scheme."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 cfg: SamplingConfig, device: str = "cpu", seed: int = 0):
        self.capacity = capacity
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

        # lazy-scheme sum tree over p_i = (priority + eps)^alpha, plus a min tree
        # so we can normalise by the buffer-wide (global) max IS weight.
        lazy_prio = cfg.is_prioritized and cfg.priority_mode == "lazy"
        self.tree = SumTree(capacity) if lazy_prio else None
        self.min_tree = MinTree(capacity) if lazy_prio else None
        self._max_prio = 1.0

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        if self.tree is not None:
            if self.min_tree is not None:
                self.min_tree.set(i, self._max_prio ** self.cfg.alpha)
            # new transitions get max priority so they are seen at least once
            self.tree.set(i, self._max_prio ** self.cfg.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # --- tensor helpers ----------------------------------------------------
    def _to_t(self, arr):
        return torch.as_tensor(arr, device=self.device)

    def _gather(self, idxs):
        return {
            "obs": self._to_t(self.obs[idxs]),
            "actions": self._to_t(self.actions[idxs]),
            "rewards": self._to_t(self.rewards[idxs]),
            "next_obs": self._to_t(self.next_obs[idxs]),
            "dones": self._to_t(self.dones[idxs]),
        }

    def _beta(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.cfg.total_anneal_steps))
        return self.cfg.beta0 + frac * (self.cfg.beta1 - self.cfg.beta0)

    # --- sampling ----------------------------------------------------------
    def sample_uniform(self, batch_size: int):
        idxs = self.rng.integers(0, self.size, size=batch_size)
        batch = self._gather(idxs)
        w = torch.ones(batch_size, device=self.device)
        return idxs, batch, w

    def sample_lazy(self, batch_size: int, step: int):
        """Sum-tree stratified sampling with p_i propto priority_i (alpha baked in)."""
        assert self.tree is not None
        total = self.tree.total()
        idxs = np.empty(batch_size, dtype=np.int64)
        seg = total / batch_size
        for j in range(batch_size):
            prefix = (self.rng.random() + j) * seg
            idxs[j] = self.tree.find(prefix)
        # p_i and IS weights w_i = (N p_i)^{-beta}
        p = np.array([self.tree.get(int(i)) for i in idxs], dtype=np.float64) / max(total, 1e-12)
        beta = self._beta(step)
        w = (self.size * p) ** (-beta)
        mode = self.cfg.normalize_mode
        if mode == "global":
            # divide by the buffer-wide max weight = (N * p_min)^{-beta}. This is a
            # per-dataset constant (independent of the drawn batch), so it is a pure
            # learning-rate rescale -- unbiased up to scale -- while still bounding
            # every weight to (0, 1]. Contrast batch-max, which is sample-correlated.
            p_min = self.min_tree.min() / max(total, 1e-12)
            if np.isfinite(p_min) and p_min > 0:
                w_max = (self.size * p_min) ** (-beta)
                w = np.minimum(w / w_max, 1.0)
            else:                     # degenerate buffer -> fall back to batch max
                w = w / w.max()
        elif mode == "batch":
            w = w / w.max()
        # mode == "none": raw weights (exactly unbiased at beta=1)
        batch = self._gather(idxs)
        return idxs, batch, torch.as_tensor(w, dtype=torch.float32, device=self.device)

    def draw_pool(self, batch_size: int):
        """Uniformly draw a candidate pool of size pool_mult*B (two-stage)."""
        pool_size = min(self.cfg.pool_mult * batch_size, self.size)
        idxs = self.rng.integers(0, self.size, size=pool_size)
        return idxs, self._gather(idxs)

    def subsample_from_pool(self, pool_idxs: np.ndarray, priorities: np.ndarray, batch_size: int):
        """LaBER-style: sample B from pool with p propto priority, weight = mean_prio/prio.

        This is an unbiased estimator of the uniform-buffer minibatch gradient:
        E_pool E_{i~p}[ w_i g_i ] = mean over pool of g_i, and the pool is uniform.
        """
        prio = priorities.astype(np.float64) + self.cfg.eps_priority
        prio_a = prio ** self.cfg.alpha
        p = prio_a / prio_a.sum()
        sel = self.rng.choice(len(pool_idxs), size=batch_size, replace=True, p=p)
        chosen = pool_idxs[sel]
        # LaBER weight: mean(prio_a)/prio_a_i  (exactly unbiased for the pool mean)
        w = prio_a.mean() / prio_a[sel]
        batch = self._gather(chosen)
        return chosen, batch, torch.as_tensor(w, dtype=torch.float32, device=self.device), sel

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray):
        """Write refreshed priorities (lazy scheme). Applies eps floor + alpha."""
        if self.tree is None:
            return
        prio = np.abs(priorities.astype(np.float64)) + self.cfg.eps_priority
        self._max_prio = max(self._max_prio, float(prio.max()))
        prio_a = prio ** self.cfg.alpha
        self.tree.set_batch(idxs, prio_a)
        if self.min_tree is not None:
            self.min_tree.set_batch(idxs, prio_a)
