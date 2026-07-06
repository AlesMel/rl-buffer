"""Profile the per-sample gradient-norm computation.

Confirms the ghost-norm trick does NOT materialise per-sample gradients and
measures its cost relative to a plain critic update, and versus the
torch.func.vmap ground-truth path (which does materialise B x P grads).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from rl_buffer.networks import SoftQNetwork
from rl_buffer.priorities import (GhostNormCalculator, adam_diag_preconditioner,
                                  per_sample_grad_norm_vmap)


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    torch.set_num_threads(1)
    obs_dim, act_dim = 17, 6           # HalfCheetah
    for B in (256, 1024):
        critic = SoftQNetwork(obs_dim, act_dim)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        s = torch.randn(B, obs_dim); a = torch.randn(B, act_dim); y = torch.randn(B, 1)
        # populate Adam state
        (critic(s, a).pow(2).mean()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        D = adam_diag_preconditioner(opt, list(critic.parameters()))
        n_params = sum(p.numel() for p in critic.parameters())

        def plain_update():
            opt.zero_grad(set_to_none=True)
            loss = (critic(s, a) - y).pow(2).mean()
            loss.backward(); opt.step()

        ghost = GhostNormCalculator(critic, metric_power=1)
        def ghost_norm():
            return ghost.grad_norm(s, a, D)

        def vmap_norm():
            return per_sample_grad_norm_vmap(critic, s, a, D, metric_power=1)

        t_plain = bench(plain_update)
        t_ghost = bench(ghost_norm)
        t_vmap = bench(vmap_norm, iters=20)
        print(f"B={B:5d}  params={n_params}  "
              f"plain_update={t_plain:.2f}ms  ghost_norm={t_ghost:.2f}ms "
              f"({t_ghost/t_plain:.2f}x)  vmap_norm={t_vmap:.2f}ms ({t_vmap/t_plain:.2f}x)")
        # memory: ghost never allocates a (B, n_params) tensor; vmap does.
        print(f"        vmap materialises B*params = {B*n_params/1e6:.1f}M floats "
              f"(~{B*n_params*4/1e6:.0f} MB); ghost materialises O(B*width) only.")


if __name__ == "__main__":
    main()
