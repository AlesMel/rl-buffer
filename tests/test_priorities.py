"""Ghost-norm per-sample gradient norms must match an autograd ground truth."""
import numpy as np
import torch

from rl_buffer.networks import SoftQNetwork
from rl_buffer.priorities import GhostNormCalculator, per_sample_grad_norm_vmap, per_sample_priority


def _random_D(critic, rng):
    """A synthetic positive diagonal preconditioner keyed by parameter."""
    D = {}
    for p in critic.parameters():
        D[p] = torch.tensor(rng.uniform(0.1, 3.0, size=tuple(p.shape)), dtype=torch.float32)
    return D


def test_ghost_norm_matches_vmap():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs_dim, act_dim, B = 11, 3, 64
    critic = SoftQNetwork(obs_dim, act_dim)
    s = torch.randn(B, obs_dim)
    a = torch.randn(B, act_dim)
    D = _random_D(critic, rng)

    for power in (0, 1, 2):
        ghost = GhostNormCalculator(critic, power).grad_norm(s, a, D)
        truth = per_sample_grad_norm_vmap(critic, s, a, D, power)
        assert ghost.shape == (B,)
        assert torch.allclose(ghost, truth, rtol=1e-4, atol=1e-5), (
            f"power={power} max abs diff {(ghost - truth).abs().max().item():.2e}")


def test_priority_factorization():
    """||g_i||_M = |delta_i| * ||grad Q_i||_M for a scalar squared-residual loss."""
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    obs_dim, act_dim, B = 7, 2, 32
    critic = SoftQNetwork(obs_dim, act_dim)
    s = torch.randn(B, obs_dim)
    a = torch.randn(B, act_dim)
    delta = torch.randn(B)
    D = _random_D(critic, rng)

    for power in (0, 1, 2):
        prio = per_sample_priority(critic, delta, s, a, D, power)
        jac = per_sample_grad_norm_vmap(critic, s, a, D, power)
        assert torch.allclose(prio, delta.abs() * jac, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_ghost_norm_matches_vmap()
    test_priority_factorization()
    print("test_priorities OK")
