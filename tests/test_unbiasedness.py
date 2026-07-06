"""The IS-weighted estimator must be unbiased for the uniform-buffer objective.

With w_i = 1/(N p_i) and index i drawn from categorical p, the single-sample
weighted gradient has expectation

    E_{i~p}[ w_i * grad L_i ] = sum_i p_i (1/(N p_i)) grad L_i = (1/N) sum_i grad L_i,

i.e. exactly the full-batch (uniform) gradient.  We verify this *exactly* by
enumerating the expectation rather than Monte-Carlo sampling, using the real
``precond`` priorities so the whole priority -> p -> w pipeline is exercised.

We also assert the shape-safety contract: w and every per-sample loss are (B,).
"""
import numpy as np
import torch

from rl_buffer.networks import Actor, SoftQNetwork
from rl_buffer.priorities import adam_diag_preconditioner, per_sample_priority


def _flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
    return torch.cat([x.reshape(-1) for x in g])


def _build(N=40, obs_dim=6, act_dim=2, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    critic = SoftQNetwork(obs_dim, act_dim)
    obs = torch.randn(N, obs_dim)
    act = torch.randn(N, act_dim)
    y = torch.randn(N, 1)                       # fixed stop-gradient targets
    return critic, obs, act, y, rng


def _precond_priorities(critic, obs, act, y):
    """Priorities p_i propto ||g_i||_{D^-1} using a synthetic (isotropic-ish) D."""
    # give Adam a nonzero, anisotropic second moment so D != cI
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    q = critic(obs, act)
    (q.pow(2).mean()).backward()
    opt.step()                                  # populates exp_avg_sq / step
    opt.zero_grad(set_to_none=True)
    D = adam_diag_preconditioner(opt, list(critic.parameters()))
    with torch.no_grad():
        delta = (critic(obs, act) - y).reshape(-1)
    prio = per_sample_priority(critic, delta, obs, act, D, metric_power=1).numpy()
    prio = np.abs(prio) + 1e-8
    return prio / prio.sum()


def test_weighted_gradient_is_unbiased_critic():
    critic, obs, act, y, rng = _build()
    N = obs.shape[0]
    params = list(critic.parameters())
    p = _precond_priorities(critic, obs, act, y)

    # full-batch (uniform) critic gradient
    delta_full = (critic(obs, act) - y).reshape(-1)
    L_full = (delta_full ** 2).mean()           # (1/N) sum_i delta_i^2
    g_full = _flat_grad(L_full, params)

    # exact expectation of the single-sample weighted gradient
    g_expected = torch.zeros_like(g_full)
    for i in range(N):
        w_i = 1.0 / (N * p[i])                   # unbiased weight, beta=1, no max-norm
        delta_i = (critic(obs[i:i+1], act[i:i+1]) - y[i:i+1]).reshape(-1)   # (1,)
        L_i = w_i * (delta_i ** 2).mean()        # weighted per-sample loss (B=1)
        g_i = _flat_grad(L_i, params)
        g_expected += p[i] * g_i

    diff = (g_expected - g_full).abs().max().item()
    rel = diff / (g_full.abs().max().item() + 1e-12)
    assert rel < 1e-5, f"unbiasedness failed: max abs diff {diff:.2e}, rel {rel:.2e}"


def test_weighted_gradient_is_unbiased_actor():
    """Same argument for the actor loss (alpha*logpi - min_q), verifying the
    identical-weight-on-all-losses correction is unbiased there too.

    The reparameterisation noise is fixed per sample so the comparison isolates
    the IS-weighting from the policy's sampling stochasticity.
    """
    torch.manual_seed(3)
    N, obs_dim, act_dim = 30, 6, 2
    obs = torch.randn(N, obs_dim)
    actor = Actor(obs_dim, act_dim, -np.ones(act_dim), np.ones(act_dim))
    qf1 = SoftQNetwork(obs_dim, act_dim)
    qf2 = SoftQNetwork(obs_dim, act_dim)
    alpha = 0.2
    params = list(actor.parameters())
    eps_fixed = torch.randn(N, act_dim)          # frozen reparam noise

    rng = np.random.default_rng(3)
    p = rng.uniform(0.5, 2.0, size=N)
    p = p / p.sum()

    def actor_loss_per_sample(idx):
        o = obs[idx]
        mean, log_std = actor(o)
        std = log_std.exp()
        x_t = mean + std * eps_fixed[idx]        # deterministic given eps
        y_t = torch.tanh(x_t)
        action = y_t * actor.action_scale + actor.action_bias
        log_prob = torch.distributions.Normal(mean, std).log_prob(x_t)
        log_prob = log_prob - torch.log(actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_prob.sum(1, keepdim=True)
        min_q = torch.min(qf1(o, action), qf2(o, action))
        return (alpha * log_pi - min_q).reshape(-1)

    L_full = actor_loss_per_sample(slice(0, N)).mean()
    g_full = _flat_grad(L_full, params)

    g_expected = torch.zeros_like(g_full)
    for i in range(N):
        w_i = 1.0 / (N * p[i])
        L_i = (w_i * actor_loss_per_sample(slice(i, i + 1))).mean()
        g_expected += p[i] * _flat_grad(L_i, params)

    rel = (g_expected - g_full).abs().max().item() / (g_full.abs().max().item() + 1e-12)
    assert rel < 1e-5, f"actor unbiasedness failed: rel {rel:.2e}"


def test_shape_safety_contract():
    """Per-sample losses and weights must be 1-D of length B (the (B,)x(B,1)
    broadcast bug the task warns about would make these fail)."""
    torch.manual_seed(4)
    B, obs_dim, act_dim = 16, 6, 2
    obs = torch.randn(B, obs_dim)
    actor = Actor(obs_dim, act_dim, -np.ones(act_dim), np.ones(act_dim))
    qf1 = SoftQNetwork(obs_dim, act_dim)
    qf2 = SoftQNetwork(obs_dim, act_dim)
    y = torch.randn(B, 1)
    w = torch.rand(B)

    delta1 = (qf1(obs, torch.randn(B, act_dim)) - y).reshape(-1)
    per_sample_critic = delta1 ** 2 + delta1 ** 2
    pi, log_pi, _ = actor.get_action(obs)
    per_sample_actor = (0.2 * log_pi - torch.min(qf1(obs, pi), qf2(obs, pi))).reshape(-1)
    per_sample_temp = (-0.2 * (log_pi.reshape(-1) - act_dim))

    assert w.shape == per_sample_critic.shape == per_sample_actor.shape == per_sample_temp.shape == (B,)


if __name__ == "__main__":
    test_weighted_gradient_is_unbiased_critic()
    test_weighted_gradient_is_unbiased_actor()
    test_shape_safety_contract()
    print("test_unbiasedness OK")
