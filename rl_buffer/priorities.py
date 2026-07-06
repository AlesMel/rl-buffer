"""Per-sample critic gradient norms in a diagonal (preconditioner) metric.

The sampling rule under test weights replay transition ``i`` by the norm of its
per-sample critic gradient measured in the metric ``M``:

    ||g_i||_M = sqrt( g_i^T M g_i ),      g_i = delta_i * grad_theta Q(s_i, a_i)

For the squared soft-Bellman residual the target ``y_i`` is stop-gradient, so
``g_i`` factorises and

    ||g_i||_M = |delta_i| * ||grad_theta Q(s_i, a_i)||_M .

We support three metrics, all diagonal in the Adam preconditioner
``D = diag(sqrt(v_hat) + eps)``:

    euclid   (D^0 ) : M = I                       -> optimal for SGD variance
    precond  (D^-1) : M = D^-1  (ours)            -> optimal for Adam variance
    precond2 (D^-2) : M = D^-2  (metric sanity)   -> double-counts D, should lose

The generic diagonal metric is ``M = D^{-power}`` with power in {0, 1, 2}.

Efficiency: we never materialise per-sample parameter gradients (B x P).  Instead
we use the *weighted ghost-norm* trick.  For a linear layer ``z = W a + b`` whose
scalar-output backprop signal is ``s = dQ/dz`` (per sample), the per-sample
gradient w.r.t. ``W`` is the outer product ``s_i a_i^T`` and w.r.t. ``b`` is
``s_i``.  The squared norm in a diagonal metric with per-weight inverse-metric
matrix ``R = M_W`` (shape out x in) and bias inverse-metric ``r_b`` (shape out) is

    ||grad_W Q_i||^2_M = sum_{jk} s_ij^2 a_ik^2 R_jk = ( s^2 * ( (a^2) @ R^T ) ).sum(1)
    ||grad_b Q_i||^2_M = sum_j  s_ij^2 r_bj          = (s^2) @ r_b

Summing these over layers gives ``||grad_theta Q_i||^2_M`` for the whole critic in
two extra forward/backward-shaped matmuls -- O(B * params), no per-sample grads.
When ``R = 1`` this reduces to the classic ``||s||^2 ||a||^2`` Goodfellow (2015)
identity; the diagonal metric is the generalisation used here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


def adam_diag_preconditioner(optimizer: torch.optim.Optimizer,
                             params: List[torch.nn.Parameter],
                             eps: Optional[float] = None) -> Dict[torch.nn.Parameter, torch.Tensor]:
    """Return D = sqrt(v_hat) + eps for each parameter, read from Adam state.

    ``v_hat`` is the bias-corrected second moment.  Before Adam has taken its
    first step (empty state) we return ``eps`` (isotropic), so ``precond`` and
    ``precond2`` gracefully reduce to a rescaled ``euclid`` early in training.
    """
    D: Dict[torch.nn.Parameter, torch.Tensor] = {}
    param_ids = {id(p) for p in params}
    # default eps / beta2 from the optimizer's param groups
    for group in optimizer.param_groups:
        g_eps = group.get("eps", 1e-8) if eps is None else eps
        beta2 = group["betas"][1]
        for p in group["params"]:
            if id(p) not in param_ids:
                continue
            state = optimizer.state.get(p, {})
            v = state.get("exp_avg_sq", None)
            step = state.get("step", 0)
            if isinstance(step, torch.Tensor):
                step = step.item()
            if v is None or step == 0:
                D[p] = torch.full_like(p, float(g_eps))
            else:
                bias_correction2 = 1.0 - beta2 ** step
                v_hat = v / bias_correction2
                D[p] = v_hat.sqrt() + g_eps
    # any requested param not owned by this optimizer -> isotropic fallback
    for p in params:
        if p not in D:
            D[p] = torch.ones_like(p)
    return D


class GhostNormCalculator:
    """Compute per-sample ``||grad_theta Q(s,a)||_M`` for a SoftQNetwork.

    ``metric_power`` selects M = D^{-power}: 0 -> euclid, 1 -> precond, 2 -> precond2.
    """

    def __init__(self, critic, metric_power: int):
        assert metric_power in (0, 1, 2)
        self.critic = critic
        self.metric_power = metric_power

    def _inv_metric(self, D_w: torch.Tensor, D_b: torch.Tensor):
        """R = D^{-power} for weight and bias (elementwise)."""
        if self.metric_power == 0:
            return torch.ones_like(D_w), torch.ones_like(D_b)
        R_w = D_w.pow(-self.metric_power)
        r_b = D_b.pow(-self.metric_power)
        return R_w, r_b

    @torch.no_grad()
    def grad_norm(self, s: torch.Tensor, a: torch.Tensor,
                  D: Dict[torch.nn.Parameter, torch.Tensor]) -> torch.Tensor:
        """Return ``||grad_theta Q(s,a)||_M`` as a 1-D tensor of length B.

        ``D`` maps the critic's parameters to their diagonal preconditioner
        (from :func:`adam_diag_preconditioner`).  Runs a manual forward pass to
        cache activations and a manual backward for the scalar output ``Q``.
        """
        c = self.critic
        q, cache = c.forward_with_cache(s, a)                 # q: (B, 1)
        B = q.shape[0]

        # Manual backward of the scalar output Q (per sample) through the 2x256 MLP.
        # s3 = dQ/dz3 = 1
        s3 = torch.ones_like(q)                               # (B, 1)
        h2 = cache["h2"]                                      # (B, 256)
        # s2 = dQ/dz2 = (s3 @ W3) * relu'(z2)
        s2 = (s3 @ c.fc3.weight) * (cache["z2"] > 0).float()  # (B, 256)
        h1 = cache["h1"]                                      # (B, 256)
        s1 = (s2 @ c.fc2.weight) * (cache["z1"] > 0).float()  # (B, 256)
        x0 = cache["x0"]                                      # (B, in0)

        sq_norm = torch.zeros(B, device=q.device, dtype=q.dtype)
        layers = [
            (c.fc1.weight, c.fc1.bias, s1, x0),
            (c.fc2.weight, c.fc2.bias, s2, h1),
            (c.fc3.weight, c.fc3.bias, s3, h2),
        ]
        for W, b, s_sig, a_in in layers:
            R_w, r_b = self._inv_metric(D[W], D[b])           # inverse-metric factors
            a2 = a_in * a_in                                  # (B, in)
            s2_ = s_sig * s_sig                               # (B, out)
            # weight term: sum_jk s_ij^2 a_ik^2 R_jk
            sq_norm = sq_norm + (s2_ * (a2 @ R_w.t())).sum(1)
            # bias term: sum_j s_ij^2 r_bj
            sq_norm = sq_norm + (s2_ @ r_b)
        return sq_norm.clamp_min(0).sqrt()                   # (B,)


def per_sample_priority(critic, delta: torch.Tensor, s: torch.Tensor, a: torch.Tensor,
                        D: Dict[torch.nn.Parameter, torch.Tensor], metric_power: int) -> torch.Tensor:
    """||g_i||_M = |delta_i| * ||grad Q(s_i,a_i)||_M, returned as a 1-D (B,) tensor."""
    calc = GhostNormCalculator(critic, metric_power)
    jac_norm = calc.grad_norm(s, a, D)                       # (B,)
    delta = delta.reshape(-1)
    assert delta.shape == jac_norm.shape, (delta.shape, jac_norm.shape)
    return delta.abs() * jac_norm


def per_sample_grad_norm_vmap(critic, s: torch.Tensor, a: torch.Tensor,
                              D: Dict[torch.nn.Parameter, torch.Tensor], metric_power: int) -> torch.Tensor:
    """Ground-truth per-sample ``||grad Q||_M`` via torch.func (materialises grads).

    Used only in tests to validate :class:`GhostNormCalculator`.  This is the
    O(B*P)-memory path the ghost-norm trick avoids in production.
    """
    from torch.func import grad, vmap, functional_call

    params = {k: v.detach() for k, v in critic.named_parameters()}

    def q_of_params(p, s_i, a_i):
        out = functional_call(critic, p, (s_i.unsqueeze(0), a_i.unsqueeze(0)))
        return out.squeeze()

    per_sample_grads = vmap(grad(q_of_params), in_dims=(None, 0, 0))(params, s, a)
    # accumulate weighted squared norm across params
    B = s.shape[0]
    sq = torch.zeros(B, dtype=s.dtype)
    name_to_param = dict(critic.named_parameters())
    for name, g in per_sample_grads.items():
        p = name_to_param[name]
        Dp = D[p]
        if metric_power == 0:
            R = torch.ones_like(Dp)
        else:
            R = Dp.pow(-metric_power)
        g2 = g * g                                           # (B, *param_shape)
        weighted = g2 * R.unsqueeze(0)
        sq = sq + weighted.reshape(B, -1).sum(1)
    return sq.clamp_min(0).sqrt()
