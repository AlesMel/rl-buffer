"""Actor and critic networks for SAC.

Architecture is a faithful copy of CleanRL's ``sac_continuous_action.py``
(2x256 ReLU MLPs, squashed-Gaussian actor, twin soft-Q critics) so that the
*only* thing this project changes relative to a trusted SAC baseline is the
replay-buffer sampling scheme.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    """Q(s, a) -> scalar. Plain 2x256 ReLU MLP over concat([s, a])."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # --- ghost-norm support -------------------------------------------------
    # Manual forward that also returns the intermediate activations and
    # pre-activations needed to compute per-sample gradient norms in a diagonal
    # metric without materialising per-sample gradients (see priorities.py).
    def forward_with_cache(self, x: torch.Tensor, a: torch.Tensor):
        x0 = torch.cat([x, a], dim=1)          # (B, in0)
        z1 = self.fc1(x0)
        h1 = F.relu(z1)
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        q = self.fc3(h2)                       # (B, 1)
        cache = {
            "x0": x0, "z1": z1, "h1": h1,
            "z2": z2, "h2": h2, "q": q,
        }
        return q, cache


class Actor(nn.Module):
    """Squashed-Gaussian policy (tanh-transformed Normal)."""

    def __init__(self, obs_dim: int, act_dim: int, action_low, action_high,
                 hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Linear(hidden, act_dim)
        # action rescaling from [-1, 1] (tanh range) to env bounds
        action_scale = (np.asarray(action_high) - np.asarray(action_low)) / 2.0
        action_bias = (np.asarray(action_high) + np.asarray(action_low)) / 2.0
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x: torch.Tensor):
        """Returns (action, log_prob, mean_action). log_prob is (B, 1)."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                 # reparameterised
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # tanh change-of-variables correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action
