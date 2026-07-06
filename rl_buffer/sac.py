"""SAC with a single knob: the replay-buffer sampling scheme.

Faithful to CleanRL's ``sac_continuous_action.py`` (same networks, same default
hyperparameters).  The only change is that a minibatch is drawn under one of five
schemes and the resulting importance weight ``w_i = 1/(N p_i)`` is applied to the
critic, actor, AND temperature per-sample losses so all three remain unbiased
estimators of the uniform-buffer objectives.

Run ``python -m rl_buffer.sac --help`` for options.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rl_buffer.buffers import ReplayBuffer, SamplingConfig
from rl_buffer.networks import Actor, SoftQNetwork
from rl_buffer.priorities import adam_diag_preconditioner, per_sample_priority


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    p.add_argument("--scheme", type=str, default="uniform",
                   choices=["uniform", "per", "euclid", "precond", "precond2"])
    p.add_argument("--priority-mode", type=str, default="lazy", choices=["lazy", "two_stage"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--target-frequency", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--autotune", type=int, default=1)
    # sampling knobs
    p.add_argument("--alpha-prio", type=float, default=0.6)
    p.add_argument("--beta0", type=float, default=0.4)
    p.add_argument("--pool-mult", type=int, default=4)
    p.add_argument("--max-normalize", type=int, default=1)
    p.add_argument("--priority-source", type=str, default="both", choices=["both", "q1"])
    # eval / logging
    p.add_argument("--eval-frequency", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--torch-threads", type=int, default=0)
    return p.parse_args()


def make_env(env_id, seed, eval_=False):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed + (10_000 if eval_ else 0))
    return env


def evaluate(actor, env_id, seed, episodes, device):
    env = make_env(env_id, seed + 999, eval_=True)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 999 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, mean_a = actor.get_action(o)
            obs, r, term, trunc, _ = env.step(mean_a.squeeze(0).cpu().numpy())
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), returns


# --------------------------------------------------------------------------- #
# Priority computation shared by lazy + two-stage
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_priorities(scheme, cfg, qf1, qf2, q_optimizer, critic_params,
                       obs, actions, delta1, delta2):
    """Return a per-sample priority (numpy, length B) for the given scheme.

    ``per``     -> sqrt(delta1^2 + delta2^2)                       (Jacobian == 1)
    gradient    -> sqrt(||g1||_M^2 + ||g2||_M^2)  with M = D^{-power}
    ``priority_source == 'q1'`` restricts to critic-1 only.
    """
    if scheme == "per":
        if cfg.__dict__.get("priority_source", "both") == "q1":
            prio = delta1.abs()
        else:
            prio = torch.sqrt(delta1 ** 2 + delta2 ** 2)
        return prio.reshape(-1).cpu().numpy()

    # gradient schemes need the Adam preconditioner
    D = adam_diag_preconditioner(q_optimizer, critic_params)
    n1 = per_sample_priority(qf1, delta1, obs, actions, D, cfg.metric_power)  # (B,)
    if cfg.__dict__.get("priority_source", "both") == "q1":
        prio = n1
    else:
        n2 = per_sample_priority(qf2, delta2, obs, actions, D, cfg.metric_power)
        prio = torch.sqrt(n1 ** 2 + n2 ** 2)
    return prio.reshape(-1).cpu().numpy()


@torch.no_grad()
def critic_deltas(actor, qf1, qf2, qf1_t, qf2_t, alpha, gamma, batch):
    """TD residuals delta1, delta2 (each (B,)) and target y ((B,1)), no grad."""
    obs, actions = batch["obs"], batch["actions"]
    next_obs, rewards, dones = batch["next_obs"], batch["rewards"], batch["dones"]
    next_a, next_logp, _ = actor.get_action(next_obs)
    q1_next = qf1_t(next_obs, next_a)
    q2_next = qf2_t(next_obs, next_a)
    min_q_next = torch.min(q1_next, q2_next) - alpha * next_logp
    y = rewards + (1.0 - dones) * gamma * min_q_next
    d1 = (qf1(obs, actions) - y).reshape(-1)
    d2 = (qf2(obs, actions) - y).reshape(-1)
    return d1, d2, y


def main():
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = args.exp_name or f"{args.env_id}__{args.scheme}__{args.priority_mode}__seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = SamplingConfig(
        scheme=args.scheme, alpha=args.alpha_prio, beta0=args.beta0, beta1=1.0,
        total_anneal_steps=args.total_steps, priority_mode=args.priority_mode,
        pool_mult=args.pool_mult, max_normalize=bool(args.max_normalize),
    )
    cfg.__dict__["priority_source"] = args.priority_source

    env = make_env(args.env_id, args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    a_low, a_high = env.action_space.low, env.action_space.high

    actor = Actor(obs_dim, act_dim, a_low, a_high).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_t = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_t = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())

    critic_params = list(qf1.parameters()) + list(qf2.parameters())
    q_optimizer = torch.optim.Adam(critic_params, lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().detach()
        a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = torch.tensor(args.alpha, device=device)

    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim, cfg, device=device, seed=args.seed)

    # logging
    log_rows = []          # eval curve
    train_returns = []     # (step, episodic_return)
    grad_evals = 0         # number of critic-gradient-shaped passes (compute accounting)
    t_start = time.time()

    obs, _ = env.reset(seed=args.seed)
    B = args.batch_size

    for global_step in range(args.total_steps):
        # --- act ----------------------------------------------------------
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _, _ = actor.get_action(o)
            action = a.squeeze(0).cpu().numpy()

        next_obs, reward, term, trunc, info = env.step(action)
        real_next = next_obs.copy()
        # store bootstrap-correct next state (truncation != termination)
        buf.add(obs, action, reward, real_next, float(term))
        obs = next_obs
        if term or trunc:
            if "episode" in info:
                train_returns.append((global_step, float(info["episode"]["r"])))
            obs, _ = env.reset()

        # --- learn --------------------------------------------------------
        if global_step < args.learning_starts:
            continue

        # sample minibatch + IS weights ----------------------------------
        pool_sel = None
        if cfg.scheme == "uniform":
            idxs, batch, w = buf.sample_uniform(B)
        elif cfg.priority_mode == "two_stage" and cfg.is_prioritized:
            pool_idxs, pool_batch = buf.draw_pool(B)
            pd1, pd2, _ = critic_deltas(actor, qf1, qf2, qf1_t, qf2_t, alpha, args.gamma, pool_batch)
            grad_evals += cfg.pool_mult  # priorities on the pool
            pool_prio = compute_priorities(cfg.scheme, cfg, qf1, qf2, q_optimizer,
                                           critic_params, pool_batch["obs"], pool_batch["actions"],
                                           pd1, pd2)
            idxs, batch, w, pool_sel = buf.subsample_from_pool(pool_idxs, pool_prio, B)
        else:  # lazy prioritized
            idxs, batch, w = buf.sample_lazy(B, global_step)

        obs_b, act_b = batch["obs"], batch["actions"]

        # --- critic update (weighted, per-sample) ------------------------
        with torch.no_grad():
            next_a, next_logp, _ = actor.get_action(batch["next_obs"])
            q1_next = qf1_t(batch["next_obs"], next_a)
            q2_next = qf2_t(batch["next_obs"], next_a)
            min_q_next = torch.min(q1_next, q2_next) - alpha * next_logp
            y = batch["rewards"] + (1.0 - batch["dones"]) * args.gamma * min_q_next

        q1 = qf1(obs_b, act_b)
        q2 = qf2(obs_b, act_b)
        delta1 = (q1 - y).reshape(-1)          # (B,)
        delta2 = (q2 - y).reshape(-1)          # (B,)
        per_sample_critic = delta1 ** 2 + delta2 ** 2      # (B,)

        assert w.shape == per_sample_critic.shape == (B,), (w.shape, per_sample_critic.shape)
        qf_loss = (w * per_sample_critic).mean()

        q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        q_optimizer.step()
        grad_evals += 1

        # refresh priorities (lazy scheme) -------------------------------
        if cfg.is_prioritized and cfg.priority_mode == "lazy":
            with torch.no_grad():
                d1n, d2n, _ = critic_deltas(actor, qf1, qf2, qf1_t, qf2_t, alpha, args.gamma, batch)
            prio = compute_priorities(cfg.scheme, cfg, qf1, qf2, q_optimizer,
                                      critic_params, obs_b, act_b, d1n, d2n)
            buf.update_priorities(idxs, prio)

        # --- actor + temperature (same w, per-sample, shape-checked) -----
        if global_step % args.policy_frequency == 0:
            for _ in range(args.policy_frequency):  # CleanRL delay compensation
                pi, log_pi, _ = actor.get_action(obs_b)
                q1_pi = qf1(obs_b, pi)
                q2_pi = qf2(obs_b, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                per_sample_actor = (alpha * log_pi - min_q_pi).reshape(-1)   # (B,)
                assert per_sample_actor.shape == w.shape == (B,)
                actor_loss = (w * per_sample_actor).mean()

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi_d, _ = actor.get_action(obs_b)
                    per_sample_temp = (-log_alpha.exp() * (log_pi_d.reshape(-1) + target_entropy))  # (B,)
                    assert per_sample_temp.shape == w.shape == (B,)
                    alpha_loss = (w * per_sample_temp).mean()
                    a_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().detach()

        # --- target networks --------------------------------------------
        if global_step % args.target_frequency == 0:
            for pt, p in zip(qf1_t.parameters(), qf1.parameters()):
                pt.data.mul_(1 - args.tau).add_(args.tau * p.data)
            for pt, p in zip(qf2_t.parameters(), qf2.parameters()):
                pt.data.mul_(1 - args.tau).add_(args.tau * p.data)

        # --- eval --------------------------------------------------------
        if (global_step + 1) % args.eval_frequency == 0:
            eval_ret, _ = evaluate(actor, args.env_id, args.seed, args.eval_episodes, device)
            sps = int((global_step + 1) / (time.time() - t_start))
            log_rows.append({
                "step": global_step + 1, "eval_return": eval_ret,
                "grad_evals": grad_evals, "wall_s": time.time() - t_start, "sps": sps,
            })
            print(f"[{exp_name}] step={global_step+1} eval={eval_ret:.1f} "
                  f"grad_evals={grad_evals} sps={sps}", flush=True)

    # --- persist ---------------------------------------------------------
    result = {
        "exp_name": exp_name,
        "args": vars(args),
        "cfg": {k: v for k, v in asdict(cfg).items()},
        "eval_curve": log_rows,
        "train_returns": train_returns,
        "total_grad_evals": grad_evals,
        "total_wall_s": time.time() - t_start,
    }
    out_path = os.path.join(args.out_dir, exp_name + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    env.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
