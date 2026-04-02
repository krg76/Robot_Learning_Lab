# import os
# import numpy as np
# import gymnasium as gym
# import gym_xarm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Normal


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOG_STD_MIN = -5.0
# LOG_STD_MAX = 2.0
# EPS = 1e-6


# class SquashedGaussianPolicy(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#         )
#         self.mean_head = nn.Linear(128, action_dim)
#         self.log_std = nn.Parameter(torch.zeros(action_dim))

#     def forward(self, states: torch.Tensor):
#         h = self.backbone(states)
#         mean = self.mean_head(h)
#         log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std).expand_as(mean)
#         return mean, std

#     def sample(self, states: torch.Tensor):
#         mean, std = self(states)
#         dist = Normal(mean, std)
#         raw_action = dist.rsample()
#         action = torch.tanh(raw_action)

#         log_prob = dist.log_prob(raw_action).sum(-1)
#         log_prob = log_prob - torch.log(1 - action.pow(2) + EPS).sum(-1)

#         return action, log_prob, raw_action

#     def evaluate_actions(self, states: torch.Tensor, raw_actions: torch.Tensor):
#         mean, std = self(states)
#         dist = Normal(mean, std)
#         action = torch.tanh(raw_actions)

#         log_prob = dist.log_prob(raw_actions).sum(-1)
#         log_prob = log_prob - torch.log(1 - action.pow(2) + EPS).sum(-1)

#         entropy = dist.entropy().sum(-1)
#         return log_prob, entropy

#     @torch.no_grad()
#     def act(self, state: np.ndarray, deterministic: bool = False):
#         state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#         mean, std = self(state_t)
#         if deterministic:
#             raw_action = mean
#         else:
#             dist = Normal(mean, std)
#             raw_action = dist.sample()
#         action = torch.tanh(raw_action)
#         return action.squeeze(0).cpu().numpy()


# class ValueNet(nn.Module):
#     def __init__(self, state_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1),
#         )

#     def forward(self, states: torch.Tensor):
#         return self.net(states).squeeze(-1)


# def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
#     T = len(rewards)
#     advantages = torch.zeros(T, dtype=torch.float32, device=device)
#     gae = torch.tensor(0.0, dtype=torch.float32, device=device)

#     for t in reversed(range(T)):
#         nonterminal = 1.0 - dones[t]
#         delta = rewards[t] + gamma * next_value * nonterminal - values[t]
#         gae = delta + gamma * lam * nonterminal * gae
#         advantages[t] = gae
#         next_value = values[t]

#     returns = advantages + values
#     return advantages, returns


# def evaluate_policy(env_name: str, policy: SquashedGaussianPolicy, n_episodes: int = 10):
#     env = gym.make(env_name)
#     returns = []
#     successes = []

#     for _ in range(n_episodes):
#         obs, _ = env.reset()
#         done = False
#         ep_ret = 0.0
#         info = {}

#         while not done:
#             action = policy.act(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             ep_ret += reward

#         returns.append(ep_ret)
#         success = 1.0 if info.get("is_success", False) else 0.0
#         successes.append(success)

#     env.close()
#     return float(np.mean(returns)), float(np.mean(successes))


# def train_expert():
#     env_name = "gym_xarm/XarmPickPlaceDense-v0"
#     env = gym.make(env_name)

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     policy = SquashedGaussianPolicy(state_dim, action_dim).to(device)
#     value_fn = ValueNet(state_dim).to(device)

#     pi_optim = optim.Adam(policy.parameters(), lr=3e-4)
#     vf_optim = optim.Adam(value_fn.parameters(), lr=1e-3)

#     gamma = 0.99
#     lam = 0.95
#     clip_eps = 0.2
#     entropy_coef = 1e-3
#     value_coef = 0.5

#     steps_per_iter = 4096
#     train_pi_iters = 10
#     train_v_iters = 10
#     minibatch_size = 256
#     num_iters = 200

#     obs, _ = env.reset()

#     for it in range(num_iters):
#         obs_buf = []
#         raw_act_buf = []
#         rew_buf = []
#         done_buf = []
#         logp_buf = []
#         val_buf = []

#         episode_returns = []
#         episode_successes = []
#         ep_ret = 0.0
#         info = {}

#         for _ in range(steps_per_iter):
#             obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

#             with torch.no_grad():
#                 value = value_fn(obs_t).squeeze(0)
#                 action_t, logp_t, raw_action_t = policy.sample(obs_t)

#             action = action_t.squeeze(0).cpu().numpy()
#             next_obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated

#             obs_buf.append(obs.copy())
#             raw_act_buf.append(raw_action_t.squeeze(0).cpu().numpy())
#             rew_buf.append(reward)
#             done_buf.append(float(done))
#             logp_buf.append(logp_t.item())
#             val_buf.append(value.item())

#             ep_ret += reward
#             obs = next_obs

#             if done:
#                 episode_returns.append(ep_ret)
#                 episode_successes.append(1.0 if info.get("is_success", False) else 0.0)
#                 obs, _ = env.reset()
#                 ep_ret = 0.0

#         with torch.no_grad():
#             obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
#             next_value = value_fn(obs_t).squeeze(0)

#         obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
#         raw_act_t = torch.tensor(np.array(raw_act_buf), dtype=torch.float32, device=device)
#         rew_t = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)
#         done_t = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)
#         old_logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
#         val_t = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)

#         adv_t, ret_t = compute_gae(
#             rewards=rew_t,
#             dones=done_t,
#             values=val_t,
#             next_value=next_value,
#             gamma=gamma,
#             lam=lam,
#         )

#         adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

#         n = obs_t.shape[0]
#         idxs = np.arange(n)

#         for _ in range(train_pi_iters):
#             np.random.shuffle(idxs)
#             for start in range(0, n, minibatch_size):
#                 mb = idxs[start:start + minibatch_size]

#                 new_logp, entropy = policy.evaluate_actions(obs_t[mb], raw_act_t[mb])
#                 ratio = torch.exp(new_logp - old_logp_t[mb])

#                 surr1 = ratio * adv_t[mb]
#                 surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
#                 pi_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

#                 pi_optim.zero_grad()
#                 pi_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
#                 pi_optim.step()

#         for _ in range(train_v_iters):
#             np.random.shuffle(idxs)
#             for start in range(0, n, minibatch_size):
#                 mb = idxs[start:start + minibatch_size]
#                 v_pred = value_fn(obs_t[mb])
#                 v_loss = value_coef * ((v_pred - ret_t[mb]) ** 2).mean()

#                 vf_optim.zero_grad()
#                 v_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(value_fn.parameters(), 1.0)
#                 vf_optim.step()

#         train_ep_return = float(np.mean(episode_returns)) if episode_returns else 0.0
#         train_success = float(np.mean(episode_successes)) if episode_successes else 0.0
#         eval_return, eval_success = evaluate_policy(env_name, policy, n_episodes=10)

#         print(
#             f"Iter {it:03d} | "
#             f"train_ep_return: {train_ep_return:.3f} | "
#             f"train_success: {train_success:.3f} | "
#             f"eval_ep_return: {eval_return:.3f} | "
#             f"eval_success: {eval_success:.3f}"
#         )

#     os.makedirs("experts", exist_ok=True)
#     torch.save(policy.state_dict(), "experts/policy.pt")
#     torch.save(value_fn.state_dict(), "experts/value.pt")
#     env.close()


# if __name__ == "__main__":
#     train_expert()

import os
import numpy as np
import gymnasium as gym
import gym_xarm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from utils.demo_collect import RuleBasedPickPlacePolicy, collect_rule_based_demos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


# =========================
# Policy (Tanh Gaussian)
# =========================
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

    def sample(self, x):
        mean, std = self(x)
        dist = Normal(mean, std)

        raw = dist.rsample()
        action = torch.tanh(raw)

        log_prob = dist.log_prob(raw).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + EPS).sum(-1)

        return action, log_prob, raw

    def evaluate(self, x, raw_action):
        mean, std = self(x)
        dist = Normal(mean, std)

        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + EPS).sum(-1)

        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        mean, std = self(obs)

        if deterministic:
            raw = mean
        else:
            dist = Normal(mean, std)
            raw = dist.sample()

        action = torch.tanh(raw)
        return action.squeeze(0).cpu().numpy()


# =========================
# Value Network
# =========================
class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# GAE
# =========================
def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T, device=device)
    gae = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
        next_value = values[t]

    returns = adv + values
    return adv, returns


# =========================
# Evaluation
# =========================
def evaluate(env_name, policy, n=10):
    env = gym.make(env_name)

    rets = []
    success = []

    for _ in range(n):
        obs, _ = env.reset()
        done = False
        total = 0

        while not done:
            action = policy.act(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            total += r

        rets.append(total)
        success.append(info.get("is_success", 0.0))

    env.close()
    return np.mean(rets), np.mean(success)

def render_policy(env_name, policy, max_steps=300):
    env = gym.make(env_name, render_mode="human")

    obs, _ = env.reset()
    total_reward = 0

    for t in range(max_steps):
        action = policy.act(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        print(f"step {t} | reward {reward:.4f}")

        if terminated or truncated:
            print("Done!")
            print("Success:", info.get("is_success", 0.0))
            break

    print("Episode return:", total_reward)
    env.close()


# =========================
# TRAIN
# =========================
def train_expert():

    env_name = "gym_xarm/XarmPickPlaceDense-v0"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim).to(device)
    value_fn = Value(state_dim).to(device)

    pi_opt = optim.Adam(policy.parameters(), lr=3e-4)
    v_opt = optim.Adam(value_fn.parameters(), lr=1e-3)

    # =========================================================
    # 🔥 STEP 1: Behavior Cloning
    # =========================================================
    print("Collecting rule-based demos for BC...")

    rb = RuleBasedPickPlacePolicy()
    obs, acts, _ = collect_rule_based_demos(env, rb, num_steps=100000)

    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    acts = torch.tensor(acts, dtype=torch.float32).to(device)

    print("BC pretraining...")

    for i in range(20000):

        mean, std = policy(obs)

        # inverse tanh
        raw_target = torch.atanh(torch.clamp(acts, -0.999, 0.999))

        loss = ((mean - raw_target) ** 2).mean()

        pi_opt.zero_grad()
        loss.backward()
        pi_opt.step()

        if i % 50 == 0:
            print(f"BC step {i} | loss {loss.item():.4f}")

    print("\n===== BC POLICY RENDER =====")
    render_policy(env_name, policy)

    print("\n===== Saving BC policy =====")

    os.makedirs("experts", exist_ok=True)
    torch.save(policy.state_dict(), "experts/policy_bc.pt")

    print("Saved BC policy to experts/policy_bc.pt")

    # =========================================================
    # 🔥 STEP 2: PPO
    # =========================================================
    gamma = 0.99
    lam = 0.95
    clip = 0.2

    steps_per_iter = 4096
    epochs = 10
    batch_size = 256

    obs, _ = env.reset()

    for it in range(200):

        obs_buf, raw_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
        ep_returns, ep_success = [], []

        ep_ret = 0

        for _ in range(steps_per_iter):

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                val = value_fn(obs_t).item()
                act, logp, raw = policy.sample(obs_t)

            action = act.squeeze(0).cpu().numpy()

            next_obs, r, term, trunc, info = env.step(action)
            done = term or trunc

            obs_buf.append(obs)
            raw_buf.append(raw.squeeze(0).cpu().numpy())
            rew_buf.append(r)
            done_buf.append(float(done))
            logp_buf.append(logp.item())
            val_buf.append(val)

            ep_ret += r
            obs = next_obs

            if done:
                ep_returns.append(ep_ret)
                ep_success.append(info.get("is_success", 0.0))
                obs, _ = env.reset()
                ep_ret = 0

        # tensors
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
        raw_t = torch.tensor(np.array(raw_buf), dtype=torch.float32).to(device)
        rew_t = torch.tensor(rew_buf, dtype=torch.float32).to(device)
        done_t = torch.tensor(done_buf, dtype=torch.float32).to(device)
        old_logp = torch.tensor(logp_buf, dtype=torch.float32).to(device)
        val_t = torch.tensor(val_buf, dtype=torch.float32).to(device)

        with torch.no_grad():
            next_val = value_fn(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)).item()

        adv, ret = compute_gae(rew_t, done_t, val_t, next_val, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO updates
        idx = np.arange(len(obs_t))

        for _ in range(epochs):
            np.random.shuffle(idx)

            for start in range(0, len(idx), batch_size):
                mb = idx[start:start + batch_size]

                new_logp, entropy = policy.evaluate(obs_t[mb], raw_t[mb])

                ratio = torch.exp(new_logp - old_logp[mb])

                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]

                pi_loss = -torch.min(s1, s2).mean() - 1e-3 * entropy.mean()

                pi_opt.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                pi_opt.step()

                v_pred = value_fn(obs_t[mb])
                v_loss = ((v_pred - ret[mb]) ** 2).mean()

                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()

        train_ret = np.mean(ep_returns) if ep_returns else 0
        train_succ = np.mean(ep_success) if ep_success else 0

        eval_ret, eval_succ = evaluate(env_name, policy)

        print(
            f"Iter {it:03d} | "
            f"train_ret {train_ret:.2f} | "
            f"train_succ {train_succ:.2f} | "
            f"eval_ret {eval_ret:.2f} | "
            f"eval_succ {eval_succ:.2f}"
        )
        if it % 50 == 0 and it != 0:
            print(f"\n===== RENDER @ ITER {it} =====")
            render_policy(env_name, policy)

    os.makedirs("experts", exist_ok=True)
    torch.save(policy.state_dict(), "experts/policy.pt")

    env.close()


if __name__ == "__main__":
    train_expert()