import numpy as np
import torch
import numpy as np
from models.nets import PolicyNetwork

# =========================
# SAC Expert Wrapper
# =========================
class SACExpertPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# =========================
# Collect SAC Demos
# =========================
def collect_sac_demos(
    env,
    policy,
    num_steps=20000,
    horizon=None,
    reward_threshold=None,
    success_only=False,
    verbose=True,
):
    """
    Collect demonstrations from a SAC policy.

    Args:
        env: gym env
        policy: SACExpertPolicy
        num_steps: total transitions to collect
        horizon: max steps per episode
        reward_threshold: keep only episodes above this reward
        success_only: keep only successful episodes
    """

    obs_buf = []
    act_buf = []
    ep_rewards = []

    steps = 0

    while steps < num_steps:
        obs, _ = env.reset()
        done = False

        ep_obs = []
        ep_acts = []
        ep_reward = 0.0
        t = 0

        while not done and steps < num_steps:
            act = policy.act(obs)
            act = np.asarray(act, dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)

            ep_obs.append(obs.copy())
            ep_acts.append(act.copy())

            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            ep_reward += reward
            t += 1
            steps += 1

            if horizon is not None and t >= horizon:
                done = True

        # -------------------------
        # Filtering logic (KEY)
        # -------------------------
        keep = True

        if success_only:
            keep = info.get("is_success", False)

        if reward_threshold is not None:
            keep = keep and (ep_reward >= reward_threshold)

        if keep:
            obs_buf.extend(ep_obs)
            act_buf.extend(ep_acts)
            ep_rewards.append(ep_reward)

        if verbose and steps % 1000 < t:
            print(f"Collected {steps} steps")

    obs_buf = np.array(obs_buf, dtype=np.float32)
    act_buf = np.array(act_buf, dtype=np.float32)

    if verbose:
        print(f"Total kept samples: {len(obs_buf)}")
        if len(ep_rewards) > 0:
            print(f"Mean episode reward: {np.mean(ep_rewards):.4f}")
        else:
            print("No episodes passed filtering!")

    return obs_buf, act_buf, ep_rewards


class TrainedExpertPolicy:
    def __init__(self, state_dim, action_dim, discrete, ckpt_path, device):
        self.device = device

        self.pi = PolicyNetwork(state_dim, action_dim, discrete).to(device)
        self.pi.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.pi.eval()

    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            dist = self.pi(obs_t)
            action = dist.sample()

            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)

        return action

class RuleBasedPickPlacePolicy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = 0
        self.close_steps = 0

    def act(self, obs):
        eef = obs[0:3]
        obj = obs[6:9]
        goal = obs[19:22]

        action = np.zeros(4, dtype=np.float32)

        # Stage 0: move above object
        if self.stage == 0:
            target = obj + np.array([0, 0, 0.05], dtype=np.float32)
            action[:3] = np.clip(5 * (target - eef), -1, 1)
            action[3] = -1

            if np.linalg.norm(target - eef) < 0.02:
                self.stage = 1

        # Stage 1: descend
        elif self.stage == 1:
            action[:3] = np.clip(5 * (obj - eef), -1, 1)
            action[3] = -1

            if np.linalg.norm(obj - eef) < 0.015:
                self.stage = 2

        # Stage 2: close gripper
        elif self.stage == 2:
            action[:] = [0, 0, 0, 1]
            self.close_steps += 1

            if self.close_steps > 40:
                self.stage = 3

        # Stage 3: lift
        elif self.stage == 3:
            action[:] = [0, 0, 1, 1]

            if obj[2] > goal[2] + 0.05:
                self.stage = 4

        # Stage 4: move to goal
        elif self.stage == 4:
            action[:3] = np.clip(5 * (goal - eef), -1, 1)
            action[3] = 1

        return action
    
def collect_rule_based_demos(env, policy, num_steps, horizon=None, render=False):
    all_obs = []
    all_acts = []
    ep_rewards = []

    steps = 0
    while steps < num_steps:
        obs, info = env.reset()
        policy.reset()

        done = False
        t = 0
        rewards = []

        while not done and steps < num_steps:
            act = policy.act(obs)

            all_obs.append(np.array(obs, copy=True))
            all_acts.append(np.array(act, copy=True))

            if render:
                env.render()

            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            rewards.append(reward)
            steps += 1
            t += 1

            if horizon is not None and t >= horizon:
                done = True

        if len(rewards) > 0:
            ep_rewards.append(np.sum(rewards))

    all_obs = np.array(all_obs, dtype=np.float32)
    all_acts = np.array(all_acts, dtype=np.float32)

    return all_obs, all_acts, ep_rewards

def collect_expert_demos(env, policy, num_steps, horizon=None, render=False):
    all_obs = []
    all_acts = []
    ep_rewards = []

    steps = 0
    while steps < num_steps:

        obs, _ = env.reset()
        done = False
        t = 0
        rewards = []

        while not done and steps < num_steps:

            act = policy.act(obs)

            all_obs.append(np.array(obs, copy=True))
            all_acts.append(np.array(act, copy=True))

            if render:
                env.render()

            obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            rewards.append(reward)

            steps += 1
            t += 1

            if horizon is not None and t >= horizon:
                done = True

        if len(rewards) > 0:
            ep_rewards.append(np.sum(rewards))

    all_obs = np.array(all_obs, dtype=np.float32)
    all_acts = np.array(all_acts, dtype=np.float32)

    return all_obs, all_acts, ep_rewards