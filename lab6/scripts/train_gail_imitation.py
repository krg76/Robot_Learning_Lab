import numpy as np
import gymnasium as gym
import gym_xarm  # registers envs on import

from stable_baselines3 import SAC
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet


# =========================
# Config
# =========================
ENV_NAME = "gym_xarm/XarmPickPlaceDense-v0"
DEMO_PATH = "ckpts/gym_xarm/XarmPickPlaceDense-v0/sac_demos.npz"

TOTAL_TIMESTEPS = 200_000
EPISODE_LENGTH = 200  # match your env horizon


# =========================
# Load demos
# =========================
def load_demos(path):
    data = np.load(path)
    obs = data["obs"]
    acts = data["acts"]

    print("Loaded demos:")
    print("obs shape:", obs.shape)
    print("acts shape:", acts.shape)

    return obs, acts


# =========================
# Convert to trajectories
# =========================
def convert_to_trajectories(obs, acts, episode_length):
    trajectories = []

    i = 0
    N = len(acts)

    while i < N:
        end = min(i + episode_length, N)

        obs_slice = obs[i:end + 1]
        act_slice = acts[i:end]

        # ensure len(obs) = len(acts) + 1
        if len(obs_slice) == len(act_slice):
            obs_slice = np.concatenate([obs_slice, obs_slice[-1:]], axis=0)

        traj = Trajectory(
            obs=obs_slice,
            acts=act_slice,
            infos=None,
            terminal=True,
        )

        trajectories.append(traj)
        i = end

    print(f"Created {len(trajectories)} trajectories")
    return trajectories


# =========================
# Train GAIL
# =========================
def train_gail(trajectories):
    rng = np.random.default_rng(0)

    venv = make_vec_env(
        ENV_NAME,
        rng=rng,
        n_envs=1,
    )

    # Generator (policy we train)
    gen_algo = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
    )

    # 🔥 REQUIRED for new imitation API
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )

    gail_trainer = GAIL(
        demonstrations=trajectories,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        demo_batch_size=1024,
        n_disc_updates_per_round=4,
    )

    print("Starting GAIL training...")
    gail_trainer.train(TOTAL_TIMESTEPS)

    return gail_trainer


# =========================
# Evaluate
# =========================
def evaluate(policy, episodes=5):
    env = gym.make(ENV_NAME, render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

            env.render()

        print(f"Episode {ep} reward: {total_reward:.2f}")

    env.close()


# =========================
# Main
# =========================
def main():
    obs, acts = load_demos(DEMO_PATH)

    trajectories = convert_to_trajectories(
        obs,
        acts,
        EPISODE_LENGTH,
    )

    gail_trainer = train_gail(trajectories)

    print("Evaluating trained policy...")
    evaluate(gail_trainer.gen_algo)


if __name__ == "__main__":
    main()