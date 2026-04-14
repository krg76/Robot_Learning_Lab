"""
Generic RL training script using Stable-Baselines3.
Supports PPO and SAC on any Gym-compatible environment.

Designed for:
- Comparing PPO vs SAC
- Testing reward functions
- Plotting reward + success curves
"""

import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

# RL algorithms
from stable_baselines3 import HerReplayBuffer, PPO, SAC

# Utilities
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

import gymnasium as gym
import gym_xarm  # registers envs on import

# import panda_gym

ENV_NAME = "gym_xarm/XarmPickPlaceSparse-v0"#"gym_xarm/XarmPickPlaceDense-v0"
#"gym_xarm/XarmPickPlaceDense-v0"

# ============================================================
# Callback for Logging Episode Metrics
# ============================================================
class MetricsCallback(BaseCallback):
    """
    Custom callback to record:
    - Episode rewards
    - Success rates (if environment provides 'is_success')

    Called automatically during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self):
        """
        This function runs at every environment step.
        We check if an episode has finished (done=True),
        then log metrics.
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done:
                ep_info = infos[i]

                # Stable-Baselines3 Monitor wrapper
                # stores episode reward under "episode"
                if "episode" in ep_info:
                    self.episode_rewards.append(ep_info["episode"]["r"])

                # Optional success metric
                if "is_success" in ep_info:
                    self.episode_success.append(float(ep_info["is_success"]))

        return True


# ============================================================
# Model Factory
# ============================================================
def make_model(algo, env, args):
    """
    Creates PPO or SAC model with user-defined hyperparameters.
    """

    # Define neural network architecture for policy
    policy_kwargs = dict(
        net_arch=[256, 256],   # e.g., [256, 256]
        activation_fn=torch.nn.ReLU  # activation function
    )

    goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

    if algo == "ppo":
        # PPO is on-policy
        model = PPO(
            "MlpPolicy",
            #"MultiInputPolicy",
            env,
            # replay_buffer_class=HerReplayBuffer,
            # # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=4,
            #     goal_selection_strategy=goal_selection_strategy,
            # ),
            learning_rate=1e-3,#args.lr,
            gamma=0.95,#args.gamma,            # discount factor
            clip_range=0.2,#args.clip_range,  # PPO clipping parameter
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    elif algo == "sac":
        # Convert ent_coef if numeric
        ent_coef = "auto"
        if ent_coef not in ["auto"] and not ent_coef.startswith("auto"):
            ent_coef = float(ent_coef)

        # SAC is off-policy with entropy regularization
        model = SAC(
            "MlpPolicy",
            #"MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=16,
            ent_coef="auto",   # 🔥 important
        )
    else:
        raise ValueError("Unsupported algorithm")

    return model


# ============================================================
# Evaluation Function
# ============================================================
def evaluate(model, env, n_rollouts=10):
    """
    Runs deterministic rollouts for evaluation.
    Returns:
        mean_reward
        mean_success (if available)
    """

    rewards = []
    success = []

    for rollout in range(n_rollouts):
        print(f"Eval rollout {rollout}")
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0
        # print(done)

        while not np.all(done):
            # Deterministic=True → no exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # print(f"done = {done}")
            ep_reward += reward

        rewards.append(ep_reward)

        # Optional success flag
        if "is_success" in info:
            success.append(float(info["is_success"]))

    mean_reward = np.mean(rewards)
    mean_success = np.mean(success) if success else None

    return mean_reward, mean_success

def make_env():
    def _init():
        env = gym.make(ENV_NAME)#"gym_xarm/XarmPickPlaceDense-v0")
        # env = gym.make("gym_xarm/XarmReach-v0")  
        env = Monitor(env)
        return env
    return _init

# ============================================================
# Main Training Function
# ============================================================
def main(args):

    os.makedirs(args.log_dir, exist_ok=True)

    # --------------------------------------------------------
    # Create Gym environment
    # --------------------------------------------------------
    # This initializes the simulation environment specified by
    # --env (e.g., "Pendulum-v1", "HalfCheetah-v4", etc.).
    # The environment defines:
    #   - Observation space (state representation)
    #   - Action space (control inputs)
    #   - Reward function
    #   - Termination conditions
    # The RL algorithm will interact with this environment
    # during training to collect experience.

    # create vectorized env
    # vec_env = make_vec_env(
    #     _make_env(args.env, "rgb_array"),
    #     n_envs=16,
    #     seed=0,
    #     vec_env_cls=SubprocVecEnv
    # )

    # # normalize observations and rewards
    # env = VecNormalize(
    #     vec_env,
    #     norm_obs=True,
    #     norm_reward=True,
    #     clip_obs=100.0
    # )

    env = SubprocVecEnv([make_env() for _ in range(16)])

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    
    model = make_model(args.algo, env, args)

    callback = MetricsCallback()

    print(f"Training {args.algo.upper()} on {args.env}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    new_logger = configure(f"./{args.log_dir}{args.algo}_{args.env}/", ["stdout", "csv"])
    model.set_logger(new_logger)
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback
    )

    # Save trained model
    os.makedirs(os.path.join(args.log_dir, f"{args.algo}_{args.env}"), exist_ok=True)
    model.save(os.path.join(args.log_dir, f"{args.algo}_{args.env}"))
    
    # --------------------------------------------------------
    # Evaluate after training
    # --------------------------------------------------------
    # print("Evaluating")
    # model_path = os.path.join(args.log_dir, f"{args.algo}_{args.env}")
    # model = PPO.load(model_path, env=env)
    # mean_reward, mean_success = evaluate(model, env)

    # print("Mean reward:", mean_reward)
    # print("Mean success:", mean_success)

   

    # --------------------------------------------------------
    # Plot Reward Curve
    # --------------------------------------------------------
    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.title("Episode Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(args.log_dir, f"{args.algo}_{args.env}/reward_curve.png"))

    # --------------------------------------------------------
    # Plot Success Curve (if available)
    # --------------------------------------------------------
    if callback.episode_success:
        plt.figure()
        plt.plot(callback.episode_success)
        plt.title("Success Rate vs Episode")
        plt.xlabel("Episode")
        plt.ylabel("Success")
        plt.savefig(os.path.join(args.log_dir, f"{args.algo}_{args.env}/success_curve.png"))

    print("Plots saved to:", args.log_dir)


# ============================================================
# Argument Parser
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Environment name (Gym registry)
    parser.add_argument("--env", type=str, default=ENV_NAME)

    # Algorithm choice
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True)

    # Training length
    parser.add_argument("--timesteps", type=int, default=1000000)

    # Learning rate
    parser.add_argument("--lr", type=float, default=1e-3)

    # Discount factor
    parser.add_argument("--gamma", type=float, default=0.95)

    # PPO clip parameter: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    parser.add_argument("--clip_range", type=float, default=0.2)

    # SAC entropy regularization: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
    parser.add_argument(
        "--ent_coef",
        type=str,
        default="auto",
        help="Entropy coefficient for SAC. Examples: 'auto', 'auto_0.1', '0.2'"
    )

    # Policy network architecture: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    parser.add_argument(
        "--policy_arch",
        nargs="+",
        type=int,
        default=[256, 256],
        help="Hidden layer sizes"
    )

    # # Reward type for pick and place
    # parser.add_argument(
    #     "reward_type",
    #     type=str,
    #     required=False,
    #     default="dense",

    # )

    # Output directory
    parser.add_argument("--log_dir", type=str, default="asset/")

    args = parser.parse_args()

    main(args)