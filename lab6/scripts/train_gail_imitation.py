import numpy as np
import gymnasium as gym
import gym_xarm  # registers envs
import os

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util import logger as imit_logger

custom_logger = imit_logger.configure(
    folder="./logs/gail/",
    format_strs=["stdout", "tensorboard", "csv"],
)


SEED = 42

# =========================
# Config
# =========================
# ENV_NAME = "gym_xarm/XarmReach-v0"
# SAC_MODEL_PATH = "experts/sac_reach"

ENV_NAME = "gym_xarm/XarmPickPlaceDense-v0"
SAC_MODEL_PATH = "experts/sac_pickplace"

SAVE_PATH = "policies/gail_ppo_pickplace_3_50_2"

N_ENVS = 8
N_EXPERT_EPISODES = 240#120#60
TOTAL_GAIL_TIMESTEPS = 20_000


# =========================
# Main
# =========================
def main():
    rng = np.random.default_rng(SEED)

    # Vec env for rollouts + GAIL
    env = make_vec_env(
        ENV_NAME,
        rng=rng,
        n_envs=N_ENVS,
        max_episode_steps=200,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    )

    # -------------------------
    # Load SAC expert
    # -------------------------
    expert = SAC.load(SAC_MODEL_PATH, env=env)

    # -------------------------
    # Collect true episodic expert rollouts
    # -------------------------
    print("Collecting SAC expert rollouts...")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=N_EXPERT_EPISODES),
        rng=np.random.default_rng(SEED),
    )
    rollouts = [t for t in rollouts if np.linalg.norm(t.obs[-1][25:28]) <= 0.015]

    #print(rollouts)
    #rollouts = [t for t in rollouts if t.infos and t.infos[-1].get("is_success", False)] 

    total_steps = sum(len(traj.acts) for traj in rollouts)
    print(f"Collected {len(rollouts)} expert trajectories")
    print(f"Total expert transitions: {total_steps}")

    if len(rollouts) > 0:
        print("Example trajectory shapes:")
        print("obs:", rollouts[0].obs.shape)
        print("acts:", rollouts[0].acts.shape)

    # -------------------------
    # PPO learner (generator)
    # -------------------------
    learner = PPO(
        policy=MlpPolicy,
        env=env,
        batch_size=256,
        ent_coef=0.0,
        learning_rate=5e-8,#4e-4,
        gamma=0.95,#0.95,
        n_epochs=5,
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/gail_tensorboard/",
    )
    if os.path.exists(SAVE_PATH + ".zip"):
        learner = PPO.load(SAVE_PATH)#ADDED!!!

    # -------------------------
    # Reward net / discriminator
    # -------------------------
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # -------------------------
    # GAIL trainer
    # -------------------------
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=custom_logger,
    )

    # -------------------------
    # Evaluate before training
    # -------------------------
    print("\nEvaluating learner before training...")
    learner_rewards_before, _ = evaluate_policy(
        learner,
        env,
        n_eval_episodes=100,
        return_episode_rewards=True,
        deterministic=True,
    )
    print("Mean reward before training:", float(np.mean(learner_rewards_before)))

    # -------------------------
    # Train GAIL
    # -------------------------
    print("\nStarting GAIL training...")
    gail_trainer.train(TOTAL_GAIL_TIMESTEPS)

    # -------------------------
    # Save trained policy
    # -------------------------
    print("\nSaving trained policy...")
    learner.save(SAVE_PATH)
    print(f"Policy saved to {SAVE_PATH}")

    # -------------------------
    # Evaluate after training
    # -------------------------
    print("\nEvaluating learner after training...")
    learner_rewards_after, _ = evaluate_policy(
        learner,
        env,
        n_eval_episodes=100,
        return_episode_rewards=True,
        deterministic=True,
    )
    print("Mean reward after training:", float(np.mean(learner_rewards_after)))

    # -------------------------
    # Optional rendered eval
    # -------------------------
    print("\nRunning rendered evaluation...")
    if True:
        print("\nRunning rendered evaluation...")
        try:
            render_env = gym.make(ENV_NAME, render_mode="human")
        except Exception:
            render_env = gym.make(ENV_NAME)

        success = 0

        for ep in range(100):
            obs, _ = render_env.reset(seed=SEED + ep)
            done = False
            total_reward = 0.0

            while not done:
                action, _ = learner.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = render_env.step(action)
                total_reward += reward
                done = terminated or truncated
                render_env.render()

            if info["is_success"]:
                success += 1

            print(f"[Rendered Eval] Episode {ep}: reward={total_reward:.2f}, success={info['is_success']}")

        print(f"\nSuccess rate: {success}/10 ({success * 10:.0f}%)")
        render_env.close()
        env.close()
    else:
        render_env = env  # assumed VecEnv
        n_envs = render_env.num_envs

        n_episodes = 100
        episode_counts = np.zeros(n_envs, dtype=int)
        episode_rewards = np.zeros(n_envs)
        success = 0
        finished_episodes = 0

        obs = render_env.reset()

        while finished_episodes < n_episodes:
            actions, _ = learner.predict(obs, deterministic=True)
            obs, rewards, dones, infos = render_env.step(actions)

            episode_rewards += rewards

            render_env.render()

            for i in range(n_envs):
                if dones[i]:
                    ep_reward = episode_rewards[i]
                    is_success = infos[i].get("is_success", False)

                    print(f"[Rendered Eval] Env {i} Episode {episode_counts[i]}: "
                        f"reward={ep_reward:.2f}, success={is_success}")

                    if is_success:
                        success += 1

                    episode_counts[i] += 1
                    finished_episodes += 1
                    episode_rewards[i] = 0.0

                    if finished_episodes >= n_episodes:
                        break

        print(f"\nSuccess rate: {success}/{n_episodes} ({100 * success / n_episodes:.1f}%)")

        render_env.close()
        env.close()

if __name__ == "__main__":
    main()