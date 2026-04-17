import gymnasium as gym
import gym_xarm
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

N_ENVS = 16

def make_env():
    def _init():
        # env = gym.make("gym_xarm/XarmPickPlaceDense-v0")
        env = gym.make("gym_xarm/XarmReach-v0")  
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=N_ENVS,
        ent_coef="auto",   # 🔥 important
    )

    model.learn(total_timesteps=50_000)

    model.save("experts/sac_reach")
    env.close()