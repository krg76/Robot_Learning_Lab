import gymnasium as gym
import gym_xarm
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

env = gym.make("gym_xarm/XarmPickPlaceDense-v0", render_mode="human")

model = SAC.load("experts/sac_pickplace")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()