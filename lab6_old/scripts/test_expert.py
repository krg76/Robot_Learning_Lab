import gymnasium as gym
import gym_xarm
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from models.gail import GAIL
import torch
import json
import numpy as np

# env = gym.make("gym_xarm/XarmPickPlaceDense-v0", render_mode="human")
env = gym.make("gym_xarm/XarmReach-v0")
model = SAC.load("experts/sac_pickplace")

state_dim = len(env.observation_space.high)

if hasattr(env.action_space, "n"):
    discrete = True
    action_dim = env.action_space.n
else:
    discrete = False
    action_dim = env.action_space.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
with open("config.json") as f:
    config = json.load(f)["gym_xarm/XarmReach-v0"]
model = GAIL(state_dim, action_dim, discrete, config)

ckpt_path = "ckpts/gym_xarm/XarmReach-v0"
model.pi.load_state_dict(torch.load(f"{ckpt_path}/policy.ckpt"))
model.v.load_state_dict(torch.load(f"{ckpt_path}/value.ckpt"))
model.d.load_state_dict(torch.load(f"{ckpt_path}/discriminator.ckpt"))

obs, _ = env.reset()
success = 0
for _ in range(1000):
    # action, _ = model.predict(obs, deterministic=True)
    act = model.act(obs)
    act = np.asarray(act, dtype=np.float32).reshape(-1)
    obs, reward, terminated, truncated, info = env.step(act)
    # env.render()

    if terminated or truncated:
        obs, _ = env.reset()

    if info.get("is_success", False):
        success += 1

print(f"Success rate = {success / 1000}")