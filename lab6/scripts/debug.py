import gymnasium as gym
import gym_xarm
import argparse
import numpy as np

from utils.xarm_pickplace_real_env import (
    XArmRealEnvConfig,
    XArmPickPlaceRealEnv,
    FixedObjectPoseProvider,

)

pick_locations = {
    "red" : [0.475,0.0847,0.172],
    "green" : [0.331,0.233,0.172],
    "yellow" : [0.432,0.199,0.172]
} # in meters

def print_obs_breakdown(obs, name="obs"):
    print(f"\n{name}")
    print("eef        :", obs[0:3])
    print("eef_velp   :", obs[3:6])
    print("obj        :", obs[6:9])
    print("obj_rot    :", obs[9:13])
    print("obj_velp   :", obs[13:16])
    print("obj_velr   :", obs[16:19])
    print("goal       :", obs[19:22])
    print("eef_to_obj :", obs[22:25])
    print("obj_to_goal:", obs[25:28])
    print("scalars    :", obs[28:34])
    print("gripper    :", obs[34:35])


pose_provider = FixedObjectPoseProvider(
    tcp_pos=pick_locations["green"]#(0.35, 0.00, 0.025)  # <-- measure this once relative to base
)

env_real = XArmPickPlaceRealEnv(
        config=XArmRealEnvConfig(ip="192.168.1.216"),
        pose_provider=pose_provider,
    )
obs_real, info = env_real.reset(seed=0)
print_obs_breakdown(obs_real,name="obs real")
env_sim = gym.make("gym_xarm/XarmPickPlaceDense-v0")
obs_sim, _ = env_sim.reset()

print_obs_breakdown(obs_sim,name="obs sim")

