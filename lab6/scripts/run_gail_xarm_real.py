import argparse
import numpy as np
from stable_baselines3 import PPO

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--robot-ip", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    pose_provider = FixedObjectPoseProvider(
    tcp_pos=pick_locations["red"]   # red cube: best sim-to-real geometry match
    )

    env = XArmPickPlaceRealEnv(
        config=XArmRealEnvConfig(ip=args.robot_ip),
        pose_provider=pose_provider,
    )

    model = PPO.load(args.model_path)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_reward += reward
            done = terminated or truncated

        print(
            f"Episode {ep}: reward={ep_reward:.3f}, "
            f"success={info.get('is_success', False)}"
        )

    env.close()


if __name__ == "__main__":
    main()