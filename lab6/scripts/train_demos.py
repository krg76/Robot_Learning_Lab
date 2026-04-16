import os
import json
import pickle
import argparse

import torch
import gymnasium as gym
import gym_xarm  # registers envs on import
import numpy as np

from models.gail import GAIL
from utils.demo_collect import RuleBasedPickPlacePolicy, TrainedExpertPolicy, collect_rule_based_demos, collect_expert_demos
from scripts.train_expert import Policy  # reuse BC policy
from stable_baselines3 import SAC
from utils.demo_collect import SACExpertPolicy, collect_sac_demos

# =========================
# Main
# =========================
def main(args):
    rule = True
    ckpt_path = "ckpts"
    os.makedirs(ckpt_path, exist_ok=True)

    # -------------------------
    # Load config
    # -------------------------
    with open("config.json") as f:
        config = json.load(f)[args.env_name]

    ckpt_path = os.path.join(ckpt_path, args.env_name)
    os.makedirs(ckpt_path, exist_ok=True)

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # -------------------------
    # Create env
    # -------------------------
    env = gym.make(args.env_name)
    env.reset()

    state_dim = len(env.observation_space.high)

    if hasattr(env.action_space, "n"):
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Collect expert demos
    # =========================
    if args.load_demos and os.path.exists(args.demo_path):

        print("Loading saved demonstrations...")
        data = np.load(args.demo_path)
        expert_obs = data["obs"]
        expert_acts = data["acts"]
        rewards = None
    
    elif args.use_sac:

        print("Collecting SAC expert demonstrations...")

        # sac_model = SAC.load("experts/sac_pickplace")
        sac_model = SAC.load("experts/sac_reach")

        sac_policy = SACExpertPolicy(sac_model)

        expert_obs, expert_acts, rewards = collect_sac_demos(
            env,
            sac_policy,
            num_steps=5000,
            horizon=config["horizon"],
        )
        # optional: save for reuse
        np.savez(os.path.join(ckpt_path, "sac_demos.npz"),
                obs=expert_obs,
                acts=expert_acts)

    elif args.use_bc:

        print("Collecting demonstrations from BC policy...")

        # load BC policy
        policy = Policy(state_dim, action_dim).to(device)
        policy.load_state_dict(torch.load(args.bc_ckpt, map_location=device))
        policy.eval()

        obs_buf, act_buf = [], []
        rewards = []

        obs, _ = env.reset()
        ep_ret = 0

        for step in range(20000):

            # IMPORTANT: stochastic sampling
            action = policy.act(obs, deterministic=False)

            next_obs, r, term, trunc, info = env.step(action)

            obs_buf.append(obs)
            act_buf.append(action)

            ep_ret += r
            obs = next_obs

            if term or trunc:
                rewards.append(ep_ret)
                obs, _ = env.reset()
                ep_ret = 0

            if step % 1000 == 0:
                print(f"Collected {step} steps")

        expert_obs = np.array(obs_buf)
        expert_acts = np.array(act_buf)

        print(f"BC demos collected: {len(expert_obs)} samples")
        print(f"Mean episode reward: {np.mean(rewards) if rewards else 0:.4f}")

        # optional: save for reuse
        np.savez(os.path.join(ckpt_path, "bc_demos.npz"),
                obs=expert_obs,
                acts=expert_acts)

    elif args.use_rule:

        print("Collecting rule-based demonstrations...")

        rb_policy = RuleBasedPickPlacePolicy()

        expert_obs, expert_acts, rewards = collect_rule_based_demos(
            env,
            rb_policy,
            num_steps=20000,
            horizon=config["horizon"],
        )

    elif args.use_trained:

        print("Collecting expert demonstrations from trained policy...")

        expert_policy = TrainedExpertPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            ckpt_path="experts/policy.pt",
            device=device,
        )

        expert_obs, expert_acts, rewards = collect_expert_demos(
            env,
            expert_policy,
            num_steps=20000,
            horizon=config["horizon"],
        )

    else:
        raise ValueError("Specify --use_rule or --use_trained or --load_demos")

    # =========================
    # Train GAIL
    # =========================
    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    results = model.train(
        env,
        expert_obs=expert_obs,
        expert_acts=expert_acts,
    )

    env.close()

    # -------------------------
    # Save results
    # -------------------------
    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(),
            os.path.join(ckpt_path, "policy.ckpt"),
        )

    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(),
            os.path.join(ckpt_path, "value.ckpt"),
        )

    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(),
            os.path.join(ckpt_path, "discriminator.ckpt"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="gym_xarm/XarmReach-v0",#"gym_xarm/XarmPickPlaceDense-v0"
    )
    parser.add_argument("--use_rule", action="store_true")
    parser.add_argument("--use_trained", action="store_true")
    parser.add_argument("--load_demos", action="store_true")
    parser.add_argument("--use_bc", action="store_true")
    parser.add_argument("--use_sac", action="store_true")
    parser.add_argument("--bc_ckpt", type=str, default="experts/policy_bc.pt")
    args = parser.parse_args()

    main(args)