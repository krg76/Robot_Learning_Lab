import pandas as pd
import matplotlib.pyplot as plt

def plot_rl_results(ppo_csv_path, sac_csv_path):
    # Load CSVs
    ppo_df = pd.read_csv(ppo_csv_path)
    sac_df = pd.read_csv(sac_csv_path)

    # Use iterations as "epochs"
    ppo_epochs = ppo_df["time/total_timesteps"]
    sac_epochs = sac_df["time/total_timesteps"]

    # Extract metrics
    ppo_success = ppo_df["rollout/success_rate"]
    sac_success = sac_df["rollout/success_rate"]

    ppo_reward = ppo_df["rollout/ep_rew_mean"]
    sac_reward = sac_df["rollout/ep_rew_mean"]

    # ---- Plot 1: Epochs vs Success Rate ----
    plt.figure(figsize=(8, 5))
    plt.plot(ppo_epochs, ppo_success, label="PPO", linewidth=2)
    plt.plot(sac_epochs, sac_success, label="SAC", linewidth=2)
    plt.xlabel("Epochs (Iterations)")
    plt.ylabel("Success Rate")
    plt.title("Epochs vs Success Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_rate_plot.png")
    plt.show()

    # ---- Plot 2: Epochs vs Reward ----
    plt.figure(figsize=(8, 5))
    plt.plot(ppo_epochs, ppo_reward, label="PPO", linewidth=2)
    plt.plot(sac_epochs, sac_reward, label="SAC", linewidth=2)
    plt.xlabel("Epochs (Iterations)")
    plt.ylabel("Mean Reward")
    plt.title("Epochs vs Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward_plot_semi_pnp.png")
    plt.show()


if __name__ == "__main__":
    # Replace with your actual file paths
    ppo_csv = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/ppo_gym_xarm/XarmPickPlaceSemi-v0/progress.csv"
    sac_csv = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/sac_gym_xarm/XarmPickPlaceSemi-v0/progress.csv"

    plot_rl_results(ppo_csv, sac_csv)