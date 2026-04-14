import pandas as pd
import matplotlib.pyplot as plt


def plot_rl_results(csvs: list):
    # Load CSVs
    ppo_sparse = pd.read_csv(csvs[0])
    sac_sparse = pd.read_csv(csvs[1])
    ppo_semi = pd.read_csv(csvs[2])
    sac_semi = pd.read_csv(csvs[3])
    ppo_dense = pd.read_csv(csvs[4])
    sac_dense = pd.read_csv(csvs[5])

    # Helper to extract safely
    def get_xy(df):
        x = df["time/total_timesteps"]
        y = df["rollout/success_rate"]
        return x, y

    # Extract data
    ppo_sparse_x, ppo_sparse_y = get_xy(ppo_sparse)
    sac_sparse_x, sac_sparse_y = get_xy(sac_sparse)

    ppo_semi_x, ppo_semi_y = get_xy(ppo_semi)
    sac_semi_x, sac_semi_y = get_xy(sac_semi)

    ppo_dense_x, ppo_dense_y = get_xy(ppo_dense)
    sac_dense_x, sac_dense_y = get_xy(sac_dense)

    # ---- Plot: Epochs vs Success Rate (6 lines) ----
    plt.figure(figsize=(10, 6))

    # Dense
    plt.plot(ppo_dense_x, ppo_dense_y, label="PPO (Dense)", linewidth=2)
    plt.plot(sac_dense_x, sac_dense_y, label="SAC (Dense)", linewidth=2)

    # Semi
    plt.plot(ppo_semi_x, ppo_semi_y, label="PPO (Semi)", linestyle="--", linewidth=2)
    plt.plot(sac_semi_x, sac_semi_y, label="SAC (Semi)", linestyle="--", linewidth=2)

    # Sparse
    plt.plot(ppo_sparse_x, ppo_sparse_y, label="PPO (Sparse)", linestyle=":", linewidth=2)
    plt.plot(sac_sparse_x, sac_sparse_y, label="SAC (Sparse)", linestyle=":", linewidth=2)

    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Timesteps (PPO vs SAC across reward types)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("success_rate_all.png")
    plt.show()


if __name__ == "__main__":
    ppo_csv_sparse = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/ppo_gym_xarm/XarmPickPlaceSparse-v0/progress.csv"
    sac_csv_sparse = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/sac_gym_xarm/XarmPickPlaceSparse-v0/progress.csv"
    ppo_csv_semi = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/ppo_gym_xarm/XarmPickPlaceSemi-v0/progress.csv"
    sac_csv_semi = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/sac_gym_xarm/XarmPickPlaceSemi-v0/progress.csv"
    ppo_csv_dense = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/ppo_gym_xarm/XarmPickPlaceDense-v0/progress.csv"
    sac_csv_dense = "/home/kyle_golobish/Desktop/Robot_learning/Robot_Learning_Lab/lab5/scripts/asset/sac_gym_xarm/XarmPickPlaceDense-v0/progress.csv"

    csvs = [
        ppo_csv_sparse, sac_csv_sparse,
        ppo_csv_semi, sac_csv_semi,
        ppo_csv_dense, sac_csv_dense
    ]

    plot_rl_results(csvs)