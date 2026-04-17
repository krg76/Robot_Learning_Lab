import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

COLUMNS = [
    "iter", "train_reward", "imitation_loss", "disc_loss",
    "disc_loss_expert", "disc_loss_generated", "test_gen_reward",
]

PANELS = [
    {
        "title": "Episode Reward",
        "keys": ["train_reward", "test_gen_reward"],
        "colors": ["#3b82f6", "#f97316"],
        "ylabel": "Reward",
    },
    {
        "title": "Imitation Loss (generator NLL on expert actions)",
        "keys": ["imitation_loss"],
        "colors": ["#8b5cf6"],
        "ylabel": "NLL",
    },
    {
        "title": "Discriminator Loss (total)",
        "keys": ["disc_loss"],
        "colors": ["#ef4444"],
        "ylabel": "BCE",
        "hlines": [("2·ln(2)", 2 * np.log(2))],
    },
    {
        "title": "Discriminator Loss (expert vs generated)",
        "keys": ["disc_loss_expert", "disc_loss_generated"],
        "colors": ["#06b6d4", "#f43f5e"],
        "ylabel": "BCE",
        "hlines": [("ln(2)", np.log(2))],
    },
]


def parse_log(path):
    data = {c: [] for c in COLUMNS}
    with open(path) as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < len(COLUMNS) + 1:
                continue
            vals = [p.strip() for p in parts[1:]]  # skip timestamp
            try:
                int(vals[0])
            except ValueError:
                continue
            for col, val in zip(COLUMNS, vals):
                data[col].append(int(val) if col == "iter" else float(val))
    return {k: np.array(v) for k, v in data.items()}


def plot_metrics(data, output_path=None):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 3.5 * len(PANELS)),
                             sharex=True)
    iters = data["iter"]

    for ax, panel in zip(axes, PANELS):
        for key, color in zip(panel["keys"], panel["colors"]):
            vals = data[key]
            mask = ~np.isnan(vals)
            style = "o-" if mask.sum() < len(vals) else "-"
            ax.plot(iters[mask], vals[mask], style, ms=2, lw=1,
                    label=key, color=color)

        for label, y in panel.get("hlines", []):
            ax.axhline(y, ls="--", color="gray", lw=0.8, label=label)

        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_file")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    data = parse_log(args.log_file)
    print(f"Parsed {len(data['iter'])} iterations")
    plot_metrics(data, args.output)