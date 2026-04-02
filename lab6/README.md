# Lab 6 — Generative Adversarial Imitation Learning (GAIL)

## Objectives

By the end of this lab, you will:

- Understand the **GAIL algorithm** and how it combines imitation learning with adversarial training
- Implement the **loss function**, **action sampling**, and **network initializations** for GAIL
- Train a GAIL policy using **pre-collected demonstration data**
- Analyze training via **imitation loss** and **discriminator loss** plots
- Deploy a trained policy to a **physical robot** and evaluate sim-to-real transfer

---

## Setup

Create the conda environment:

```bash
conda env create -f lab6.yml
conda activate lab6
```

Install the xArm gym environment:

```bash
git clone git@github.com:iqr-lab/gym-xarm.git
cd gym_xarm
pip install -e .
```

Verify the environment renders correctly:

```bash
MUJOCO_GL=glfw python scripts/test_expert.py
```

> Hit Tab once when the sim appears if the camera is not aligned, to switch the camera view.

---

## Code Structure

```
lab6/
├── models/
│   ├── gail.py          # GAIL training loop (generator + discriminator + value network)
│   └── nets.py          # PolicyNetwork, ValueNetwork, Discriminator, Expert
├── scripts/
│   ├── train.py         # Entry point for training
│   └── plot.py          # Plotting utilities for training logs
├── utils/
│   └── funcs.py         # TRPO helpers (conjugate gradient, line search, etc.)
├── experts/             # Pre-trained expert checkpoints
├── ckpts/               # Saved model checkpoints from training
├── logs/                # Training logs (parsed by plot.py)
├── config.json          # Hyperparameters per environment
└── lab6.yml             # Conda environment file
```

---

# Part 1 — Implement Missing GAIL Code

Before editing any code, you should:

1. **Revisit the GAIL pseudocode** (Ho & Ermon, 2016)
2. **Read through the full implementation** in `models/gail.py` and `models/nets.py`
3. **Write comments** in the code explaining how each section maps to the pseudocode

Then fill in the following missing pieces:

### 1. Loss Function

In `models/gail.py`, locate the discriminator training step. The discriminator is trained to distinguish expert trajectories (label 0) from generated trajectories (label 1).

The loss is binary cross-entropy applied to both:

- Expert state-action pairs → label `0` (real)
- Generator state-action pairs → label `1` (fake)

Implement this combined loss and call `.backward()` and `.step()`.

### 2. Sample Actions from the Generator Policy

In `models/gail.py`, inside the `act()` function, implement how actions are sampled from the policy:

- Call `self.pi(state)` to get the action distribution
- Sample using `.sample()`
- Detach and convert to numpy
- Remove the batch dimension
- Clip actions to [-1, 1]

This function is used during trajectory collection.

### 3. Initialize Generator, Discriminator, and Value Networks

In `models/gail.py` (the `__init__` method), initialize:

- `self.pi` — a `PolicyNetwork(state_dim, action_dim, discrete)`
- `self.v` — a `ValueNetwork(state_dim)`
- `self.d` — a `Discriminator(state_dim, action_dim, discrete)`

These classes are defined in `models/nets.py`.

---

### Run Training

You must specify how to obtain demonstrations:

```bash
# Option 1: Use SAC expert (recommended)
python scripts/train.py --use_sac

# Option 2: Use saved demonstrations
python scripts/train.py --load_demos --demo_path ckpts/<env>/sac_demos.npz
```

Supported environments:

| Environment                     | Type       |
|---------------------------------|------------|
| `CartPole-v1`                   | Discrete   |
| `Pendulum-v0`                   | Continuous |
| `BipedalWalker-v3`              | Continuous |
| `gym_xarm/XarmPickPlaceDense-v0`| Continuous |
| `gym_xarm/XarmReach-v0`         | Continuous |

Checkpoints are saved to `ckpts/<env_name>/`:

```
ckpts/
└── CartPole-v1/
    ├── policy.ckpt
    ├── value.ckpt
    ├── discriminator.ckpt
    ├── results.pkl
    └── model_config.json
```

Training logs are written to `logs/` and can be plotted with `scripts/plot.py`.

---

# Part 2 — Visualize Training in Simulation

Using pre-collected demonstration data for **reach**, train the GAIL model and evaluate it in simulation.

### Train the Model

```bash
python scripts/train.py --use_sac
```

### Plot Training Metrics

```bash
python scripts/plot.py -l logs/reach.log -o plots/reach_training.png
```

The plot script generates four panels:

| Panel | Metric | Description |
|-------|--------|-------------|
| 1 | Episode Reward | Train reward and test generator reward over iterations |
| 2 | Imitation Loss | Generator NLL on expert actions (lower = better imitation) |
| 3 | Discriminator Loss (total) | Sum of expert + generated BCE losses |
| 4 | Discriminator Loss (expert vs generated) | Per-component BCE; at equilibrium both approach `ln(2)` |

### Deliverable — Plots

Submit the following plots:

- **Imitation loss** for the generator over training iterations
- **Discriminator loss** for generated trajectories vs. demonstration trajectories

Include a brief written interpretation (1–2 paragraphs):

- Does the generator improve at imitating the expert?
- Does the discriminator reach equilibrium (losses converge to `ln(2)`)?
- Are there signs of instability or mode collapse?

---

# Part 3 — Test on Physical Robot

After achieving good performance in simulation, perform **sim-to-real testing**.

### Safety Requirements

- Set robot speed to low
- Define workspace limits before running
- Have emergency stop ready
- Operate under supervision at all times

### Procedure

1. Verify simulation rollouts look safe and consistent
2. Transfer the trained policy checkpoint to the robot deployment system
3. Run the policy on the physical robot for the pick-and-place task

### Deliverable — Video

Record a video of the robot's performance:

```
videos/real_robot_test.mp4
```

Include in your write-up:

- How did real performance compare to simulation?
- Were motions slower or less accurate?
- Did any failures occur? What were the likely causes (dynamics mismatch, friction, sensor noise, delay)?
- How would you reduce the sim-to-real gap?

---

## Reflection Questions

- How does the discriminator shape the generator's reward signal?
- What happens to training if the discriminator becomes too strong too quickly?
- How does GAIL differ from behavioral cloning? When would you prefer one over the other?
- What role does the causal entropy regularization term (`lambda`) play?
- How does GAE (Generalized Advantage Estimation) improve the policy update?

---

## Submission Checklist

- [ ] Commented code aligning implementation to GAIL pseudocode
- [ ] Completed loss function implementation
- [ ] Completed action sampling implementation
- [ ] Completed network initializations
- [ ] Imitation loss plot over training iterations
- [ ] Discriminator loss plot (generated vs. demonstration)
- [ ] Written analysis of training curves (1–2 paragraphs)
- [ ] Video of real robot performance
- [ ] Written sim-to-real reflection
- [ ] GitHub repo link
