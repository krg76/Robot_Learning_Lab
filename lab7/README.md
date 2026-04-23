# Lab 7 — Vision-Language-Action Models (VLAs)

## Objectives

By the end of this lab, you will:

- Evaluate **Pi0-FAST** and **Pi0.5** out-of-the-box on a physical robot
- Analyze how **fine-tuning** affects model performance on in-distribution tasks
- Probe **distribution shift** robustness by varying objects and language instructions
- Test **out-of-distribution generalization** on novel tasks not seen during fine-tuning
- Compare the two models' behavior qualitatively and quantitatively across all conditions

---

## Setup

Add setup instructions (conda env, OpenPi repo version, robot connection, etc.)

```bash
conda env create -f lab6.yml
conda activate lab6
```

Make sure you have the correct version of the OpenPi repos before starting.

---

## Code Structure
```
lab7/
├── policies/
│   ├── pi0_fast/        # Pi0-FAST checkpoints
│   └── pi0_5/           # Pi0.5 checkpoints (base + fine-tuned)
├── scripts/
│   ├── run_pi0_fast.py  # Inference script for Pi0-FAST
│   └── run_pi0_5.py     # Inference script for Pi0.5
├── tasks/
│   └── task_list.md     # Task definitions and language instructions
├── videos/              # Recorded rollout videos
└── logs/                # Success rate logs
```

---

## Tasks

| Task ID | Description | Language Instruction | Datasize |
|---------|-------------|----------------------|---------------------------|
| T1      | Lift object | "lift the eggplant, put the eggplant in the pot"  | High  |
| T2      | Knock object | "knock the white condiment bottle" | Medium |
| T3      | Wipe surface | "wipe the stove top" | Low |

Some other tasks include:

```
{"task_index": 0, "task": "put the pot in the sink"}
{"task_index": 1, "task": "stir the pot"}
{"task_index": 2, "task": "lift the corn"}
{"task_index": 3, "task": "lift the tomato sauce can"}
{"task_index": 4, "task": "knock the tomato sauce can"}
{"task_index": 5, "task": "wipe the stove top"}
{"task_index": 6, "task": "lift the pot"}
{"task_index": 7, "task": "put the eggplant in the pot"}
{"task_index": 8, "task": "put the eggplant in the sink"}
{"task_index": 9, "task": "put the corn in the sink"}
{"task_index": 10, "task": "put the apple in the pot"}
{"task_index": 11, "task": "lift the apple"}
{"task_index": 12, "task": "lift the white condiment bottle"}
{"task_index": 13, "task": "put the corn in the pot"}
{"task_index": 14, "task": "lift the green rag"}
{"task_index": 15, "task": "lift the eggplant"}
{"task_index": 16, "task": "knock the eggplant"}
{"task_index": 17, "task": "put the egglant in the pot"}
{"task_index": 18, "task": ""}
{"task_index": 19, "task": "knock the white condiment bottle"}
{"task_index": 20, "task": "put the eggplant in the sinkput the eggplant in the sink"}
{"task_index": 21, "task": " lift the eggplant"}
```

---

# Part 1 — Evaluate Out-of-the-Box Models

### Background

Before running any trials, review:

- The **data mixture** that Pi0-FAST and Pi0.5 were trained over
- The **tasks** defined above, paying attention to differences in robot embodiment, environment observations, and task specifications between the training and test domains

### Reflection 1

Before running any trials, write 1–2 paragraphs addressing:

- How do you expect the two models to perform out-of-the-box?
- How do you expect their performance to differ from each other?
- How do you expect each model to perform on each individual task?

### Procedure

Run **3 trials** of each model on each task. Keep objects the same across trials but vary their positions slightly.

```bash
# Pi0-FAST
python scripts/run_pi0_fast.py --task T1

# Pi0.5
python scripts/run_pi0_5.py --task T1
```

### Deliverables

**Deliverable 1 — Videos:** Record one video per model per task (6 videos total).
```
videos/part1/
├── pi0_fast_T1.mp4
├── pi0_fast_T2.mp4
├── pi0_fast_T3.mp4
├── pi0_5_T1.mp4
├── pi0_5_T2.mp4
└── pi0_5_T3.mp4
```
**Deliverable 2 — Success Rate Table:** Report success rates (successes / 3 trials) for each model and task.

|           | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Pi0-FAST  |        |        |        |
| Pi0.5     |        |        |        |

### Reflection 2

- Did the models perform as you expected? Why or why not?
- In the failure cases, why did the robot fail?

### Reflection 3

- Qualitatively, how does the robot's behavior differ between the two models?
- How consistent is each model's behavior across rollouts?

---

# Part 2 — Evaluate Fine-Tuned Models

### Background

Review the **data mixture** used to fine-tune the two models before running any trials.

### Reflection 4

Before running any trials, write 1–2 paragraphs addressing:

- How do you expect the two models to perform after fine-tuning?
- How do you expect their performance to differ from each other?

---

## Part 2A — Within-Distribution

Run the same 3-trial protocol from Part 1, using the fine-tuned checkpoints.

```bash
# Pi0-FAST (fine-tuned)
python scripts/run_pi0_fast.py --task T1 --checkpoint finetuned

# Pi0.5 (fine-tuned)
python scripts/run_pi0_5.py --task T1 --checkpoint finetuned
```

### Deliverables

**Deliverable — Videos:** Record one video per model per task (6 videos total).
```
videos/part2a/
├── pi0_fast_T1.mp4
...
```

**Deliverable — Success Rate Table:**

|           | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Pi0-FAST  |        |        |        |
| Pi0.5     |        |        |        |

### Reflection

- Did the models perform as you expected? Why or why not?
- In the failure cases, why did the robot fail?

---

## Part 2B — Distribution Shift

Repeat the 3-trial protocol, but introduce variations of your choice — for example, replacing objects with visually similar alternatives or rephrasing the language instructions.

### Reflection

- What variations did you choose, and why do you consider them a "distribution shift"?

### Deliverables

**Deliverable — Videos:** 6 videos (one per model per task).

videos/part2b/
├── pi0_fast_T1.mp4
...

**Deliverable — Success Rate Table:**

|           | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Pi0-FAST  |        |        |        |
| Pi0.5     |        |        |        |

### Reflection

- Did the models perform as you expected under distribution shift? Why or why not?
- In the failure cases, what was the likely cause?

---

## Part 2C — Out-of-Distribution Generalization

Pick a **new task** not included in the fine-tuning data that you believe is still plausibly within the model's capabilities. This should be a mixture between two known tasks. E.g. "put the bowl in the sink" and "put the carrot in the pot" -> "put the pot in the sink". "Put the rubber duck into the sink". Run 3 trials per model.

### Reflection

- What OOD task did you choose, and why do you think it is feasible for the model?

### Deliverables

**Deliverable — Videos:** 2 videos (one per model).
```
videos/part2c/
├── pi0_fast_OOD.mp4
└── pi0_5_OOD.mp4
```
**Deliverable — Success Rate Table:**

|           | OOD Task |
|-----------|----------|
| Pi0-FAST  |          |
| Pi0.5     |          |

### Reflection

- Did the models perform as you expected? Why or why not?
- In the failure cases, why did the robot fail?

---

## Reflection Questions

- How does fine-tuning affect each model's behavior compared to the base checkpoint?
- What types of distribution shift had the largest effect on performance? Why?
- How does each model handle novel language instructions — does it generalize or fail gracefully?
- What are the key differences between Pi0-FAST and Pi0.5 in terms of architecture and training? How do those differences manifest in behavior?
- What would you need to do to improve robustness to distribution shift?

---

## Submission Checklist

- [ ] Reflection 1: Pre-trial expectations for out-of-the-box performance
- [ ] 6 videos from Part 1 (out-of-the-box evaluation)
- [ ] Part 1 success rate table
- [ ] Reflections 2 & 3: Analysis of out-of-the-box results
- [ ] Reflection 4: Pre-trial expectations for fine-tuned performance
- [ ] 6 videos from Part 2A (within-distribution)
- [ ] Part 2A success rate table + reflection
- [ ] 6 videos from Part 2B (distribution shift)
- [ ] Part 2B success rate table + reflection
- [ ] 2 videos from Part 2C (OOD generalization)
- [ ] Part 2C success rate + reflection
- [ ] GitHub repo link
