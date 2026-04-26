---
title: API Drift Gym
emoji: "🌀"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.31.0"
python_version: "3.11"
app_file: app.py
pinned: false
tags:
  - openenv
---

# 🌀 API Drift Gym

### Can a 0.5B model learn to survive when the API keeps lying to it?

We gave a tiny language model a broken API, a schema that silently mutates mid-session, and zero documentation. No hardcoded rules. No lookup tables. Just a malformed response and four tools to figure it out.

By episode 10, it had learned to stop blindly retrying. It started inspecting first, transforming the request, then calling again — exactly the way a senior engineer would debug a production incident at 2 AM.

**This is API Drift Gym** — an adversarial environment where an RL-style agent learns structured fault recovery by navigating real-world API breakage patterns: silent schema drift, partial observability, and cascading failures that punish guessing.

> Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv/tree/v0.2.1) | Trained on [HF Jobs](https://huggingface.co/docs/hub/jobs) with T4 GPU | Model: `Qwen2.5-0.5B-Instruct` + LoRA SFT | Adapter: [shivamkr1353/api-drift-sft-qwen](https://huggingface.co/shivamkr1353/api-drift-sft-qwen)

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| **Live Demo (HF Space)** | [huggingface.co/spaces/shivamkr1353/api-drift-gym](https://huggingface.co/spaces/shivamkr1353/api-drift-gym) |
| **Trained Adapter** | [huggingface.co/shivamkr1353/api-drift-sft-qwen](https://huggingface.co/shivamkr1353/api-drift-sft-qwen) |
| **Source Code** | [github.com/shivamkr1353/Meta_Hack_Round2](https://github.com/shivamkr1353/Meta_Hack_Round2) |

---

## What Problem Are We Solving?

Real-world API integrations fail in ways that are invisible at the surface. The status code is 200. The response arrives. But the schema shifted — a field was renamed, a type was changed, or a key silently disappeared. A naive agent (or engineer) retries. It fails again. It retries harder. Still fails.

**The standard retry loop is not a debugging strategy. It is a panic response.**

API Drift Gym forces an agent to learn the correct mental model:

```
bad:  call_api -> fail -> retry -> fail -> retry -> timeout

good: call_api -> fail -> inspect_schema -> transform_request -> call_api -> success
```

This is not a toy problem. Schema drift is how production systems actually break, and the ability to recover without human intervention is what makes an agent useful.

---

## Results

### Baseline vs Trained Agent

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| Random Agent | 0% | 0% | 0% |
| SFT Agent (Qwen 0.5B) | 70% | 50% | 45% |
| Expert Policy (Teacher) | 100% | 100% | 100% |

### Success Rate Comparison

![Success Rate Comparison](plots/success_rate_comparison.png)

### Reward Curves by Difficulty

![Reward Per Episode](plots/reward_per_episode.png)

### Training Loss

![Training Loss](plots/training_loss.png)

---

### Trajectory Comparison

**Before training (Random Agent):**
```
Step  1 │ ✗ call_api              │ -1.0  │ /user
        │   ⚠ Schema mismatch: missing fields=['full_name']
Step  2 │ ✗ retry                 │ -1.0  │ /user
        │   ⚠ retry requires a previously transformed payload
Step  3 │ ✗ skip_step             │ -0.5  │ /orders
Step  4 │ ✗ call_api              │ -1.0  │ /orders
        │   ⚠ Schema mismatch: missing fields=['max_results']
...
[timeout — episode failed]
```

**After training (SFT Agent):**
```
Step  1 │ · call_api              │ -0.5  │ /user
        │   ⚠ Schema mismatch (expected drift)
Step  2 │ ✓ inspect_schema        │ +1.0  │ /user
Step  3 │ ✓ transform_request     │ +1.0  │ /user
Step  4 │ ✓ call_api              │ +1.0  │ /orders
        │   200 OK — stage resolved
...
Step 12 │ ✓ call_api              │ +1.0  │ /summary
[episode resolved — 4/4 stages complete]
```

The difference is not speed. It is **reasoning structure**.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                      API DRIFT GYM LOOP                         │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Schema Drift│───►│  Environment │───►│       Agent        │  │
│  │ Generator   │    │ (OpenEnv +   │    │ (Qwen 0.5B + LoRA) │  │
│  │ Easy/Med/   │    │ api_drift_gym)│   │ inspect_schema     │  │
│  │ Hard drift  │    │              │    │ transform_request  │  │
│  └─────────────┘    └──────────────┘    │ call_api / retry   │  │
│                            │            └────────┬───────────┘  │
│                            │                     │              │
│                     API response           reward signal         │
│                                                 │               │
│                                                 ▼               │
│                                        ┌────────────────┐       │
│                                        │  SFT Trainer   │       │
│                                        │  (TRL + LoRA)  │       │
│                                        └────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### The Loop (Step by Step)

1. **Schema drift generator** creates mismatches (rename, partial drift, extra noise fields) by difficulty.
2. **Environment** emits partial observations (error, response fragments, hints).
3. **Agent** picks from `inspect_schema`, `transform_request`, `call_api`, `retry`.
4. **Reward engine** scores correctness, workflow progress, order quality, and terminal outcomes.
5. **SFT trainer** learns from teacher-guided trajectories.
6. **Difficulty escalates** with stronger hidden drift and multi-stage workflows.

### What Makes This Different

- **Hidden schema changes mid-episode**
- **Partial observability by design**
- **No hardcoded recovery policy in the model**
- **Difficulty tiers with real behavioral differences**
- **Teacher trajectories that teach sequence quality, not just endpoint success**

---

## Reward Design

The environment uses additive reward shaping with a strict terminal-failure override:

```python
reward = (
    per_step_correctness
    + workflow_progress
    + repeat_penalty
    + phase_order_bonus
    + resolution_bonus
)

if terminal_failure:
    reward = -2.0
```

Key properties:

- encourages action quality, not only final success
- rewards workflow-stage completion and full resolution
- penalizes invalid or repetitive behavior
- gives a clear terminal negative signal on timeout/failure

---

## The Agent's Journey

### Day 1: Blind Retrying

The agent receives a partial API response with a field mismatch error. It has no schema documentation, only the error message and four available actions.

It calls the API again. Fails. Calls again. Fails.

### Day 4: First Inspection

Instead of immediately retrying, the agent calls `inspect_schema`. It sees the contract the API expects, notices mismatch, calls `transform_request`, then calls again.

Response: 200. Fields match. Episode resolved.

### Day 10: Systematic Recovery

The pattern becomes consistent. Failure triggers inspection before action. The agent learns that retrying without understanding is wasted compute.

The environment escalates: drift can happen mid-episode. A schema valid on step 1 may become invalid by step 3. The agent must detect and re-inspect dynamically.

---

## Architecture

```
HF Jobs (T4 GPU)                         API Drift Environment
┌─────────────────────────────┐          ┌──────────────────────────┐
│  train_clean.py             │          │  ApiDriftGymEnv          │
│  ├─ SFTTrainer (TRL)        │◄────────►│  ├─ ApiDriftSimulator    │
│  ├─ Qwen2.5-0.5B-Instruct   │          │  ├─ Hidden drift logic   │
│  ├─ LoRA (r=16, alpha=32)   │          │  ├─ Partial hints        │
│  └─ 2 epochs, batch=2       │          │  └─ RewardEngine         │
│  Save: ./final_model        │          │                          │
└─────────────────────────────┘          │  Difficulty: easy/med/hard│
                                         └──────────────────────────┘
```

---

## Demo

The [HF Space](https://huggingface.co/spaces/shivamkr1353/api-drift-gym) provides an interactive Gradio UI with three tabs:

1. **Single Episode** — Run any agent (random/expert/SFT) on any difficulty with step-by-step trace
2. **Side-by-Side Comparison** — Compare all three agents on the same seed
3. **Batch Evaluation** — Run N episodes and see success rate statistics

---

## Quick Start

```python
from api_drift_gym import ApiDriftGymEnv

env = ApiDriftGymEnv(max_steps=20, seed=11, difficulty="medium")
obs = env.reset()
print(obs)

# Example endpoint-aware flow
obs, reward, done, info = env.step("inspect_schema:/user")
obs, reward, done, info = env.step('transform_request:/user:{"id":1,"full_name":"Ada"}')
obs, reward, done, info = env.step("retry")
```

---

## Training

**Install**
```bash
git clone https://github.com/shivamkr1353/Meta_Hack_Round2.git
cd Meta_Hack_Round2
pip install -r requirements.txt
```

**Launch training job (HF Jobs)**
```bash
hf jobs run -d \
  --namespace shivamkr1353 \
  --flavor t4-small \
  --timeout 1800 \
  -s HF_TOKEN \
  huggingface/transformers-pytorch-gpu:latest \
  bash -lc "git clone https://github.com/shivamkr1353/Meta_Hack_Round2.git && \
           cd Meta_Hack_Round2 && \
           pip install -r requirements.txt && \
           PYTHONUNBUFFERED=1 python3 train_clean.py"
```

**Monitor**
```bash
hf jobs inspect --namespace shivamkr1353 <job_id>
```

**Expected training logs (example)**
```
Epoch 1 | Loss: 0.2667
Epoch 2 | Loss: 0.0448
Training complete! Saving to final_model
Done! Saved model artifacts to final_model
```

**Upload adapter**
```bash
huggingface-cli upload shivamkr1353/api-drift-sft-qwen ./final_model \
  --repo-type model \
  --commit-message "Upload API Drift SFT Qwen adapter"
```

---

## Evaluation

```bash
# Full evaluation (random vs SFT vs expert, all difficulties)
python evaluate.py

# Generate plots
python generate_plots.py

# Run interactive demo
python app.py

# Environment behavior tests
python -m unittest discover -s tests -v
```

---

## HF Space Deployment

The Space can be deployed by pushing this repository to a HF Space:

```bash
# Create and push to HF Space
huggingface-cli repo create api-drift-gym --type space --space-sdk gradio
git remote add space https://huggingface.co/spaces/shivamkr1353/api-drift-gym
git push space main
```

The `app.py` file serves as the Gradio-based Space entry point, while `api_drift_env/server/app.py` provides the OpenEnv-compliant FastAPI server with REST endpoints (`/reset`, `/step`, `/state`, `/health`).

---

## Configuration

| Variable / Param | Description | Default |
|------------------|-------------|---------|
| `difficulty` | `easy`, `medium`, or `hard` | sampled / optional |
| `max_steps` | max actions per episode | `20` |
| `NUM_EPOCHS` | SFT epochs (`train_clean.py`) | `2` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | per-device batch size | `2` |
| `GRADIENT_ACCUMULATION_STEPS` | grad accumulation | `4` |
| `HF_TOKEN` | Hugging Face access token | required for jobs/upload |

---

## Project Structure

```
Meta_Hack_Round2/
├── app.py                  # Gradio demo UI (HF Space entry point)
├── evaluate.py             # Full evaluation: random vs SFT vs expert
├── generate_plots.py       # Publication-quality plot generation
├── train_clean.py          # SFT training (TRL + LoRA, T4-optimized)
├── train_grpo.py           # GRPO extension (reward-based, experimental)
├── inference.py            # Adapter inference sanity checks
├── baseline.py             # Simple random-action baseline
├── demo.py                 # Scripted walkthrough
├── stage_aware_policy.py   # Expert policy module
├── requirements.txt        # Pinned dependencies
├── api_drift_gym/          # Core environment package
│   ├── env.py              # Env loop: reset -> step -> reward
│   ├── api_simulator.py    # Drift + schema/workflow simulation
│   ├── reward.py           # Reward shaping engine
│   ├── logger.py           # Trajectory logger
│   └── __init__.py
├── api_drift_env/          # OpenEnv deployment package
│   ├── server/
│   │   ├── app.py          # FastAPI server (OpenEnv endpoints)
│   │   ├── api_drift_env_environment.py  # OpenEnv Environment adapter
│   │   └── Dockerfile      # Container deployment
│   ├── client.py           # OpenEnv client
│   ├── models.py           # Action/Observation Pydantic models
│   ├── openenv.yaml        # OpenEnv spec
│   └── pyproject.toml      # pip-installable package config
├── tests/
│   └── test_api_drift_gym.py
├── plots/                  # Generated evaluation plots
│   ├── success_rate_comparison.png
│   ├── reward_per_episode.png
│   └── training_loss.png
├── results/                # Generated evaluation data
│   ├── metrics.json
│   ├── trajectory_failed.json
│   └── trajectory_success.json
└── final_model/            # Saved LoRA adapter + tokenizer
```

---

## Key Design Decisions

**1) Inspect-first over retry loops**
The core learned behavior is inspect → transform → act, not retry spam.

**2) Partial observability by design**
Hiding full schema forces reasoning over observation history.

**3) SFT first for stability**
Teacher trajectories provide a stable policy prior before reward-only optimization.

**4) Real difficulty tiers**
Easy/Medium/Hard change observability and drift timing, not just thresholds.

**5) GRPO is a natural extension**
SFT builds structured behavior; GRPO can improve robustness under unseen drift.

---

## Extending to GRPO

`train_grpo.py` extends the same environment and action semantics with reward-based optimization.

```python
# train_grpo.py (simplified intent)
# - same environment semantics
# - rollout + reward collection
# - policy updates on relative advantage
```

---

## What We Learned

1. Observability calibration matters more than cosmetic difficulty labels.
2. Trajectory quality beats raw trajectory count.
3. Stable reward shaping is upstream of stable training.
4. Training surfaces environment bugs quickly (which is good).
5. A small model with disciplined structure can outperform larger but poorly trained systems.
