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

> 🏆 **Submission for Theme #3.2: Personalized Tasks**  
> 🎯 **Targeting Bonus Prize:** Patronus AI - *Consumer Workflows with Schema Drift*

> Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv/tree/v0.2.1) | Trained on [HF Jobs](https://huggingface.co/docs/hub/jobs) with T4 GPU | Model: `Qwen2.5-0.5B-Instruct` + LoRA SFT | Training via [HF TRL](https://github.com/huggingface/trl) in [Colab](train.ipynb)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      API DRIFT GYM LOOP                          │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐   │
│  │ Schema Drift│───►│  Environment │───►│       Agent        │   │
│  │ Generator   │    │ (OpenEnv +   │    │ (Qwen 0.5B + LoRA) │   │
│  │ Easy/Med/   │    │ api_drift_gym)│   │ inspect_schema     │   │
│  │ Hard drift  │    │              │    │ transform_request  │   │
│  └─────────────┘    └──────────────┘    │ call_api / retry   │   │
│                            │            └────────┬───────────┘   │
│                            │                     │               │
│                     API response           reward signal         │
│                                                 │                │
│                                                 ▼                │
│                                        ┌────────────────┐        │
│                                        │  SFT Trainer   │        │
│                                        │  (TRL + LoRA)  │        │
│                                        └────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

## Drift Types & Difficulty Escalation

| Difficulty | What Gets Injected | What Agent Must Do |
|------------|--------------------|---------------------|
| `easy` | Simple missing field (e.g. `full_name`) | Map available fields to schema (`inspect` -> `transform`) |
| `medium` | Nested dictionary drift & missing required keys | Deep JSON restructuring before `call_api` |
| `hard` | Multi-endpoint workflows where stage 1 schema is valid, but stage 2 drifts | Maintain context, detect failure mid-workflow, and re-inspect dynamically |


## 🔗 Links

| Resource | URL |
|----------|-----|
| **Live Demo (HF Space)** | [huggingface.co/spaces/shivamkr1353/api-drift-gym](https://huggingface.co/spaces/shivamkr1353/api-drift-gym) |
| **Trained Adapter** | [huggingface.co/shivamkr1353/api-drift-sft-qwen](https://huggingface.co/shivamkr1353/api-drift-sft-qwen) |
| **Source Code** | [github.com/shivamkr1353/Meta_Hack_Round2](https://github.com/shivamkr1353/Meta_Hack_Round2) |

---

## 📝 Read The Full Writeup

Curious about how we designed the reward engine, implemented schema drift, and taught a 0.5B model to debug like a senior engineer?

👉 **[Read the Full Hackathon Blog Post Here](blog_post.md)**

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

---

## Training with HF TRL (Colab)

A complete training notebook is provided at [`train.ipynb`](train.ipynb) using **HF TRL's SFTTrainer**. This satisfies the Hackathon requirement for a minimal Colab training script. The notebook covers:

1. Initializing the `ApiDriftGymEnv` locally.
2. Generating teacher trajectories using the `StageAwarePolicy`.
3. Configuring LoRA (r=16, alpha=32) for `Qwen2.5-0.5B-Instruct`.
4. Running SFT training and saving checkpoints.

Alternatively, you can launch a training job on Hugging Face Jobs:

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
           PYTHONUNBUFFERED=1 python3 train.py"
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
python eval.py

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
| `NUM_EPOCHS` | SFT epochs (`train.py`) | `2` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | per-device batch size | `2` |
| `GRADIENT_ACCUMULATION_STEPS` | grad accumulation | `4` |
| `HF_TOKEN` | Hugging Face access token | required for jobs/upload |

---

## Project Structure

```
Meta_Hack_Round2/
├── app.py                  # Gradio demo UI (HF Space entry point)
├── eval.py                 # Full evaluation: random vs SFT vs expert
├── generate_plots.py       # Publication-quality plot generation
├── train.py                # SFT training (TRL + LoRA, T4-optimized)
├── stage_aware_policy.py   # Expert policy module
├── requirements.txt        # Pinned dependencies
├── api_drift_gym/          # Core environment package
│   ├── env.py              # Env loop: reset -> step -> reward
│   ├── api_simulator.py    # Drift + schema/workflow simulation
│   ├── reward.py           # Reward shaping engine
│   ├── logger.py           # Trajectory logger
│   └── __init__.py
├── server/
│   ├── app.py          # FastAPI server (OpenEnv endpoints)
│   ├── api_drift_env_environment.py  # OpenEnv Environment adapter
│   └── Dockerfile      # Container deployment
├── client.py           # OpenEnv client
├── models.py           # Action/Observation Pydantic models
├── openenv.yaml        # OpenEnv spec
├── pyproject.toml      # pip-installable package config
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

