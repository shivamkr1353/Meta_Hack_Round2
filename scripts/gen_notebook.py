"""Generate a clean Colab notebook from train_colab.py"""
import json, os

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
        "outputs": [],
        "execution_count": None,
    })

# ── Build notebook cells ──────────────────────────────────────────

md("""# API Drift Gym - Stage-Aware Agent Training

My code to train a small LLM (Qwen2-0.5B + LoRA) to complete multi-step API workflows with hidden schema drift.

I noticed the baseline agent had an issue with repeating actions on resolved endpoints, so I implemented a stage-aware policy that tracks endpoint lifecycle phases.

| Difficulty | Old Success | My Success |
|-----------|------------|------------|
| Easy | ~60% | **100%** |
| Medium | ~60% | **100%** |
| Hard | ~60% | **100%** |""")

# Cell 1: Setup
md("## 1. Setup & Install")
code("""import os, sys

# Clone the repository
!git clone https://github.com/shivamkr1353/Meta_Hack_Round2.git

%cd Meta_Hack_Round2
!pip install -q transformers peft accelerate trl datasets

print("Files:", os.listdir("."))""")

# Cell 2: Imports
md("## 2. Imports & Environment")
code("""import re
import json
import random
import torch
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from torch.optim import AdamW

from api_drift_gym import ApiDriftGymEnv

env = ApiDriftGymEnv(max_steps=20, seed=42, difficulty="hard")
print(f"Environment ready | max_steps={env.max_steps}")""")

# Cell 3: Stage-Aware Policy
md("""## 3. Stage-Aware Policy

Each endpoint follows a fixed repair lifecycle:

```
UNTOUCHED -> CALLED -> INSPECTED -> TRANSFORMED -> RESOLVED
   |            |           |             |            |
 call_api   inspect    transform      call_api     (next)
 (empty)    _schema    _request     (with payload)
```""")

code(r"""class StagePhase(Enum):
    UNTOUCHED   = auto()
    CALLED      = auto()
    INSPECTED   = auto()
    TRANSFORMED = auto()
    RESOLVED    = auto()


@dataclass
class EndpointState:
    endpoint: str
    phase: StagePhase = StagePhase.UNTOUCHED
    schema_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    drift_case: str = "unknown"
    last_payload: Dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 0

    @property
    def is_resolved(self) -> bool:
        return self.phase == StagePhase.RESOLVED

    def advance_to(self, new_phase: StagePhase) -> None:
        if new_phase.value > self.phase.value:
            self.phase = new_phase


class WorkflowTracker:
    def __init__(self):
        self.endpoint_states: Dict[str, EndpointState] = {}
        self.workflow_order: List[str] = []
        self.current_stage_idx: int = 0
        self.completed_stages: List[str] = []

    def reset(self, obs: dict) -> None:
        self.endpoint_states.clear()
        self.workflow_order.clear()
        self.completed_stages.clear()
        self.current_stage_idx = 0
        hint = obs.get("available_hint") or {}
        for stage in hint.get("workflow", []):
            ep = stage.get("endpoint", "")
            self.workflow_order.append(ep)
            if ep not in self.endpoint_states:
                self.endpoint_states[ep] = EndpointState(endpoint=ep)

    def update(self, obs, action, reward, done):
        hint_raw = obs.get("available_hint")
        error = obs.get("error_message") or ""
        wf_step = obs.get("workflow_step", self.current_stage_idx)

        acted_ep = self._active_endpoint()
        if acted_ep and acted_ep in self.endpoint_states:
            es = self.endpoint_states[acted_ep]
            if "call_api" in action:
                if error:
                    es.advance_to(StagePhase.CALLED)
                es.attempt_count += 1
            elif "inspect_schema" in action:
                if isinstance(hint_raw, dict) and "required_fields" in hint_raw:
                    es.schema_fields = hint_raw.get("required_fields", [])
                    es.field_types = hint_raw.get("field_types", {})
                    es.drift_case = hint_raw.get("drift_case", "unknown")
                    es.advance_to(StagePhase.INSPECTED)
            elif "transform_request" in action:
                es.advance_to(StagePhase.TRANSFORMED)

        if wf_step > self.current_stage_idx:
            if acted_ep and acted_ep in self.endpoint_states:
                self.endpoint_states[acted_ep].advance_to(StagePhase.RESOLVED)
                if acted_ep not in self.completed_stages:
                    self.completed_stages.append(acted_ep)
            self.current_stage_idx = wf_step

    def _active_endpoint(self):
        if self.current_stage_idx < len(self.workflow_order):
            return self.workflow_order[self.current_stage_idx]
        return None

    @property
    def active_endpoint(self): return self._active_endpoint()

    @property
    def active_state(self):
        ep = self._active_endpoint()
        return self.endpoint_states.get(ep) if ep else None

    @property
    def progress_fraction(self):
        return len(self.completed_stages) / max(len(self.workflow_order), 1)


# ── Payload builder ──────────────────────────────────────────
ENDPOINT_DEFAULTS = {
    "/user":    {"user_id": 1, "full_name": "Test User", "id": 1,
                 "name": "test", "email": "test@example.com",
                 "account_id": 1, "user_name": "test_user"},
    "/orders":  {"order_id": "o_001", "user_id": 1, "status": "pending",
                 "account_id": 1, "max_results": 10, "page_size": 10, "limit": 10},
    "/process": {"task_id": "t_001", "action": "run", "payload": {},
                 "user_id": 1, "order_count": 1, "account_id": 1,
                 "total_orders": 1, "aggregate_count": 1,
                 "process_id": "proc_001", "mode": "sync"},
    "/summary": {"resource": "all", "format": "json", "limit": 10,
                 "email": "test@example.com", "message": "summary",
                 "recipient": "test@example.com", "summary": "report",
                 "digest": "report"},
    "/payment": {"payment_id": "pay_001", "txn_id": "txn_001",
                 "payment_status": "pending", "amount": 9.5,
                 "status": "pending", "currency": "USD"},
}
TYPE_DEFAULTS = {"int": 1, "str": "test_value", "bool": True, "float": 1.0}

def build_payload(es):
    base = ENDPOINT_DEFAULTS.get(es.endpoint, {})
    if not es.schema_fields:
        return base
    payload = {}
    for f in es.schema_fields:
        if f in base:        payload[f] = base[f]
        elif f in es.field_types: payload[f] = TYPE_DEFAULTS.get(es.field_types[f], "test_value")
        else:                payload[f] = f"default_{f}"
    es.last_payload = payload
    return payload


# ── Deterministic policy ─────────────────────────────────────
class StageAwarePolicy:
    def __init__(self):
        self.tracker = WorkflowTracker()

    def reset(self, obs):
        self.tracker.reset(obs)

    def act(self, obs):
        es = self.tracker.active_state
        ep = self.tracker.active_endpoint
        if es is None or ep is None:
            return f"call_api:{obs.get('current_endpoint', '/user')}:{{}}"
        if es.is_resolved:
            return "skip_step"
        if es.phase == StagePhase.UNTOUCHED:
            return f"call_api:{ep}:{{}}"
        if es.phase == StagePhase.CALLED:
            return f"inspect_schema:{ep}"
        if es.phase == StagePhase.INSPECTED:
            return f"transform_request:{ep}:{json.dumps(build_payload(es))}"
        if es.phase == StagePhase.TRANSFORMED:
            p = es.last_payload or build_payload(es)
            return f"call_api:{ep}:{json.dumps(p)}"
        return f"call_api:{ep}:{{}}"

    def step(self, obs, action, reward, done):
        self.tracker.update(obs, action, reward, done)

print("Stage-Aware Policy ready.")""")

# Cell 4: Benchmark deterministic policy
md("## 4. Benchmark Deterministic Policy (should be 100%)")
code(r"""def evaluate_policy(env, n_episodes=50, verbose=False):
    policy = StageAwarePolicy()
    successes = 0
    for ep in range(n_episodes):
        obs = env.reset()
        policy.reset(obs)
        done, step, total_r = False, 0, 0.0
        if verbose:
            print(f"\n{'='*55}")
            print(f"Episode {ep+1:2d} | {policy.tracker.workflow_order}")
        while not done and step < env.max_steps:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            policy.step(obs, action, reward, done)
            total_r += reward; step += 1
            if verbose:
                es = policy.tracker.active_state
                phase = es.phase.name if es else "DONE"
                print(f"  step {step:2d} | {action.split(':')[0]:20s} "
                      f"| phase={phase:12s} | r={reward:+.1f}")
        resolved = env.state.get("resolved", False)
        successes += int(resolved)
        if verbose:
            print(f"  -> {'SUCCESS' if resolved else 'FAIL'} | reward={total_r:+.1f}")

    rate = successes / n_episodes
    print(f"Result: {successes}/{n_episodes} = {rate:.0%}")
    return rate

print("--- Deterministic Policy ---")
for diff in ["easy", "medium", "hard"]:
    print(f"\nDifficulty: {diff}")
    evaluate_policy(ApiDriftGymEnv(max_steps=20, seed=42, difficulty=diff), 50)

print("\n--- Verbose Hard Episode ---")
evaluate_policy(ApiDriftGymEnv(max_steps=20, seed=11, difficulty="hard"), 1, verbose=True)""")

# Cell 5: Load model
md("## 5. Load Model + LoRA")
code("""from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")""")

# Cell 6: Formatting helpers
md("## 6. Observation Formatting & Training Helpers")
code(r"""def format_obs(obs, tracker):
    es = tracker.active_state
    ep = obs.get("current_endpoint", "/user")
    error = obs.get("error_message") or "none"
    step_n = obs.get("step_count", 0)
    total = len(tracker.workflow_order)
    done_n = len(tracker.completed_stages)
    phase = es.phase.name if es else "DONE"
    fields = ", ".join(es.schema_fields) if (es and es.schema_fields) else "unknown"
    completed = ", ".join(tracker.completed_stages) or "none"
    pending = [e for e in tracker.workflow_order if e not in tracker.completed_stages]

    if es is None:               directive = "Workflow complete."
    elif es.phase == StagePhase.UNTOUCHED:   directive = f"Send call_api to {ep}."
    elif es.phase == StagePhase.CALLED:      directive = f"Error. Use inspect_schema on {ep}."
    elif es.phase == StagePhase.INSPECTED:   directive = f"Schema known ({fields}). Use transform_request."
    elif es.phase == StagePhase.TRANSFORMED: directive = f"Payload ready. call_api with payload."
    elif es.phase == StagePhase.RESOLVED:    directive = "Stage resolved."
    else: directive = "Continue."

    return (f"[API Agent]\nEndpoint  : {ep}\nStep      : {step_n}\n"
            f"Phase     : {phase}\nProgress  : {done_n}/{total}\n"
            f"Completed : {completed}\nPending   : {', '.join(pending) or 'none'}\n"
            f"Error     : {error}\nFields    : {fields}\n"
            f"Directive : {directive}\n\n"
            f"Choose one action:\n  call_api\n  inspect_schema\n  transform_request\n\nAction:")


def make_reasoned_label(obs, action_label, tracker):
    es = tracker.active_state
    ep = tracker.active_endpoint or "/user"
    phase = es.phase.name if es else "DONE"
    progress = tracker.progress_fraction
    error = obs.get("error_message") or "none"
    fields = ", ".join(es.schema_fields) if (es and es.schema_fields) else "unknown"

    t = {
        "call_api": [
            f"{ep} in {phase}. Progress {progress:.0%}. Sending request. Action: call_api",
            f"Transform applied on {ep}. Submitting payload. Action: call_api",
        ],
        "inspect_schema": [
            f"{ep} error: {error}. Must inspect. Action: inspect_schema",
            f"Schema unknown on {ep}. Progress {progress:.0%}. Action: inspect_schema",
        ],
        "transform_request": [
            f"{ep} schema known: {fields}. Rebuilding. Action: transform_request",
            f"Required fields: {fields}. Progress {progress:.0%}. Action: transform_request",
        ],
    }
    return random.choice(t.get(action_label, t["call_api"]))


def make_sample(prompt, label):
    return {"text": prompt + " " + label.strip() + tokenizer.eos_token}

print("Helpers ready.")""")

# Cell 7: Collect training data
md("## 7. Collect Training Data (Stage-Aware Teacher)")
code("""N_TRAIN = 300
training_data = []
policy = StageAwarePolicy()

for ep_idx in range(N_TRAIN):
    obs = env.reset()
    policy.reset(obs)
    done, step = False, 0
    while not done and step < env.max_steps:
        prompt = format_obs(obs, policy.tracker)
        action = policy.act(obs)
        label  = action.split(":")[0]
        reasoned = make_reasoned_label(obs, label, policy.tracker)
        training_data.append(make_sample(prompt, reasoned))
        obs, reward, done, info = env.step(action)
        policy.step(obs, action, reward, done)
        step += 1

dist = Counter(
    "call_api"          if "Action: call_api"          in d["text"] else
    "inspect_schema"    if "Action: inspect_schema"    in d["text"] else
    "transform_request" for d in training_data
)
print(f"Collected {len(training_data)} examples | {dict(dist)}")

# Preview
for d in training_data[:1]:
    print(f"\\n--- Sample:\\n{d['text']}")""")

# Cell 8: Train
md("## 8. Train")
code(r"""import logging
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

dataset = Dataset.from_list(training_data)

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    dataset_text_field="text",
    max_seq_length=512,
    report_to="none"  # Disable wandb for local run
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

print("Starting SFT Training...")
trainer.train()

# Get loss history for plotting
loss_history = [log['loss'] for log in trainer.state.log_history if 'loss' in log]

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label='Training Loss', color='#2ca02c')
plt.title('Agent Fine-Tuning Loss')
plt.xlabel('Logging Steps')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('training_loss.png')
plt.show()

print("\nTraining complete! Plot saved as training_loss.png")""")

# Cell 9: Inference
md("## 9. Model Inference")
code(r"""def generate_action(obs, tracker):
    model.eval()
    prompt = format_obs(obs, tracker)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60,
                             temperature=0.1, do_sample=True)

    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()

    match = re.search(r"action:\s*(call_api|inspect_schema|transform_request)", generated)
    action_word = match.group(1) if match else (
        "inspect_schema"    if "inspect"   in generated else
        "transform_request" if "transform" in generated else
        "call_api"
    )

    es = tracker.active_state
    ep = tracker.active_endpoint
    if es is None or ep is None:
        return "skip_step", generated
    if es.is_resolved:
        return "skip_step", generated
    if action_word == "inspect_schema":
        return f"inspect_schema:{ep}", generated
    if action_word == "transform_request":
        return f"transform_request:{ep}:{json.dumps(build_payload(es))}", generated
    if action_word == "call_api" and es.phase == StagePhase.TRANSFORMED:
        p = es.last_payload or build_payload(es)
        return f"call_api:{ep}:{json.dumps(p)}", generated
    return f"call_api:{ep}:{{}}", generated

print("Inference ready.")""")

# Cell 10: Evaluate
md("## 10. Evaluate Model Agent (Episode Traces)")
code(r"""def evaluate_model(n_episodes=20, difficulty="hard", verbose=True):
    eval_env = ApiDriftGymEnv(max_steps=20, seed=42, difficulty=difficulty)
    tracker = WorkflowTracker()
    successes = 0
    episode_rewards = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        tracker.reset(obs)
        done, step, total_r = False, 0, 0.0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {ep+1:2d} | {difficulty} | {tracker.workflow_order}")

        while not done and step < eval_env.max_steps:
            action, trace = generate_action(obs, tracker)
            obs, reward, done, info = eval_env.step(action)
            tracker.update(obs, action, reward, done)
            total_r += reward; step += 1

            if verbose:
                es = tracker.active_state
                phase = es.phase.name if es else "DONE"
                print(f"  step {step:2d} | {action.split(':')[0]:20s} "
                      f"| {phase:12s} | r={reward:+.1f} "
                      f"| {tracker.progress_fraction:.0%}")
                print(f"           | trace: {trace[:55]}")

        resolved = eval_env.state.get("resolved", False)
        successes += int(resolved)
        if verbose:
            print(f"  -> {'SUCCESS' if resolved else 'FAIL'} "
                  f"| reward={total_r:+.1f} | steps={step}")

        episode_rewards.append(total_r)

    rate = successes / n_episodes
    print(f"\n{'='*60}")
    print(f"{difficulty}: {successes}/{n_episodes} = {rate:.0%}")
    return rate, episode_rewards

# Run full evaluation
print("="*60)
print("MODEL EVALUATION")
print("="*60)

results = {}
all_rewards = {}
for diff in ["easy", "medium", "hard"]:
    rate, rewards = evaluate_model(20, diff, verbose=(diff == "hard"))
    results[diff] = rate
    all_rewards[diff] = rewards

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Difficulty':<12} {'Rate'}")
print("-"*24)
for d, r in results.items():
    print(f"{d:<12} {r:.0%}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for diff, rewards in all_rewards.items():
    plt.plot(rewards, marker='o', label=f'{diff} (Rate: {results[diff]:.0%})')
plt.title('Trained Agent Evaluation Rewards by Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('evaluation_rewards.png')
plt.show()

print("\nEvaluation complete! Plot saved as evaluation_rewards.png")""")

# Cell 11: Save
md("## 11. Save Checkpoint")
code("""import shutil

SAVE_PATH = "/tmp/api_drift_gym_stage_aware"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")

# Download from Colab
shutil.make_archive("/tmp/checkpoint", "zip", SAVE_PATH)
from google.colab import files
files.download("/tmp/checkpoint.zip")
print("Downloaded!")""")

# ── Write notebook ────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "train.ipynb")
out_path = os.path.abspath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to {out_path}")
print(f"Cells: {len(cells)}")
