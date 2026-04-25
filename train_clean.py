import os
import re
import json
import random
import torch
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from api_drift_gym import ApiDriftGymEnv

# Ensure logs flow immediately to HF Jobs console
os.environ["PYTHONUNBUFFERED"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Initializing Environment...")
env = ApiDriftGymEnv(max_steps=20, seed=42, difficulty="hard")

class StagePhase(Enum):
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

def format_obs(obs, tracker):
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def make_sample(prompt, label):
    return {"text": prompt + " " + label.strip() + tokenizer.eos_token}

print("Generating Data using Stage-Aware Policy...")
N_TRAIN = 300
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

print(f"Collected {len(training_data)} examples.")

print(f"Loading model {MODEL_NAME} for LoRA...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[HF Jobs Log] Step {state.global_step}: Loss = {logs['loss']:.4f}")

dataset = Dataset.from_list(training_data)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    dataset_text_field="text",
    max_seq_length=512,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
    callbacks=[ProgressLoggingCallback()]
)

print("Starting Optimized HF Jobs SFT Training...")
trainer.train()

# Final clean save, replacing colab logic
print("Training complete! Saving to ./trained_model")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Done! You can now use the model locally or push to Hub.")
