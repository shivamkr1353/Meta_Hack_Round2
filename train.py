import os
import re
import json
import math
import copy
import random
import torch
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

try:
    from api_drift_gym import ApiDriftGymEnv
except ImportError:
    class ApiDriftGymEnv:
        BASE_SCHEMAS = {
            "/user": {"id": "int", "name": "str"},
            "/orders": {"user_id": "int", "limit": "int"},
            "/payment": {"txn_id": "str", "amount": "float"},
            "/process": {"user_id": "int", "order_count": "int"},
            "/summary": {"email": "str", "message": "str"},
        }
        DRIFT_OPTIONS = {
            "/user": [
                {"drifted_schema": {"id": "int", "full_name": "str"}, "drift_case": "rename_field"},
                {"drifted_schema": {"user_id": "int", "full_name": "str"}, "drift_case": "partial_schema_drift"},
            ],
            "/orders": [
                {"drifted_schema": {"account_id": "int", "max_results": "int"}, "drift_case": "rename_field"},
                {"drifted_schema": {"user_id": "int", "page_size": "int"}, "drift_case": "partial_schema_drift"},
            ],
            "/payment": [
                {"drifted_schema": {"payment_id": "str", "status": "str"}, "drift_case": "rename_field"},
                {"drifted_schema": {"txn_id": "str", "payment_status": "str"}, "drift_case": "partial_schema_drift"},
            ],
            "/process": [
                {"drifted_schema": {"account_id": "int", "total_orders": "int"}, "drift_case": "rename_field"},
                {"drifted_schema": {"user_id": "int", "aggregate_count": "int"}, "drift_case": "partial_schema_drift"},
            ],
            "/summary": [
                {"drifted_schema": {"recipient": "str", "summary": "str"}, "drift_case": "rename_field"},
                {"drifted_schema": {"email": "str", "digest": "str"}, "drift_case": "partial_schema_drift"},
            ],
        }
        WORKFLOWS = {
            "hard": [
                {
                    "task": "Build an enterprise customer summary across services",
                    "steps": [
                        {"name": "fetch_user", "endpoint": "/user"},
                        {"name": "fetch_orders", "endpoint": "/orders"},
                        {"name": "process_data", "endpoint": "/process"},
                        {"name": "send_summary", "endpoint": "/summary"},
                    ],
                }
            ],
        }
        UNUSED_FIELD_POOL = ["legacy_id", "debug_mode", "trace_token", "deprecated_flag"]

        def __init__(self, max_steps: int = 20, seed: Optional[int] = None, difficulty: Optional[str] = None):
            self.max_steps = max_steps
            self.default_difficulty = difficulty or "hard"
            self.rng = random.Random(seed)
            self.state: Dict[str, Any] = {}

        def reset(self, seed: Optional[int] = None, difficulty: Optional[str] = None) -> Dict[str, Any]:
            if seed is not None:
                self.rng.seed(seed)
            episode_difficulty = difficulty or self.default_difficulty
            template = copy.deepcopy(self.rng.choice(self.WORKFLOWS[episode_difficulty]))
            workflow = [
                {**step, "completed": False, "skipped": False, "attempts": 0}
                for step in template["steps"]
            ]
            api_states = {}
            for step in workflow:
                endpoint = step["endpoint"]
                drift = copy.deepcopy(self.rng.choice(self.DRIFT_OPTIONS[endpoint]))
                extra_count = 1 if self.rng.random() < 0.5 else 2
                api_states[endpoint] = {
                    "endpoint": endpoint,
                    "original_schema": copy.deepcopy(self.BASE_SCHEMAS[endpoint]),
                    "drifted_schema": drift["drifted_schema"],
                    "drift_case": drift["drift_case"],
                    "extra_unused_fields": self.rng.sample(self.UNUSED_FIELD_POOL, k=extra_count),
                    "saw_failure": False,
                    "pending_payload": None,
                }
            self.state = {
                "task": template["task"],
                "difficulty": episode_difficulty,
                "workflow": workflow,
                "workflow_step": 0,
                "api_states": api_states,
                "step_count": 0,
                "last_endpoint": None,
                "resolved": False,
            }
            return self._obs("reset", "", "", {
                "task": self.state["task"],
                "difficulty": self.state["difficulty"],
                "workflow": [{"name": s["name"], "endpoint": s["endpoint"]} for s in workflow],
                "known_schema": copy.deepcopy(self.BASE_SCHEMAS[self._current_endpoint()]),
            })

        def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
            self.state["step_count"] += 1
            name, endpoint, payload = self._parse_action(action)
            endpoint = endpoint or self._current_endpoint()
            reward = 0.0
            response = ""
            error = ""
            hint = None

            if name == "call_api":
                response, error, reward = self._call_api(endpoint, payload)
            elif name == "inspect_schema":
                hint = self._inspect_schema(endpoint)
                response = "Schema inspection completed."
                reward = 1.0
            elif name == "transform_request":
                if isinstance(payload, dict) and endpoint in self.state["api_states"]:
                    self.state["api_states"][endpoint]["pending_payload"] = copy.deepcopy(payload)
                    hint = {"pending_payload": copy.deepcopy(payload)}
                    response = "Request transformed."
                    reward = 1.0
                else:
                    error = "transform_request requires a JSON object payload."
                    reward = -1.0
            elif name == "skip_step":
                self.state["workflow_step"] += 1
                response = "Skipped workflow step."
                reward = -0.5
            else:
                error = "Invalid action format."
                reward = -1.0

            done = self.state["resolved"] or self.state["step_count"] >= self.max_steps
            done = done or self.state["workflow_step"] >= len(self.state["workflow"])
            return self._obs(action, response, error, hint), reward, done, {"resolved": self.state["resolved"]}

        def _parse_action(self, action: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
            if action == "skip_step":
                return "skip_step", self._current_endpoint(), None
            if action.startswith("inspect_schema:"):
                return "inspect_schema", self._normalize_endpoint(action.split(":", 1)[1]), None
            for name in ("call_api", "transform_request"):
                prefix = f"{name}:"
                if action.startswith(prefix):
                    endpoint, _, payload_raw = action[len(prefix):].partition(":")
                    payload = self._parse_payload(payload_raw) if payload_raw else None
                    return name, self._normalize_endpoint(endpoint), payload
            return "invalid", None, None

        def _call_api(self, endpoint: Optional[str], payload: Optional[Dict[str, Any]]) -> Tuple[str, str, float]:
            if endpoint not in self.state["api_states"] or not isinstance(payload, dict):
                return "", "API call requires a known endpoint and JSON object payload.", -1.0
            api_state = self.state["api_states"][endpoint]
            matched, details = self._matches_schema(api_state["drifted_schema"], payload)
            if not matched:
                api_state["saw_failure"] = True
                parts = []
                if details["missing"]:
                    parts.append(f"missing fields={details['missing']}")
                if details["unexpected"]:
                    parts.append(f"unexpected fields={details['unexpected']}")
                error = "Schema mismatch: " + "; ".join(parts or ["unknown contract drift"])
                return "", error, -0.5

            if endpoint == self._current_endpoint():
                self.state["workflow"][self.state["workflow_step"]]["completed"] = True
                self.state["workflow_step"] += 1
            self.state["last_endpoint"] = endpoint
            self.state["resolved"] = self.state["workflow_step"] >= len(self.state["workflow"])
            response = json.dumps({"status": "success", "endpoint": endpoint}, sort_keys=True)
            return response, "", 1.0

        def _inspect_schema(self, endpoint: Optional[str]) -> Dict[str, Any]:
            api_state = self.state["api_states"][endpoint]
            if api_state["saw_failure"]:
                return {
                    "endpoint": endpoint,
                    "required_fields": sorted(api_state["drifted_schema"].keys()),
                    "field_types": copy.deepcopy(api_state["drifted_schema"]),
                    "drift_case": api_state["drift_case"],
                    "deprecated_candidates": copy.deepcopy(api_state["extra_unused_fields"]),
                }
            return {
                "endpoint": endpoint,
                "field_count": len(api_state["drifted_schema"]),
                "notes": "Partial compatibility metadata only. Trigger a failure to reveal more.",
            }

        def _matches_schema(self, schema: Dict[str, str], payload: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
            expected = set(schema.keys())
            actual = set(payload.keys())
            details = {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}
            return not details["missing"] and not details["unexpected"], details

        def _obs(self, action: str, response: str, error: str, hint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                "last_action": action,
                "api_response": response,
                "error_message": error,
                "available_hint": hint,
                "step_count": self.state["step_count"],
                "workflow_step": self.state["workflow_step"],
                "current_endpoint": self._current_endpoint(),
                "difficulty": self.state["difficulty"],
            }

        def _current_endpoint(self) -> Optional[str]:
            index = self.state.get("workflow_step", 0)
            workflow = self.state.get("workflow", [])
            if index >= len(workflow):
                return self.state.get("last_endpoint")
            return workflow[index]["endpoint"]

        def _normalize_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
            if not endpoint:
                return self._current_endpoint()
            return endpoint if endpoint.startswith("/") else f"/{endpoint}"

        def _parse_payload(self, raw: str) -> Optional[Dict[str, Any]]:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None

# Ensure logs flow immediately to HF Jobs console
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

SEED = 42
N_TRAIN = 200
NUM_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 512

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Initializing Environment...")
env = ApiDriftGymEnv(max_steps=20, seed=SEED, difficulty="hard")

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
model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss", logs.get("train_loss"))
        if loss is None:
            return
        epoch_value = logs.get("epoch", state.epoch or 0)
        epoch = max(1, min(int(math.ceil(float(epoch_value))), int(args.num_train_epochs)))
        print(f"Epoch {epoch} | Loss: {float(loss):.4f}", flush=True)

dataset = Dataset.from_list(training_data)

training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=2e-4,
    logging_strategy="epoch",
    save_strategy="no",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    bf16=False,
    seed=SEED,
    data_seed=SEED,
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
print("Training complete! Saving to final_model")
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
print("Done! Saved model artifacts to final_model")
