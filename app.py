"""
app.py — Gradio Demo UI for API Drift Gym
============================================
Interactive HF Space interface that lets users:
  1. Watch the SFT agent run episodes step-by-step
  2. Compare random vs trained agent behavior
  3. See rewards, actions, and final outcomes

Deployed as the primary HF Space interface.
"""

import json
import os
import random
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gradio as gr

from api_drift_gym import ApiDriftGymEnv

# ─────────────────────────────────────────────────────────────────
# SHARED COMPONENTS (same as evaluate.py — kept self-contained)
# ─────────────────────────────────────────────────────────────────

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
        wf_step = obs.get("workflow_step", self.current_stage_idx)
        acted_ep = self._active_endpoint()
        if acted_ep and acted_ep in self.endpoint_states:
            es = self.endpoint_states[acted_ep]
            if "call_api" in action:
                if obs.get("error_message"):
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


def build_payload(es: EndpointState) -> dict:
    base = ENDPOINT_DEFAULTS.get(es.endpoint, {})
    if not es.schema_fields:
        return base
    payload = {}
    for f in es.schema_fields:
        if f in base:
            payload[f] = base[f]
        elif f in es.field_types:
            payload[f] = TYPE_DEFAULTS.get(es.field_types[f], "test_value")
        else:
            payload[f] = f"default_{f}"
    es.last_payload = payload
    return payload


# ─────────────────────────────────────────────────────────────────
# EXPERT POLICY (deterministic)
# ─────────────────────────────────────────────────────────────────

class ExpertPolicy:
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


class RandomAgent:
    ACTIONS = [
        "call_api:/user:{}", "call_api:/orders:{}", "call_api:/payment:{}",
        "inspect_schema:/user", "inspect_schema:/orders",
        "retry", "skip_step",
    ]
    def act(self, obs):
        return random.choice(self.ACTIONS)


# ─────────────────────────────────────────────────────────────────
# OPTIONAL: SFT AGENT (loads LoRA adapter if available)
# ─────────────────────────────────────────────────────────────────

def format_obs_for_model(obs, tracker):
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

    if es is None:                            directive = "Workflow complete."
    elif es.phase == StagePhase.UNTOUCHED:    directive = f"Send call_api to {ep}."
    elif es.phase == StagePhase.CALLED:       directive = f"Error. Use inspect_schema on {ep}."
    elif es.phase == StagePhase.INSPECTED:    directive = f"Schema known ({fields}). Use transform_request."
    elif es.phase == StagePhase.TRANSFORMED:  directive = f"Payload ready. call_api with payload."
    elif es.phase == StagePhase.RESOLVED:     directive = "Stage resolved."
    else:                                     directive = "Continue."

    return (f"[API Agent]\nEndpoint  : {ep}\nStep      : {step_n}\n"
            f"Phase     : {phase}\nProgress  : {done_n}/{total}\n"
            f"Completed : {completed}\nPending   : {', '.join(pending) or 'none'}\n"
            f"Error     : {error}\nFields    : {fields}\n"
            f"Directive : {directive}\n\n"
            f"Choose one action:\n  call_api\n  inspect_schema\n  transform_request\n\nAction:")


_sft_model = None
_sft_tokenizer = None
_sft_load_attempted = False  # Cache failure so we don't retry every step


def load_sft_model():
    """Lazy-load the SFT model. Returns (model, tokenizer) or (None, None).
    Skips loading entirely on CPU — model requires GPU to run at usable speed."""
    global _sft_model, _sft_tokenizer, _sft_load_attempted
    if _sft_load_attempted:
        return _sft_model, _sft_tokenizer
    _sft_load_attempted = True
    try:
        import torch
        # Skip download on CPU — would hang trying to load a 500M param model
        if not torch.cuda.is_available():
            return None, None

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        adapter_path = "./final_model"
        base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        if not os.path.exists(adapter_path):
            return None, None

        _sft_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        _sft_model = PeftModel.from_pretrained(base, adapter_path)
        _sft_model.eval()
        return _sft_model, _sft_tokenizer
    except Exception:
        return None, None


def sft_agent_act(obs, tracker):
    """Generate action using SFT model. Falls back to expert if model unavailable."""
    model, tokenizer = load_sft_model()
    if model is None:
        # Fallback to expert
        policy = ExpertPolicy()
        policy.tracker = tracker
        return policy.act(obs), "expert_fallback"

    import torch
    prompt = format_obs_for_model(obs, tracker)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=60,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

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


# ─────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────

def run_episode(agent_type: str, difficulty: str, seed: int) -> str:
    """Run a single episode and return formatted step-by-step output."""
    env = ApiDriftGymEnv(max_steps=20, seed=seed, difficulty=difficulty)
    obs = env.reset()

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"🎮 API Drift Gym — {agent_type.upper()} Agent")
    lines.append(f"   Difficulty: {difficulty.capitalize()} | Seed: {seed}")

    # Get workflow info
    hint = obs.get("available_hint", {})
    workflow = hint.get("workflow", [])
    task = hint.get("task", "Unknown")
    lines.append(f"   Task: {task}")
    lines.append(f"   Workflow: {' → '.join(s['endpoint'] for s in workflow)}")
    lines.append(f"{'='*60}")
    lines.append("")

    if agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "expert":
        agent = ExpertPolicy()
        agent.reset(obs)
    else:
        # SFT agent
        tracker = WorkflowTracker()
        tracker.reset(obs)

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < env.max_steps:
        if agent_type == "random":
            action = agent.act(obs)
        elif agent_type == "expert":
            action = agent.act(obs)
        else:
            action, _ = sft_agent_act(obs, tracker)

        obs, reward, done, info = env.step(action)

        if agent_type == "expert":
            agent.step(obs, action, reward, done)
        elif agent_type == "sft":
            tracker.update(obs, action, reward, done)

        total_reward += reward
        step += 1

        # Format step output
        action_name = action.split(":")[0]
        error = obs.get("error_message") or ""
        endpoint = obs.get("current_endpoint", "")
        r_sign = "+" if reward >= 0 else ""

        icon = "✓" if reward > 0 else ("✗" if reward < 0 else "·")
        lines.append(f"  Step {step:2d} │ {icon} {action_name:<20s} │ {r_sign}{reward:.1f} │ {endpoint}")
        if error:
            lines.append(f"         │   ⚠ {error[:55]}")

    resolved = env.state.get("resolved", False)
    lines.append("")
    lines.append(f"{'─'*60}")

    if resolved:
        lines.append(f"  ✅ EPISODE RESOLVED — Total Reward: {total_reward:+.1f} | Steps: {step}")
    else:
        lines.append(f"  ❌ EPISODE FAILED  — Total Reward: {total_reward:+.1f} | Steps: {step}")

    lines.append(f"{'─'*60}")
    return "\n".join(lines)


def run_comparison(difficulty: str, seed: int) -> str:
    """Run all three agents on the same episode for comparison."""
    results = []
    for agent_type in ["random", "expert", "sft"]:
        result = run_episode(agent_type, difficulty, seed)
        results.append(result)
    return "\n\n".join(results)


def run_batch_eval(difficulty: str, n_episodes: int):  # noqa: C901
    """Run batch evaluation and show summary statistics."""
    n_episodes = min(int(n_episodes), 50)   # Cap at 50 for speed
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"📊 Batch Evaluation — {difficulty.capitalize()} | {n_episodes} episodes")
    lines.append(f"{'='*60}")
    lines.append("")

    yield "\n".join(lines) + "\n\nInitializing..."

    results = {}
    for agent_type in ["random", "expert", "sft"]:
        label = {"random": "Random Agent", "expert": "Expert Policy", "sft": "SFT Agent (Qwen 0.5B)"}[agent_type]
        
        status_idx = len(lines)
        lines.append(f"  ⏳ Running {label} (0/{n_episodes})...")
        yield "\n".join(lines)

        # Use max_steps=10 for batch eval speed; full 20 only in single-episode view
        env = ApiDriftGymEnv(max_steps=10, seed=42, difficulty=difficulty)
        successes = 0
        total_r = 0.0

        if agent_type == "random":
            agent_obj = RandomAgent()
        elif agent_type == "expert":
            agent_obj = ExpertPolicy()
        else:
            tracker_obj = WorkflowTracker()

        for ep in range(n_episodes):
            obs = env.reset()
            if agent_type == "expert":
                agent_obj.reset(obs)
            elif agent_type == "sft":
                tracker_obj = WorkflowTracker()
                tracker_obj.reset(obs)

            done, step, ep_r = False, 0, 0.0
            while not done and step < env.max_steps:
                if agent_type == "random":
                    action = agent_obj.act(obs)
                elif agent_type == "expert":
                    action = agent_obj.act(obs)
                else:
                    action, _ = sft_agent_act(obs, tracker_obj)
                obs, reward, done, info = env.step(action)
                if agent_type == "expert":
                    agent_obj.step(obs, action, reward, done)
                elif agent_type == "sft":
                    tracker_obj.update(obs, action, reward, done)
                ep_r += reward
                step += 1

            resolved = env.state.get("resolved", False)
            successes += int(resolved)
            total_r += ep_r
            
            if (ep + 1) % max(1, n_episodes // 5) == 0:
                lines[status_idx] = f"  ⏳ Running {label} ({ep+1}/{n_episodes})..."
                yield "\n".join(lines)

        rate = successes / n_episodes * 100
        avg_r = total_r / n_episodes
        results[agent_type] = (rate, avg_r)
        
        lines[status_idx] = f"  {label:<30s} │ Success: {rate:5.1f}% │ Avg Reward: {avg_r:+.2f}"
        yield "\n".join(lines)

    lines.append("")
    lines.append(f"{'─'*60}")
    yield "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🌀 API Drift Gym — Interactive Demo

**Can a 0.5B model learn to survive when the API keeps lying to it?**

This environment simulates real-world API schema drift — fields get renamed, types change,
and the agent must learn to **inspect → transform → retry** instead of blindly retrying.

### Agents
- **Random**: Picks actions at random (baseline)
- **Expert**: Deterministic stage-aware teacher (upper bound)
- **SFT**: Fine-tuned Qwen2.5-0.5B with LoRA adapter (our trained agent)

### Actions
| Action | Description |
|--------|-------------|
| `call_api` | Send a request to the current endpoint |
| `inspect_schema` | Examine the actual API contract |
| `transform_request` | Rebuild the payload to match the drifted schema |
| `retry` | Retry with previously transformed payload |
| `skip_step` | Skip the current workflow stage |
"""


def build_demo():
    with gr.Blocks(
        title="API Drift Gym",
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tab("🎮 Single Episode"):
            with gr.Row():
                agent_type = gr.Dropdown(
                    choices=["random", "expert", "sft"],
                    value="expert",
                    label="Agent Type",
                    info="Which agent to run",
                )
                difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Difficulty",
                    info="Number of workflow stages",
                )
                seed = gr.Number(value=42, label="Seed", precision=0, info="Random seed for reproducibility")
            run_btn = gr.Button("▶ Run Episode", variant="primary", size="lg")
            output = gr.Textbox(
                label="Episode Trace",
                lines=25,
                max_lines=40,
            )
            run_btn.click(fn=run_episode, inputs=[agent_type, difficulty, seed], outputs=output)

        with gr.Tab("⚔️ Side-by-Side Comparison"):
            with gr.Row():
                cmp_difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Difficulty",
                )
                cmp_seed = gr.Number(value=42, label="Seed", precision=0)
            cmp_btn = gr.Button("▶ Compare All Agents", variant="primary", size="lg")
            cmp_output = gr.Textbox(
                label="Comparison Results",
                lines=40,
                max_lines=80,
            )
            cmp_btn.click(fn=run_comparison, inputs=[cmp_difficulty, cmp_seed], outputs=cmp_output)

        with gr.Tab("📊 Batch Evaluation"):
            with gr.Row():
                batch_difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Difficulty",
                )
                batch_n = gr.Number(value=5, label="Episodes", precision=0, info="Number of episodes (max 50)")
            batch_btn = gr.Button("▶ Run Batch Evaluation", variant="primary", size="lg")
            batch_output = gr.Textbox(
                label="Evaluation Summary",
                lines=15,
                max_lines=25,
            )
            batch_btn.click(fn=run_batch_eval, inputs=[batch_difficulty, batch_n], outputs=batch_output)

        gr.Markdown("""
---
**Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv)** |
Model: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA SFT |
[GitHub](https://github.com/shivamkr1353/Meta_Hack_Round2) |
[Model Adapter](https://huggingface.co/shivamkr1353/api-drift-sft-qwen)
""")

    return demo


if __name__ == "__main__":
    demo = build_demo()
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme)
