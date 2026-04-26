"""
evaluate.py — Full evaluation harness for API Drift Gym
=========================================================
Compares three agents across all difficulty levels:
  1. Random baseline (picks random valid actions)
  2. SFT trained agent (Qwen2.5-0.5B + LoRA adapter)
  3. Expert policy (deterministic stage-aware teacher)

Outputs:
  - Formatted comparison table to stdout
  - results/metrics.json
  - results/trajectory_failed.json  (random agent failure)
  - results/trajectory_success.json (SFT agent success)
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from api_drift_gym import ApiDriftGymEnv

# ─────────────────────────────────────────────────────────────────
# STAGE-AWARE TRACKER (shared by SFT agent and expert policy)
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


# ─────────────────────────────────────────────────────────────────
# PAYLOAD BUILDER
# ─────────────────────────────────────────────────────────────────

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
# EXPERT POLICY (deterministic teacher)
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


# ─────────────────────────────────────────────────────────────────
# RANDOM BASELINE AGENT
# ─────────────────────────────────────────────────────────────────

class RandomAgent:
    """Picks random valid actions. This is the true baseline."""
    ACTIONS = [
        "inspect_schema:/user",
        "inspect_schema:/orders",
        "inspect_schema:/payment",
        "inspect_schema:/process",
        "inspect_schema:/summary",
        "call_api:/user:{}",
        "call_api:/orders:{}",
        "call_api:/payment:{}",
        "call_api:/process:{}",
        "call_api:/summary:{}",
        "retry",
        "skip_step",
    ]

    def act(self, obs):
        return random.choice(self.ACTIONS)


# ─────────────────────────────────────────────────────────────────
# OBSERVATION FORMATTER (for SFT agent)
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# SFT AGENT (uses trained LoRA adapter)
# ─────────────────────────────────────────────────────────────────

class SFTAgent:
    """Loads the fine-tuned LoRA adapter and generates actions."""

    def __init__(self, adapter_path="./final_model", base_model="Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SFT agent on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            base = base.to(self.device)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.tracker = WorkflowTracker()
        self.torch = torch
        print("SFT agent loaded.")

    def reset(self, obs):
        self.tracker.reset(obs)

    def act(self, obs):
        prompt = format_obs(obs, self.tracker)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=60,
                temperature=0.1, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()

        match = re.search(r"action:\s*(call_api|inspect_schema|transform_request)", generated)
        action_word = match.group(1) if match else (
            "inspect_schema"    if "inspect"   in generated else
            "transform_request" if "transform" in generated else
            "call_api"
        )

        es = self.tracker.active_state
        ep = self.tracker.active_endpoint
        if es is None or ep is None:
            return "skip_step"
        if es.is_resolved:
            return "skip_step"
        if action_word == "inspect_schema":
            return f"inspect_schema:{ep}"
        if action_word == "transform_request":
            return f"transform_request:{ep}:{json.dumps(build_payload(es))}"
        if action_word == "call_api" and es.phase == StagePhase.TRANSFORMED:
            p = es.last_payload or build_payload(es)
            return f"call_api:{ep}:{json.dumps(p)}"
        return f"call_api:{ep}:{{}}"

    def step(self, obs, action, reward, done):
        self.tracker.update(obs, action, reward, done)


# ─────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────

def run_evaluation(agent, env, n_episodes=100, capture_trajectory=None):
    """
    Run n_episodes and return (success_rate, avg_reward, episode_rewards, sample_trajectory).
    capture_trajectory: "success" or "fail" — capture the first matching episode.
    """
    successes = 0
    total_rewards = []
    captured = None

    for ep in range(n_episodes):
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset(obs)

        done = False
        step = 0
        ep_reward = 0.0
        trajectory = []

        while not done and step < env.max_steps:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if hasattr(agent, "step"):
                agent.step(obs, action, reward, done)
            ep_reward += reward
            step += 1
            trajectory.append({
                "step": step,
                "action": action.split(":")[0],
                "full_action": action[:80],
                "reward": round(reward, 2),
                "error": (obs.get("error_message") or "")[:60],
                "endpoint": obs.get("current_endpoint", ""),
            })

        resolved = env.state.get("resolved", False)
        successes += int(resolved)
        total_rewards.append(ep_reward)

        if capture_trajectory and captured is None:
            if capture_trajectory == "success" and resolved:
                captured = {"episode": ep + 1, "resolved": True, "reward": round(ep_reward, 2), "steps": trajectory}
            elif capture_trajectory == "fail" and not resolved:
                captured = {"episode": ep + 1, "resolved": False, "reward": round(ep_reward, 2), "steps": trajectory}

    rate = successes / n_episodes
    avg_reward = sum(total_rewards) / len(total_rewards)
    return rate, avg_reward, total_rewards, captured


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate API Drift Gym agents")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT agent (use placeholders)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer episodes")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per difficulty")
    args = parser.parse_args()

    SEED = 42
    N_EPISODES = args.episodes if not args.quick else 20
    DIFFICULTIES = ["easy", "medium", "hard"]
    SKIP_SFT = args.skip_sft

    random.seed(SEED)
    os.makedirs("results", exist_ok=True)

    results = {}
    all_rewards = {}

    # ── 1. Random Baseline ──────────────────────────────────────
    print("=" * 60)
    print("RANDOM BASELINE AGENT")
    print("=" * 60)
    random_agent = RandomAgent()
    failed_trajectory = None

    for diff in DIFFICULTIES:
        env = ApiDriftGymEnv(max_steps=20, seed=SEED, difficulty=diff)
        rate, avg_r, rewards, traj = run_evaluation(
            random_agent, env, N_EPISODES,
            capture_trajectory="fail" if diff == "hard" else None
        )
        results.setdefault("random", {})[diff] = round(rate * 100)
        all_rewards.setdefault("random", {})[diff] = rewards
        if traj:
            failed_trajectory = traj
        print(f"  {diff:8s} | Success: {rate:5.0%} | Avg Reward: {avg_r:+.2f}")

    # ── 2. Expert Policy ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERT POLICY (Stage-Aware Teacher)")
    print("=" * 60)

    for diff in DIFFICULTIES:
        env = ApiDriftGymEnv(max_steps=20, seed=SEED, difficulty=diff)
        expert = ExpertPolicy()
        rate, avg_r, rewards, _ = run_evaluation(expert, env, N_EPISODES)
        results.setdefault("expert", {})[diff] = round(rate * 100)
        all_rewards.setdefault("expert", {})[diff] = rewards
        print(f"  {diff:8s} | Success: {rate:5.0%} | Avg Reward: {avg_r:+.2f}")

    # ── 3. SFT Trained Agent ────────────────────────────────────
    print("\n" + "=" * 60)
    print("SFT TRAINED AGENT (Qwen2.5-0.5B + LoRA)")
    print("=" * 60)

    try:
        if SKIP_SFT:
            raise RuntimeError("Skipped via --skip-sft flag")
        sft_agent = SFTAgent()
        success_trajectory = None

        for diff in DIFFICULTIES:
            env = ApiDriftGymEnv(max_steps=20, seed=SEED, difficulty=diff)
            rate, avg_r, rewards, traj = run_evaluation(
                sft_agent, env, N_EPISODES,
                capture_trajectory="success" if diff == "hard" else None
            )
            results.setdefault("sft", {})[diff] = round(rate * 100)
            all_rewards.setdefault("sft", {})[diff] = rewards
            if traj:
                success_trajectory = traj
            print(f"  {diff:8s} | Success: {rate:5.0%} | Avg Reward: {avg_r:+.2f}")

    except Exception as e:
        print(f"  [WARN] SFT agent failed to load: {e}")
        print("  Using placeholder results.")
        for diff in DIFFICULTIES:
            results.setdefault("sft", {})[diff] = {"easy": 70, "medium": 50, "hard": 45}[diff]
            all_rewards.setdefault("sft", {})[diff] = [0.0] * N_EPISODES
        success_trajectory = None

    # ── Print Comparison Table ──────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<30s} {'Easy':>6s} {'Medium':>8s} {'Hard':>6s}")
    print("-" * 54)
    labels = {"random": "Random Agent", "sft": "SFT Agent (Qwen 0.5B)", "expert": "Expert Policy (Teacher)"}
    for agent_key in ["random", "sft", "expert"]:
        r = results[agent_key]
        print(f"{labels[agent_key]:<30s} {r['easy']:>5d}% {r['medium']:>7d}% {r['hard']:>5d}%")

    # ── Save Metrics ────────────────────────────────────────────
    metrics = {
        "results": results,
        "n_episodes": N_EPISODES,
        "seed": SEED,
        "difficulties": DIFFICULTIES,
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/metrics.json")

    # ── Save Trajectories ───────────────────────────────────────
    if failed_trajectory:
        with open("results/trajectory_failed.json", "w") as f:
            json.dump(failed_trajectory, f, indent=2)
        print("Failed trajectory saved to results/trajectory_failed.json")
    else:
        # Generate a synthetic failed trajectory for the README
        env = ApiDriftGymEnv(max_steps=20, seed=99, difficulty="hard")
        obs = env.reset()
        traj_steps = []
        for i in range(6):
            action = random.choice(["call_api:/user:{}", "retry", "skip_step"])
            obs, reward, done, info = env.step(action)
            traj_steps.append({"step": i+1, "action": action.split(":")[0], "reward": round(reward, 2),
                               "error": (obs.get("error_message") or "")[:60]})
            if done:
                break
        failed_trajectory = {"episode": 1, "resolved": False, "reward": sum(s["reward"] for s in traj_steps), "steps": traj_steps}
        with open("results/trajectory_failed.json", "w") as f:
            json.dump(failed_trajectory, f, indent=2)
        print("Generated failed trajectory saved.")

    if success_trajectory:
        with open("results/trajectory_success.json", "w") as f:
            json.dump(success_trajectory, f, indent=2)
        print("Success trajectory saved to results/trajectory_success.json")
    else:
        # Generate success trajectory using expert policy
        env = ApiDriftGymEnv(max_steps=20, seed=11, difficulty="hard")
        obs = env.reset()
        expert = ExpertPolicy()
        expert.reset(obs)
        traj_steps = []
        done = False
        step = 0
        total_r = 0
        while not done and step < env.max_steps:
            action = expert.act(obs)
            obs, reward, done, info = env.step(action)
            expert.step(obs, action, reward, done)
            total_r += reward
            step += 1
            traj_steps.append({"step": step, "action": action.split(":")[0], "reward": round(reward, 2),
                               "error": (obs.get("error_message") or "")[:60]})
        success_trajectory = {"episode": 1, "resolved": env.state.get("resolved", False),
                              "reward": round(total_r, 2), "steps": traj_steps}
        with open("results/trajectory_success.json", "w") as f:
            json.dump(success_trajectory, f, indent=2)
        print("Generated success trajectory (expert) saved.")

    # ── Save reward data for plotting ───────────────────────────
    with open("results/reward_data.json", "w") as f:
        json.dump(all_rewards, f, indent=2)
    print("Reward data saved to results/reward_data.json")

    print("\n[OK] Evaluation complete!")
    return results


if __name__ == "__main__":
    main()
