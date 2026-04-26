"""
Stage-Aware Policy for api_drift_gym
=====================================
My solution to improve the agent's performance on multi-step workflows.
I noticed the original agent failed because:
  1. It didn't remember which stages it already completed.
  2. It wasted steps retrying endpoints that were already fixed.
  3. It got stuck between transitions.

I built this as a drop-in module so I wouldn't have to break the original env.py.
"""

import copy
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# 1. STAGE PHASE TRACKING
# ─────────────────────────────────────────────────────────────────

class StagePhase(Enum):
    """Each endpoint goes through a fixed repair lifecycle."""
    UNTOUCHED  = auto()  # Haven't interacted yet
    CALLED     = auto()  # Sent initial call (expect error due to drift)
    INSPECTED  = auto()  # Schema inspected after failure
    TRANSFORMED = auto() # Payload rebuilt with correct fields
    RESOLVED   = auto()  # Endpoint successfully completed


@dataclass
class EndpointState:
    """Tracks the repair lifecycle for a single endpoint."""
    endpoint: str
    phase: StagePhase = StagePhase.UNTOUCHED
    schema_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    drift_case: str = "unknown"
    last_payload: Dict[str, Any] = field(default_factory=dict)
    error_history: List[str] = field(default_factory=list)
    attempt_count: int = 0

    @property
    def is_resolved(self) -> bool:
        return self.phase == StagePhase.RESOLVED

    @property
    def has_schema(self) -> bool:
        return self.phase in (StagePhase.INSPECTED, StagePhase.TRANSFORMED)

    def advance_to(self, new_phase: StagePhase) -> None:
        """Only advance forward in the lifecycle — never regress."""
        if new_phase.value > self.phase.value:
            self.phase = new_phase


# ─────────────────────────────────────────────────────────────────
# 2. WORKFLOW TRACKER — the core state machine
# ─────────────────────────────────────────────────────────────────

class WorkflowTracker:
    """
    I created this tracker to give the agent some memory. 
    It keeps track of which workflow stages are done, which endpoint is currently
    active, and what phase we're in so we don't repeat mistakes.
    """

    def __init__(self):
        self.endpoint_states: Dict[str, EndpointState] = {}
        self.workflow_order: List[str] = []
        self.current_stage_idx: int = 0
        self.completed_stages: List[str] = []
        self.episode_active: bool = False

    def reset(self, obs: dict) -> None:
        """Initialize tracker from the reset observation."""
        self.endpoint_states.clear()
        self.workflow_order.clear()
        self.completed_stages.clear()
        self.current_stage_idx = 0
        self.episode_active = True

        # Extract workflow from the reset hint
        hint = obs.get("available_hint") or {}
        workflow = hint.get("workflow", [])
        for stage in workflow:
            ep = stage.get("endpoint", "")
            self.workflow_order.append(ep)
            if ep not in self.endpoint_states:
                self.endpoint_states[ep] = EndpointState(endpoint=ep)

    def update(self, obs: dict, action_taken: str, reward: float,
               done: bool) -> None:
        """Update internal state after an env.step() call."""
        hint_raw = obs.get("available_hint")
        error = obs.get("error_message") or ""
        api_response = obs.get("api_response") or ""
        workflow_step = obs.get("workflow_step", self.current_stage_idx)

        # --- Identify which endpoint we just acted on ---
        # Before advancing, this is the endpoint we targeted
        acted_ep = self._active_endpoint()

        # --- Update the ACTED endpoint's phase from action result ---
        if acted_ep and acted_ep in self.endpoint_states:
            es = self.endpoint_states[acted_ep]

            if "call_api" in action_taken:
                if error:
                    es.advance_to(StagePhase.CALLED)
                    es.error_history.append(error)
                es.attempt_count += 1

            elif "inspect_schema" in action_taken:
                if isinstance(hint_raw, dict) and "required_fields" in hint_raw:
                    es.schema_fields = hint_raw.get("required_fields", [])
                    es.field_types = hint_raw.get("field_types", {})
                    es.drift_case = hint_raw.get("drift_case", "unknown")
                    es.advance_to(StagePhase.INSPECTED)

            elif "transform_request" in action_taken:
                es.advance_to(StagePhase.TRANSFORMED)

        # --- Detect stage advancement (env moved workflow_step) ---
        if workflow_step > self.current_stage_idx:
            resolved_ep = acted_ep
            if resolved_ep and resolved_ep in self.endpoint_states:
                self.endpoint_states[resolved_ep].advance_to(
                    StagePhase.RESOLVED
                )
                if resolved_ep not in self.completed_stages:
                    self.completed_stages.append(resolved_ep)
            self.current_stage_idx = workflow_step

        if done:
            self.episode_active = False

    def _active_endpoint(self) -> Optional[str]:
        if self.current_stage_idx < len(self.workflow_order):
            return self.workflow_order[self.current_stage_idx]
        return None

    @property
    def active_endpoint(self) -> Optional[str]:
        return self._active_endpoint()

    @property
    def active_state(self) -> Optional[EndpointState]:
        ep = self._active_endpoint()
        return self.endpoint_states.get(ep) if ep else None

    @property
    def all_resolved(self) -> bool:
        return all(es.is_resolved for es in self.endpoint_states.values())

    @property
    def progress_fraction(self) -> float:
        if not self.workflow_order:
            return 0.0
        return len(self.completed_stages) / len(self.workflow_order)


# ─────────────────────────────────────────────────────────────────
# 3. PAYLOAD BUILDER — type-aware defaults per endpoint
# ─────────────────────────────────────────────────────────────────

_ENDPOINT_DEFAULTS = {
    "/user":    {"user_id": 1, "full_name": "Test User", "id": 1,
                 "name": "test", "email": "test@example.com",
                 "account_id": 1, "user_name": "test_user"},
    "/orders":  {"order_id": "o_001", "user_id": 1, "status": "pending",
                 "account_id": 1, "max_results": 10, "page_size": 10,
                 "limit": 10},
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
                 "status": "pending", "currency": "USD",
                 "payer_id": "payer_001"},
}

_TYPE_DEFAULTS = {"int": 1, "str": "test_value", "bool": True, "float": 1.0}


def build_payload_from_state(es: EndpointState) -> dict:
    """Build a type-correct payload from tracked endpoint state."""
    base = _ENDPOINT_DEFAULTS.get(es.endpoint, {})
    if not es.schema_fields:
        return base

    payload = {}
    for f in es.schema_fields:
        if f in base:
            payload[f] = base[f]
        elif f in es.field_types:
            payload[f] = _TYPE_DEFAULTS.get(es.field_types[f], "test_value")
        else:
            payload[f] = f"default_{f}"
    es.last_payload = payload
    return payload


# ─────────────────────────────────────────────────────────────────
# 4. STAGE-AWARE POLICY — the decision engine
# ─────────────────────────────────────────────────────────────────

class StageAwarePolicy:
    """
    My deterministic policy to guide the agent. It essentially acts as an expert
    system to generate perfect training data.
    
    The logic enforces a strict order:
      call_api -> inspect_schema -> transform_request -> call_api (with correct payload)
    """

    def __init__(self):
        self.tracker = WorkflowTracker()

    def reset(self, obs: dict) -> None:
        self.tracker.reset(obs)

    def act(self, obs: dict) -> str:
        """
        Returns a fully-formed environment action string.
        Decision tree based on the active endpoint's phase.
        """
        es = self.tracker.active_state
        endpoint = self.tracker.active_endpoint

        # Fallback: no active stage (workflow exhausted or unknown)
        if es is None or endpoint is None:
            ep = obs.get("current_endpoint", "/user")
            return f"call_api:{ep}:{{}}"

        # ── GUARD: never re-act on a resolved endpoint ──
        if es.is_resolved:
            # This shouldn't happen if the env advances workflow_step,
            # but guard against it anyway
            return "skip_step"

        # ── PHASE-BASED DECISION ──
        phase = es.phase

        if phase == StagePhase.UNTOUCHED:
            # Step 1: Send initial call (will likely fail due to drift)
            return f"call_api:{endpoint}:{{}}"

        if phase == StagePhase.CALLED:
            # Step 2: We got an error → inspect the schema
            return f"inspect_schema:{endpoint}"

        if phase == StagePhase.INSPECTED:
            # Step 3: We know the schema → build and send transform
            payload = build_payload_from_state(es)
            return f"transform_request:{endpoint}:{json.dumps(payload)}"

        if phase == StagePhase.TRANSFORMED:
            # Step 4: Transform stored → call_api WITH the correct payload
            payload = es.last_payload or build_payload_from_state(es)
            return f"call_api:{endpoint}:{json.dumps(payload)}"

        # Unreachable but safe
        return f"call_api:{endpoint}:{{}}"

    def step(self, obs: dict, action: str, reward: float,
             done: bool) -> None:
        """Call after env.step() to update internal tracking."""
        self.tracker.update(obs, action, reward, done)


# ─────────────────────────────────────────────────────────────────
# 5. FORMAT_OBS — stage-aware prompt for model-based agents
# ─────────────────────────────────────────────────────────────────

def format_obs_stage_aware(obs: dict, tracker: WorkflowTracker) -> str:
    """
    I enhanced the observation string to inject my tracker's state so the model
    actually knows what's going on and what it should do next.
    """
    es = tracker.active_state
    endpoint = obs.get("current_endpoint", "/user")
    error = obs.get("error_message") or "none"
    step_num = obs.get("step_count", 0)

    # Stage progress summary
    total = len(tracker.workflow_order)
    done_count = len(tracker.completed_stages)
    progress = f"{done_count}/{total}"
    completed_str = ", ".join(tracker.completed_stages) if tracker.completed_stages else "none"
    pending = [ep for ep in tracker.workflow_order
               if ep not in tracker.completed_stages]
    pending_str = ", ".join(pending) if pending else "none"

    # Current phase
    phase_str = es.phase.name if es else "UNKNOWN"
    fields_str = ", ".join(es.schema_fields) if (es and es.schema_fields) else "unknown"

    # Directive
    if es is None:
        directive = "Workflow complete."
    elif es.phase == StagePhase.UNTOUCHED:
        directive = f"Send initial call_api to {endpoint}."
    elif es.phase == StagePhase.CALLED:
        directive = f"Error received. Use inspect_schema on {endpoint}."
    elif es.phase == StagePhase.INSPECTED:
        directive = f"Schema known (fields: {fields_str}). Use transform_request."
    elif es.phase == StagePhase.TRANSFORMED:
        directive = f"Payload ready. Use call_api with payload on {endpoint}."
    elif es.phase == StagePhase.RESOLVED:
        directive = "Stage resolved. Transition to next endpoint."
    else:
        directive = "Continue."

    return (
        f"[API Agent - Stage Aware]\n"
        f"Endpoint     : {endpoint}\n"
        f"Step         : {step_num}\n"
        f"Phase        : {phase_str}\n"
        f"Progress     : {progress}\n"
        f"Completed    : {completed_str}\n"
        f"Pending      : {pending_str}\n"
        f"Error        : {error}\n"
        f"Known fields : {fields_str}\n"
        f"Directive    : {directive}\n\n"
        f"Choose one action:\n"
        f"  call_api\n"
        f"  inspect_schema\n"
        f"  transform_request\n\n"
        f"Action:"
    )


# ─────────────────────────────────────────────────────────────────
# 6. MODEL-BASED AGENT WRAPPER
# ─────────────────────────────────────────────────────────────────

def make_model_action(action_word: str, tracker: WorkflowTracker) -> str:
    """
    Helper function I wrote to safely translate the model's text output into a 
    valid environment action. It uses my tracker to fill in the correct payload.
    """
    es = tracker.active_state
    endpoint = tracker.active_endpoint

    if es is None or endpoint is None:
        return "skip_step"

    # Override: if model picks wrong action for the phase, correct it
    if es.is_resolved:
        return "skip_step"

    if action_word == "inspect_schema":
        return f"inspect_schema:{endpoint}"

    if action_word == "transform_request":
        payload = build_payload_from_state(es)
        return f"transform_request:{endpoint}:{json.dumps(payload)}"

    if action_word == "call_api":
        if es.phase == StagePhase.TRANSFORMED:
            payload = es.last_payload or build_payload_from_state(es)
            return f"call_api:{endpoint}:{json.dumps(payload)}"
        return f"call_api:{endpoint}:{{}}"

    return f"call_api:{endpoint}:{{}}"


# ─────────────────────────────────────────────────────────────────
# 7. TEACHER POLICY — generates training data with stage awareness
# ─────────────────────────────────────────────────────────────────

def teacher_policy_stage_aware(
    obs: dict, tracker: WorkflowTracker
) -> Tuple[str, str]:
    """
    Wraps my deterministic policy so I can generate perfectly labeled trajectories
    to train the Qwen model.
    """
    policy = StageAwarePolicy()
    policy.tracker = tracker  # Share tracker state
    env_action = policy.act(obs)
    action_label = env_action.split(":")[0]
    return action_label, env_action


# ─────────────────────────────────────────────────────────────────
# 8. EVALUATION HARNESS
# ─────────────────────────────────────────────────────────────────

def evaluate_stage_aware(env, n_episodes: int = 50,
                         verbose: bool = False) -> float:
    """
    My evaluation script to prove that this expert policy achieves 100% success.
    """
    policy = StageAwarePolicy()
    successes = 0

    for ep in range(n_episodes):
        obs = env.reset()
        policy.reset(obs)
        done = False
        step = 0
        total_reward = 0.0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {ep+1:3d} | "
                  f"workflow: {policy.tracker.workflow_order}")

        while not done and step < env.max_steps:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            policy.step(obs, action, reward, done)
            total_reward += reward
            step += 1

            if verbose:
                es = policy.tracker.active_state
                phase = es.phase.name if es else "DONE"
                print(f"  step {step:2d} | {action.split(':')[0]:20s} | "
                      f"phase={phase:12s} | r={reward:+.1f} | "
                      f"progress={policy.tracker.progress_fraction:.0%}")

        resolved = env.state.get("resolved", False)
        successes += int(resolved)

        if verbose:
            tag = "SUCCESS" if resolved else "FAIL"
            print(f"  -> {tag} "
                  f"| total_reward={total_reward:+.1f} "
                  f"| steps={step}")

    rate = successes / n_episodes
    print(f"\nStage-Aware Policy: {successes}/{n_episodes} = {rate:.0%}")
    return rate


# ─────────────────────────────────────────────────────────────────
# 9. TRAINING DATA GENERATOR — stage-aware trajectories
# ─────────────────────────────────────────────────────────────────

def collect_stage_aware_trajectories(
    env, n_episodes: int = 300
) -> List[Tuple[str, str]]:
    """
    I run my teacher policy here to gather training data. I inject the stage
    progress into the reasoning trace so the model learns the step-by-step logic.
    """
    policy = StageAwarePolicy()
    data = []

    for _ in range(n_episodes):
        obs = env.reset()
        policy.reset(obs)
        done = False
        step = 0

        while not done and step < env.max_steps:
            prompt = format_obs_stage_aware(obs, policy.tracker)
            action = policy.act(obs)
            action_label = action.split(":")[0]

            # Reasoned label with stage context
            es = policy.tracker.active_state
            ep = policy.tracker.active_endpoint or "/user"
            phase = es.phase.name if es else "DONE"
            progress = policy.tracker.progress_fraction

            reason = (
                f"Stage {policy.tracker.current_stage_idx+1}/"
                f"{len(policy.tracker.workflow_order)}: "
                f"{ep} is in {phase} phase. "
                f"Progress: {progress:.0%}. "
                f"Action: {action_label}"
            )
            data.append((prompt, reason))

            obs, reward, done, info = env.step(action)
            policy.step(obs, action, reward, done)
            step += 1

    return data


# ─────────────────────────────────────────────────────────────────
# 10. MAIN — demo and benchmark
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from api_drift_gym import ApiDriftGymEnv

    print("=" * 60)
    print("STAGE-AWARE POLICY BENCHMARK")
    print("=" * 60)

    # Test across all difficulties
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- Difficulty: {difficulty} ---")
        env = ApiDriftGymEnv(max_steps=20, seed=42, difficulty=difficulty)
        evaluate_stage_aware(env, n_episodes=50, verbose=False)

    # Verbose demo on hard
    print("\n" + "=" * 60)
    print("VERBOSE HARD EPISODE")
    print("=" * 60)
    env = ApiDriftGymEnv(max_steps=20, seed=11, difficulty="hard")
    evaluate_stage_aware(env, n_episodes=3, verbose=True)
