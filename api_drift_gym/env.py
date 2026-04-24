from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .api_simulator import ApiDriftSimulator, JsonDict, Schema
from .logger import TrajectoryLogger
from .reward import RewardBreakdown, RewardEngine


@dataclass(frozen=True)
class Action:
    raw: str
    name: str
    endpoint: Optional[str] = None
    payload: Optional[JsonDict] = None
    legacy_format: bool = False

    def signature(self) -> str:
        return f"{self.name}:{self.endpoint or ''}"


@dataclass
class Observation:
    last_action: str
    api_response: str
    error_message: str
    available_hint: Optional[JsonDict]
    step_count: int
    workflow_step: int
    current_endpoint: Optional[str]
    difficulty: Optional[str]

    def to_dict(self) -> JsonDict:
        return asdict(self)


class ApiDriftGymEnv:
    ACTION_CALL_API = "call_api"
    ACTION_INSPECT_SCHEMA = "inspect_schema"
    ACTION_TRANSFORM_REQUEST = "transform_request"
    ACTION_RETRY = "retry"
    ACTION_SKIP_STEP = "skip_step"
    ACTION_NOOP = "noop"

    VALID_ACTIONS = {
        ACTION_CALL_API,
        ACTION_INSPECT_SCHEMA,
        ACTION_TRANSFORM_REQUEST,
        ACTION_RETRY,
        ACTION_SKIP_STEP,
        ACTION_NOOP,
    }

    def __init__(
        self,
        max_steps: int = 12,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.max_steps = max_steps
        self.default_difficulty = difficulty
        self.rng = random.Random(seed)
        self.simulator = ApiDriftSimulator(self.rng)
        self.reward_engine = RewardEngine()
        self.logger = TrajectoryLogger()
        self.episode_id = 0
        self.state: JsonDict = {}
        self.last_observation = Observation("", "", "", None, 0, 0, None, None)

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> JsonDict:
        if seed is not None:
            self.rng.seed(seed)

        episode_difficulty = difficulty or self.default_difficulty or self.simulator.choose_difficulty()
        episode = self.simulator.generate_episode(episode_difficulty)
        self.episode_id += 1

        self.state = {
            "task": episode["task"],
            "difficulty": episode_difficulty,
            "workflow": episode["workflow"],
            "workflow_step": 0,
            "api_states": episode["api_states"],
            "history": [],
            "step_count": 0,
            "max_steps": self.max_steps,
            "resolved": False,
            "failures": 0,
            "invalid_calls": 0,
            "last_action_signature": None,
            "last_endpoint": None,
            "retry_endpoint": self._current_endpoint_from_state(episode["workflow"], 0),
            "successful_response": None,
            "current_schema": {},
            "drifted_schema": {},
        }
        self._sync_compatibility_state()
        self.logger.start_episode(
            episode_id=self.episode_id,
            difficulty=episode_difficulty,
            workflow=[stage["name"] for stage in self.state["workflow"]],
        )

        self.last_observation = Observation(
            last_action="reset",
            api_response="",
            error_message="",
            available_hint={
                "task": self.state["task"],
                "difficulty": self.state["difficulty"],
                "workflow": [
                    {"name": stage["name"], "endpoint": stage["endpoint"]}
                    for stage in self.state["workflow"]
                ],
                "known_schema": copy.deepcopy(self.state["current_schema"]),
            },
            step_count=0,
            workflow_step=0,
            current_endpoint=self._current_endpoint(),
            difficulty=self.state["difficulty"],
        )
        return self.last_observation.to_dict()

    def step(self, action: str) -> Tuple[JsonDict, float, bool, JsonDict]:
        if not self.state:
            self.reset()

        parsed_action = self.parse_action(action)
        previous_signature = self.state.get("last_action_signature")
        self.state["step_count"] += 1

        execution, observation = self._execute_action(parsed_action)
        reward_components = self.calculate_reward(
            parsed_action=parsed_action,
            result=execution["result"],
            previous_action=previous_signature,
            execution=execution,
        )

        done = self._episode_done()
        terminal_failure = done and not self.state["resolved"]
        if terminal_failure:
            reward_components.timeout_penalty = -2.0
        reward = reward_components.compute_total(terminal_failure=terminal_failure)

        history_item = {
            "step": self.state["step_count"],
            "workflow_step": observation.workflow_step,
            "action": action,
            "action_name": parsed_action.name,
            "endpoint": execution["endpoint"],
            "result": execution["result"],
            "reward": reward,
            "reward_components": reward_components.to_dict(),
            "observation": observation.to_dict(),
        }
        self.state["history"].append(history_item)
        self.state["last_action_signature"] = parsed_action.signature()
        self.last_observation = observation

        self.logger.log_step(
            step_number=self.state["step_count"],
            workflow_step=observation.workflow_step,
            action=action,
            endpoint=execution["endpoint"],
            result=execution["result"],
            response=execution["response"] or observation.error_message,
            reward=reward,
            reward_components=reward_components.to_dict(),
            observation=observation.to_dict(),
        )
        if done:
            self.logger.finish(self.state["resolved"])

        info = {
            "result": execution["result"],
            "resolved": self.state["resolved"],
            "difficulty": self.state["difficulty"],
            "reward_components": reward_components.to_dict(),
            "trajectory": copy.deepcopy(self.state["history"]),
            "episode_summary": self.logger.summary.to_dict(),
            "log": self.logger.render_text(),
        }
        return observation.to_dict(), reward, done, info

    def parse_action(self, action: str) -> Action:
        if not isinstance(action, str) or not action:
            return Action(raw=str(action), name="invalid")

        if action == self.ACTION_RETRY:
            return Action(raw=action, name=self.ACTION_RETRY, endpoint=self.state.get("retry_endpoint"))

        if action == self.ACTION_SKIP_STEP:
            return Action(raw=action, name=self.ACTION_SKIP_STEP, endpoint=self._current_endpoint())

        if action == self.ACTION_NOOP:
            return Action(raw=action, name=self.ACTION_NOOP, endpoint=self._current_endpoint())

        if action == self.ACTION_INSPECT_SCHEMA:
            return Action(
                raw=action,
                name=self.ACTION_INSPECT_SCHEMA,
                endpoint=self._current_endpoint(),
                legacy_format=True,
            )

        if action.startswith(f"{self.ACTION_INSPECT_SCHEMA}:"):
            endpoint = self._normalize_endpoint(action.split(":", 1)[1])
            return Action(raw=action, name=self.ACTION_INSPECT_SCHEMA, endpoint=endpoint)

        for name in (self.ACTION_CALL_API, self.ACTION_TRANSFORM_REQUEST):
            prefix = f"{name}:"
            if action.startswith(prefix):
                remainder = action[len(prefix) :]
                if remainder.startswith("{"):
                    return Action(
                        raw=action,
                        name=name,
                        endpoint=self._current_endpoint(),
                        payload=self._parse_payload(remainder),
                        legacy_format=True,
                    )

                endpoint, separator, payload_raw = remainder.partition(":")
                if not separator:
                    return Action(raw=action, name=name, endpoint=self._normalize_endpoint(endpoint))

                return Action(
                    raw=action,
                    name=name,
                    endpoint=self._normalize_endpoint(endpoint),
                    payload=self._parse_payload(payload_raw),
                )

        if action in self.VALID_ACTIONS:
            return Action(raw=action, name=action, endpoint=self._current_endpoint())

        return Action(raw=action, name="invalid")

    def calculate_reward(
        self,
        parsed_action: Action,
        result: Optional[str] = None,
        previous_action: Optional[str] = None,
        execution: Optional[JsonDict] = None,
    ) -> RewardBreakdown:
        if execution is None:
            execution = {
                "result": result or "NOOP",
                "endpoint": parsed_action.endpoint,
                "target_current_step": parsed_action.endpoint == self._current_endpoint(),
                "after_failure": False,
                "inspected_before": False,
                "transformed_before": False,
                "invalid_usage": parsed_action.name == "invalid",
                "step_completed": False,
                "workflow_completed": False,
            }
        return self.reward_engine.calculate(
            state=self.state,
            action=parsed_action,
            execution=execution,
            previous_signature=previous_action,
        )

    def verify_success(self, payload: Optional[JsonDict], endpoint: Optional[str] = None) -> bool:
        if payload is None:
            return False

        target_endpoint = self._normalize_endpoint(endpoint) if endpoint else self._current_endpoint()
        if target_endpoint is None:
            return False

        api_state = self.state["api_states"].get(target_endpoint)
        if api_state is None:
            return False

        matched, _ = self.simulator.matches_schema(api_state["drifted_schema"], payload)
        return matched

    def verify_episode_success(self) -> bool:
        if self.state["invalid_calls"] > 0:
            return False

        for stage in self.state["workflow"]:
            if stage["skipped"] or not stage["completed"]:
                return False
        return True

    def get_log(self) -> str:
        return self.logger.render_text()

    def _execute_action(self, parsed_action: Action) -> Tuple[JsonDict, Observation]:
        expected_endpoint = self._current_endpoint()
        endpoint = parsed_action.endpoint or expected_endpoint
        api_state_before = copy.deepcopy(self.state["api_states"].get(endpoint, {}))
        execution = {
            "result": "INVALID",
            "endpoint": endpoint,
            "response": "",
            "error": "",
            "target_current_step": endpoint == expected_endpoint,
            "after_failure": bool(api_state_before.get("saw_failure")),
            "inspected_before": bool(api_state_before.get("inspected_after_failure")),
            "transformed_before": bool(api_state_before.get("transformed_after_failure")),
            "invalid_usage": False,
            "step_completed": False,
            "workflow_completed": False,
        }

        if parsed_action.name == "invalid":
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["error"] = "Invalid action format."
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        if parsed_action.name == self.ACTION_CALL_API:
            return self._execute_call(parsed_action, execution)

        if parsed_action.name == self.ACTION_INSPECT_SCHEMA:
            return self._execute_inspect(parsed_action, execution)

        if parsed_action.name == self.ACTION_TRANSFORM_REQUEST:
            return self._execute_transform(parsed_action, execution)

        if parsed_action.name == self.ACTION_RETRY:
            return self._execute_retry(parsed_action, execution)

        if parsed_action.name == self.ACTION_SKIP_STEP:
            return self._execute_skip(parsed_action, execution)

        execution["result"] = "NOOP"
        execution["response"] = "No operation performed."
        return execution, self._make_observation(parsed_action.raw, execution["response"], "", None)

    def _execute_call(self, parsed_action: Action, execution: JsonDict) -> Tuple[JsonDict, Observation]:
        endpoint = execution["endpoint"]
        api_state = self.state["api_states"].get(endpoint)
        if api_state is None:
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["result"] = "INVALID"
            execution["error"] = f"Unknown endpoint: {endpoint}"
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        stage = self._current_stage()
        if stage is not None and endpoint == stage["endpoint"]:
            stage["attempts"] += 1

        call_result = self.simulator.call_api(api_state, parsed_action.payload, stage["name"] if stage else "out_of_band")
        execution["result"] = call_result["result"]
        execution["response"] = call_result["response"]
        execution["error"] = call_result["error"]

        if call_result["result"] == "INVALID":
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        self.state["last_endpoint"] = endpoint
        self.state["retry_endpoint"] = endpoint
        if call_result["result"] == "ERROR":
            self.state["failures"] += 1
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        if stage is not None and endpoint == stage["endpoint"] and not stage["completed"]:
            stage["completed"] = True
            stage["response"] = call_result["response"]
            execution["step_completed"] = True
            self.state["workflow_step"] += 1
            self._sync_compatibility_state()

        if self.verify_episode_success():
            self.state["resolved"] = True
            self.state["successful_response"] = call_result["response_obj"]
            execution["workflow_completed"] = True

        return execution, self._make_observation(parsed_action.raw, execution["response"], "", None)

    def _execute_inspect(self, parsed_action: Action, execution: JsonDict) -> Tuple[JsonDict, Observation]:
        endpoint = execution["endpoint"]
        api_state = self.state["api_states"].get(endpoint)
        if api_state is None:
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["result"] = "INVALID"
            execution["error"] = f"Unknown endpoint: {endpoint}"
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        hint = self.simulator.inspect_schema(api_state)
        execution["result"] = "SCHEMA_HINT"
        execution["response"] = "Schema inspection completed."
        self.state["retry_endpoint"] = endpoint
        return execution, self._make_observation(parsed_action.raw, execution["response"], "", hint)

    def _execute_transform(self, parsed_action: Action, execution: JsonDict) -> Tuple[JsonDict, Observation]:
        endpoint = execution["endpoint"]
        api_state = self.state["api_states"].get(endpoint)
        if api_state is None:
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["result"] = "INVALID"
            execution["error"] = f"Unknown endpoint: {endpoint}"
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        if parsed_action.payload is None:
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["result"] = "INVALID"
            execution["error"] = "transform_request requires a JSON object payload."
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        api_state["pending_payload"] = copy.deepcopy(parsed_action.payload)
        if api_state["saw_failure"]:
            api_state["transformed_after_failure"] = True
        self.state["retry_endpoint"] = endpoint
        execution["result"] = "REQUEST_TRANSFORMED"
        execution["response"] = "Request transformed."
        return execution, self._make_observation(
            parsed_action.raw,
            execution["response"],
            "",
            {"pending_payload": copy.deepcopy(parsed_action.payload)},
        )

    def _execute_retry(self, parsed_action: Action, execution: JsonDict) -> Tuple[JsonDict, Observation]:
        endpoint = execution["endpoint"] or self._current_endpoint()
        api_state = self.state["api_states"].get(endpoint)
        if api_state is None or api_state.get("pending_payload") is None:
            execution["endpoint"] = endpoint
            self.state["invalid_calls"] += 1
            execution["invalid_usage"] = True
            execution["result"] = "INVALID"
            execution["error"] = "retry requires a previously transformed payload."
            return execution, self._make_observation(parsed_action.raw, "", execution["error"], None)

        retry_action = Action(
            raw=parsed_action.raw,
            name=self.ACTION_CALL_API,
            endpoint=endpoint,
            payload=copy.deepcopy(api_state["pending_payload"]),
        )
        execution["endpoint"] = endpoint
        return self._execute_call(retry_action, execution)

    def _execute_skip(self, parsed_action: Action, execution: JsonDict) -> Tuple[JsonDict, Observation]:
        stage = self._current_stage()
        if stage is None:
            execution["result"] = "SKIPPED"
            execution["response"] = "Workflow already exhausted."
            return execution, self._make_observation(parsed_action.raw, execution["response"], "", None)

        stage["skipped"] = True
        self.state["workflow_step"] += 1
        self._sync_compatibility_state()
        execution["result"] = "SKIPPED"
        execution["endpoint"] = stage["endpoint"]
        execution["response"] = f"Skipped workflow step {stage['name']}."
        return execution, self._make_observation(parsed_action.raw, execution["response"], "", None)

    def _make_observation(
        self,
        last_action: str,
        api_response: str,
        error_message: str,
        hint: Optional[JsonDict],
    ) -> Observation:
        return Observation(
            last_action=last_action,
            api_response=api_response,
            error_message=error_message,
            available_hint=hint,
            step_count=self.state["step_count"],
            workflow_step=self.state["workflow_step"],
            current_endpoint=self._current_endpoint(),
            difficulty=self.state["difficulty"],
        )

    def _episode_done(self) -> bool:
        if self.state["resolved"]:
            return True
        if self.state["step_count"] >= self.max_steps:
            return True
        return self.state["workflow_step"] >= len(self.state["workflow"])

    def _current_stage(self) -> Optional[JsonDict]:
        index = self.state["workflow_step"]
        if index >= len(self.state["workflow"]):
            return None
        return self.state["workflow"][index]

    def _current_endpoint(self) -> Optional[str]:
        stage = self._current_stage()
        if stage is None:
            return self.state.get("last_endpoint")
        return stage["endpoint"]

    def _current_endpoint_from_state(self, workflow: Any, workflow_step: int) -> Optional[str]:
        if workflow_step >= len(workflow):
            return None
        return workflow[workflow_step]["endpoint"]

    def _normalize_endpoint(self, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is None or endpoint == "":
            return self._current_endpoint()
        if endpoint.startswith("/"):
            return endpoint
        return f"/{endpoint}"

    def _parse_payload(self, raw_payload: str) -> Optional[JsonDict]:
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _sync_compatibility_state(self) -> None:
        endpoint = self._current_endpoint()
        if endpoint is None:
            return
        api_state = self.state["api_states"][endpoint]
        self.state["current_schema"] = copy.deepcopy(api_state["original_schema"])
        self.state["drifted_schema"] = copy.deepcopy(api_state["drifted_schema"])
        self.state["retry_endpoint"] = endpoint
