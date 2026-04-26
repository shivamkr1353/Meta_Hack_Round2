from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


JsonDict = Dict[str, Any]


@dataclass
class RewardBreakdown:
    per_step_correctness: float = 0.0
    workflow_progress: float = 0.0
    repeat_penalty: float = 0.0
    phase_order_bonus: float = 0.0
    resolution_bonus: float = 0.0
    timeout_penalty: float = 0.0
    total: float = 0.0

    def compute_total(self, terminal_failure: bool = False) -> float:
        if terminal_failure:
            self.total = -2.0
            return self.total

        self.total = (
            self.per_step_correctness
            + self.workflow_progress
            + self.repeat_penalty
            + self.phase_order_bonus
            + self.resolution_bonus
            + self.timeout_penalty
        )
        return self.total

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RewardEngine:
    def calculate(
        self,
        state: JsonDict,
        action: Any,
        execution: JsonDict,
        previous_signature: Optional[str],
    ) -> RewardBreakdown:
        components = RewardBreakdown()
        components.per_step_correctness = self._score_correctness(action, execution)

        if previous_signature and previous_signature == action.signature():
            components.repeat_penalty = -0.2

        if execution["step_completed"]:
            components.workflow_progress = 1.5

        if execution["workflow_completed"]:
            components.resolution_bonus = 3.0

        components.phase_order_bonus = self._phase_order_bonus(action, execution)
        return components

    def _score_correctness(self, action: Any, execution: JsonDict) -> float:
        if action.name == "invalid" or execution["result"] == "INVALID" or execution["invalid_usage"]:
            return -1.0

        if action.name == "noop":
            return 0.0

        if action.name == "skip_step":
            return -0.5

        if action.name == "call_api":
            if execution["step_completed"]:
                return 1.0
            if execution["target_current_step"] and execution["result"] == "ERROR":
                return -0.5
            if execution["target_current_step"]:
                return 0.5
            return -0.5

        if action.name == "inspect_schema":
            if not execution["target_current_step"]:
                return -0.5
            if execution["after_failure"]:
                return 1.0
            return 0.5

        if action.name == "transform_request":
            if not execution["target_current_step"]:
                return -0.5
            if execution["after_failure"] and execution["inspected_before"]:
                return 1.0
            if execution["after_failure"] or execution["inspected_before"]:
                return 0.5
            return -0.5

        if action.name == "retry":
            if execution["step_completed"]:
                return 1.0
            if execution["target_current_step"]:
                return -0.5
            return -1.0

        return -0.5

    def _phase_order_bonus(self, action: Any, execution: JsonDict) -> float:
        if action.name == "inspect_schema" and execution["target_current_step"] and execution["after_failure"]:
            return 0.3

        if (
            action.name == "transform_request"
            and execution["target_current_step"]
            and execution["after_failure"]
            and execution["inspected_before"]
        ):
            return 0.3

        if (
            action.name == "retry"
            and execution["target_current_step"]
            and execution["inspected_before"]
            and execution["transformed_before"]
            and execution["step_completed"]
        ):
            return 0.3

        if action.name == "retry" and not execution["inspected_before"]:
            return -0.4

        if action.name == "transform_request" and not execution["after_failure"]:
            return -0.4

        if action.name == "skip_step":
            return -0.4

        return 0.0
