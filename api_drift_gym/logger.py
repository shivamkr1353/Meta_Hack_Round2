from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class StepLog:
    step_number: int
    workflow_step: int
    action: str
    endpoint: Optional[str]
    result: str
    response: str
    reward: float
    reward_components: JsonDict
    observation: JsonDict


@dataclass
class EpisodeSummary:
    episode_id: int = 0
    difficulty: str = ""
    workflow: List[str] = field(default_factory=list)
    steps_taken: int = 0
    status: str = "IN_PROGRESS"

    def to_dict(self) -> JsonDict:
        return {
            "episode_id": self.episode_id,
            "difficulty": self.difficulty,
            "workflow": copy.deepcopy(self.workflow),
            "steps_taken": self.steps_taken,
            "status": self.status,
        }


@dataclass
class TrajectoryLogger:
    summary: EpisodeSummary = field(default_factory=EpisodeSummary)
    steps: List[StepLog] = field(default_factory=list)

    def start_episode(self, episode_id: int, difficulty: str, workflow: List[str]) -> None:
        self.summary = EpisodeSummary(
            episode_id=episode_id,
            difficulty=difficulty,
            workflow=copy.deepcopy(workflow),
            steps_taken=0,
            status="IN_PROGRESS",
        )
        self.steps.clear()

    def log_step(
        self,
        step_number: int,
        workflow_step: int,
        action: str,
        endpoint: Optional[str],
        result: str,
        response: str,
        reward: float,
        reward_components: JsonDict,
        observation: JsonDict,
    ) -> None:
        self.summary.steps_taken = step_number
        self.steps.append(
            StepLog(
                step_number=step_number,
                workflow_step=workflow_step,
                action=action,
                endpoint=endpoint,
                result=result,
                response=response,
                reward=reward,
                reward_components=copy.deepcopy(reward_components),
                observation=copy.deepcopy(observation),
            )
        )

    def finish(self, resolved: bool) -> None:
        self.summary.status = "RESOLVED" if resolved else "FAILED"

    def render_text(self) -> str:
        lines = [f"Episode {self.summary.episode_id}:"]
        lines.append(f"Difficulty: {self.summary.difficulty}")
        lines.append(f"Workflow: {' -> '.join(self.summary.workflow)}")
        for step in self.steps:
            reward_text = f"{step.reward:+.2f}"
            response = step.response or step.observation.get("error_message", "")
            lines.append(
                f"Step {step.step_number}: workflow_step={step.workflow_step} "
                f"action={step.action} endpoint={step.endpoint or '-'} "
                f"-> {step.result} response={response} reward={reward_text}"
            )
        lines.append("")
        lines.append("Final:")
        lines.append(f"-> {self.summary.status}")
        lines.append(f"Steps taken: {self.summary.steps_taken}")
        return "\n".join(lines)
