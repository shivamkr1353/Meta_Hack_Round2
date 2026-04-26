# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Api Drift Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ApiDriftAction, ApiDriftObservation


class ApiDriftEnv(
    EnvClient[ApiDriftAction, ApiDriftObservation, State]
):
    """Client for the Api Drift Env Environment."""

    def _step_payload(self, action: ApiDriftAction) -> Dict:
        return {
            "raw": action.raw,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ApiDriftObservation]:
        obs_data = payload.get("observation", {})
        observation = ApiDriftObservation(
            last_action=obs_data.get("last_action", ""),
            api_response=obs_data.get("api_response", ""),
            error_message=obs_data.get("error_message", ""),
            available_hint=obs_data.get("available_hint"),
            step_count=obs_data.get("step_count", 0),
            workflow_step=obs_data.get("workflow_step", 0),
            current_endpoint=obs_data.get("current_endpoint"),
            difficulty=obs_data.get("difficulty"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
