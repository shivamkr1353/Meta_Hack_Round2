# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Api Drift Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import json
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ApiDriftAction, ApiDriftObservation
    from ..api_drift_gym.env import ApiDriftGymEnv
except ImportError:
    from models import ApiDriftAction, ApiDriftObservation
    from api_drift_gym.env import ApiDriftGymEnv

class ApiDriftEnvironment(Environment):
    """
    OpenEnv wrapper for ApiDriftGymEnv.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.gym_env = ApiDriftGymEnv(difficulty="hard")

    def reset(self) -> ApiDriftObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs_dict = self.gym_env.reset()
        
        return ApiDriftObservation(
            last_action=obs_dict.get("last_action", ""),
            api_response=obs_dict.get("api_response", ""),
            error_message=obs_dict.get("error_message", ""),
            available_hint=obs_dict.get("available_hint"),
            step_count=obs_dict.get("step_count", 0),
            workflow_step=obs_dict.get("workflow_step", 0),
            current_endpoint=obs_dict.get("current_endpoint"),
            difficulty=obs_dict.get("difficulty"),
            done=False,
            reward=0.0
        )

    def step(self, action: ApiDriftAction) -> ApiDriftObservation:  # type: ignore[override]
        self._state.step_count += 1
        raw_action = action.raw
        
        obs_dict, reward, done, info = self.gym_env.step(raw_action)
        
        return ApiDriftObservation(
            last_action=obs_dict.get("last_action", ""),
            api_response=obs_dict.get("api_response", ""),
            error_message=obs_dict.get("error_message", ""),
            available_hint=obs_dict.get("available_hint"),
            step_count=obs_dict.get("step_count", 0),
            workflow_step=obs_dict.get("workflow_step", 0),
            current_endpoint=obs_dict.get("current_endpoint"),
            difficulty=obs_dict.get("difficulty"),
            done=done,
            reward=reward,
            metadata=info
        )

    @property
    def state(self) -> State:
        return self._state
