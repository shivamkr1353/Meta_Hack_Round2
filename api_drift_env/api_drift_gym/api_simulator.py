from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]
Schema = Dict[str, str]


@dataclass
class WorkflowStage:
    name: str
    endpoint: str
    description: str
    completed: bool = False
    skipped: bool = False
    attempts: int = 0
    response: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "description": self.description,
            "completed": self.completed,
            "skipped": self.skipped,
            "attempts": self.attempts,
            "response": self.response,
        }


class ApiDriftSimulator:
    SCHEMA_TYPES = {
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
    }

    BASE_SCHEMAS: Dict[str, Schema] = {
        "/user": {"id": "int", "name": "str"},
        "/orders": {"user_id": "int", "limit": "int"},
        "/payment": {"txn_id": "str", "amount": "float"},
        "/process": {"user_id": "int", "order_count": "int"},
        "/summary": {"email": "str", "message": "str"},
    }

    DRIFT_OPTIONS: Dict[str, List[JsonDict]] = {
        "/user": [
            {
                "drifted_schema": {"id": "int", "full_name": "str"},
                "drift_case": "rename_field",
            },
            {
                "drifted_schema": {"user_id": "int", "full_name": "str"},
                "drift_case": "partial_schema_drift",
            },
        ],
        "/orders": [
            {
                "drifted_schema": {"account_id": "int", "max_results": "int"},
                "drift_case": "rename_field",
            },
            {
                "drifted_schema": {"user_id": "int", "page_size": "int"},
                "drift_case": "partial_schema_drift",
            },
        ],
        "/payment": [
            {
                "drifted_schema": {"payment_id": "str", "status": "str"},
                "drift_case": "rename_field",
            },
            {
                "drifted_schema": {"txn_id": "str", "payment_status": "str"},
                "drift_case": "partial_schema_drift",
            },
        ],
        "/process": [
            {
                "drifted_schema": {"account_id": "int", "total_orders": "int"},
                "drift_case": "rename_field",
            },
            {
                "drifted_schema": {"user_id": "int", "aggregate_count": "int"},
                "drift_case": "partial_schema_drift",
            },
        ],
        "/summary": [
            {
                "drifted_schema": {"recipient": "str", "summary": "str"},
                "drift_case": "rename_field",
            },
            {
                "drifted_schema": {"email": "str", "digest": "str"},
                "drift_case": "partial_schema_drift",
            },
        ],
    }

    WORKFLOWS: Dict[str, List[JsonDict]] = {
        "easy": [
            {
                "task": "Fetch user data through the enterprise directory API",
                "steps": [
                    {
                        "name": "fetch_user",
                        "endpoint": "/user",
                        "description": "Load the target user profile.",
                    }
                ],
            },
            {
                "task": "Check a payment record through the billing API",
                "steps": [
                    {
                        "name": "fetch_payment",
                        "endpoint": "/payment",
                        "description": "Load the payment transaction.",
                    }
                ],
            },
        ],
        "medium": [
            {
                "task": "Collect user and order data for an account audit",
                "steps": [
                    {
                        "name": "fetch_user",
                        "endpoint": "/user",
                        "description": "Load the target user profile.",
                    },
                    {
                        "name": "fetch_orders",
                        "endpoint": "/orders",
                        "description": "Load the user's recent orders.",
                    },
                ],
            },
            {
                "task": "Validate user identity before checking payment status",
                "steps": [
                    {
                        "name": "fetch_user",
                        "endpoint": "/user",
                        "description": "Load the target user profile.",
                    },
                    {
                        "name": "fetch_payment",
                        "endpoint": "/payment",
                        "description": "Load the payment transaction.",
                    },
                ],
            },
        ],
        "hard": [
            {
                "task": "Build an enterprise customer summary across services",
                "steps": [
                    {
                        "name": "fetch_user",
                        "endpoint": "/user",
                        "description": "Load the target user profile.",
                    },
                    {
                        "name": "fetch_orders",
                        "endpoint": "/orders",
                        "description": "Load the user's recent orders.",
                    },
                    {
                        "name": "process_data",
                        "endpoint": "/process",
                        "description": "Aggregate user and order counts.",
                    },
                    {
                        "name": "send_summary",
                        "endpoint": "/summary",
                        "description": "Send the final summary message.",
                    },
                ],
            }
        ],
    }

    UNUSED_FIELD_POOL = ["legacy_id", "debug_mode", "trace_token", "deprecated_flag"]

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_difficulty(self) -> str:
        return self.rng.choice(["easy", "medium", "hard"])

    def generate_episode(self, difficulty: str) -> JsonDict:
        workflow_template = copy.deepcopy(self.rng.choice(self.WORKFLOWS[difficulty]))
        workflow = [WorkflowStage(**step).to_dict() for step in workflow_template["steps"]]

        api_states: Dict[str, JsonDict] = {}
        for step in workflow:
            endpoint = step["endpoint"]
            if endpoint in api_states:
                continue

            drift = copy.deepcopy(self.rng.choice(self.DRIFT_OPTIONS[endpoint]))
            extra_unused_fields = self._sample_unused_fields(difficulty)
            api_states[endpoint] = {
                "endpoint": endpoint,
                "original_schema": copy.deepcopy(self.BASE_SCHEMAS[endpoint]),
                "drifted_schema": copy.deepcopy(drift["drifted_schema"]),
                "drift_case": drift["drift_case"],
                "misleading_errors": difficulty != "easy" and self.rng.random() < 0.5,
                "extra_unused_fields": extra_unused_fields,
                "pending_payload": None,
                "last_payload": None,
                "last_error": "",
                "last_response": "",
                "failure_count": 0,
                "inspect_count": 0,
                "saw_failure": False,
                "inspected_after_failure": False,
                "transformed_after_failure": False,
                "resolved": False,
            }

        return {
            "task": workflow_template["task"],
            "difficulty": difficulty,
            "workflow": workflow,
            "api_states": api_states,
        }

    def inspect_schema(self, api_state: JsonDict) -> JsonDict:
        api_state["inspect_count"] += 1
        original_schema = api_state["original_schema"]
        drifted_schema = api_state["drifted_schema"]
        changed_fields = sorted(
            set(original_schema.keys()).symmetric_difference(set(drifted_schema.keys()))
        )
        shared_fields = sorted(set(original_schema.keys()) & set(drifted_schema.keys()))

        if api_state["saw_failure"]:
            api_state["inspected_after_failure"] = True
            return {
                "endpoint": api_state["endpoint"],
                "required_fields": sorted(drifted_schema.keys()),
                "field_types": copy.deepcopy(drifted_schema),
                "changed_fields": changed_fields,
                "drift_case": api_state["drift_case"],
                "deprecated_candidates": copy.deepcopy(api_state["extra_unused_fields"]),
            }

        return {
            "endpoint": api_state["endpoint"],
            "field_count": len(drifted_schema),
            "stable_fields": shared_fields,
            "suspected_changes": len(changed_fields),
            "notes": "Partial compatibility metadata only. Trigger a failure to reveal more.",
        }

    def call_api(self, api_state: JsonDict, payload: Optional[JsonDict], stage_name: str) -> JsonDict:
        if payload is None or not isinstance(payload, dict):
            return {
                "result": "INVALID",
                "response": "",
                "error": "API call requires a JSON object payload.",
                "matched_schema": False,
                "response_obj": None,
            }

        api_state["last_payload"] = copy.deepcopy(payload)
        matched_schema, details = self.matches_schema(api_state["drifted_schema"], payload)
        if matched_schema:
            response_obj = {
                "status": "success",
                "endpoint": api_state["endpoint"],
                "workflow_step": stage_name,
                "accepted_fields": sorted(payload.keys()),
            }
            api_state["last_error"] = ""
            api_state["last_response"] = json.dumps(response_obj, sort_keys=True)
            api_state["resolved"] = True
            return {
                "result": "SUCCESS",
                "response": api_state["last_response"],
                "error": "",
                "matched_schema": True,
                "response_obj": response_obj,
            }

        error = self.describe_mismatch(api_state, details)
        api_state["saw_failure"] = True
        api_state["failure_count"] += 1
        api_state["last_error"] = error
        api_state["last_response"] = ""
        return {
            "result": "ERROR",
            "response": "",
            "error": error,
            "matched_schema": False,
            "response_obj": None,
        }

    def matches_schema(self, schema: Schema, payload: JsonDict) -> Tuple[bool, JsonDict]:
        expected_keys = set(schema.keys())
        actual_keys = set(payload.keys())
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        wrong_types = []

        for key in sorted(expected_keys & actual_keys):
            if not self.value_matches_type(payload[key], schema[key]):
                wrong_types.append(f"{key}: expected {schema[key]}")

        details = {
            "missing": missing,
            "unexpected": unexpected,
            "wrong_types": wrong_types,
        }
        return not any(details.values()), details

    def value_matches_type(self, value: Any, expected_type: str) -> bool:
        if expected_type == "bool":
            return type(value) is bool
        if expected_type == "float":
            return isinstance(value, (float, int)) and type(value) is not bool
        if expected_type == "int":
            return type(value) is int
        if expected_type == "str":
            return isinstance(value, str)
        return False

    def describe_mismatch(self, api_state: JsonDict, details: JsonDict) -> str:
        if api_state["misleading_errors"] and api_state["failure_count"] == 0:
            base = "Downstream contract mismatch detected in the request envelope."
        else:
            parts = []
            if details["missing"]:
                parts.append(f"missing fields={details['missing']}")
            if details["unexpected"]:
                parts.append(f"unexpected fields={details['unexpected']}")
            if details["wrong_types"]:
                parts.append(f"type errors={details['wrong_types']}")
            base = "Schema mismatch: " + "; ".join(parts or ["unknown contract drift"])

        if api_state["extra_unused_fields"]:
            base += f" Deprecated candidates={api_state['extra_unused_fields']}."
        return base

    def _sample_unused_fields(self, difficulty: str) -> List[str]:
        if difficulty == "easy":
            return []
        if difficulty == "medium":
            count = 1 if self.rng.random() < 0.6 else 0
        else:
            count = 1 if self.rng.random() < 0.5 else 2
        return copy.deepcopy(self.rng.sample(self.UNUSED_FIELD_POOL, k=count))
