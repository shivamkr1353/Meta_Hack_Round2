# api_drift_gym

`api_drift_gym` is an OpenEnv-style reinforcement learning environment for
enterprise workflow adaptation under API schema drift.

## What Changed

The environment now supports:

- multi-step workflows with `state["workflow_step"]`
- multiple APIs with separate original and drifted schemas
- hidden schema drift revealed through failures and schema inspection
- curriculum difficulty levels: `easy`, `medium`, `hard`
- adversarial cases such as misleading errors and extra unused fields
- modular structure:
  - `api_simulator.py`
  - `reward.py`
  - `logger.py`
  - `env.py`

## Action Formats

New endpoint-aware actions:

- `call_api:<endpoint>:<payload_json>`
- `inspect_schema:<endpoint>`
- `transform_request:<endpoint>:<payload_json>`
- `retry`
- `skip_step`

Legacy compatibility is preserved:

- `call_api:<payload_json>`
- `inspect_schema`
- `transform_request:<payload_json>`
- `noop`

Legacy actions automatically target the current workflow endpoint.

## Observation

```python
{
    "last_action": str,
    "api_response": str,
    "error_message": str,
    "available_hint": dict | None,
    "step_count": int,
    "workflow_step": int,
    "current_endpoint": str | None,
    "difficulty": str | None,
}
```

## Reward Components

`step()` returns a scalar reward built from deterministic components:

- `per_step_correctness`
- `workflow_progress`
- `repeat_penalty`
- `phase_order_bonus`
- `resolution_bonus`
- `timeout_penalty`

Terminal workflow failure returns `-2.0`.

## Run

```bash
python -m unittest discover -s tests -v
python demo.py
```
