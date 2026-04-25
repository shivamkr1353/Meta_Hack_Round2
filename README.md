# api_drift_gym

`api_drift_gym` is a lightweight reinforcement-learning style environment for one specific problem: teaching an agent to recover when enterprise APIs silently change their request schema.

The environment simulates workflow execution under API drift. An agent starts with an old schema, makes calls that can fail, inspects the schema after failure, transforms the request, retries, and tries to finish a full business workflow before running out of steps.

This repository is small, but it already captures the main research idea:

- API contracts can drift without warning.
- Failures may initially be vague or misleading.
- Work is often multi-step, not single-call.
- The agent must adapt while preserving workflow progress.
- Good behavior depends on action order, not only final success.

## Problem This Repo Simulates

Imagine an internal enterprise system that depends on several services such as:

- `/user`
- `/orders`
- `/payment`
- `/process`
- `/summary`

The client still believes the old request schema is valid, but one or more services have changed field names or field requirements. The drift is hidden at the start of the episode. The agent only discovers it through failed calls and later inspection.

The environment therefore models an adaptation loop:

1. Try the request using the original schema.
2. Observe a failure.
3. Inspect the schema for better hints.
4. Transform the payload to match the drifted schema.
5. Retry.
6. Continue until the full workflow is resolved or the episode ends.

This makes the task useful for testing:

- sequential decision-making
- recovery from partial failure
- hidden-state reasoning
- reward shaping for repair behavior
- curriculum learning from easy to hard workflows

## High-Level Design

The main class is `ApiDriftGymEnv` in `api_drift_gym/env.py`.

An episode contains:

- a task description
- a selected difficulty level
- a workflow made of one or more ordered stages
- per-endpoint API state
- failure/inspection/transform history
- reward-tracked trajectory logs

The episode state is stored in `env.state` and includes fields such as:

- `task`
- `difficulty`
- `workflow`
- `workflow_step`
- `api_states`
- `history`
- `step_count`
- `resolved`
- `failures`
- `invalid_calls`
- `retry_endpoint`

## Core Scenario Mechanics

### 1. Workflows Are Multi-Step

The environment is not just a single API call benchmark.

Depending on difficulty, the agent may need to complete:

- one stage in `easy`
- two stages in `medium`
- four stages in `hard`

Example hard workflow:

1. `fetch_user` via `/user`
2. `fetch_orders` via `/orders`
3. `process_data` via `/process`
4. `send_summary` via `/summary`

Each stage must be completed in order for the episode to fully resolve.

### 2. Every Endpoint Has Original and Drifted Schemas

For each endpoint, the simulator keeps:

- `original_schema`: what the client thinks is correct
- `drifted_schema`: what the API currently expects

Example kinds of drift already implemented:

- field rename drift
- partial schema drift

Examples:

- `/user`: `name` may become `full_name`
- `/orders`: `limit` may become `max_results`
- `/summary`: `message` may become `summary`

### 3. Drift Is Hidden at Reset Time

When the environment resets, the observation does not expose the drifted schema directly.

The initial hint includes:

- task
- difficulty
- workflow stage names and endpoints
- the currently known schema

The actual drift is intended to emerge through interaction.

### 4. Failure Unlocks Better Inspection

Inspection before failure gives only partial metadata.

After a failed request, inspection becomes much more informative and can reveal:

- required fields
- field types
- changed fields
- drift case
- deprecated candidate fields

This is one of the central design ideas in the repo: the agent should not get full schema knowledge for free.

### 5. Errors Can Be Misleading

For `medium` and `hard` cases, some failures intentionally use vague error messages on the first mismatch, such as a generic downstream contract warning instead of a precise missing-field list.

This makes the task closer to real-world production debugging, where errors are not always clean or direct.

### 6. Extra Unused Fields Add Noise

The simulator may attach deprecated or unused field candidates like:

- `legacy_id`
- `debug_mode`
- `trace_token`
- `deprecated_flag`

These are not necessarily the fix. They are distractors meant to make the adaptation problem less trivial.

## Environment API

The package exports `ApiDriftGymEnv` from `api_drift_gym/__init__.py`.

Typical usage:

```python
from api_drift_gym import ApiDriftGymEnv

env = ApiDriftGymEnv(max_steps=20, seed=11, difficulty="hard")
obs = env.reset()
obs, reward, done, info = env.step("inspect_schema:/user")
```

Constructor arguments:

- `max_steps`: maximum actions allowed before terminal failure
- `seed`: optional deterministic random seed
- `difficulty`: optional fixed difficulty; otherwise the simulator samples one

Main methods:

- `reset(seed=None, difficulty=None) -> dict`
- `step(action: str) -> (observation, reward, done, info)`
- `get_log() -> str`
- `verify_success(payload, endpoint=None) -> bool`
- `verify_episode_success() -> bool`

## Action Space

The environment accepts string-based actions.

### Endpoint-Aware Actions

- `call_api:<endpoint>:<payload_json>`
- `inspect_schema:<endpoint>`
- `transform_request:<endpoint>:<payload_json>`
- `retry`
- `skip_step`

### Legacy Compatibility Actions

These still work and automatically target the current workflow endpoint:

- `call_api:<payload_json>`
- `inspect_schema`
- `transform_request:<payload_json>`
- `noop`

### What Each Action Means

`call_api`

- Sends a JSON payload to the target endpoint.
- If the payload matches the drifted schema exactly, the stage succeeds.
- If not, the environment returns an error and marks that endpoint as having seen a failure.

`inspect_schema`

- Before failure: returns partial compatibility metadata only.
- After failure: returns much richer schema information.

`transform_request`

- Stores a pending payload for retry.
- If done after failure, it is treated as part of a meaningful repair sequence.

`retry`

- Replays the last transformed payload against the target endpoint.
- Fails if no transformed payload exists.

`skip_step`

- Skips the current workflow stage.
- Lets the agent move on, but hurts reward and prevents full episode success.

`noop`

- Valid but unhelpful placeholder action.
- Useful mostly for negative-path testing.

### Action Parsing Details

`ApiDriftGymEnv.parse_action()` supports:

- endpoint-normalized actions such as `user` becoming `/user`
- payload parsing from JSON strings
- legacy formats
- invalid-action detection

Invalid actions increase `invalid_calls` and receive negative reward.

## Observation Format

`reset()` and `step()` return a dictionary with this shape:

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

What matters most:

- `available_hint` changes based on what the agent has already learned
- `workflow_step` shows current stage progress
- `current_endpoint` shows where the agent is operating now
- `error_message` carries the failure clue when a call fails

## Reward Design

Rewards are calculated in `api_drift_gym/reward.py` through `RewardEngine`.

The reward is decomposed into:

- `per_step_correctness`
- `workflow_progress`
- `repeat_penalty`
- `phase_order_bonus`
- `resolution_bonus`
- `timeout_penalty`

### Reward Intuition

The environment is rewarding more than just "eventual success."

It also rewards whether the agent behaves in a sensible adaptation order:

1. fail or detect mismatch
2. inspect
3. transform
4. retry
5. progress the workflow

### Important Reward Behaviors

- completing a workflow stage gives `workflow_progress = 1.5`
- resolving the entire workflow gives `resolution_bonus = 3.0`
- repeating the same action signature gives `repeat_penalty = -0.2`
- skipping a step is penalized
- retrying too early is penalized
- transforming before observing failure is penalized
- invalid actions get strong negative feedback

### Terminal Failure

If the episode ends without full resolution, the environment forces:

- `timeout_penalty = -2.0`
- total reward = `-2.0`

That behavior is explicitly tested.

## Difficulty Levels

The simulator has three curriculum levels.

### `easy`

- single-step workflows
- simpler failure patterns
- no misleading errors
- no noisy deprecated fields

### `medium`

- two-step workflows
- may include misleading errors
- may include one distractor field

### `hard`

- four-step workflow
- hidden drift across several endpoints
- misleading errors are possible
- one or two distractor fields may appear

Difficulty can be fixed explicitly or sampled randomly.

## Simulator Behavior

The schema and workflow generation logic lives in `api_drift_gym/api_simulator.py`.

It defines:

- base schemas for every endpoint
- drift options for each endpoint
- workflow templates by difficulty
- schema matching rules
- mismatch description rules

Success requires an exact schema match:

- missing fields fail
- unexpected fields fail
- wrong types fail

That exactness is important because it turns the task into real repair, not approximate matching.

## Logging and Trajectory Output

The environment logs episode progress through `TrajectoryLogger` in `api_drift_gym/logger.py`.

The log captures:

- episode id
- difficulty
- workflow summary
- each action taken
- endpoint touched
- result
- response or error
- scalar reward
- reward breakdown
- final status

`env.get_log()` returns a readable text trace that is useful for debugging agents and inspecting trajectories manually.

## Repository Layout

```text
api_drift_gym/
  __init__.py          # public exports
  api_simulator.py     # schemas, drift generation, workflow templates, matching
  env.py               # environment state machine, action execution, observations
  logger.py            # trajectory logging and episode summaries
  reward.py            # reward shaping logic
tests/
  test_api_drift_gym.py
baseline.py            # random-action baseline
demo.py                # scripted walkthrough of a hard episode
README.md
```

## What Each Top-Level Script Does

### `baseline.py`

This is a very weak baseline that randomly chooses from:

- `inspect_schema:/user`
- `inspect_schema:/orders`
- `retry`
- `skip_step`

It runs 20 episodes and prints a success rate.

Important note: this is not a trained baseline. It is only a quick sanity script showing that naive behavior performs poorly.

### `demo.py`

This is the best file to run if someone wants to understand the environment behavior quickly.

It:

- creates a hard-difficulty environment
- constructs both original and drifted payloads
- intentionally tries the old payload first
- then inspects schema
- then transforms the request
- then retries
- prints observations, rewards, and reward components at each step
- prints the final trajectory log

If you want another model to understand the intended interaction loop, `demo.py` is the clearest executable example in the repo.

## Tests

`tests/test_api_drift_gym.py` currently checks:

- legacy single-endpoint behavior still works
- hard multi-endpoint workflows can be completed
- hidden drift is only fully exposed after failure plus inspection
- terminal failure produces the exact `-2.0` penalty
- `verify_success()` requires an exact endpoint schema match

These tests are a strong summary of the current contract of the environment.

## How To Run

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the walkthrough demo:

```bash
python demo.py
```

Run the simple baseline:

```bash
python baseline.py
```

## Current Strengths

What is already implemented well:

- hidden schema drift instead of fully supervised schema reveal
- multi-step workflows instead of isolated API calls
- endpoint-aware action format
- compatibility support for older single-endpoint actions
- reward shaping that encourages repair order
- deterministic tests for key behavior
- text trajectory logging for debugging

## Current Limitations

What is not implemented yet:

- no formal Gymnasium wrapper or spaces object
- no learned agent or training pipeline in tracked repo files
- no vectorized environment support
- no stochastic response payload semantics beyond schema validity
- no observation masking beyond the current hint design
- no benchmark harness for comparing policies
- no persistence layer or dataset export for trajectories

So the repo is best understood as a clean prototype environment, not a full training framework.

## Best Short Description For Another Model

If you want to brief Claude Opus or any other model quickly, use something close to this:

> This repo implements a reinforcement-learning style environment called `api_drift_gym` for studying how an agent adapts to hidden API schema drift in enterprise workflows. Each episode contains one or more ordered workflow stages across endpoints like `/user`, `/orders`, `/process`, and `/summary`. The agent starts with old schemas, may fail with misleading errors, can inspect schemas for partial or full hints, transform request payloads, retry, and is rewarded for following a sensible repair order. Success requires exact matching of the hidden drifted schema and completion of the whole workflow before the step budget runs out.

## Detailed Handoff Notes

If another model is going to continue work on this project, these are the most important facts to keep in mind:

- The authoritative environment logic is in `api_drift_gym/env.py`.
- Drift generation and workflow templates are in `api_drift_gym/api_simulator.py`.
- The reward contract is in `api_drift_gym/reward.py`.
- The current public behavior is best summarized by the tests.
- `demo.py` is the clearest executable narrative of the intended adaptation loop.
- Legacy action formats are intentionally preserved and should not be broken accidentally.
- Hidden drift should remain hidden at reset time; full schema detail should only surface after the right interaction pattern.

## Suggested Next Steps

Natural extensions from here would be:

- add a real RL training loop and policy baseline
- add Gymnasium-compatible wrappers
- export trajectories for supervised or offline RL experiments
- create harder drift types such as type drift, optional fields, and nested payload changes
- add evaluation metrics across curricula and seeds
- introduce partial observability experiments and ablations

## Summary

This project is a compact but thoughtfully structured environment for testing API-repair behavior under hidden schema drift. The main novelty is not raw complexity; it is the combination of:

- workflow-level adaptation
- delayed schema revelation
- adversarial error signals
- reward shaping around repair order

That makes it a good starting point for experimentation with agents that must diagnose, adapt, and recover inside changing software systems.
