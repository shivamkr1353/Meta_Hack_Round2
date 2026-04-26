# 🌀 API Drift Gym: Teaching an Agent to Debug Like a Senior Engineer

### Can a 0.5B model learn to survive when the API keeps lying to it?

We gave a tiny language model a broken API, a schema that silently mutates mid-session, and zero documentation. No hardcoded rules. No lookup tables. Just a malformed response and four tools to figure it out.

By episode 10, it had learned to stop blindly retrying. It started inspecting first, transforming the request, then calling again — exactly the way a senior engineer would debug a production incident at 2 AM.

**This is API Drift Gym** — an adversarial environment where an RL-style agent learns structured fault recovery by navigating real-world API breakage patterns: silent schema drift, partial observability, and cascading failures that punish guessing.

> 🏆 **Submission for Theme #3.2: Personalized Tasks**  
> 🎯 **Targeting Bonus Prize:** Patronus AI - *Consumer Workflows with Schema Drift*

---

## What Problem Are We Solving?

Real-world API integrations fail in ways that are invisible at the surface. The status code is 200. The response arrives. But the schema shifted — a field was renamed, a type was changed, or a key silently disappeared. A naive agent (or engineer) retries. It fails again. It retries harder. Still fails.

**The standard retry loop is not a debugging strategy. It is a panic response.**

API Drift Gym forces an agent to learn the correct mental model:

```
bad:  call_api -> fail -> retry -> fail -> retry -> timeout

good: call_api -> fail -> inspect_schema -> transform_request -> call_api -> success
```

This is not a toy problem. Schema drift is how production systems actually break, and the ability to recover without human intervention is what makes an agent useful.

---

## Results

### Baseline vs Trained Agent

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| Random Agent | 0% | 0% | 0% |
| SFT Agent (Qwen 0.5B) | 70% | 50% | 45% |
| Expert Policy (Teacher) | 100% | 100% | 100% |

### Success Rate Comparison

![Success Rate Comparison](plots/success_rate_comparison.png)

### Reward Curves by Difficulty

![Reward Per Episode](plots/reward_per_episode.png)

### Training Loss

![Training Loss](plots/training_loss.png)

---

### Trajectory Comparison

**Before training (Random Agent):**
```
Step  1 │ ✗ call_api              │ -1.0  │ /user
        │   ⚠ Schema mismatch: missing fields=['full_name']
Step  2 │ ✗ retry                 │ -1.0  │ /user
        │   ⚠ retry requires a previously transformed payload
Step  3 │ ✗ skip_step             │ -0.5  │ /orders
Step  4 │ ✗ call_api              │ -1.0  │ /orders
        │   ⚠ Schema mismatch: missing fields=['max_results']
...
[timeout — episode failed]
```

**After training (SFT Agent):**
```
Step  1 │ · call_api              │ -0.5  │ /user
        │   ⚠ Schema mismatch (expected drift)
Step  2 │ ✓ inspect_schema        │ +1.0  │ /user
Step  3 │ ✓ transform_request     │ +1.0  │ /user
Step  4 │ ✓ call_api              │ +1.0  │ /orders
        │   200 OK — stage resolved
...
Step 12 │ ✓ call_api              │ +1.0  │ /summary
[episode resolved — 4/4 stages complete]
```

The difference is not speed. It is **reasoning structure**.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                      API DRIFT GYM LOOP                         │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Schema Drift│───►│  Environment │───►│       Agent        │  │
│  │ Generator   │    │ (OpenEnv +   │    │ (Qwen 0.5B + LoRA) │  │
│  │ Easy/Med/   │    │ api_drift_gym)│   │ inspect_schema     │  │
│  │ Hard drift  │    │              │    │ transform_request  │  │
│  └─────────────┘    └──────────────┘    │ call_api / retry   │  │
│                            │            └────────┬───────────┘  │
│                            │                     │              │
│                     API response           reward signal         │
│                                                 │               │
│                                                 ▼               │
│                                        ┌────────────────┐       │
│                                        │  SFT Trainer   │       │
│                                        │  (TRL + LoRA)  │       │
│                                        └────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### The Loop (Step by Step)

1. **Schema drift generator** creates mismatches (rename, partial drift, extra noise fields) by difficulty.
2. **Environment** emits partial observations (error, response fragments, hints).
3. **Agent** picks from `inspect_schema`, `transform_request`, `call_api`, `retry`.
4. **Reward engine** scores correctness, workflow progress, order quality, and terminal outcomes.
5. **SFT trainer** learns from teacher-guided trajectories.
6. **Difficulty escalates** with stronger hidden drift and multi-stage workflows.

### What Makes This Different

- **Hidden schema changes mid-episode**
- **Partial observability by design**
- **No hardcoded recovery policy in the model**
- **Difficulty tiers with real behavioral differences**
- **Teacher trajectories that teach sequence quality, not just endpoint success**

---

## Reward Design

The environment uses additive reward shaping with a strict terminal-failure override:

```python
reward = (
    per_step_correctness
    + workflow_progress
    + repeat_penalty
    + phase_order_bonus
    + resolution_bonus
)

if terminal_failure:
    reward = -2.0
```

Key properties:

- encourages action quality, not only final success
- rewards workflow-stage completion and full resolution
- penalizes invalid or repetitive behavior
- gives a clear terminal negative signal on timeout/failure

---

## The Agent's Journey

### Day 1: Blind Retrying

The agent receives a partial API response with a field mismatch error. It has no schema documentation, only the error message and four available actions.

It calls the API again. Fails. Calls again. Fails.

### Day 4: First Inspection

Instead of immediately retrying, the agent calls `inspect_schema`. It sees the contract the API expects, notices mismatch, calls `transform_request`, then calls again.

Response: 200. Fields match. Episode resolved.

### Day 10: Systematic Recovery

The pattern becomes consistent. Failure triggers inspection before action. The agent learns that retrying without understanding is wasted compute.

The environment escalates: drift can happen mid-episode. A schema valid on step 1 may become invalid by step 3. The agent must detect and re-inspect dynamically.

---

## Key Design Decisions

**1) Inspect-first over retry loops**
The core learned behavior is inspect → transform → act, not retry spam.

**2) Partial observability by design**
Hiding full schema forces reasoning over observation history.

**3) SFT first for stability**
Teacher trajectories provide a stable policy prior before reward-only optimization.

**4) Real difficulty tiers**
Easy/Medium/Hard change observability and drift timing, not just thresholds.

**5) GRPO is a natural extension**
SFT builds structured behavior; GRPO can improve robustness under unseen drift.

---

## What We Learned

1. Observability calibration matters more than cosmetic difficulty labels.
2. Trajectory quality beats raw trajectory count.
3. Stable reward shaping is upstream of stable training.
4. Training surfaces environment bugs quickly (which is good).
5. A small model with disciplined structure can outperform larger but poorly trained systems.

> Built with [OpenEnv v0.2.3](https://github.com/meta-pytorch/OpenEnv/tree/v0.2.3) | Trained on [HF Jobs](https://huggingface.co/docs/hub/jobs) with T4 GPU | Model: `Qwen2.5-0.5B-Instruct` + LoRA SFT | Adapter: [shivamkr1353/api-drift-sft-qwen](https://huggingface.co/shivamkr1353/api-drift-sft-qwen)
