import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./final_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load ─────────────────────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(f"Model loaded on {DEVICE}")

# ── Environment State ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an API recovery agent. You interact with an API that has dynamic schema changes.
Your available actions are:
- inspect_schema: Examine the current API schema
- transform_request: Modify the request to match the current schema
- call_api: Make the API call
- retry: Retry the last failed call

Always respond with ONLY one of these four actions. Nothing else."""

def get_next_action(observation: str, history: list = []) -> str:
    """Given an API observation, return the next action."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": observation})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Validate action
    valid_actions = ["inspect_schema", "transform_request", "call_api", "retry"]
    for action in valid_actions:
        if action in response.lower():
            return action
    return "inspect_schema"  # safe default

# ── Test Scenarios ───────────────────────────────────────────────────
TEST_CASES = [
    {
        "name": "Easy - Schema Mismatch",
        "observation": "API Error: Field 'user_id' not found. Response: {'error': 'schema_mismatch', 'expected': 'userId'}",
        "expected": "inspect_schema"
    },
    {
        "name": "Medium - Mid-episode Drift",
        "observation": "Previous call succeeded. New call failed: Field type changed from string to integer. Schema may have drifted.",
        "expected": "inspect_schema"
    },
    {
        "name": "Hard - Partial Observability",
        "observation": "API returned 200 but payload is incomplete. Missing nested fields. Last known schema is stale.",
        "expected": "transform_request"
    },
]

if __name__ == "__main__":
    print("\n" + "="*50)
    print("API Drift Gym — Inference Test")
    print("="*50 + "\n")

    results = []
    correct = 0

    for i, test in enumerate(TEST_CASES):
        action = get_next_action(test["observation"])
        is_correct = action == test["expected"]
        if is_correct:
            correct += 1

        result = {
            "test": test["name"],
            "observation": test["observation"][:60] + "...",
            "predicted_action": action,
            "expected_action": test["expected"],
            "correct": is_correct
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"[{status}] {test['name']}")
        print(f"    Predicted : {action}")
        print(f"    Expected  : {test['expected']}")
        print()

    print(f"Score: {correct}/{len(TEST_CASES)} ({100*correct//len(TEST_CASES)}%)")
    print("\nFull results:")
    print(json.dumps(results, indent=2))