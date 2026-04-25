import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

from api_drift_gym.env import ApiDriftGymEnv

print("Initializing Environment...")
env = ApiDriftGymEnv(max_steps=20, difficulty="hard")

def build_prompt(obs_dict):
    hint = obs_dict.get("available_hint", "")
    if isinstance(hint, dict):
        hint = json.dumps(hint)
    return f"Observation: {obs_dict.get('current_endpoint')} | Hint: {hint}\nAction:"

# Generate dataset of initial prompts
print("Generating dataset...")
prompts = []
for _ in range(200):
    obs = env.reset()
    prompts.append({"prompt": build_prompt(obs)})
dataset = Dataset.from_list(prompts)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA for fast training
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        obs = env.reset()
        total_reward = 0.0
        done = False
        
        # Split model output into actions (take first line)
        action_text = completion.strip().split('\n')[0].strip()
        
        if action_text:
            _, reward, done, _ = env.step(action_text)
            total_reward += reward
            
        rewards.append(total_reward)
    return rewards

# GRPO Config optimized for T4 GPU
config = GRPOConfig(
    output_dir="./grpo_results",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=2,
    generation_batch_size=2,
    max_completion_length=20,
    max_steps=200,   # Keep small as requested!
    save_strategy="no",
    report_to="none",
    bf16=False,
    fp16=True,
    logging_steps=10
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer
)

print("Starting GRPO training...")
trainer.train()
print("Training complete!")

# Save
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
print("Model saved to final_model directory.")
