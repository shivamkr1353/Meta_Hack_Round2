import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

from api_drift_gym.env import ApiDriftGymEnv

# -------------------------
# LOAD MODEL
# -------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

# -------------------------
# ENV WRAPPER
# -------------------------
env = ApiDriftGymEnv()

def generate_episode():
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        prompt = f"Observation: {obs}\nAction:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)

        action = tokenizer.decode(out[0], skip_special_tokens=True)
        action = action.split("Action:")[-1].strip()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward

# -------------------------
# GRPO CONFIG
# -------------------------
config = GRPOConfig(
    learning_rate=5e-6,
    batch_size=2,
    num_rollouts=2,
    max_steps=200,   # keep small!
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
)

# -------------------------
# TRAIN LOOP
# -------------------------
print("Starting GRPO training...")

for step in range(config.max_steps):
    reward = generate_episode()

    print(f"Step {step} | Reward: {reward}")

print("Training complete!")

# Save
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
