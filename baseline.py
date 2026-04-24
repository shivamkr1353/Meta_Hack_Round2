import random
from api_drift_gym.env import ApiDriftGymEnv

env = ApiDriftGymEnv()

actions = [
    "inspect_schema:/user",
    "inspect_schema:/orders",
    "retry",
    "skip_step"
]

success = 0
episodes = 20

for _ in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action = random.choice(actions)
        obs, reward, done, info = env.step(action)

    if env.state.get("resolved", False):
        success += 1

print("Baseline success:", success / episodes)