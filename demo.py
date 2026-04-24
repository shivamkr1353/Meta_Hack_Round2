import json

from api_drift_gym import ApiDriftGymEnv


def payload_for(schema):
    values = {
        "int": 1,
        "str": "Ada Lovelace",
        "bool": True,
        "float": 9.5,
    }
    return {key: values[field_type] for key, field_type in schema.items()}


def main() -> None:
    env = ApiDriftGymEnv(max_steps=20, seed=11, difficulty="hard")
    observation = env.reset()
    print("Initial observation:")
    print(observation)

    done = False
    while not done:
        stage_index = env.state["workflow_step"]
        stage = env.state["workflow"][env.state["workflow_step"]]
        endpoint = stage["endpoint"]
        original_payload = payload_for(env.state["api_states"][endpoint]["original_schema"])
        drifted_payload = payload_for(env.state["api_states"][endpoint]["drifted_schema"])
        actions = [
            f"call_api:{endpoint}:{json.dumps(original_payload)}",
            f"inspect_schema:{endpoint}",
            f"transform_request:{endpoint}:{json.dumps(drifted_payload)}",
            "retry",
        ]

        for action in actions:
            observation, reward, done, info = env.step(action)
            print(f"\naction={action}")
            print(f"reward={reward:+.2f}, done={done}")
            print(f"observation={observation}")
            print(f"components={info['reward_components']}")
            if done or env.state["workflow_step"] > stage_index:
                break

    print("\nTrajectory log:")
    print(env.get_log())


if __name__ == "__main__":
    main()
