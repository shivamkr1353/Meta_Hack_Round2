import json
import unittest

from api_drift_gym import ApiDriftGymEnv


def payload_for(schema):
    values = {
        "int": 1,
        "str": "Ada",
        "bool": True,
        "float": 5.5,
    }
    return {key: values[field_type] for key, field_type in schema.items()}


class ApiDriftGymTests(unittest.TestCase):
    def test_legacy_single_api_flow_still_works(self) -> None:
        env = ApiDriftGymEnv(max_steps=8, seed=2, difficulty="easy")
        env.reset()

        original_payload = payload_for(env.state["current_schema"])
        drifted_payload = payload_for(env.state["drifted_schema"])
        actions = [
            f"call_api:{json.dumps(original_payload)}",
            "inspect_schema",
            f"transform_request:{json.dumps(drifted_payload)}",
            "retry",
        ]

        done = False
        total = 0.0
        info = {}
        for action in actions:
            _, reward, done, info = env.step(action)
            total += reward
            if done:
                break

        self.assertTrue(done)
        self.assertTrue(info["resolved"])
        self.assertEqual(info["reward_components"]["resolution_bonus"], 3.0)
        self.assertGreater(total, 0.0)
        self.assertIn("Difficulty: easy", env.get_log())

    def test_hard_workflow_completes_across_multiple_endpoints(self) -> None:
        env = ApiDriftGymEnv(max_steps=24, seed=5, difficulty="hard")
        env.reset()

        done = False
        info = {}
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
                _, _, done, info = env.step(action)
                if done or env.state["workflow_step"] > stage_index:
                    break

        self.assertTrue(done)
        self.assertTrue(info["resolved"])
        self.assertEqual(env.state["workflow_step"], len(env.state["workflow"]))
        self.assertEqual(info["difficulty"], "hard")
        self.assertIn("fetch_user -> fetch_orders -> process_data -> send_summary", env.get_log())

    def test_hidden_drift_only_surfaces_through_hints(self) -> None:
        env = ApiDriftGymEnv(max_steps=12, seed=3, difficulty="medium")
        reset_observation = env.reset()
        self.assertNotIn("drifted_schema", reset_observation.get("available_hint", {}))

        stage = env.state["workflow"][env.state["workflow_step"]]
        endpoint = stage["endpoint"]
        pre_failure_obs, _, _, _ = env.step(f"inspect_schema:{endpoint}")
        self.assertNotIn("field_types", pre_failure_obs["available_hint"])

        original_payload = payload_for(env.state["api_states"][endpoint]["original_schema"])
        env.step(f"call_api:{endpoint}:{json.dumps(original_payload)}")
        post_failure_obs, _, _, _ = env.step(f"inspect_schema:{endpoint}")
        self.assertIn("field_types", post_failure_obs["available_hint"])
        self.assertEqual(post_failure_obs["available_hint"]["endpoint"], endpoint)

    def test_terminal_failure_uses_negative_two_penalty(self) -> None:
        env = ApiDriftGymEnv(max_steps=2, seed=4, difficulty="easy")
        env.reset()

        _, _, done, _ = env.step("noop")
        self.assertFalse(done)
        _, reward, done, info = env.step("noop")

        self.assertTrue(done)
        self.assertFalse(info["resolved"])
        self.assertEqual(reward, -2.0)
        self.assertEqual(info["reward_components"]["timeout_penalty"], -2.0)

    def test_verify_success_requires_exact_endpoint_schema(self) -> None:
        env = ApiDriftGymEnv(max_steps=8, seed=2, difficulty="easy")
        env.reset()
        endpoint = env.state["workflow"][0]["endpoint"]
        drifted_payload = payload_for(env.state["api_states"][endpoint]["drifted_schema"])
        extra_payload = dict(drifted_payload)
        extra_payload["extra"] = "field"

        self.assertTrue(env.verify_success(drifted_payload, endpoint=endpoint))
        self.assertFalse(env.verify_success(extra_payload, endpoint=endpoint))
        self.assertFalse(env.verify_success(payload_for(env.state["api_states"][endpoint]["original_schema"]), endpoint=endpoint))


if __name__ == "__main__":
    unittest.main()
