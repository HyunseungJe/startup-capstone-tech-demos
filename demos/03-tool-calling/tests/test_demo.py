import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import configured_model
from world import World
from agent import Agent


def response(*calls, content=None):
    class Message:
        tool_calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=n, arguments=a))
                      for i, (n, a) in enumerate(calls)]

        def model_dump(self, **kwargs):
            result = {"role": "assistant", "content": content}
            if self.tool_calls:
                result["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": vars(c.function)}
                    for c in self.tool_calls]
            return result
    msg = Message()
    msg.content = content
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class DemoTests(unittest.TestCase):
    def test_current_supported_model_is_used_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_model(), "openai/gpt-oss-120b")
    def test_asset_lifecycle_and_independent_instances(self):
        world = World()
        asset = world.execute("search_asset", {"query": "나무"})["assets"][0]
        first = world.execute("place_asset", {"asset_id": asset["id"], "position": [0, 0, 0]})
        second = world.execute("place_asset", {"asset_id": asset["id"], "position": [1, 0, 2]})
        world.execute("move_asset", {"object_id": first["id"], "position": [3, 0, -1]})
        scene = world.execute("inspect_scene", {})
        self.assertEqual(scene["objects"][first["id"]]["position"], [3, 0, -1])
        self.assertEqual(scene["objects"][second["id"]]["position"], [1, 0, 2])
        scene["objects"].clear()
        self.assertEqual(len(world.execute("inspect_scene", {})["objects"]), 2)
        world.execute("delete_asset", {"object_id": first["id"]})
        self.assertEqual(list(world.objects), [second["id"]])
        self.assertIn(asset["id"], world.assets)

    def test_generated_metadata_can_be_placed(self):
        world = World()
        asset = world.execute("generate_asset", {"description": "빛나는 수정"})
        obj = world.execute("place_asset", {"asset_id": asset["id"], "position": [2, 0, 4]})
        self.assertEqual(world.assets[obj["asset_id"]]["description"], "빛나는 수정")
        self.assertEqual(asset["source"], "generated_placeholder")

    def test_invalid_calls_do_not_mutate_world(self):
        world = World()
        before = world.execute("inspect_scene", {})
        for name, args in [
            ("place_asset", {"asset_id": "missing", "position": [0, 0, 0]}),
            ("place_asset", {"asset_id": "tree", "position": [0, 0]}),
            ("place_asset", {"asset_id": "tree", "position": [0, float("inf"), 0]}),
            ("move_asset", {"object_id": "missing", "position": [0, 0, 0]}),
            ("delete_asset", {"object_id": "missing"}),
            ("inspect_scene", {"unexpected": True}), ("__dict__", {}),
            ("generate_asset", {"description": " "}),
        ]:
            with self.subTest(name=name, args=args):
                self.assertIn("error", world.execute(name, args))
                self.assertEqual(world.execute("inspect_scene", {}), before)

    def test_loop_returns_results_and_keeps_conversation(self):
        replies = iter([
            response(("search_asset", '{"query":"tree"}')),
            response(("place_asset", '{"asset_id":"tree","position":[0,0,0]}')),
            response(content="배치했습니다."), response(content="나무가 있습니다."),
        ])
        requests = []
        def create(**kwargs):
            requests.append(json.loads(json.dumps(kwargs)))
            return next(replies)
        world = World()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent = Agent(client, "test-model", world, emit=lambda text: None)
        self.assertEqual(agent.run("나무 배치"), "배치했습니다.")
        self.assertEqual(len(world.objects), 1)
        tool_result = requests[1]["messages"][-1]
        self.assertEqual(tool_result["tool_call_id"], "call_0")
        self.assertEqual(json.loads(tool_result["content"])["assets"][0]["id"], "tree")
        agent.run("뭐가 있어?")
        self.assertEqual(sum(m["role"] == "user" for m in requests[-1]["messages"]), 2)

    def test_malformed_arguments_and_multiple_calls_get_results(self):
        replies = iter([response(("place_asset", "{"), ("inspect_scene", "[]")),
                        response(content="다시 요청해 주세요.")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: next(replies))))
        agent = Agent(client, "test", World(), emit=lambda text: None)
        agent.run("배치")
        results = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], ["call_0", "call_1"])
        self.assertTrue(all("error" in json.loads(m["content"]) for m in results))

    def test_loop_limit_leaves_no_unanswered_tool_calls(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: response(("inspect_scene", "{}")))))
        agent = Agent(client, "test", World(), emit=lambda text: None, max_rounds=2)
        self.assertIn("한도", agent.run("계속 확인"))
        self.assertEqual(sum(m["role"] == "tool" for m in agent.messages), 2)
        self.assertEqual(agent.messages[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
