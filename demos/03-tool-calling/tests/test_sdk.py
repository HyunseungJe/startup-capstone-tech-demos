"""Exercise actual OpenAI SDK serialization without a network/API key."""
import json
import unittest

import httpx
from openai import OpenAI

from agent import Agent
from world import World


class SDKTests(unittest.TestCase):
    def test_sdk_round_trip_changes_world_and_sends_matching_result(self):
        payloads = []

        def handle(request):
            body = json.loads(request.content)
            payloads.append(body)
            if len(payloads) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "place_123", "type": "function", "function": {
                        "name": "place_asset", "arguments": '{"asset_id":"rock","position":[1,0,2]}'}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "바위를 배치했습니다."}
                finish = "stop"
            return httpx.Response(200, json={"id": "completion_1", "object": "chat.completion",
                "created": 1, "model": "test-model", "choices": [
                    {"index": 0, "message": message, "finish_reason": finish}]})

        world = World()
        with OpenAI(api_key="offline-test", base_url="https://example.invalid/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            agent = Agent(client, "test-model", world, emit=lambda text: None)
            self.assertEqual(agent.run("바위 배치"), "바위를 배치했습니다.")
        self.assertEqual(world.objects["object_1"]["position"], [1, 0, 2])
        self.assertEqual(payloads[1]["messages"][-1]["tool_call_id"], "place_123")
        self.assertEqual(json.loads(payloads[1]["messages"][-1]["content"])["id"], "object_1")
        self.assertEqual(len(payloads[0]["tools"]), 6)
