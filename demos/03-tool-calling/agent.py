"""OpenAI-compatible chat -> local tools -> chat, with a bounded loop."""
import json

from world import TOOLS

SYSTEM = """You assist with a small 3D game world represented as data.
Use tools for all scene queries and edits; never claim an edit without a successful tool result.
Assets are reusable catalog entries; objects are placed instances with separate IDs.
Search using a short keyword first. If no suitable asset exists, generate placeholder metadata.
Use actual IDs returned by tools, never invented IDs. Inspect the scene when IDs or positions are unknown.
Coordinates are absolute [x,y,z], y up, +x right, +z forward. For relative moves, inspect first and calculate.
If a position is unspecified, choose a reasonable one and report it. Ask if the target is ambiguous.
Execute dependent operations in separate rounds, using previous tool results.
Generated assets are metadata only, not real 3D models. Report errors honestly.
Reply concisely in the user's language, with object IDs and positions for edits.
"""


class Agent:
    def __init__(self, client, model, world, emit=print, max_rounds=10):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.client = client
        self.model = model
        self.world = world
        self.emit = emit
        self.max_rounds = max_rounds
        self.messages = [{"role": "system", "content": SYSTEM}]

    def run(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_rounds):
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=TOOLS, tool_choice="auto",
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content or "모델이 텍스트 없이 응답했습니다."
            # Execute every call in order and pair each result with its original ID.
            for call in message.tool_calls:
                name, raw = call.function.name, call.function.arguments
                self.emit(f"\n[tool] {name}({raw})")
                try:
                    arguments = json.loads(raw)
                    result = self.world.execute(name, arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    result = {"error": f"Invalid tool arguments: {exc}"}
                content = json.dumps(result, ensure_ascii=False, allow_nan=False)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                self.emit(f"[result] {content}")
        text = "툴 호출 반복 한도에 도달했습니다. 일부 작업만 반영됐을 수 있습니다. /scene으로 확인하세요."
        self.messages.append({"role": "assistant", "content": text})
        return text
