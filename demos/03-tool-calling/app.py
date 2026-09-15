"""Run with: uv run python app.py"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI

from agent import Agent
from world import World


DEFAULT_MODEL = "openai/gpt-oss-120b"


def configured_model():
    return os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Groq tool-calling world demo")
    parser.add_argument("--prompt", help="Run one natural-language request and print the scene")
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key":
        parser.exit(1, "GROQ_API_KEY를 설정하세요. .env.example을 .env로 복사해서 키를 입력하면 됩니다.\n")
    model = configured_model()
    world = World()
    with OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=1) as client:
        agent = Agent(client, model, world)

        def run(text):
            try:
                print(f"\nassistant> {agent.run(text)}")
                return True
            except APIError as exc:
                # Avoid echoing server payloads or credentials into the terminal.
                status = getattr(exc, "status_code", None)
                print(f"\nAPI 요청 실패: {type(exc).__name__} (status={status}). "
                      f"model={model}. 키, 모델 권한, 사용 한도, 네트워크를 확인하세요. 기존 월드 변경은 유지됩니다.")
                return False

        def scene():
            print(json.dumps(world.execute("inspect_scene", {}), ensure_ascii=False, indent=2))

        if args.prompt:
            ok = run(args.prompt)
            scene()
            return 0 if ok else 1
        print(f"World assistant | {model}\n/scene: 월드 JSON  /quit: 종료\n상태는 종료 시 초기화됩니다.")
        while True:
            try:
                text = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text == "/quit":
                break
            if text == "/scene":
                scene()
            elif text:
                run(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
