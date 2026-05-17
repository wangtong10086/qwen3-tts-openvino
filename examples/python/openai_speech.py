from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def post_json(url: str, payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /v1/audio/speech and save WAV or PCM output.")
    parser.add_argument("--server", default="http://127.0.0.1:17860")
    parser.add_argument("--request", default="examples/openai_speech_request.example.json")
    parser.add_argument("--output", default="outputs/example_openai.pcm")
    args = parser.parse_args()

    with open(args.request, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    audio = post_json(args.server.rstrip("/") + "/v1/audio/speech", payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio)
    print(f"wrote {output} ({len(audio)} bytes)")


if __name__ == "__main__":
    main()
