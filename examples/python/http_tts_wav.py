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
    parser = argparse.ArgumentParser(description="Call /v1/tts and save a WAV file.")
    parser.add_argument("--server", default="http://127.0.0.1:17860")
    parser.add_argument("--text", default="你好，这是一个 HTTP WAV 示例。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    parser.add_argument("--output", default="outputs/example_http.wav")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    payload = {
        "mode": "voice_design",
        "text": args.text,
        "language": args.language,
        "instruct": args.instruct,
        "generation": {"max_new_tokens": args.max_new_tokens},
    }
    audio = post_json(args.server.rstrip("/") + "/v1/tts", payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(audio)
    print(f"wrote {output} ({len(audio)} bytes)")


if __name__ == "__main__":
    main()
