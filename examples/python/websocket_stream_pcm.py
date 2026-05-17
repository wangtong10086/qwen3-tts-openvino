from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path
from urllib.parse import urlparse


def websocket_url(server: str) -> str:
    parsed = urlparse(server)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc.rstrip('/')}/v1/tts/stream"


async def run(server: str, request_path: str, output_path: str) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit("Install the optional dependency first: uv run --with websockets ...") from exc

    with open(request_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pcm_chunks: list[bytes] = []
    sample_rate = 24000
    url = websocket_url(server)
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(payload, ensure_ascii=False))
        async for message in ws:
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "metadata":
                sample_rate = int(event.get("sample_rate", sample_rate))
                print(f"metadata: sample_rate={sample_rate}, strategy={event.get('strategy')}")
            elif event_type == "final":
                print(f"final: elapsed={event.get('elapsed')}")
                break
            elif event_type == "error":
                raise RuntimeError(str(event.get("message")))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(pcm_chunks))
    print(f"wrote {output} ({sum(len(chunk) for chunk in pcm_chunks)} pcm bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /v1/tts/stream over WebSocket and save a WAV file.")
    parser.add_argument("--server", default="http://127.0.0.1:17860")
    parser.add_argument("--request", default="examples/stream_request.example.json")
    parser.add_argument("--output", default="outputs/example_ws.wav")
    args = parser.parse_args()
    asyncio.run(run(args.server, args.request, args.output))


if __name__ == "__main__":
    main()
