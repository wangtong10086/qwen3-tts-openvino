#!/usr/bin/env python3
"""Run short/long TTS streaming benchmarks with TTFT and TPS metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SHORT_TEXT = "你好，这是一次短文本实时合成性能测试，用于测量首 token 延迟和稳定生成速度。"
LONG_PARAGRAPH = (
    "今天我们系统性验证长文本语音合成的实时性能。"
    "测试会保持完整上下文自回归，不做文本切段，重点观察不同输出长度下的首 token 延迟、"
    "首个音频块延迟、codec token 生成速度、流式 RTF，以及 decoder 是否出现 fallback。"
    "这种测试比单次短文本 RTF 更能反映桌面应用中连续朗读、长段内容播报和多场景集成时的真实性能。"
)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def built_in_long_text(repeat: int) -> str:
    repeat = max(1, int(repeat))
    return "\n".join(f"{index + 1}. {LONG_PARAGRAPH}" for index in range(repeat))


def load_long_text(path: str | None, repeat: int) -> tuple[str, str | None]:
    if path:
        text_path = Path(path)
        if text_path.exists():
            return text_path.read_text(encoding="utf-8").strip(), str(text_path)
    return built_in_long_text(repeat), None


def run_case(args: argparse.Namespace, case_name: str, text: str, text_file: str | None, token_set: str) -> dict:
    output = Path(args.output_dir) / f"{case_name}.json"
    cmd = [
        sys.executable,
        "scripts/benchmark_streaming_realtime.py",
        "--ir-dir",
        args.ir_dir,
        "--device",
        args.device,
        "--profiles",
        args.profiles,
        "--runs",
        str(args.runs),
        "--case-name",
        case_name,
        "--max-new-tokens-set",
        token_set,
        "--min-new-tokens",
        str(args.min_new_tokens),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--chunk-strategy",
        args.chunk_strategy,
        "--warmup-generations",
        str(args.warmup_generations),
        "--worker-timeout-sec",
        str(args.worker_timeout_sec),
        "--output-json",
        str(output),
    ]
    if args.decoder_device:
        cmd.extend(["--decoder-device", args.decoder_device])
    if args.ov_cache_dir:
        cmd.extend(["--ov-cache-dir", args.ov_cache_dir])
    if text_file:
        cmd.extend(["--text-file", text_file])
    else:
        cmd.extend(["--text", text])
    completed = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        text=True,
        capture_output=True,
        timeout=float(args.worker_timeout_sec) * max(1, len(parse_csv(token_set))) * max(1, int(args.runs)) + 60,
    )
    result = {
        "case_name": case_name,
        "text_chars": len(text),
        "token_set": token_set,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_json": str(output),
    }
    if output.exists():
        result["result"] = json.loads(output.read_text(encoding="utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--profiles", default="fastest")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--cases", default="short,long")
    parser.add_argument("--short-max-new-tokens-set", default="48,128")
    parser.add_argument("--long-max-new-tokens-set", default="256,768")
    parser.add_argument("--long-text-file", default="tmp/test.txt")
    parser.add_argument("--long-repeat", type=int, default=24)
    parser.add_argument("--min-new-tokens", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--chunk-strategy", default="smooth")
    parser.add_argument("--warmup-generations", type=int, default=0)
    parser.add_argument("--worker-timeout-sec", type=float, default=900.0)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/realtime_bench/length_scaling")
    parser.add_argument("--output-json", default="outputs/realtime_bench/length_scaling_summary.json")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cases = parse_csv(args.cases)
    combined = {"cases": [], "runs": [], "summaries": []}
    if "short" in cases:
        case = run_case(args, "short", SHORT_TEXT, None, args.short_max_new_tokens_set)
        combined["cases"].append({k: v for k, v in case.items() if k != "result"})
        result = case.get("result") or {}
        combined["runs"].extend(result.get("runs", []))
        combined["summaries"].extend(result.get("summaries", []))
    if "long" in cases:
        long_text, long_file = load_long_text(args.long_text_file, args.long_repeat)
        case = run_case(args, "long", long_text, long_file, args.long_max_new_tokens_set)
        combined["cases"].append({k: v for k, v in case.items() if k != "result"})
        result = case.get("result") or {}
        combined["runs"].extend(result.get("runs", []))
        combined["summaries"].extend(result.get("summaries", []))

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined["summaries"], ensure_ascii=False, indent=2))
    print(f"wrote {output}", flush=True)
    failed = [case for case in combined["cases"] if case.get("returncode") != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
