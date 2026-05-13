from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run_tool(tool: Path, input_path: Path, output_path: Path, args: argparse.Namespace, kv_cache_heads: int) -> dict:
    cmd = [
        str(tool),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--kv-cache-heads",
        str(kv_cache_heads),
        "--kv-cache-block-size",
        str(args.kv_cache_block_size),
        "--kv-cache-head-dim",
        str(args.kv_cache_head_dim),
        "--kv-cache-precision",
        str(args.kv_cache_precision),
        "--kv-cache-input-precision",
        str(args.kv_cache_input_precision),
    ]
    if args.no_score_aggregation:
        cmd.append("--no-score-aggregation")
    if args.compile_device:
        cmd.extend(["--compile-device", args.compile_device])
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "paged-kv tool returned non-JSON output "
            f"(returncode={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr!r})"
        ) from exc
    summary["returncode"] = proc.returncode
    summary["stderr_tail"] = proc.stderr[-4000:]
    return summary


def compact_summary(key: str, graph_name: str, summary: dict) -> dict:
    conversion = summary.get("attention_conversion") or {}
    scoped_after = summary.get("scoped_op_counts_after") or {}
    talker_after = scoped_after.get("talker_attention") or {}
    return {
        "key": key,
        "graph": graph_name,
        "is_unroll": "unroll" in key,
        "returncode": summary.get("returncode"),
        "ok": bool(summary.get("ok")),
        "paged_kv_ready": bool(summary.get("paged_kv_ready")),
        "validation_warning": summary.get("validation_warning"),
        "talker_sdpa_before": int(conversion.get("talker_sdpa_before") or 0),
        "talker_sdpa_after": int(conversion.get("talker_sdpa_after") or 0),
        "talker_paged_after": int(conversion.get("talker_paged_after") or 0),
        "talker_conversion_complete": bool(conversion.get("talker_conversion_complete")),
        "remaining_talker_attention_ops": {
            "ScaledDotProductAttention": int(talker_after.get("ScaledDotProductAttention") or 0),
            "PagedAttentionExtension": int(talker_after.get("PagedAttentionExtension") or 0),
            "MatMul": int(talker_after.get("MatMul") or 0),
        },
        "compile_ok": summary.get("compile_ok"),
        "compile_error": summary.get("compile_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OpenVINO SDPAToPagedAttention coverage for Qwen3-TTS seed graphs.")
    parser.add_argument("--ir-dir", required=True, type=Path)
    parser.add_argument("--tool", type=Path, default=Path("native/build/qwen3_tts_ov_paged_kv_tool"))
    parser.add_argument("--kv-cache-heads", type=int, default=16)
    parser.add_argument("--kv-cache-gqa-heads", type=int, default=8)
    parser.add_argument("--kv-cache-block-size", type=int, default=8)
    parser.add_argument("--kv-cache-head-dim", type=int, default=128)
    parser.add_argument("--kv-cache-precision", choices=["f16", "bf16", "u8"], default="f16")
    parser.add_argument("--kv-cache-input-precision", choices=["f32", "f16", "bf16", "u8"], default="f32")
    parser.add_argument("--no-score-aggregation", action="store_true")
    parser.add_argument("--compile-device", default=None)
    parser.add_argument("--keys", default="", help="Comma-separated manifest graphs.paged_kv_seed keys. Empty means all.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--keep-converted", action="store_true", help="Keep temporary converted XML/BIN files under --work-dir.")
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not args.tool.exists():
        raise FileNotFoundError(f"paged-kv tool not found: {args.tool}; run `uv run python scripts/build_paged_kv_tool.py`")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = dict((manifest.get("graphs", {}) or {}).get("paged_kv_seed", {}) or {})
    if not seeds:
        raise RuntimeError("manifest has no graphs.paged_kv_seed; export with --export-paged-kv-seed first")
    selected_keys = [item.strip() for item in args.keys.split(",") if item.strip()]
    if selected_keys:
        missing = [key for key in selected_keys if key not in seeds]
        if missing:
            raise KeyError(f"unknown paged_kv_seed keys: {', '.join(missing)}")
        seeds = {key: seeds[key] for key in selected_keys}

    temp_context = None
    if args.work_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="qwen3_tts_paged_audit_")
        work_dir = Path(temp_context.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw: dict[str, dict] = {}
        compact = []
        for key, graph_name in seeds.items():
            input_path = args.ir_dir / graph_name
            output_path = work_dir / f"{Path(graph_name).stem}.paged_audit.xml"
            kv_cache_heads = args.kv_cache_gqa_heads if key.endswith("_gqa") or "_gqa_" in key else args.kv_cache_heads
            summary = run_tool(args.tool, input_path, output_path, args, kv_cache_heads)
            raw[key] = summary
            compact.append(compact_summary(key, graph_name, summary))
            if not args.keep_converted:
                output_path.unlink(missing_ok=True)
                output_path.with_suffix(".bin").unlink(missing_ok=True)

        failed = [item for item in compact if not item["paged_kv_ready"]]
        incomplete = [item for item in compact if item["talker_sdpa_after"] > 0]
        result = {
            "ir_dir": str(args.ir_dir),
            "kv_cache_precision": args.kv_cache_precision,
            "kv_cache_input_precision": args.kv_cache_input_precision,
            "total_graphs": len(compact),
            "ready_graphs": len(compact) - len(failed),
            "complete_talker_attention_graphs": len(compact) - len(incomplete),
            "incomplete_talker_attention_graphs": [item["key"] for item in incomplete],
            "graphs": compact,
            "raw": raw,
        }
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
