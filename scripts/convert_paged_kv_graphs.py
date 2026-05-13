from __future__ import annotations

import argparse
import json
import subprocess
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
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stderr:
        print(proc.stderr, end="")
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "paged-kv tool returned non-JSON output "
            f"(returncode={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr!r})"
        ) from exc
    summary["returncode"] = proc.returncode
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS SDPA seed graphs to OpenVINO paged KV graphs.")
    parser.add_argument("--ir-dir", required=True, type=Path)
    parser.add_argument("--tool", type=Path, default=Path("native/build/qwen3_tts_ov_paged_kv_tool"))
    parser.add_argument("--kv-cache-heads", type=int, default=16)
    parser.add_argument("--kv-cache-gqa-heads", type=int, default=8)
    parser.add_argument("--kv-cache-block-size", type=int, default=8)
    parser.add_argument("--kv-cache-head-dim", type=int, default=128)
    parser.add_argument("--kv-cache-precision", choices=["f16", "bf16", "u8"], default="f16")
    parser.add_argument("--kv-cache-input-precision", choices=["f32", "f16", "bf16", "u8"], default="f32")
    parser.add_argument("--no-score-aggregation", action="store_true")
    parser.add_argument(
        "--include-unroll",
        action="store_true",
        help=(
            "Also convert fused_cache_step_unroll* seeds. These are experimental: "
            "current OpenVINO SDPAToPagedAttention only converts the first internal step, "
            "so the resulting graph is not a default correctness path."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ir_dir = args.ir_dir
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not args.tool.exists():
        raise FileNotFoundError(f"paged-kv tool not found: {args.tool}; run `uv run python scripts/build_paged_kv_tool.py`")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = manifest.get("graphs", {}).get("paged_kv_seed", {})
    if not seeds:
        raise RuntimeError("manifest has no graphs.paged_kv_seed; export with --export-paged-kv-seed first")

    converted: dict[str, str] = {}
    summaries: dict[str, dict] = {}
    for key, graph_name in seeds.items():
        is_unroll = "unroll" in key
        if is_unroll and not args.include_unroll:
            summaries[key] = {
                "skipped": True,
                "reason": "unroll paged-KV seeds are incomplete with current SDPAToPagedAttention conversion",
            }
            continue
        input_path = ir_dir / graph_name
        graph_path = Path(graph_name)
        output_stem = graph_path.stem[:-5] if graph_path.stem.endswith("_seed") else f"{graph_path.stem}_converted"
        output_name = f"{output_stem}{graph_path.suffix}"
        output_path = ir_dir / output_name
        if input_path.resolve() == output_path.resolve():
            raise RuntimeError(f"refusing to overwrite paged-KV seed graph in-place: {input_path}")
        if output_path.exists() and output_path.with_suffix(".bin").exists() and not args.force:
            converted[key] = output_name
            continue
        kv_cache_heads = args.kv_cache_gqa_heads if key.endswith("_gqa") or "_gqa_" in key else args.kv_cache_heads
        summary = run_tool(args.tool, input_path, output_path, args, kv_cache_heads)
        summaries[key] = summary
        remaining_sdpa = int((summary.get("op_counts_after") or {}).get("ScaledDotProductAttention") or 0)
        converted_paged = int((summary.get("op_counts_after") or {}).get("PagedAttentionExtension") or 0)
        if summary.get("returncode") != 0:
            if is_unroll:
                summaries[key]["skipped"] = True
                summaries[key]["reason"] = "conversion tool reported an incomplete unroll graph"
                continue
            raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
        if is_unroll and remaining_sdpa > converted_paged:
            summaries[key]["skipped"] = True
            summaries[key]["reason"] = (
                f"incomplete unroll conversion: {converted_paged} PagedAttention ops, "
                f"{remaining_sdpa} ScaledDotProductAttention ops remain"
            )
            continue
        converted[key] = output_name

    manifest.setdefault("graphs", {})["paged_kv"] = converted
    manifest["paged_kv"] = {
        "enabled": bool(converted),
        "cache_layout": "gpu_f16_paged_attention",
        "kv_cache_heads": args.kv_cache_heads,
        "kv_cache_gqa_heads": args.kv_cache_gqa_heads,
        "kv_cache_block_size": args.kv_cache_block_size,
        "kv_cache_head_dim": args.kv_cache_head_dim,
        "kv_cache_precision": args.kv_cache_precision,
        "kv_cache_input_precision": args.kv_cache_input_precision,
        "note": (
            "Converted graphs contain PagedAttentionExtension. Runtime support is experimental."
            if converted
            else "No paged-KV graphs were converted; all available seeds were skipped."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"converted": converted, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
