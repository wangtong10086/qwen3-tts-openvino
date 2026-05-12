import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OV_TELEMETRY_DISABLE", "1")

import nncf
import openvino as ov

from qwen3_tts_ov.manifest import resolve_ir_dir


def add_suffix(path: str, suffix: str) -> str:
    item = Path(path)
    return f"{item.stem}{suffix}{item.suffix}"


def compress_model(source: Path, target: Path, mode, ignored_scope, force: bool) -> None:
    if target.exists() and target.with_suffix(".bin").exists() and not force:
        print(f"exists {target}; skipping", flush=True)
        return

    started = time.time()
    print(f"compressing {source.name} -> {target.name}", flush=True)
    core = ov.Core()
    model = core.read_model(source)
    compressed = nncf.compress_weights(
        model,
        mode=mode,
        ignored_scope=ignored_scope,
    )
    ov.save_model(compressed, target, compress_to_fp16=False)
    print(f"saved {target} in {time.time() - started:.1f}s", flush=True)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def merge_nested_graphs(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_nested_graphs(target[key], value)
        else:
            target[key] = value


def update_variant(manifest: dict, variant: str, variant_graphs: dict, mode_name: str) -> None:
    variants = manifest.setdefault("graph_variants", {})
    entry = variants.setdefault(variant, {"precision": f"{mode_name}_weights", "graphs": {}})
    entry["precision"] = f"{mode_name}_weights"
    merge_nested_graphs(entry.setdefault("graphs", {}), variant_graphs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="auto")
    parser.add_argument("--variant", default="int8_fused")
    parser.add_argument(
        "--source-variant",
        default=None,
        help="Read source graph paths from an existing manifest graph variant, for example fp16_fused_rms.",
    )
    parser.add_argument("--mode", default="int8_asym", choices=["int8_asym", "int8_sym"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-no-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-subcode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-cached-subcode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-sdpa-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-fused-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-fused-unroll", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-fused-decode-unroll", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-cache-kernels", default="exact")
    parser.add_argument("--fused-cache-unroll-steps", default="4,6,8,12")
    parser.add_argument(
        "--compress-gather",
        action="store_true",
        help="Also compress Gather-backed embedding weights. Defaults to off to reduce TTS quality risk.",
    )
    args = parser.parse_args()

    ir_dir = resolve_ir_dir(args.ir_dir, fallback_to_local_voice_design=True, warn=True)
    manifest_path = ir_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mode = {
        "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
        "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    }[args.mode]
    ignored_scope = None if args.compress_gather else nncf.IgnoredScope(types=["Gather"])

    graphs = copy.deepcopy(manifest["graphs"])
    if args.source_variant:
        variants = manifest.get("graph_variants", {})
        if args.source_variant not in variants:
            available = ", ".join(sorted(variants)) or "none"
            raise ValueError(f"source graph variant {args.source_variant!r} not found in manifest; available variants: {available}")
        merge_nested_graphs(graphs, copy.deepcopy(variants[args.source_variant].get("graphs", {})))
    variant_graphs = {}
    selected_jobs = 0

    if args.include_no_cache:
        selected_jobs += 1
        source = graphs["talker"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["talker"] = target

    if args.include_subcode:
        selected_jobs += 1
        source = graphs["subcode_greedy"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["subcode_greedy"] = target

    if args.include_cached_subcode:
        selected_jobs += 1
        source = graphs["subcode_greedy_cached"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["subcode_greedy_cached"] = target

    if args.include_sdpa_cache:
        selected_jobs += 1
        sdpa_buckets = graphs.get("talker_stateful_buckets", {}).get("sdpa", {})
        if not sdpa_buckets:
            raise ValueError("manifest has no graphs.talker_stateful_buckets.sdpa section")
        compressed_buckets = {}
        for bucket, source in sorted(sdpa_buckets.items(), key=lambda item: int(item[0])):
            target = add_suffix(source, f"_{args.variant}")
            compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
            compressed_buckets[str(bucket)] = target
        variant_graphs["talker_stateful_buckets"] = {"sdpa": compressed_buckets}

    if args.include_fused_cache:
        selected_jobs += 1
        fused_section = graphs.get("fused_cache_step_buckets", {})
        compressed_by_kernel = {}
        kernels = parse_csv(args.fused_cache_kernels)
        if not kernels:
            raise ValueError("--fused-cache-kernels must list at least one kernel when --include-fused-cache is enabled")
        for kernel in kernels:
            buckets = fused_section.get(kernel, {})
            if not buckets:
                raise ValueError(f"manifest has no graphs.fused_cache_step_buckets.{kernel} section")
            compressed_buckets = {}
            for bucket, source in sorted(buckets.items(), key=lambda item: int(item[0])):
                target = add_suffix(source, f"_{args.variant}")
                compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
                compressed_buckets[str(bucket)] = target
            compressed_by_kernel[kernel] = compressed_buckets
        variant_graphs["fused_cache_step_buckets"] = compressed_by_kernel

    def compress_unroll_section(section_name: str, warning_name: str):
        unroll_section = graphs.get(section_name, {})
        if unroll_section:
            compressed_by_kernel = {}
            kernels = parse_csv(args.fused_cache_kernels)
            unroll_steps = parse_csv(args.fused_cache_unroll_steps)
            for kernel in kernels:
                by_kernel = unroll_section.get(kernel, {})
                if not by_kernel:
                    continue
                compressed_by_step = {}
                for unroll in unroll_steps:
                    buckets = by_kernel.get(str(unroll), {})
                    if not buckets:
                        continue
                    compressed_buckets = {}
                    for bucket, source in sorted(buckets.items(), key=lambda item: int(item[0])):
                        target = add_suffix(source, f"_{args.variant}")
                        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
                        compressed_buckets[str(bucket)] = target
                    compressed_by_step[str(unroll)] = compressed_buckets
                if compressed_by_step:
                    compressed_by_kernel[kernel] = compressed_by_step
            if compressed_by_kernel:
                variant_graphs[section_name] = compressed_by_kernel
                return True
            else:
                print(f"warning: no matching {warning_name} graphs found; skipping compression", flush=True)
                return False
        print(f"warning: manifest has no graphs.{section_name}; skipping {warning_name} compression", flush=True)
        return False

    if args.include_fused_unroll:
        selected_jobs += 1
        if not parse_csv(args.fused_cache_kernels):
            raise ValueError("--fused-cache-kernels must list at least one kernel when --include-fused-unroll is enabled")
        if not parse_csv(args.fused_cache_unroll_steps):
            raise ValueError("--fused-cache-unroll-steps must list at least one step when --include-fused-unroll is enabled")
        compress_unroll_section("fused_cache_step_unroll_buckets", "fused cache unroll")
        compress_unroll_section("fused_cache_step_unroll_norepeat_buckets", "fused cache no-repeat unroll")

    if args.include_fused_decode_unroll:
        selected_jobs += 1
        if not parse_csv(args.fused_cache_kernels):
            raise ValueError("--fused-cache-kernels must list at least one kernel when --include-fused-decode-unroll is enabled")
        if not parse_csv(args.fused_cache_unroll_steps):
            raise ValueError("--fused-cache-unroll-steps must list at least one step when --include-fused-decode-unroll is enabled")
        compress_unroll_section("fused_cache_decode_unroll_buckets", "fused cache decode unroll")
        compress_unroll_section("fused_cache_decode_unroll_stateful_mask_buckets", "fused cache decode unroll stateful mask")
        compress_unroll_section("fused_cache_decode_unroll_norepeat_buckets", "fused cache decode unroll no-repeat")

    if selected_jobs == 0:
        raise ValueError("no graph groups selected for compression")

    update_variant(manifest, args.variant, variant_graphs, args.mode)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
