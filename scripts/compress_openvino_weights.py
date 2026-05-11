import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OV_TELEMETRY_DISABLE", "1")

import nncf
import openvino as ov


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


def update_variant(manifest: dict, variant: str, variant_graphs: dict, mode_name: str) -> None:
    variants = manifest.setdefault("graph_variants", {})
    variants[variant] = {
        "precision": f"{mode_name}_weights",
        "graphs": variant_graphs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--variant", default="int8")
    parser.add_argument("--mode", default="int8_asym", choices=["int8_asym", "int8_sym"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-no-cache", action="store_true", default=True)
    parser.add_argument("--include-subcode", action="store_true", default=True)
    parser.add_argument("--include-cached-subcode", action="store_true")
    parser.add_argument("--include-sdpa-cache", action="store_true", default=True)
    parser.add_argument(
        "--compress-gather",
        action="store_true",
        help="Also compress Gather-backed embedding weights. Defaults to off to reduce TTS quality risk.",
    )
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    manifest_path = ir_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mode = {
        "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
        "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    }[args.mode]
    ignored_scope = None if args.compress_gather else nncf.IgnoredScope(types=["Gather"])

    graphs = manifest["graphs"]
    variant_graphs = {}

    if args.include_no_cache:
        source = graphs["talker"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["talker"] = target

    if args.include_subcode:
        source = graphs["subcode_greedy"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["subcode_greedy"] = target

    if args.include_cached_subcode:
        source = graphs["subcode_greedy_cached"]
        target = add_suffix(source, f"_{args.variant}")
        compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
        variant_graphs["subcode_greedy_cached"] = target

    if args.include_sdpa_cache:
        sdpa_buckets = graphs.get("talker_stateful_buckets", {}).get("sdpa", {})
        if not sdpa_buckets:
            raise ValueError("manifest has no graphs.talker_stateful_buckets.sdpa section")
        compressed_buckets = {}
        for bucket, source in sorted(sdpa_buckets.items(), key=lambda item: int(item[0])):
            target = add_suffix(source, f"_{args.variant}")
            compress_model(ir_dir / source, ir_dir / target, mode, ignored_scope, args.force)
            compressed_buckets[str(bucket)] = target
        variant_graphs["talker_stateful_buckets"] = {"sdpa": compressed_buckets}

    update_variant(manifest, args.variant, variant_graphs, args.mode)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
