import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import openvino as ov

from .cache import merge_compile_config_with_cache_mode, normalize_ov_cache_mode, resolve_ov_cache_dir
from .manifest import load_manifest, resolve_ir_dir
from .runtime import compile_model


@dataclass(frozen=True)
class WarmupTask:
    label: str
    graph: str
    device_role: str = "runtime"


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def effective_runtime_options(mode: str, cache_kernel: str, cache_step: str, graph_variant: str) -> tuple[str, str, str, str]:
    if mode == "fast-cache":
        return "cache", "sdpa", "split", "int8_cachedsub"
    return mode, cache_kernel, cache_step, graph_variant


def graph_name(graphs: dict, variant_graphs: dict, name: str) -> str | None:
    if name in variant_graphs:
        return variant_graphs[name]
    value = graphs.get(name)
    return value if isinstance(value, str) else None


def load_graph_variant(manifest: dict, graph_variant: str) -> dict:
    if graph_variant in {"", "fp16", None}:
        return {}
    variants = manifest.get("graph_variants", {})
    if graph_variant not in variants:
        available = ", ".join(sorted(variants)) or "none"
        raise ValueError(f"graph variant {graph_variant!r} not found in manifest; available variants: {available}")
    return variants[graph_variant].get("graphs", {})


def load_bucket_graphs(bucket_section: dict, kernel: str) -> dict[int, str]:
    if not bucket_section:
        return {}
    if kernel in bucket_section and isinstance(bucket_section[kernel], dict):
        return {int(length): graph for length, graph in bucket_section[kernel].items() if graph}
    if all(str(key).isdigit() for key in bucket_section):
        return {int(length): graph for length, graph in bucket_section.items() if graph}
    return {}


def merged_bucket_graphs(graphs: dict, variant_graphs: dict, section: str, kernel: str) -> dict[int, str]:
    buckets = load_bucket_graphs(graphs.get(section, {}), kernel)
    buckets.update(load_bucket_graphs(variant_graphs.get(section, {}), kernel))
    return dict(sorted(buckets.items()))


def select_buckets(available: dict[int, str], preload_buckets: str) -> dict[int, str]:
    mode = str(preload_buckets or "warmup").strip().lower()
    if mode == "all":
        return available
    if mode in {"", "none", "off", "false", "0"}:
        return {}
    if mode in {"warmup", "auto", "required", "first"}:
        return {min(available): available[min(available)]} if available else {}
    requested = [int(item) for item in parse_csv(mode)]
    return {bucket: available[bucket] for bucket in requested if bucket in available}


def streaming_contexts(manifest: dict, graphs: dict) -> dict[int, dict[int, str]]:
    stream_config = manifest.get("streaming_decoder", {})
    contexts = stream_config.get("contexts")
    if contexts:
        return {
            int(context): {int(chunk): graph for chunk, graph in chunk_graphs.items() if graph}
            for context, chunk_graphs in contexts.items()
            if chunk_graphs
        }
    stream_graphs = stream_config.get("graphs") or graphs.get("streaming_decoder", {})
    if not stream_graphs:
        return {}
    if all(str(key).isdigit() and isinstance(value, dict) for key, value in stream_graphs.items()):
        return {
            int(context): {int(chunk): graph for chunk, graph in chunk_graphs.items() if graph}
            for context, chunk_graphs in stream_graphs.items()
            if chunk_graphs
        }
    left_context = int(stream_config.get("left_context_frames", 25))
    return {left_context: {int(chunk): graph for chunk, graph in stream_graphs.items() if graph}}


def select_stream_graph(chunk_graphs: dict[int, str], frames: int) -> tuple[int, str] | None:
    if frames in chunk_graphs:
        return frames, chunk_graphs[frames]
    larger = [chunk for chunk in sorted(chunk_graphs) if chunk >= frames]
    if larger:
        return larger[0], chunk_graphs[larger[0]]
    if chunk_graphs:
        chunk = max(chunk_graphs)
        return chunk, chunk_graphs[chunk]
    return None


def collect_warmup_tasks(
    ir_dir: str | Path,
    *,
    graphs: str = "core,stream,buckets",
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    preload_buckets: str = "warmup",
    stream_decoders: str = "strategy",
    warmup_strategy: str = "low_latency",
) -> tuple[list[WarmupTask], dict]:
    ir_dir = resolve_ir_dir(ir_dir, fallback_to_local_voice_design=True, warn=True)
    manifest = load_manifest(ir_dir)
    effective_mode, effective_kernel, effective_step, effective_variant = effective_runtime_options(
        mode,
        cache_kernel,
        cache_step,
        graph_variant,
    )
    graph_sections = set(parse_csv(graphs) or ["core", "stream", "buckets"])
    manifest_graphs = manifest["graphs"]
    variant_graphs = load_graph_variant(manifest, effective_variant)
    tasks: list[WarmupTask] = []

    def add(label: str, graph: str | None, device_role: str = "runtime"):
        if graph:
            tasks.append(WarmupTask(label=label, graph=graph, device_role=device_role))

    if "all" in graph_sections:
        graph_sections.update({"core", "stream", "buckets", "decoder"})

    if "core" in graph_sections:
        add("core:text_embedding", graph_name(manifest_graphs, variant_graphs, "text_embedding"))
        add("core:codec_embedding", graph_name(manifest_graphs, variant_graphs, "codec_embedding"))
        add("core:code_frame_embedding", graph_name(manifest_graphs, variant_graphs, "code_frame_embedding"))
        if effective_mode == "no-cache":
            add("core:talker", graph_name(manifest_graphs, variant_graphs, "talker"))
            add("core:subcode_greedy", graph_name(manifest_graphs, variant_graphs, "subcode_greedy"))
        elif effective_mode == "fused-no-cache":
            add("core:fused_no_cache_step", graph_name(manifest_graphs, variant_graphs, "fused_no_cache_step"))
        elif effective_mode == "cache" and effective_step == "split":
            add("core:subcode_greedy", graph_name(manifest_graphs, variant_graphs, "subcode_greedy"))
        add("core:speech_encoder", manifest_graphs.get("speech_encoder"), "decoder")
        add("core:speaker_encoder", manifest_graphs.get("speaker_encoder"))

    if "buckets" in graph_sections and effective_mode == "cache":
        section = "fused_cache_step_buckets" if effective_step == "fused" else "talker_stateful_buckets"
        for bucket, graph in select_buckets(
            merged_bucket_graphs(manifest_graphs, variant_graphs, section, effective_kernel),
            preload_buckets,
        ).items():
            add(f"bucket:{section}:{bucket}", graph)

    if "stream" in graph_sections:
        contexts = streaming_contexts(manifest, manifest_graphs)
        if stream_decoders == "all":
            for context, chunk_graphs in sorted(contexts.items()):
                for chunk, graph in sorted(chunk_graphs.items()):
                    add(f"stream:c{context}_t{chunk}", graph, "decoder")
        else:
            stream_config = manifest.get("streaming_decoder", {})
            strategies = stream_config.get("strategies", {})
            strategy = strategies.get(warmup_strategy) or strategies.get(warmup_strategy.replace("-", "_")) or {}
            initial_frames = int(strategy.get("initial_chunk_frames", 8))
            chunk_frames = int(strategy.get("chunk_frames", 12))
            left_context = int(strategy.get("left_context_frames", stream_config.get("left_context_frames", 25)))
            first = select_stream_graph(contexts.get(0, {}), initial_frames)
            if first:
                add(f"stream:c0_t{first[0]}", first[1], "decoder")
            steady = select_stream_graph(contexts.get(left_context, {}), chunk_frames)
            if steady:
                add(f"stream:c{left_context}_t{steady[0]}", steady[1], "decoder")

    if "decoder" in graph_sections:
        for tokens, graph in sorted((manifest_graphs.get("speech_decoder") or {}).items(), key=lambda item: int(item[0])):
            add(f"decoder:t{tokens}", graph, "decoder")

    deduped = []
    seen = set()
    for task in tasks:
        key = (task.graph, task.device_role)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped, manifest


def compile_warmup_task(
    ir_dir: str | Path,
    manifest: dict,
    task: WarmupTask,
    *,
    device: str,
    decoder_device: str | None,
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
    precision_hint: str,
    compile_config: dict,
    ov_cache_dir: str | Path | None,
    ov_cache_mode: str | None,
    disable_ov_cache: bool,
    allow_cpu_fallback: bool,
) -> dict:
    ir_dir = Path(ir_dir)
    effective_mode, effective_kernel, effective_step, effective_variant = effective_runtime_options(
        mode,
        cache_kernel,
        cache_step,
        graph_variant,
    )
    effective_compile_config = merge_compile_config_with_cache_mode(
        compile_config,
        ov_cache_mode=ov_cache_mode,
        disable_ov_cache=disable_ov_cache,
    )
    cache_dir = resolve_ov_cache_dir(
        ir_dir,
        manifest,
        device=device,
        decoder_device=decoder_device,
        mode=effective_mode,
        cache_kernel=effective_kernel,
        cache_step=effective_step,
        graph_variant=effective_variant,
        precision_hint=precision_hint,
        compile_config=effective_compile_config,
        ov_cache_dir=ov_cache_dir,
        disable_ov_cache=disable_ov_cache,
    )
    task_device = (decoder_device or device) if task.device_role == "decoder" else device
    started = time.time()
    core = ov.Core()
    compiled = compile_model(
        core,
        ir_dir / task.graph,
        task_device,
        cache_dir,
        allow_cpu_fallback,
        False,
        precision_hint,
        effective_compile_config,
        ov_cache_mode=ov_cache_mode,
        disable_ov_cache=disable_ov_cache,
    )
    del compiled
    del core
    gc.collect()
    return {
        **asdict(task),
        "device": task_device,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "elapsed": time.time() - started,
        "status": "ok",
    }


def run_single_task(args: argparse.Namespace) -> dict:
    ir_dir = resolve_ir_dir(args.ir_dir, fallback_to_local_voice_design=True, warn=True)
    manifest = load_manifest(ir_dir)
    task = WarmupTask(**json.loads(args.single_task_json))
    compile_config = json.loads(args.compile_config_json or "{}")
    return compile_warmup_task(
        ir_dir,
        manifest,
        task,
        device=args.device,
        decoder_device=args.decoder_device,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=args.cache_step,
        graph_variant=args.graph_variant,
        precision_hint=args.precision_hint,
        compile_config=compile_config,
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )


def subprocess_base_args(args: argparse.Namespace, compile_config: dict) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "qwen3_tts_ov",
        "cache-warmup",
        "--ir-dir",
        str(args.ir_dir),
        "--device",
        args.device,
        "--mode",
        args.mode,
        "--cache-kernel",
        args.cache_kernel,
        "--cache-step",
        args.cache_step,
        "--graph-variant",
        args.graph_variant,
        "--precision-hint",
        args.precision_hint,
        "--ov-cache-mode",
        args.ov_cache_mode,
        "--compile-config-json",
        json.dumps(compile_config, sort_keys=True),
        "--no-subprocess",
    ]
    if args.decoder_device:
        cmd.extend(["--decoder-device", args.decoder_device])
    if args.ov_cache_dir:
        cmd.extend(["--ov-cache-dir", str(args.ov_cache_dir)])
    if args.disable_ov_cache:
        cmd.append("--disable-ov-cache")
    if args.allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    return cmd


def run_cache_warmup(args: argparse.Namespace, compile_config: dict) -> dict:
    args.ir_dir = resolve_ir_dir(args.ir_dir, fallback_to_local_voice_design=True, warn=True)
    tasks, manifest = collect_warmup_tasks(
        args.ir_dir,
        graphs=args.graphs,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=args.cache_step,
        graph_variant=args.graph_variant,
        preload_buckets=args.preload_buckets,
        stream_decoders=args.stream_decoders,
        warmup_strategy=args.warmup_strategy,
    )
    cache_dir = resolve_ov_cache_dir(
        args.ir_dir,
        manifest,
        device=args.device,
        decoder_device=args.decoder_device,
        mode=effective_runtime_options(args.mode, args.cache_kernel, args.cache_step, args.graph_variant)[0],
        cache_kernel=effective_runtime_options(args.mode, args.cache_kernel, args.cache_step, args.graph_variant)[1],
        cache_step=effective_runtime_options(args.mode, args.cache_kernel, args.cache_step, args.graph_variant)[2],
        graph_variant=effective_runtime_options(args.mode, args.cache_kernel, args.cache_step, args.graph_variant)[3],
        precision_hint=args.precision_hint,
        compile_config=merge_compile_config_with_cache_mode(
            compile_config,
            ov_cache_mode=args.ov_cache_mode,
            disable_ov_cache=args.disable_ov_cache,
        ),
        ov_cache_dir=args.ov_cache_dir,
        disable_ov_cache=args.disable_ov_cache,
    )
    summary = {
        "ir_dir": str(Path(args.ir_dir).resolve()),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "ov_cache_mode": normalize_ov_cache_mode(args.ov_cache_mode),
        "task_count": len(tasks),
        "tasks": [asdict(task) for task in tasks],
        "results": [],
    }
    if args.dry_run:
        return summary

    started = time.time()
    if args.subprocess:
        base = subprocess_base_args(args, compile_config)
        for task in tasks:
            cmd = base + ["--single-task-json", json.dumps(asdict(task), sort_keys=True)]
            task_started = time.time()
            completed = subprocess.run(cmd, cwd=str(Path.cwd()), text=True, capture_output=True)
            if completed.returncode != 0:
                result = {
                    **asdict(task),
                    "elapsed": time.time() - task_started,
                    "status": "error",
                    "stderr": completed.stderr.strip(),
                    "stdout": completed.stdout.strip(),
                }
            else:
                result = json.loads(completed.stdout.strip().splitlines()[-1])
            summary["results"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        for task in tasks:
            try:
                result = compile_warmup_task(
                    args.ir_dir,
                    manifest,
                    task,
                    device=args.device,
                    decoder_device=args.decoder_device,
                    mode=args.mode,
                    cache_kernel=args.cache_kernel,
                    cache_step=args.cache_step,
                    graph_variant=args.graph_variant,
                    precision_hint=args.precision_hint,
                    compile_config=compile_config,
                    ov_cache_dir=args.ov_cache_dir,
                    ov_cache_mode=args.ov_cache_mode,
                    disable_ov_cache=args.disable_ov_cache,
                    allow_cpu_fallback=args.allow_cpu_fallback,
                )
            except Exception as exc:
                result = {**asdict(task), "status": "error", "error": str(exc)}
            summary["results"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary["elapsed"] = time.time() - started
    summary["ok"] = all(item.get("status") == "ok" for item in summary["results"])
    return summary
