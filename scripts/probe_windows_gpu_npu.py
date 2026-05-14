#!/usr/bin/env python3
"""Probe Windows GPU+NPU OpenVINO support for the release TTS path."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def normalize_device_name(value: str | None) -> str:
    return str(value or "").strip().upper()


def device_available(available_devices: list[str], required: str) -> bool:
    required_name = normalize_device_name(required)
    for item in available_devices:
        name = normalize_device_name(item)
        if name == required_name or name.startswith(f"{required_name}."):
            return True
    return False


def load_manifest(model_root: Path) -> tuple[Path, dict]:
    ir_dir = model_root / "voice_design"
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"voice_design manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return ir_dir, json.load(handle)


def load_optional_manifest(model_root: Path, mode_dir: str) -> tuple[Path, dict] | None:
    ir_dir = model_root / mode_dir
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return ir_dir, json.load(handle)


def select_stream_decoder_graphs(manifest: dict) -> tuple[str, str]:
    contexts = (manifest.get("streaming_decoder") or {}).get("contexts") or {}
    first_context = contexts.get("0") or contexts.get(0) or {}
    steady_context = contexts.get("25") or contexts.get(25) or {}
    first_graph = first_context.get("8") or first_context.get(8) or first_context.get("12") or first_context.get(12)
    steady_graph = steady_context.get("24") or steady_context.get(24) or steady_context.get("12") or steady_context.get(12)
    if not first_graph or not steady_graph:
        raise RuntimeError("runtime-minimal IR must include streaming decoder contexts c0_t8/c0_t12 and c25_t24/c25_t12")
    return str(first_graph), str(steady_graph)


def select_audio_encoder_graphs(manifest: dict) -> list[tuple[str, str]]:
    graphs = manifest.get("graphs") or {}
    selected = []
    for key in ("speech_encoder", "speaker_encoder"):
        graph = graphs.get(key)
        if isinstance(graph, str) and graph:
            selected.append((key, graph))
    return selected


def select_prompt_graphs(manifest: dict) -> list[tuple[str, str]]:
    graphs = manifest.get("graphs") or {}
    selected = []
    for key in ("text_embedding", "codec_embedding"):
        graph = graphs.get(key)
        if isinstance(graph, str) and graph:
            selected.append((key, graph))
    return selected


def compile_named_graphs(
    core,
    ir_dir: Path,
    graphs: list[tuple[str, str]],
    device: str,
    cache_dir: Path | None,
) -> list[dict]:
    config = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache_dir)
        config["CACHE_MODE"] = "OPTIMIZE_SPEED"
    compiled = []
    for label, graph in graphs:
        path = ir_dir / graph
        if not path.exists():
            raise FileNotFoundError(f"{label} graph not found: {path}")
        started = time.time()
        model = core.compile_model(str(path), device, config)
        request = model.create_infer_request()
        compiled.append(
            {
                "label": label,
                "graph": graph,
                "device": device,
                "compile_ms": round((time.time() - started) * 1000.0, 3),
                "inputs": [str(item.get_any_name()) for item in model.inputs],
                "outputs": [str(item.get_any_name()) for item in model.outputs],
                "request_created": request is not None,
            }
        )
    return compiled


def compile_decoder_graphs(core, ir_dir: Path, graphs: list[str], decoder_device: str, cache_dir: Path | None) -> list[dict]:
    return compile_named_graphs(
        core,
        ir_dir,
        [(f"stream_decoder:{graph}", graph) for graph in graphs],
        decoder_device,
        cache_dir,
    )


def zero_copy_probe(core, devices: list[str]) -> dict:
    result = {
        "status": "info_only",
        "available_devices": devices,
        "python_api": {
            "core_create_context": hasattr(core, "create_context"),
            "core_create_tensor": hasattr(core, "create_tensor"),
        },
        "contexts": {},
        "note": (
            "This probe checks OpenVINO Python remote-context API visibility only. "
            "Actual GPU/NPU shared-handle zero-copy requires native handles and is not enabled by this smoke path."
        ),
    }
    for device in ("GPU", "NPU"):
        if not device_available(devices, device):
            result["contexts"][device] = {"status": "missing_device"}
            continue
        if not hasattr(core, "create_context"):
            result["contexts"][device] = {"status": "unavailable", "reason": "Core.create_context is not exposed"}
            continue
        try:
            context = core.create_context(device, {})
            result["contexts"][device] = {"status": "ok", "type": type(context).__name__}
        except Exception as exc:  # pragma: no cover - depends on installed OpenVINO plugin support
            result["contexts"][device] = {"status": "failed", "error": str(exc)}
    return result


def write_summary(summary: dict, output_json: str | None) -> None:
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default="NPU")
    parser.add_argument("--skip-prompt-graphs", action="store_true")
    parser.add_argument("--skip-audio-encoders", action="store_true")
    parser.add_argument("--skip-if-missing-devices", action="store_true")
    parser.add_argument("--require-zero-copy", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    summary = {
        "status": "failed",
        "platform": platform.platform(),
        "model_root": str(Path(args.model_root).resolve()),
        "device": args.device,
        "decoder_device": args.decoder_device,
        "available_devices": [],
        "decoder_compile": [],
        "prompt_compile": {"status": "not_run", "graphs": []},
        "audio_encoder_compile": {"status": "not_run", "graphs": []},
        "zero_copy_probe": {},
    }

    try:
        import openvino as ov

        summary["openvino_version"] = getattr(ov, "__version__", "unknown")
        core = ov.Core()
        available_devices = [str(item) for item in core.available_devices]
        summary["available_devices"] = available_devices
        required_devices = [args.device, args.decoder_device]
        missing = [item for item in required_devices if not device_available(available_devices, item)]
        if missing:
            summary["status"] = "skipped" if args.skip_if_missing_devices else "failed"
            summary["skip_reason"] = f"missing required OpenVINO devices: {', '.join(missing)}"
            write_summary(summary, args.output_json)
            raise SystemExit(0 if args.skip_if_missing_devices else 2)

        ir_dir, manifest = load_manifest(Path(args.model_root).resolve())
        first_graph, steady_graph = select_stream_decoder_graphs(manifest)
        try:
            summary["decoder_compile"] = compile_decoder_graphs(
                core,
                ir_dir,
                [first_graph, steady_graph],
                args.decoder_device,
                Path(args.cache_dir).resolve() if args.cache_dir else None,
            )
        except Exception as exc:
            if args.skip_if_missing_devices:
                summary["status"] = "skipped"
                summary["skip_reason"] = f"NPU decoder compile failed: {exc}"
                write_summary(summary, args.output_json)
                raise SystemExit(0)
            raise
        if args.skip_prompt_graphs:
            summary["prompt_compile"] = {"status": "skipped", "reason": "disabled", "graphs": []}
        else:
            prompt_graphs = select_prompt_graphs(manifest)
            if not prompt_graphs:
                summary["prompt_compile"] = {
                    "status": "skipped",
                    "reason": "voice_design manifest has no prompt graphs",
                    "graphs": [],
                }
            else:
                try:
                    summary["prompt_compile"] = {
                        "status": "ok",
                        "graphs": compile_named_graphs(
                            core,
                            ir_dir,
                            prompt_graphs,
                            args.decoder_device,
                            Path(args.cache_dir).resolve() / "prompt" if args.cache_dir else None,
                        ),
                    }
                except Exception as exc:
                    if args.skip_if_missing_devices:
                        summary["prompt_compile"] = {
                            "status": "skipped",
                            "reason": f"NPU prompt graph compile failed: {exc}",
                            "graphs": [],
                        }
                    else:
                        raise
        if args.skip_audio_encoders:
            summary["audio_encoder_compile"] = {"status": "skipped", "reason": "disabled", "graphs": []}
        else:
            base_manifest = load_optional_manifest(Path(args.model_root).resolve(), "base")
            if base_manifest is None:
                summary["audio_encoder_compile"] = {
                    "status": "skipped",
                    "reason": "base manifest not found",
                    "graphs": [],
                }
            else:
                base_ir_dir, base_payload = base_manifest
                audio_graphs = select_audio_encoder_graphs(base_payload)
                if not audio_graphs:
                    summary["audio_encoder_compile"] = {
                        "status": "skipped",
                        "reason": "base manifest has no audio encoder graphs",
                        "graphs": [],
                    }
                else:
                    try:
                        summary["audio_encoder_compile"] = {
                            "status": "ok",
                            "graphs": compile_named_graphs(
                                core,
                                base_ir_dir,
                                audio_graphs,
                                args.decoder_device,
                                Path(args.cache_dir).resolve() / "audio-encoders" if args.cache_dir else None,
                            ),
                        }
                    except Exception as exc:
                        if args.skip_if_missing_devices:
                            summary["audio_encoder_compile"] = {
                                "status": "skipped",
                                "reason": f"NPU audio encoder compile failed: {exc}",
                                "graphs": [],
                            }
                        else:
                            raise
        summary["zero_copy_probe"] = zero_copy_probe(core, available_devices)
        if args.require_zero_copy:
            contexts = summary["zero_copy_probe"].get("contexts") or {}
            if not all((contexts.get(device) or {}).get("status") == "ok" for device in ("GPU", "NPU")):
                summary["status"] = "failed"
                summary["error"] = "zero-copy remote context probe did not pass for both GPU and NPU"
                write_summary(summary, args.output_json)
                raise SystemExit(3)
        summary["status"] = "ok"
        write_summary(summary, args.output_json)
    except SystemExit:
        raise
    except Exception as exc:
        summary["error"] = str(exc)
        write_summary(summary, args.output_json)
        raise


if __name__ == "__main__":
    main()
