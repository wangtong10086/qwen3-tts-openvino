#!/usr/bin/env python3
"""Package an exported OpenVINO IR directory as a user-distributable artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


IR_PROFILES = ("full", "runtime-minimal")
MINIMAL_GRAPH_VARIANT = "int8_sym_paged_talker_split"
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def walk_manifest_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_manifest_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_manifest_strings(item)


def manifest_referenced_files(ir_dir: Path, manifest: dict) -> list[Path]:
    names = {"manifest.json"}
    for value in walk_manifest_strings(manifest):
        if value.endswith((".xml", ".bin", ".json")) and not Path(value).is_absolute():
            names.add(value)
            if value.endswith(".xml"):
                names.add(str(Path(value).with_suffix(".bin")))
    paths = []
    missing = []
    for name in sorted(names):
        path = ir_dir / name
        if path.exists():
            paths.append(path)
        elif name.endswith((".xml", ".bin")):
            missing.append(name)
    if missing:
        raise FileNotFoundError("IR manifest references missing files: " + ", ".join(missing[:20]))
    return paths


def _require_graph(graphs: dict, key: str) -> str:
    value = graphs.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime-minimal IR requires graphs.{key}")
    return value


def _require_stream_graph(stream_config: dict, context_frames: int, chunk_frames: int) -> str:
    contexts = stream_config.get("contexts") or {}
    graph = ((contexts.get(str(context_frames)) or {}).get(str(chunk_frames)))
    if not graph and context_frames != 0:
        graph = ((stream_config.get("graphs") or {}).get(str(chunk_frames)))
    if not graph:
        raise ValueError(f"runtime-minimal IR requires streaming decoder c{context_frames}_t{chunk_frames}")
    return graph


def runtime_minimal_manifest(manifest: dict, model_type: str) -> dict:
    source_graphs = manifest.get("graphs") or {}
    source_variants = manifest.get("graph_variants") or {}
    source_stream = manifest.get("streaming_decoder") or {}
    variant = source_variants.get(MINIMAL_GRAPH_VARIANT)
    if not isinstance(variant, dict):
        available = ", ".join(sorted(source_variants)) or "none"
        raise ValueError(
            f"runtime-minimal IR requires graph variant {MINIMAL_GRAPH_VARIANT!r}; available variants: {available}"
        )
    variant_graphs = variant.get("graphs") or {}
    variant_paged = variant_graphs.get("paged_kv_seed") or {}
    if not isinstance(variant_paged, dict) or not variant_paged:
        raise ValueError(f"runtime-minimal IR requires graph_variants.{MINIMAL_GRAPH_VARIANT}.graphs.paged_kv_seed")

    first_decoder = _require_stream_graph(source_stream, 0, 8)
    steady_decoder = _require_stream_graph(source_stream, 25, 24)
    graphs = {
        "text_embedding": _require_graph(source_graphs, "text_embedding"),
        "codec_embedding": _require_graph(source_graphs, "codec_embedding"),
        "paged_kv_seed": {},
        "subcode_greedy_cached": _require_graph(source_graphs, "subcode_greedy_cached"),
        "speech_decoder": {},
        "streaming_decoder": {"24": steady_decoder},
    }

    normalized_model_type = str(manifest.get("tts_model_type") or model_type).replace("-", "_").lower()
    if normalized_model_type in {"base", "voice_clone"} or model_type == "base":
        graphs["code_frame_embedding"] = _require_graph(source_graphs, "code_frame_embedding")
        graphs["speech_encoder"] = _require_graph(source_graphs, "speech_encoder")
        graphs["speaker_encoder"] = _require_graph(source_graphs, "speaker_encoder")

    minimal = copy.deepcopy(manifest)
    minimal["model_dir"] = "."
    minimal.pop("tokenizer_ir", None)
    minimal["graphs"] = graphs
    minimal["graph_variants"] = {
        MINIMAL_GRAPH_VARIANT: {
            **{key: value for key, value in variant.items() if key != "graphs"},
            "graphs": {"paged_kv_seed": variant_paged},
        }
    }
    minimal["streaming_decoder"] = {
        "left_context_frames": 25,
        "chunk_frames": [24],
        "first_chunk_frames": [8],
        "default_strategy": "smooth",
        "strategies": {"smooth": {"initial_chunk_frames": 8, "chunk_frames": 24, "left_context_frames": 25}},
        "graphs": {"24": steady_decoder},
        "contexts": {"0": {"8": first_decoder}, "25": {"24": steady_decoder}},
        "output_format": source_stream.get("output_format", "pcm_f32"),
    }
    return minimal


def manifest_for_profile(manifest: dict, profile: str, model_type: str) -> dict:
    if profile == "runtime-minimal":
        return runtime_minimal_manifest(manifest, model_type)
    packaged = copy.deepcopy(manifest)
    packaged["model_dir"] = "."
    return packaged


def tokenizer_file_sources(ir_dir: Path, source_manifest: dict) -> dict[str, Path]:
    roots = [ir_dir]
    model_dir = source_manifest.get("model_dir")
    if model_dir:
        model_path = Path(model_dir)
        if model_path.exists():
            roots.append(model_path)
    result = {}
    for name in TOKENIZER_FILES:
        for root in roots:
            path = root / name
            if path.exists():
                result[name] = path
                break
    missing = [name for name in TOKENIZER_FILES if name not in result]
    if missing:
        raise FileNotFoundError("missing tokenizer files required for portable release IR: " + ", ".join(missing))
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_archive(staging_root: Path, output: Path, archive_format: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in staging_root.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(staging_root.parent).as_posix())
        return output
    if archive_format == "tar.gz":
        with tarfile.open(output, "w:gz") as tf:
            tf.add(staging_root, arcname=staging_root.name)
        return output
    if archive_format == "tar.zst":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise SystemExit("tar.zst packaging requires `uv pip install -e .[release]`") from exc
        with tempfile.NamedTemporaryFile(suffix=".tar") as raw:
            with tarfile.open(raw.name, "w") as tf:
                tf.add(staging_root, arcname=staging_root.name)
            raw.seek(0)
            cctx = zstd.ZstdCompressor(level=10)
            with output.open("wb") as dst:
                cctx.copy_stream(raw, dst)
        return output
    raise ValueError(f"unsupported archive format: {archive_format}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--model-type", default="voice_design", choices=["voice_design", "custom_voice", "base"])
    parser.add_argument("--version", default="0.1.2")
    parser.add_argument("--out-dir", default="dist/release")
    parser.add_argument("--format", default="auto", choices=["auto", "zip", "tar.gz", "tar.zst"])
    parser.add_argument("--profile", default="runtime-minimal", choices=IR_PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir).resolve()
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = manifest_for_profile(source_manifest, args.profile, args.model_type)
    files = manifest_referenced_files(ir_dir, manifest)
    tokenizer_sources = tokenizer_file_sources(ir_dir, source_manifest)
    archive_format = "tar.zst" if args.format == "auto" else args.format
    suffix = {"zip": ".zip", "tar.gz": ".tar.gz", "tar.zst": ".tar.zst"}[archive_format]
    profile_suffix = "" if args.profile == "full" else f"-{args.profile}"
    package_name = f"qwen3-tts-openvino-ir-{args.model_type}-{args.version}{profile_suffix}"
    output = Path(args.out_dir) / f"{package_name}{suffix}"
    print(
        json.dumps(
            {"package": str(output), "profile": args.profile, "file_count": len(files) + len(tokenizer_sources), "format": archive_format},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging_root = Path(tmp) / package_name
        target_ir = staging_root / "openvino" / args.model_type
        target_ir.mkdir(parents=True)
        for source in files:
            if source.name == "manifest.json":
                continue
            rel = source.relative_to(ir_dir)
            dest = target_ir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        for rel, source in tokenizer_sources.items():
            shutil.copy2(source, target_ir / rel)
        (target_ir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = staging_root / "README_RELEASE_IR.md"
        readme.write_text(
            "\n".join(
                [
                    f"# Qwen3-TTS OpenVINO IR {args.model_type} {args.version}",
                    "",
                    "Extract this package next to the app package so the server can find `openvino/`.",
                    "",
                    "Example:",
                    "",
                    "```bash",
                    "./qwen3-tts-ov-server --model-root openvino --device GPU",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        write_checksums(staging_root)
        make_archive(staging_root, output, archive_format)
    print(output, flush=True)


if __name__ == "__main__":
    main()
