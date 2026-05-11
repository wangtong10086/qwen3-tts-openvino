import hashlib
import json
import os
import re
import sys
from pathlib import Path

import openvino as ov


OV_CACHE_MODE_VALUES = {
    "optimize_speed": "OPTIMIZE_SPEED",
    "speed": "OPTIMIZE_SPEED",
    "optimize_size": "OPTIMIZE_SIZE",
    "size": "OPTIMIZE_SIZE",
}


def normalize_ov_cache_mode(value: str | None) -> str | None:
    if value is None:
        return "OPTIMIZE_SPEED"
    key = str(value).strip().replace("-", "_").lower()
    if key in {"", "default", "none"}:
        return None
    if key not in OV_CACHE_MODE_VALUES:
        supported = ", ".join(sorted({"optimize_speed", "optimize_size"}))
        raise ValueError(f"unsupported ov_cache_mode={value!r}; supported modes: {supported}")
    return OV_CACHE_MODE_VALUES[key]


def default_ov_cache_root() -> Path:
    override = os.environ.get("QWEN3_TTS_OV_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "qwen3-tts-ov" / "openvino-cache"
        return Path.home() / "AppData" / "Local" / "qwen3-tts-ov" / "openvino-cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "qwen3-tts-ov" / "openvino-cache"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "qwen3-tts-ov" / "openvino-cache"


def sanitize_cache_part(value: object) -> str:
    text = str(value or "default")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text[:80] or "default"


def manifest_cache_fingerprint(
    ir_dir: str | Path,
    manifest: dict,
    *,
    device: str,
    decoder_device: str | None,
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
    precision_hint: str,
    compile_config: dict | None,
) -> str:
    ir_dir = Path(ir_dir)
    manifest_bytes = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload = {
        "ir_dir": str(ir_dir.resolve()),
        "openvino_version": getattr(ov, "__version__", "unknown"),
        "device": device,
        "decoder_device": decoder_device or device,
        "mode": mode,
        "cache_kernel": cache_kernel,
        "cache_step": cache_step,
        "graph_variant": graph_variant,
        "precision_hint": precision_hint,
        "compile_config": compile_config or {},
    }
    digest = hashlib.sha256()
    digest.update(manifest_bytes)
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def resolve_ov_cache_dir(
    ir_dir: str | Path,
    manifest: dict,
    *,
    device: str,
    decoder_device: str | None = None,
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    precision_hint: str = "f16",
    compile_config: dict | None = None,
    ov_cache_dir: str | Path | None = None,
    disable_ov_cache: bool = False,
) -> Path | None:
    if disable_ov_cache:
        return None
    if ov_cache_dir:
        return Path(ov_cache_dir).expanduser()
    model_type = sanitize_cache_part(manifest.get("tts_model_type") or "qwen3_tts")
    ov_version = sanitize_cache_part(getattr(ov, "__version__", "unknown").split("-")[0])
    device_part = sanitize_cache_part(f"{device}-{decoder_device or device}")
    fingerprint = manifest_cache_fingerprint(
        ir_dir,
        manifest,
        device=device,
        decoder_device=decoder_device,
        mode=mode,
        cache_kernel=cache_kernel,
        cache_step=cache_step,
        graph_variant=graph_variant,
        precision_hint=precision_hint,
        compile_config=compile_config,
    )[:16]
    return default_ov_cache_root() / ov_version / device_part / f"{model_type}-{fingerprint}"


def build_ov_cache_config(
    cache_dir: str | Path | None,
    *,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
) -> dict:
    if disable_ov_cache or cache_dir is None:
        return {}
    config = {"CACHE_DIR": str(Path(cache_dir).expanduser())}
    normalized_mode = normalize_ov_cache_mode(ov_cache_mode)
    if normalized_mode:
        config["CACHE_MODE"] = normalized_mode
    return config


def merge_compile_config_with_cache_mode(
    compile_config: dict | None,
    *,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
) -> dict:
    config = dict(compile_config or {})
    normalized_mode = normalize_ov_cache_mode(ov_cache_mode)
    if not disable_ov_cache and normalized_mode:
        config.setdefault("CACHE_MODE", normalized_mode)
    return config
