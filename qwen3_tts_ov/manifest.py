import json
import sys
from pathlib import Path


LIKELY_LOCAL_IR_DIRS = (
    "openvino/voice_design",
    "openvino/custom_voice",
    "openvino/base",
    "openvino_full",
)
DEFAULT_VOICE_DESIGN_IR_DIR = "openvino/voice_design"
LEGACY_VOICE_DESIGN_IR_DIR = "openvino_full"


def local_manifest_candidates(cwd: str | Path | None = None) -> list[str]:
    root = Path.cwd() if cwd is None else Path(cwd)
    candidates = []
    for item in LIKELY_LOCAL_IR_DIRS:
        manifest_path = root / item / "manifest.json"
        if manifest_path.exists():
            candidates.append(item)
    return candidates


def has_manifest(ir_dir: str | Path) -> bool:
    return (Path(ir_dir) / "manifest.json").exists()


def path_text(path: str | Path) -> str:
    return Path(path).as_posix().rstrip("/")


def resolve_ir_dir(ir_dir: str | Path, *, fallback_to_local_voice_design: bool = False, warn: bool = False) -> Path:
    path = Path(ir_dir)
    if has_manifest(path):
        return path
    if fallback_to_local_voice_design and path_text(path) == DEFAULT_VOICE_DESIGN_IR_DIR:
        fallback = Path(LEGACY_VOICE_DESIGN_IR_DIR)
        if has_manifest(fallback):
            if warn:
                print(
                    f"warning: {DEFAULT_VOICE_DESIGN_IR_DIR}/manifest.json not found; using {fallback}/manifest.json",
                    file=sys.stderr,
                )
            return fallback
    return path


def manifest_missing_message(ir_dir: str | Path) -> str:
    ir_dir = Path(ir_dir)
    manifest_path = ir_dir / "manifest.json"
    candidates = local_manifest_candidates()
    candidate_text = ", ".join(candidates) if candidates else "none found"
    return (
        f"OpenVINO IR manifest not found: {manifest_path}\n"
        "--ir-dir must point to an exported OpenVINO IR directory that contains manifest.json.\n"
        "This source repository does not include model weights or OpenVINO IR files. "
        "Export first with `uv run python -m qwen3_tts_ov export ...`, or pass an existing local IR directory.\n"
        f"Local manifest candidates: {candidate_text}"
    )


def load_manifest(ir_dir: str | Path, *, fallback_to_local_voice_design: bool = False, warn: bool = False) -> dict:
    resolved = resolve_ir_dir(ir_dir, fallback_to_local_voice_design=fallback_to_local_voice_design, warn=warn)
    manifest_path = resolved / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_missing_message(ir_dir))
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
