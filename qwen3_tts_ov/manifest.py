import json
from pathlib import Path


LIKELY_LOCAL_IR_DIRS = (
    "openvino/voice_design",
    "openvino/custom_voice",
    "openvino/base",
    "openvino_full",
)


def local_manifest_candidates(cwd: str | Path | None = None) -> list[str]:
    root = Path.cwd() if cwd is None else Path(cwd)
    candidates = []
    for item in LIKELY_LOCAL_IR_DIRS:
        manifest_path = root / item / "manifest.json"
        if manifest_path.exists():
            candidates.append(item)
    return candidates


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


def load_manifest(ir_dir: str | Path) -> dict:
    manifest_path = Path(ir_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_missing_message(ir_dir))
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
