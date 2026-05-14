from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .manifest import has_manifest


DEFAULT_RELEASE_MODEL_REPO = "waston10086/qwen3-tts-openvino-voice-design"
DEFAULT_RELEASE_MODEL_REVISION = "main"
DEFAULT_RELEASE_MODEL_SUBDIR = "openvino_realtime"
MODEL_DIR_NAMES = ("voice_design", "custom_voice", "base")
AUTO_DOWNLOAD_ENV = "QWEN3_TTS_OV_AUTO_DOWNLOAD_MODEL"
MODEL_CACHE_ENV = "QWEN3_TTS_OV_MODEL_CACHE_DIR"


@dataclass(frozen=True)
class ModelDownloadResult:
    model_root: Path
    status: str
    repo_id: str
    revision: str
    subdir: str
    cache_dir: Path
    message: str


def env_flag_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def model_root_has_manifest(model_root: str | Path) -> bool:
    root = Path(model_root)
    if has_manifest(root):
        return True
    return any(has_manifest(root / item) for item in MODEL_DIR_NAMES)


def default_model_cache_dir() -> Path:
    override = os.environ.get(MODEL_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "qwen3-tts-openvino" / "models"
        return Path.home() / "AppData" / "Local" / "qwen3-tts-openvino" / "models"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "qwen3-tts-openvino" / "models"
    return Path.home() / ".cache" / "qwen3-tts-openvino" / "models"


def _repo_cache_name(repo_id: str, revision: str) -> str:
    safe_repo = repo_id.replace("/", "--").replace("\\", "--")
    safe_revision = (revision or DEFAULT_RELEASE_MODEL_REVISION).replace("/", "--").replace("\\", "--")
    return f"{safe_repo}--{safe_revision}"


def _snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    subdir: str,
) -> Path:
    # PyInstaller release bundles are more reliable with the pure HTTP path than
    # with the optional hf-xet native helper, which may not be collected on all
    # platforms.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - exercised only in stripped release envs
        raise RuntimeError(
            "automatic model download requires the `huggingface_hub` package. "
            "Install it or manually download the OpenVINO IR."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision or None,
            local_dir=str(local_dir),
            allow_patterns=[f"{subdir}/**"],
        )
    )


def ensure_release_model_root(
    model_root: str | Path,
    *,
    auto_download: bool = True,
    repo_id: str = DEFAULT_RELEASE_MODEL_REPO,
    revision: str = DEFAULT_RELEASE_MODEL_REVISION,
    subdir: str = DEFAULT_RELEASE_MODEL_SUBDIR,
    cache_dir: str | Path | None = None,
) -> ModelDownloadResult:
    requested_root = Path(model_root).expanduser()
    effective_cache_dir = Path(cache_dir).expanduser() if cache_dir else default_model_cache_dir()
    revision = revision or DEFAULT_RELEASE_MODEL_REVISION
    subdir = subdir.strip().strip("/\\") or DEFAULT_RELEASE_MODEL_SUBDIR

    if model_root_has_manifest(requested_root):
        return ModelDownloadResult(
            model_root=requested_root,
            status="local",
            repo_id=repo_id,
            revision=revision,
            subdir=subdir,
            cache_dir=effective_cache_dir,
            message=f"using local OpenVINO IR at {requested_root}",
        )

    if not auto_download or not env_flag_enabled(AUTO_DOWNLOAD_ENV, True):
        return ModelDownloadResult(
            model_root=requested_root,
            status="missing",
            repo_id=repo_id,
            revision=revision,
            subdir=subdir,
            cache_dir=effective_cache_dir,
            message=f"OpenVINO IR was not found at {requested_root}; automatic download is disabled",
        )

    download_root = effective_cache_dir / _repo_cache_name(repo_id, revision)
    resolved_model_root = download_root / subdir
    if model_root_has_manifest(resolved_model_root):
        return ModelDownloadResult(
            model_root=resolved_model_root,
            status="cached",
            repo_id=repo_id,
            revision=revision,
            subdir=subdir,
            cache_dir=effective_cache_dir,
            message=f"using cached OpenVINO IR at {resolved_model_root}",
        )

    effective_cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "OpenVINO IR not found; downloading "
        f"{repo_id}/{subdir} ({revision}) to {download_root}",
        file=sys.stderr,
        flush=True,
    )
    snapshot_root = _snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=download_root,
        subdir=subdir,
    )
    downloaded_model_root = snapshot_root / subdir
    if not model_root_has_manifest(downloaded_model_root):
        raise FileNotFoundError(
            "automatic download completed, but no OpenVINO manifest was found under "
            f"{downloaded_model_root}. Check --model-repo/--model-subdir or download the IR manually."
        )
    return ModelDownloadResult(
        model_root=downloaded_model_root,
        status="downloaded",
        repo_id=repo_id,
        revision=revision,
        subdir=subdir,
        cache_dir=effective_cache_dir,
        message=f"downloaded OpenVINO IR to {downloaded_model_root}",
    )
