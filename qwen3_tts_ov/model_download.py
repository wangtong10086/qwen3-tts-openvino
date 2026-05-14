from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .manifest import has_manifest


DEFAULT_RELEASE_MODEL_REPO = "waston10086/qwen3-tts-openvino-voice-design"
DEFAULT_RELEASE_MODEL_REVISION = "main"
DEFAULT_RELEASE_MODEL_SUBDIR = "openvino_realtime"
MODEL_DIR_NAMES = ("voice_design", "custom_voice", "base")
MODE_TO_MODEL_DIR = {
    "voice_design": "voice_design",
    "custom_voice": "custom_voice",
    "voice_clone": "base",
}
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
    mode: str = ""
    target_dir: Path | None = None


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
    allow_patterns: list[str] | None = None,
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
            allow_patterns=allow_patterns or [f"{subdir}/**"],
        )
    )


def normalize_download_mode(mode: str) -> str:
    key = (mode or "").strip().replace("-", "_")
    if key == "base":
        key = "voice_clone"
    if key not in MODE_TO_MODEL_DIR:
        raise ValueError("mode must be voice_design, custom_voice, or voice_clone")
    return key


def mode_env_suffix(mode: str) -> str:
    return normalize_download_mode(mode).upper()


def mode_download_config(
    mode: str,
    *,
    repo_id: str = DEFAULT_RELEASE_MODEL_REPO,
    revision: str = DEFAULT_RELEASE_MODEL_REVISION,
    subdir: str = DEFAULT_RELEASE_MODEL_SUBDIR,
) -> dict:
    mode = normalize_download_mode(mode)
    suffix = mode_env_suffix(mode)
    return {
        "mode": mode,
        "model_dir": MODE_TO_MODEL_DIR[mode],
        "repo_id": os.environ.get(f"QWEN3_TTS_OV_MODEL_REPO_{suffix}") or repo_id,
        "revision": os.environ.get(f"QWEN3_TTS_OV_MODEL_REVISION_{suffix}") or revision or DEFAULT_RELEASE_MODEL_REVISION,
        "subdir": (
            os.environ.get(f"QWEN3_TTS_OV_MODEL_SUBDIR_{suffix}") or subdir or DEFAULT_RELEASE_MODEL_SUBDIR
        ).strip().strip("/\\"),
    }


def manifest_supports_download_mode(ir_dir: Path, mode: str) -> bool:
    if not has_manifest(ir_dir):
        return False
    try:
        import json

        manifest = json.loads((ir_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    model_type = str(manifest.get("tts_model_type") or "").replace("-", "_").lower()
    mode = normalize_download_mode(mode)
    if mode == "voice_design":
        return model_type in {"", "voice_design"}
    if mode == "custom_voice":
        return model_type == "custom_voice"
    if mode == "voice_clone":
        return model_type in {"base", "voice_clone"}
    return False


def download_mode_ir(
    model_root: str | Path,
    mode: str,
    *,
    repo_id: str = DEFAULT_RELEASE_MODEL_REPO,
    revision: str = DEFAULT_RELEASE_MODEL_REVISION,
    subdir: str = DEFAULT_RELEASE_MODEL_SUBDIR,
    cache_dir: str | Path | None = None,
) -> ModelDownloadResult:
    mode = normalize_download_mode(mode)
    config = mode_download_config(mode, repo_id=repo_id, revision=revision, subdir=subdir)
    repo_id = config["repo_id"]
    revision = config["revision"]
    subdir = config["subdir"] or DEFAULT_RELEASE_MODEL_SUBDIR
    model_dir_name = config["model_dir"]
    root = Path(model_root).expanduser()
    target_dir = root / model_dir_name
    effective_cache_dir = Path(cache_dir).expanduser() if cache_dir else default_model_cache_dir()

    if manifest_supports_download_mode(target_dir, mode):
        return ModelDownloadResult(
            model_root=root,
            status="local",
            repo_id=repo_id,
            revision=revision,
            subdir=subdir,
            cache_dir=effective_cache_dir,
            message=f"{mode} OpenVINO IR already exists at {target_dir}",
            mode=mode,
            target_dir=target_dir,
        )
    if manifest_supports_download_mode(root, mode):
        return ModelDownloadResult(
            model_root=root,
            status="local",
            repo_id=repo_id,
            revision=revision,
            subdir=subdir,
            cache_dir=effective_cache_dir,
            message=f"{mode} OpenVINO IR already exists at {root}",
            mode=mode,
            target_dir=root,
        )

    download_root = effective_cache_dir / _repo_cache_name(repo_id, revision)
    source_root = download_root / subdir
    candidate_sources = [source_root / model_dir_name, source_root]
    source_dir = next((candidate for candidate in candidate_sources if manifest_supports_download_mode(candidate, mode)), None)
    if source_dir is None:
        effective_cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"downloading {mode} OpenVINO IR from {repo_id}/{subdir} ({revision}) to {download_root}",
            file=sys.stderr,
            flush=True,
        )
        subdir_is_mode_dir = Path(subdir).name == model_dir_name
        allow_patterns = [f"{subdir}/**"] if subdir_is_mode_dir else [f"{subdir}/{model_dir_name}/**"]
        snapshot_root = _snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=download_root,
            subdir=subdir,
            allow_patterns=allow_patterns,
        )
        source_root = snapshot_root / subdir
        candidate_sources = [source_root / model_dir_name, source_root]
        source_dir = next((candidate for candidate in candidate_sources if manifest_supports_download_mode(candidate, mode)), None)

    if source_dir is None:
        raise FileNotFoundError(
            f"download source {repo_id}/{subdir} does not contain a compatible {mode} OpenVINO IR. "
            f"Expected {subdir}/{model_dir_name}/manifest.json or a direct {subdir}/manifest.json."
        )

    if source_dir.resolve() != target_dir.resolve():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    if not manifest_supports_download_mode(target_dir, mode):
        raise FileNotFoundError(f"download completed but {target_dir / 'manifest.json'} is missing or incompatible")

    return ModelDownloadResult(
        model_root=root,
        status="downloaded",
        repo_id=repo_id,
        revision=revision,
        subdir=subdir,
        cache_dir=effective_cache_dir,
        message=f"downloaded {mode} OpenVINO IR to {target_dir}",
        mode=mode,
        target_dir=target_dir,
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
