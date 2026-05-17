from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


class NativeCodegenUnavailable(RuntimeError):
    pass


def float_audio_view_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2", copy=False).tobytes()


_DLL_DIRECTORY_HANDLES = []


def native_library_name() -> str:
    if os.name == "nt":
        return "qwen3_tts_ov_genai.dll"
    if sys.platform == "darwin":
        return "libqwen3_tts_ov_genai.dylib"
    return "libqwen3_tts_ov_genai.so"


def native_library_candidates() -> list[Path]:
    name = native_library_name()
    roots: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parents[1])

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "native" / "build" / name,
                root / "native" / name,
                root / name,
            ]
        )
    return candidates


def default_library_path() -> Path:
    for candidate in native_library_candidates():
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[1] / "native" / "build" / native_library_name()


def ensure_openvino_tokenizers_extension_env() -> None:
    if os.environ.get("OPENVINO_TOKENIZERS_PATH_GENAI"):
        return
    try:
        import openvino_tokenizers
    except Exception:
        return
    extension_path = getattr(openvino_tokenizers, "_ext_path", None)
    if extension_path:
        os.environ["OPENVINO_TOKENIZERS_PATH_GENAI"] = str(extension_path)


def ensure_windows_dll_search_paths(library_path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    roots = [library_path.parent]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    for module_name, subdir in (
        ("openvino", "libs"),
        ("openvino_genai", ""),
        ("openvino_tokenizers", "lib"),
    ):
        try:
            module = __import__(module_name)
        except Exception:
            continue
        module_dir = Path(getattr(module, "__file__", "")).resolve().parent
        roots.append(module_dir / subdir if subdir else module_dir)

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(root)))


class NativeCodegenRunner:
    def __init__(
        self,
        prefill_graph: Path,
        decode_graph: Path,
        device: str,
        cache_dir: Path | None = None,
        cache_mode: str = "OPTIMIZE_SPEED",
        library_path: Path | None = None,
        paged_kv: bool = False,
        kv_cache_precision: str = "f16",
        kv_cache_heads: int = 16,
        kv_cache_block_size: int = 8,
        kv_cache_head_dim: int = 128,
        paged_kv_subcode_graph: Path | None = None,
    ):
        library_path = Path(library_path or os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_LIB") or default_library_path())
        if not library_path.exists():
            raise NativeCodegenUnavailable(
                f"native codegen library not found: {library_path}; build it with `uv run python scripts/build_native_codegen.py`"
            )
        ensure_openvino_tokenizers_extension_env()
        ensure_windows_dll_search_paths(library_path)
        self.library_path = library_path
        self.lib = ctypes.CDLL(str(library_path))
        self._configure_api()
        self.handle = ctypes.c_void_p()
        err = ctypes.c_char_p()
        if paged_kv and paged_kv_subcode_graph is not None:
            rc = self.lib.qwen3_tts_codegen_create_paged_kv_split(
                str(prefill_graph).encode("utf-8"),
                str(paged_kv_subcode_graph).encode("utf-8"),
                str(device).encode("utf-8"),
                str(cache_dir or "").encode("utf-8"),
                str(cache_mode or "OPTIMIZE_SPEED").encode("utf-8"),
                str(kv_cache_precision or "f16").encode("utf-8"),
                int(kv_cache_heads),
                int(kv_cache_block_size),
                int(kv_cache_head_dim),
                ctypes.byref(self.handle),
                ctypes.byref(err),
            )
        elif paged_kv:
            rc = self.lib.qwen3_tts_codegen_create_paged_kv(
                str(prefill_graph).encode("utf-8"),
                str(device).encode("utf-8"),
                str(cache_dir or "").encode("utf-8"),
                str(cache_mode or "OPTIMIZE_SPEED").encode("utf-8"),
                str(kv_cache_precision or "f16").encode("utf-8"),
                int(kv_cache_heads),
                int(kv_cache_block_size),
                int(kv_cache_head_dim),
                ctypes.byref(self.handle),
                ctypes.byref(err),
            )
        else:
            rc = self.lib.qwen3_tts_codegen_create(
                str(prefill_graph).encode("utf-8"),
                str(decode_graph).encode("utf-8"),
                str(device).encode("utf-8"),
                str(cache_dir or "").encode("utf-8"),
                str(cache_mode or "OPTIMIZE_SPEED").encode("utf-8"),
                ctypes.byref(self.handle),
                ctypes.byref(err),
            )
        self._check(rc, err)
        self.closed = False
        self.last_remote_embed = False
        self.paged_kv = bool(paged_kv)
        self.paged_kv_split_subcode = bool(paged_kv and paged_kv_subcode_graph is not None)

    def _configure_api(self) -> None:
        self.lib.qwen3_tts_codegen_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_create.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_create_paged_kv.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_create_paged_kv.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_create_paged_kv_split.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_create_paged_kv_split.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_destroy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.qwen3_tts_codegen_destroy.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask.restype = ctypes.c_int
        self._frame_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        )
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            self._frame_callback_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_stream.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_set_stream_decoders.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_set_stream_decoders.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_configure_voice_design_prompt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_configure_voice_design_prompt.restype = ctypes.c_int
        self._audio_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_void_p,
        )
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_audio_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            self._audio_callback_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_audio_stream.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_run_voice_design_audio_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            self._audio_callback_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_voice_design_audio_stream.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_run_paged_kv_repeat_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_paged_kv_repeat_batch.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch_codes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch_codes.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_online_batch_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_online_batch_reset.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_online_batch_add_sequence.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_online_batch_add_sequence.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_online_batch_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_online_batch_step.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_online_batch_cancel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_online_batch_cancel.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_online_batch_get_stats_json.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_online_batch_get_stats_json.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_get_last_remote_embed_used.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_get_last_remote_embed_used.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_reset_profile.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_reset_profile.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_get_profile_json.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_get_profile_json.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_get_last_timing_json.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_get_last_timing_json.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_release_run_buffers.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.lib.qwen3_tts_codegen_release_run_buffers.restype = ctypes.c_int
        self.lib.qwen3_tts_codegen_free_error.argtypes = [ctypes.c_char_p]
        self.lib.qwen3_tts_codegen_free_error.restype = None

    def _check(self, rc: int, err: ctypes.c_char_p) -> None:
        if rc == 0:
            return
        message = err.value.decode("utf-8", errors="replace") if err.value else "native codegen failed"
        if err.value:
            self.lib.qwen3_tts_codegen_free_error(err)
        raise RuntimeError(message)

    def last_remote_embed_used(self) -> bool:
        used = ctypes.c_int64(0)
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_get_last_remote_embed_used(
            self.handle,
            ctypes.byref(used),
            ctypes.byref(err),
        )
        self._check(rc, err)
        return bool(used.value)

    def reset_profile(self) -> None:
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_reset_profile(self.handle, ctypes.byref(err))
        self._check(rc, err)

    def profile_json(self) -> dict | None:
        out = ctypes.c_char_p()
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_get_profile_json(
            self.handle,
            ctypes.byref(out),
            ctypes.byref(err),
        )
        self._check(rc, err)
        try:
            if not out.value:
                return None
            return json.loads(out.value.decode("utf-8", errors="replace"))
        finally:
            if out.value:
                self.lib.qwen3_tts_codegen_free_error(out)

    def timing_json(self) -> dict | None:
        out = ctypes.c_char_p()
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_get_last_timing_json(
            self.handle,
            ctypes.byref(out),
            ctypes.byref(err),
        )
        self._check(rc, err)
        try:
            if not out.value:
                return None
            return json.loads(out.value.decode("utf-8", errors="replace"))
        finally:
            if out.value:
                self.lib.qwen3_tts_codegen_free_error(out)

    def release_run_buffers(self) -> None:
        if getattr(self, "closed", True):
            return
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_release_run_buffers(self.handle, ctypes.byref(err))
        self._check(rc, err)

    def run(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> tuple[np.ndarray, float]:
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError("sequence must have shape [1, prompt_len, hidden]")
        if tts_pad_embed.shape != (1, 1, sequence.shape[-1]):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")
        out = np.empty((int(max_new_tokens), int(num_code_groups)), dtype=np.int64)
        out_count = ctypes.c_int64(0)
        elapsed_ms = ctypes.c_double(0.0)
        err = ctypes.c_char_p()
        self.reset_profile()
        rc = self.lib.qwen3_tts_codegen_run_unroll4_statefulmask(
            self.handle,
            sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(sequence.shape[1]),
            ctypes.c_int64(sequence.shape[2]),
            tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(max_new_tokens),
            ctypes.c_int64(min_new_tokens),
            ctypes.c_float(repetition_penalty),
            ctypes.c_int64(vocab_size),
            ctypes.c_int64(num_code_groups),
            ctypes.c_int64(eos_token_id),
            ctypes.c_int64(1 if do_sample else 0),
            ctypes.c_int64(top_k),
            ctypes.c_float(top_p),
            ctypes.c_float(temperature),
            ctypes.c_uint64(seed),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(out_count),
            ctypes.byref(elapsed_ms),
            ctypes.byref(err),
        )
        self._check(rc, err)
        self.last_remote_embed = self.last_remote_embed_used()
        self.last_timing_json = self.timing_json()
        return out[: int(out_count.value)].copy(), float(elapsed_ms.value)

    def run_paged_kv_repeat_batch(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        batch_size: int,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> dict:
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        batch_size = int(batch_size)
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError("sequence must have shape [1, prompt_len, hidden]")
        if tts_pad_embed.shape != (1, 1, sequence.shape[-1]):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        out_counts = np.zeros((batch_size,), dtype=np.int64)
        out_ttft_ms = np.full((batch_size,), -1.0, dtype=np.float64)
        out_last_token_ms = np.zeros((batch_size,), dtype=np.float64)
        elapsed_ms = ctypes.c_double(0.0)
        err = ctypes.c_char_p()
        self.reset_profile()
        rc = self.lib.qwen3_tts_codegen_run_paged_kv_repeat_batch(
            self.handle,
            sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(sequence.shape[1]),
            ctypes.c_int64(sequence.shape[2]),
            tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(batch_size),
            ctypes.c_int64(max_new_tokens),
            ctypes.c_int64(min_new_tokens),
            ctypes.c_float(repetition_penalty),
            ctypes.c_int64(vocab_size),
            ctypes.c_int64(num_code_groups),
            ctypes.c_int64(eos_token_id),
            ctypes.c_int64(1 if do_sample else 0),
            ctypes.c_int64(top_k),
            ctypes.c_float(top_p),
            ctypes.c_float(temperature),
            ctypes.c_uint64(seed),
            out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_ttft_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_last_token_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(elapsed_ms),
            ctypes.byref(err),
        )
        self._check(rc, err)
        self.last_remote_embed = self.last_remote_embed_used()
        self.last_profile_json = self.profile_json()
        self.last_timing_json = self.timing_json()
        return {
            "batch_size": batch_size,
            "counts": out_counts.copy(),
            "ttft_ms": out_ttft_ms.copy(),
            "last_token_ms": out_last_token_ms.copy(),
            "elapsed_ms": float(elapsed_ms.value),
            "profile": self.last_profile_json,
            "timing": self.last_timing_json,
        }

    def run_paged_kv_sequence_batch(
        self,
        sequences: list[np.ndarray] | tuple[np.ndarray, ...],
        tts_pad_embed: np.ndarray,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
        return_codes: bool = False,
    ) -> dict:
        if not sequences:
            raise ValueError("sequences must not be empty")
        prepared: list[np.ndarray] = []
        prompt_lens: list[int] = []
        hidden_size: int | None = None
        for sequence in sequences:
            array = np.ascontiguousarray(sequence, dtype=np.float32)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array.reshape(array.shape[1], array.shape[2])
            if array.ndim != 2:
                raise ValueError("each sequence must have shape [1, prompt_len, hidden] or [prompt_len, hidden]")
            if hidden_size is None:
                hidden_size = int(array.shape[-1])
            elif int(array.shape[-1]) != hidden_size:
                raise ValueError("all sequences must have the same hidden size")
            if int(array.shape[0]) <= 0:
                raise ValueError("prompt_len must be positive")
            prepared.append(array)
            prompt_lens.append(int(array.shape[0]))
        assert hidden_size is not None
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        if tts_pad_embed.shape != (1, 1, hidden_size):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")
        flat = np.ascontiguousarray(np.concatenate(prepared, axis=0), dtype=np.float32)
        prompt_lens_array = np.ascontiguousarray(prompt_lens, dtype=np.int64)
        batch_size = int(prompt_lens_array.shape[0])
        out_counts = np.zeros((batch_size,), dtype=np.int64)
        out_codes = np.full((batch_size, max_new_tokens, num_code_groups), -1, dtype=np.int64)
        out_ttft_ms = np.full((batch_size,), -1.0, dtype=np.float64)
        out_last_token_ms = np.zeros((batch_size,), dtype=np.float64)
        elapsed_ms = ctypes.c_double(0.0)
        err = ctypes.c_char_p()
        self.reset_profile()
        common_args = (
            self.handle,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(flat.shape[0]),
            prompt_lens_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(hidden_size),
            tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(batch_size),
            ctypes.c_int64(max_new_tokens),
            ctypes.c_int64(min_new_tokens),
            ctypes.c_float(repetition_penalty),
            ctypes.c_int64(vocab_size),
            ctypes.c_int64(num_code_groups),
            ctypes.c_int64(eos_token_id),
            ctypes.c_int64(1 if do_sample else 0),
            ctypes.c_int64(top_k),
            ctypes.c_float(top_p),
            ctypes.c_float(temperature),
            ctypes.c_uint64(seed),
            out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        if return_codes:
            rc = self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch_codes(
                *common_args,
                out_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                out_ttft_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_last_token_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(elapsed_ms),
                ctypes.byref(err),
            )
        else:
            rc = self.lib.qwen3_tts_codegen_run_paged_kv_sequence_batch(
                *common_args,
                out_ttft_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_last_token_ms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(elapsed_ms),
                ctypes.byref(err),
            )
        self._check(rc, err)
        self.last_remote_embed = self.last_remote_embed_used()
        self.last_profile_json = self.profile_json()
        self.last_timing_json = self.timing_json()
        result = {
            "batch_size": batch_size,
            "prompt_lens": prompt_lens_array.copy(),
            "counts": out_counts.copy(),
            "ttft_ms": out_ttft_ms.copy(),
            "last_token_ms": out_last_token_ms.copy(),
            "elapsed_ms": float(elapsed_ms.value),
            "profile": self.last_profile_json,
            "timing": self.last_timing_json,
        }
        if return_codes:
            result["codes"] = out_codes.copy()
        return result

    def online_batch_reset(self, max_cache_blocks: int) -> None:
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_online_batch_reset(
            self.handle,
            ctypes.c_int64(int(max_cache_blocks)),
            ctypes.byref(err),
        )
        self._check(rc, err)

    def online_batch_add_sequence(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> int:
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        if sequence.ndim == 3 and sequence.shape[0] == 1:
            sequence = sequence.reshape(sequence.shape[1], sequence.shape[2])
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape [1, prompt_len, hidden] or [prompt_len, hidden]")
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        if tts_pad_embed.shape != (1, 1, sequence.shape[-1]):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")
        request_id = ctypes.c_int64(0)
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_online_batch_add_sequence(
            self.handle,
            sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(sequence.shape[0]),
            ctypes.c_int64(sequence.shape[1]),
            tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(max_new_tokens),
            ctypes.c_int64(min_new_tokens),
            ctypes.c_float(repetition_penalty),
            ctypes.c_int64(vocab_size),
            ctypes.c_int64(num_code_groups),
            ctypes.c_int64(eos_token_id),
            ctypes.c_int64(1 if do_sample else 0),
            ctypes.c_int64(top_k),
            ctypes.c_float(top_p),
            ctypes.c_float(temperature),
            ctypes.c_uint64(seed),
            ctypes.byref(request_id),
            ctypes.byref(err),
        )
        self._check(rc, err)
        return int(request_id.value)

    def online_batch_step(self, max_decode_batch: int, max_events: int, num_code_groups: int) -> dict:
        out_ids = np.zeros((max_events,), dtype=np.int64)
        out_kinds = np.zeros((max_events,), dtype=np.int64)
        out_codes = np.full((max_events, num_code_groups), -1, dtype=np.int64)
        out_count = ctypes.c_int64(0)
        elapsed_ms = ctypes.c_double(0.0)
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_online_batch_step(
            self.handle,
            ctypes.c_int64(int(max_decode_batch)),
            ctypes.c_int64(int(max_events)),
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(out_count),
            ctypes.byref(elapsed_ms),
            ctypes.byref(err),
        )
        self._check(rc, err)
        count = int(out_count.value)
        self.last_profile_json = self.profile_json()
        self.last_timing_json = self.timing_json()
        return {
            "ids": out_ids[:count].copy(),
            "kinds": out_kinds[:count].copy(),
            "codes": out_codes[:count].copy(),
            "elapsed_ms": float(elapsed_ms.value),
            "profile": self.last_profile_json,
            "timing": self.last_timing_json,
        }

    def online_batch_cancel(self, request_id: int) -> None:
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_online_batch_cancel(
            self.handle,
            ctypes.c_int64(int(request_id)),
            ctypes.byref(err),
        )
        self._check(rc, err)

    def online_batch_stats(self) -> dict:
        out = ctypes.c_char_p()
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_online_batch_get_stats_json(self.handle, ctypes.byref(out), ctypes.byref(err))
        self._check(rc, err)
        try:
            raw = out.value.decode("utf-8") if out.value else "{}"
            return json.loads(raw)
        finally:
            if out:
                self.lib.qwen3_tts_codegen_free_error(out)

    def set_stream_decoders(
        self,
        first_decoder_graph: Path,
        steady_decoder_graph: Path,
        decoder_device: str,
        cache_dir: Path | None = None,
        cache_mode: str = "OPTIMIZE_SPEED",
        first_context_frames: int = 0,
        first_chunk_frames: int = 8,
        steady_context_frames: int = 25,
        steady_chunk_frames: int = 12,
        num_code_groups: int = 16,
        decode_upsample_rate: int = 2000,
    ) -> None:
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_set_stream_decoders(
            self.handle,
            str(first_decoder_graph).encode("utf-8"),
            str(steady_decoder_graph).encode("utf-8"),
            str(decoder_device).encode("utf-8"),
            str(cache_dir or "").encode("utf-8"),
            str(cache_mode or "OPTIMIZE_SPEED").encode("utf-8"),
            ctypes.c_int64(first_context_frames),
            ctypes.c_int64(first_chunk_frames),
            ctypes.c_int64(steady_context_frames),
            ctypes.c_int64(steady_chunk_frames),
            ctypes.c_int64(num_code_groups),
            ctypes.c_int64(decode_upsample_rate),
            ctypes.byref(err),
        )
        self._check(rc, err)

    def configure_voice_design_prompt(
        self,
        tokenizer_dir: Path,
        text_embedding_graph: Path,
        codec_embedding_graph: Path,
        device: str,
        ids: dict,
        cache_dir: Path | None = None,
        cache_mode: str = "OPTIMIZE_SPEED",
    ) -> None:
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_configure_voice_design_prompt(
            self.handle,
            str(tokenizer_dir).encode("utf-8"),
            str(text_embedding_graph).encode("utf-8"),
            str(codec_embedding_graph).encode("utf-8"),
            str(device).encode("utf-8"),
            str(cache_dir or "").encode("utf-8"),
            str(cache_mode or "OPTIMIZE_SPEED").encode("utf-8"),
            ctypes.c_int64(int(ids["tts_bos_token_id"])),
            ctypes.c_int64(int(ids["tts_eos_token_id"])),
            ctypes.c_int64(int(ids["tts_pad_token_id"])),
            ctypes.c_int64(int(ids["codec_pad_id"])),
            ctypes.c_int64(int(ids["codec_bos_id"])),
            ctypes.byref(err),
        )
        self._check(rc, err)

    def iter_batches(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ):
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError("sequence must have shape [1, prompt_len, hidden]")
        if tts_pad_embed.shape != (1, 1, sequence.shape[-1]):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")

        out_queue: queue.Queue[object] = queue.Queue()
        self.reset_profile()

        def callback(ptr, num_frames, num_groups, _user_data):
            flat = np.ctypeslib.as_array(ptr, shape=(int(num_frames) * int(num_groups),))
            batch = flat.copy().reshape(int(num_frames), int(num_groups))
            out_queue.put(batch)
            return 0

        c_callback = self._frame_callback_type(callback)

        def worker():
            out_count = ctypes.c_int64(0)
            elapsed_ms = ctypes.c_double(0.0)
            err = ctypes.c_char_p()
            try:
                rc = self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_stream(
                    self.handle,
                    sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(sequence.shape[1]),
                    ctypes.c_int64(sequence.shape[2]),
                    tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(max_new_tokens),
                    ctypes.c_int64(min_new_tokens),
                    ctypes.c_float(repetition_penalty),
                    ctypes.c_int64(vocab_size),
                    ctypes.c_int64(num_code_groups),
                    ctypes.c_int64(eos_token_id),
                    ctypes.c_int64(1 if do_sample else 0),
                    ctypes.c_int64(top_k),
                    ctypes.c_float(top_p),
                    ctypes.c_float(temperature),
                    ctypes.c_uint64(seed),
                    c_callback,
                    None,
                    ctypes.byref(out_count),
                    ctypes.byref(elapsed_ms),
                    ctypes.byref(err),
                )
                self._check(rc, err)
                remote_embed = self.last_remote_embed_used()
                profile = self.profile_json()
                timing = self.timing_json()
                out_queue.put(("done", int(out_count.value), float(elapsed_ms.value), remote_embed, profile, timing))
            except BaseException as exc:
                out_queue.put(exc)

        thread = threading.Thread(target=worker, name="qwen3-tts-native-codegen", daemon=True)
        thread.start()
        while True:
            item = out_queue.get()
            if isinstance(item, BaseException):
                thread.join(timeout=0.1)
                raise item
            if isinstance(item, tuple) and item and item[0] == "done":
                self.last_stream_count = int(item[1])
                self.last_stream_elapsed_ms = float(item[2])
                self.last_remote_embed = bool(item[3]) if len(item) > 3 else False
                self.last_profile_json = item[4] if len(item) > 4 else None
                self.last_timing_json = item[5] if len(item) > 5 else None
                thread.join(timeout=0.1)
                return
            yield item

    def iter_audio_chunks(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        prefix_codes: np.ndarray | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ):
        sequence = np.ascontiguousarray(sequence, dtype=np.float32)
        tts_pad_embed = np.ascontiguousarray(tts_pad_embed, dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError("sequence must have shape [1, prompt_len, hidden]")
        if tts_pad_embed.shape != (1, 1, sequence.shape[-1]):
            raise ValueError("tts_pad_embed must have shape [1, 1, hidden]")
        if prefix_codes is None:
            prefix = np.empty((0, int(num_code_groups)), dtype=np.int64)
        else:
            prefix = np.ascontiguousarray(prefix_codes, dtype=np.int64).reshape(-1, int(num_code_groups))

        out_queue: queue.Queue[object] = queue.Queue()
        self.reset_profile()

        def callback(audio_ptr, num_samples, codes_ptr, num_frames, num_groups, is_final, codegen_ms, decode_ms, _user_data):
            if int(num_samples) > 0:
                audio_view = np.ctypeslib.as_array(audio_ptr, shape=(int(num_samples),))
                pcm_started = time.perf_counter()
                pcm_s16le = float_audio_view_to_pcm16_bytes(audio_view)
                pcm_convert_ms = (time.perf_counter() - pcm_started) * 1000.0
                audio = audio_view.copy()
            else:
                audio = np.zeros((0,), dtype=np.float32)
                pcm_s16le = b""
                pcm_convert_ms = 0.0
            if int(num_frames) > 0:
                flat_codes = np.ctypeslib.as_array(codes_ptr, shape=(int(num_frames) * int(num_groups),))
                codes = flat_codes.copy().reshape(int(num_frames), int(num_groups))
            else:
                codes = np.empty((0, int(num_groups)), dtype=np.int64)
            remote_embed = self.last_remote_embed_used()
            self.last_remote_embed = remote_embed
            out_queue.put(
                {
                    "audio": audio,
                    "codes": codes,
                    "is_final": bool(is_final),
                    "codegen_ms": float(codegen_ms),
                    "decode_ms": float(decode_ms),
                    "remote_embed": remote_embed,
                    "pcm_s16le": pcm_s16le,
                    "pcm_convert_ms": pcm_convert_ms,
                }
            )
            return 0

        c_callback = self._audio_callback_type(callback)

        def worker():
            out_count = ctypes.c_int64(0)
            elapsed_ms = ctypes.c_double(0.0)
            err = ctypes.c_char_p()
            try:
                rc = self.lib.qwen3_tts_codegen_run_unroll4_statefulmask_audio_stream(
                    self.handle,
                    sequence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(sequence.shape[1]),
                    ctypes.c_int64(sequence.shape[2]),
                    tts_pad_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int64(max_new_tokens),
                    ctypes.c_int64(min_new_tokens),
                    ctypes.c_float(repetition_penalty),
                    ctypes.c_int64(vocab_size),
                    ctypes.c_int64(num_code_groups),
                    ctypes.c_int64(eos_token_id),
                    ctypes.c_int64(1 if do_sample else 0),
                    ctypes.c_int64(top_k),
                    ctypes.c_float(top_p),
                    ctypes.c_float(temperature),
                    ctypes.c_uint64(seed),
                    prefix.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(prefix.shape[0]),
                    c_callback,
                    None,
                    ctypes.byref(out_count),
                    ctypes.byref(elapsed_ms),
                    ctypes.byref(err),
                )
                self._check(rc, err)
                remote_embed = self.last_remote_embed_used()
                profile = self.profile_json()
                timing = self.timing_json()
                out_queue.put(("done", int(out_count.value), float(elapsed_ms.value), remote_embed, profile, timing))
            except BaseException as exc:
                out_queue.put(exc)
            finally:
                if os.environ.get("QWEN3_TTS_OV_NATIVE_RELEASE_RUN_BUFFERS_AFTER_RUN", "0").strip().lower() in {
                    "1",
                    "true",
                    "on",
                    "yes",
                }:
                    try:
                        self.release_run_buffers()
                    except BaseException as exc:
                        out_queue.put(exc)

        thread = threading.Thread(target=worker, name="qwen3-tts-native-audio-pipeline", daemon=True)
        thread.start()
        pending_final = None
        while True:
            item = out_queue.get()
            if isinstance(item, BaseException):
                thread.join(timeout=0.1)
                raise item
            if isinstance(item, tuple) and item and item[0] == "done":
                self.last_audio_stream_count = int(item[1])
                self.last_audio_stream_elapsed_ms = float(item[2])
                self.last_remote_embed = bool(item[3]) if len(item) > 3 else False
                self.last_profile_json = item[4] if len(item) > 4 else None
                self.last_timing_json = item[5] if len(item) > 5 else None
                thread.join(timeout=0.1)
                if pending_final is not None:
                    if self.last_profile_json is not None:
                        pending_final["native_ov_profile"] = self.last_profile_json
                    if self.last_timing_json is not None:
                        pending_final["native_timing"] = self.last_timing_json
                    yield pending_final
                return
            if isinstance(item, dict) and item.get("is_final"):
                pending_final = item
                continue
            yield item

    def iter_voice_design_audio_chunks(
        self,
        text: str,
        instruct: str,
        codec_prefill: list[int] | np.ndarray,
        max_prompt_tokens: int,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        vocab_size: int,
        num_code_groups: int,
        eos_token_id: int,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ):
        codec_prefill_array = np.ascontiguousarray(codec_prefill, dtype=np.int64).reshape(-1)
        if codec_prefill_array.size == 0:
            raise ValueError("codec_prefill must not be empty")

        out_queue: queue.Queue[object] = queue.Queue()
        self.reset_profile()

        def callback(audio_ptr, num_samples, codes_ptr, num_frames, num_groups, is_final, codegen_ms, decode_ms, _user_data):
            if int(num_samples) > 0:
                audio_view = np.ctypeslib.as_array(audio_ptr, shape=(int(num_samples),))
                pcm_started = time.perf_counter()
                pcm_s16le = float_audio_view_to_pcm16_bytes(audio_view)
                pcm_convert_ms = (time.perf_counter() - pcm_started) * 1000.0
                audio = audio_view.copy()
            else:
                audio = np.zeros((0,), dtype=np.float32)
                pcm_s16le = b""
                pcm_convert_ms = 0.0
            if int(num_frames) > 0:
                flat_codes = np.ctypeslib.as_array(codes_ptr, shape=(int(num_frames) * int(num_groups),))
                codes = flat_codes.copy().reshape(int(num_frames), int(num_groups))
            else:
                codes = np.empty((0, int(num_groups)), dtype=np.int64)
            remote_embed = self.last_remote_embed_used()
            self.last_remote_embed = remote_embed
            out_queue.put(
                {
                    "audio": audio,
                    "codes": codes,
                    "is_final": bool(is_final),
                    "codegen_ms": float(codegen_ms),
                    "decode_ms": float(decode_ms),
                    "remote_embed": remote_embed,
                    "pcm_s16le": pcm_s16le,
                    "pcm_convert_ms": pcm_convert_ms,
                }
            )
            return 0

        c_callback = self._audio_callback_type(callback)

        def worker():
            out_count = ctypes.c_int64(0)
            elapsed_ms = ctypes.c_double(0.0)
            err = ctypes.c_char_p()
            try:
                rc = self.lib.qwen3_tts_codegen_run_voice_design_audio_stream(
                    self.handle,
                    str(text).encode("utf-8"),
                    str(instruct or "").encode("utf-8"),
                    codec_prefill_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(codec_prefill_array.size),
                    ctypes.c_int64(max_prompt_tokens),
                    ctypes.c_int64(max_new_tokens),
                    ctypes.c_int64(min_new_tokens),
                    ctypes.c_float(repetition_penalty),
                    ctypes.c_int64(vocab_size),
                    ctypes.c_int64(num_code_groups),
                    ctypes.c_int64(eos_token_id),
                    ctypes.c_int64(1 if do_sample else 0),
                    ctypes.c_int64(top_k),
                    ctypes.c_float(top_p),
                    ctypes.c_float(temperature),
                    ctypes.c_uint64(seed),
                    c_callback,
                    None,
                    ctypes.byref(out_count),
                    ctypes.byref(elapsed_ms),
                    ctypes.byref(err),
                )
                self._check(rc, err)
                remote_embed = self.last_remote_embed_used()
                profile = self.profile_json()
                timing = self.timing_json()
                out_queue.put(("done", int(out_count.value), float(elapsed_ms.value), remote_embed, profile, timing))
            except BaseException as exc:
                out_queue.put(exc)
            finally:
                if os.environ.get("QWEN3_TTS_OV_NATIVE_RELEASE_RUN_BUFFERS_AFTER_RUN", "0").strip().lower() in {
                    "1",
                    "true",
                    "on",
                    "yes",
                }:
                    try:
                        self.release_run_buffers()
                    except BaseException as exc:
                        out_queue.put(exc)

        thread = threading.Thread(target=worker, name="qwen3-tts-native-voice-design-pipeline", daemon=True)
        thread.start()
        pending_final = None
        while True:
            item = out_queue.get()
            if isinstance(item, BaseException):
                thread.join(timeout=0.1)
                raise item
            if isinstance(item, tuple) and item and item[0] == "done":
                self.last_audio_stream_count = int(item[1])
                self.last_audio_stream_elapsed_ms = float(item[2])
                self.last_remote_embed = bool(item[3]) if len(item) > 3 else False
                self.last_profile_json = item[4] if len(item) > 4 else None
                self.last_timing_json = item[5] if len(item) > 5 else None
                thread.join(timeout=0.1)
                if pending_final is not None:
                    if self.last_profile_json is not None:
                        pending_final["native_ov_profile"] = self.last_profile_json
                    if self.last_timing_json is not None:
                        pending_final["native_timing"] = self.last_timing_json
                    yield pending_final
                return
            if isinstance(item, dict) and item.get("is_final"):
                pending_final = item
                continue
            yield item

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        err = ctypes.c_char_p()
        rc = self.lib.qwen3_tts_codegen_destroy(self.handle, ctypes.byref(err))
        self.closed = True
        self._check(rc, err)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
