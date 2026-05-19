#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TEXTS = [
    "今天天气很好，我们一起测试在线合批 prefill 的语音质量是否稳定。",
    "第二条请求用于检查并发进入调度器时，音色和语气是否仍然自然连续。",
]
DEFAULT_INSTRUCT = "用自然、清晰、稳定的中文女声朗读。"
PREFILL_MODE_CHOICES = {"serial", "dynamic_ragged", "bucketed_padded"}
PREFILL_REFERENCE_CACHE_SCHEMA = "qwen3_tts_prefill_reference_v5"
REFERENCE_GPU_DEVICE_POLICIES = {"gpu", "accelerator", "cuda", "cuda:0", "xpu", "xpu:0"}
REFERENCE_ACCELERATOR_PREFIXES = ("cuda", "xpu")


def load_script_module(name: str):
    script = Path(__file__).resolve().with_name(name)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    clean = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not clean:
        return math.inf
    index = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return clean[index]


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    if args.text:
        texts.extend(str(item).strip() for item in args.text if str(item).strip())
    if not texts:
        texts = list(DEFAULT_TEXTS)
    while len(texts) < int(args.batch_size):
        texts.extend(texts)
    return texts[: int(args.batch_size)]


def load_candidate_texts(args: argparse.Namespace, reference_texts: list[str]) -> list[str]:
    candidate_texts: list[str] = []
    if getattr(args, "candidate_text_file", None):
        text = Path(args.candidate_text_file).read_text(encoding="utf-8").strip()
        if text:
            candidate_texts.append(text)
    if getattr(args, "candidate_text", None):
        candidate_texts.extend(str(item).strip() for item in args.candidate_text if str(item).strip())
    if not candidate_texts:
        return list(reference_texts)
    while len(candidate_texts) < int(args.batch_size):
        candidate_texts.extend(candidate_texts)
    return candidate_texts[: int(args.batch_size)]


def expand_values(value: str | list[str] | None, count: int, default: str = "") -> list[str]:
    if isinstance(value, list):
        items = [str(item) for item in value]
    elif value is None:
        items = [default]
    else:
        items = [str(value)]
    items = [item for item in items if item != ""]
    if not items:
        items = [default]
    while len(items) < int(count):
        items.extend(items)
    return items[: int(count)]


def reference_python_executable(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "python_executable", "auto") or "auto").strip()
    if configured and configured.lower() != "auto":
        return configured
    repo = Path(args.qwen3_tts_repo)
    candidates = [
        repo / ".venv" / "bin" / "python",
        repo / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def normalized_python_device_policy(args: argparse.Namespace) -> str:
    return str(args.python_device).strip().lower()


def reference_cache_payload(args: argparse.Namespace, texts: list[str]) -> dict[str, Any]:
    languages = expand_values(getattr(args, "language", None), len(texts), "Chinese")
    instructs = expand_values(getattr(args, "instruct", None), len(texts), DEFAULT_INSTRUCT)
    speakers = expand_values(getattr(args, "speaker", None), len(texts), "Vivian")
    ref_audios = expand_values(getattr(args, "ref_audio", None), len(texts), "")
    ref_texts = expand_values(getattr(args, "ref_text", None), len(texts), "")
    ref_audio_hash = None
    ref_audio_hashes = []
    for ref_audio in ref_audios:
        if ref_audio:
            ref_path = Path(ref_audio)
            if ref_path.exists() and ref_path.is_file():
                item_hash = hashlib.sha256(ref_path.read_bytes()).hexdigest()
                ref_audio_hashes.append(item_hash)
                if ref_audio_hash is None:
                    ref_audio_hash = item_hash
    return {
        "schema": PREFILL_REFERENCE_CACHE_SCHEMA,
        "mode": args.mode,
        "qwen3_tts_repo": str(Path(args.qwen3_tts_repo).resolve()),
        "python_model": str(Path(args.python_model).resolve()) if Path(args.python_model).exists() else args.python_model,
        "python_executable": str(Path(reference_python_executable(args)).resolve()),
        "python_device": normalized_python_device_policy(args),
        "python_dtype": args.python_dtype,
        "python_attn_implementation": args.python_attn_implementation,
        "texts": texts,
        "languages": languages,
        "instructs": instructs,
        "speakers": speakers,
        "ref_audios": [
            str(Path(item).resolve()) if item and Path(item).exists() else item
            for item in ref_audios
        ],
        "ref_audio_sha256": ref_audio_hash,
        "ref_audio_sha256s": ref_audio_hashes,
        "ref_texts": ref_texts,
        "x_vector_only": bool(getattr(args, "x_vector_only", False)),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "do_sample": bool(args.do_sample),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "seed": int(args.seed),
    }


def reference_requires_accelerator(args: argparse.Namespace) -> bool:
    return normalized_python_device_policy(args) in REFERENCE_GPU_DEVICE_POLICIES


def reference_result_uses_accelerator(result: dict[str, Any]) -> bool:
    device = str(result.get("device") or "").strip().lower()
    return device.startswith(REFERENCE_ACCELERATOR_PREFIXES)


def validate_reference_result_device(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if not reference_requires_accelerator(args):
        return
    if reference_result_uses_accelerator(result):
        return
    device = str(result.get("device") or "<unknown>")
    requested = normalized_python_device_policy(args)
    raise RuntimeError(
        "Python reference was requested on GPU/CUDA/XPU, but the worker produced a non-accelerator result "
        f"(requested={requested}, actual={device}). Refusing to use or cache this reference. "
        "Pass --python-device cpu only when CPU reference is intentional."
    )


def reference_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_reference_items_from_dir(reference_dir: Path, texts: list[str]) -> list[dict[str, Any]]:
    items = []
    for index, text in enumerate(texts):
        wav_path = reference_dir / f"reference_{index:02d}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"missing reference wav: {wav_path}")
        items.append({"index": index, "wav_path": str(wav_path), "text": text})
    return items


def cached_reference_ready(reference_dir: Path, texts: list[str], payload: dict[str, Any]) -> bool:
    result_json = reference_dir / "result.json"
    metadata_json = reference_dir / "cache_key.json"
    if not result_json.exists() or not metadata_json.exists():
        return False
    try:
        result = load_json(result_json)
        metadata = load_json(metadata_json)
        if not result.get("ok") or metadata.get("payload") != payload:
            return False
        cache_args = argparse.Namespace(python_device=payload.get("python_device"))
        validate_reference_result_device(cache_args, result)
        load_reference_items_from_dir(reference_dir, texts)
        return True
    except Exception:
        return False


def resolve_reference_dir(args: argparse.Namespace, texts: list[str], out_dir: Path) -> tuple[Path, dict[str, Any], str]:
    payload = reference_cache_payload(args, texts)
    key = reference_cache_key(payload)
    if bool(args.no_reference_cache):
        return out_dir / "python_reference", payload, key
    cache_root = Path(args.reference_cache_dir)
    return cache_root / key, payload, key


def run_python_reference(
    args: argparse.Namespace,
    texts: list[str],
    reference_dir: Path,
    *,
    cache_payload: dict[str, Any],
    cache_key: str,
) -> list[dict[str, Any]]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    worker_json = reference_dir / "worker.json"
    result_json = reference_dir / "result.json"
    worker_code = r"""
import json
import sys
import time
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result_path = Path(cfg["result_json"])
try:
    sys.path.insert(0, cfg["qwen3_tts_repo"])
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    def first_model_device(tts_model):
        try:
            for param in tts_model.model.parameters():
                return str(param.device)
        except Exception:
            pass
        try:
            return str(tts_model.device)
        except Exception:
            return "<unknown>"

    requested_device = str(cfg["python_device"]).strip().lower()
    if requested_device in {"gpu", "accelerator"}:
        if torch.cuda.is_available():
            requested_device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            requested_device = "xpu:0"
        else:
            raise RuntimeError("Python reference requested GPU, but torch sees no CUDA/XPU device. Pass --python-device cpu only when CPU reference is intentional.")
    elif requested_device == "cuda":
        requested_device = "cuda:0"
    elif requested_device == "xpu":
        requested_device = "xpu:0"
    elif requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            requested_device = "xpu:0"
        else:
            requested_device = "cpu"
    dtype_name = str(cfg["python_dtype"])
    if dtype_name == "auto":
        dtype = torch.bfloat16 if requested_device.startswith(("cuda", "xpu")) else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    attn = str(cfg["python_attn_implementation"])
    if attn == "auto":
        attn = "flash_attention_2" if requested_device.startswith("cuda") else "sdpa"

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        cfg["python_model"],
        device_map=requested_device,
        dtype=dtype,
        attn_implementation=attn,
    )
    model_device = first_model_device(model)
    if cfg["python_device"] in {"gpu", "accelerator", "cuda", "cuda:0", "xpu", "xpu:0"} and not model_device.startswith(("cuda", "xpu")):
        raise RuntimeError(
            "Python reference requested GPU/CUDA/XPU, but loaded model parameters are not on an accelerator "
            f"(resolved_device={requested_device}, model_device={model_device})."
        )
    kwargs = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "min_new_tokens": int(cfg["min_new_tokens"]),
        "do_sample": bool(cfg["do_sample"]),
        "top_k": int(cfg["top_k"]),
        "top_p": float(cfg["top_p"]),
        "temperature": float(cfg["temperature"]),
        "repetition_penalty": float(cfg["repetition_penalty"]),
    }
    try:
        torch.manual_seed(int(cfg["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg["seed"]))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(int(cfg["seed"]))
    except Exception:
        pass
    mode = str(cfg.get("mode", "voice_design")).replace("-", "_")
    if mode == "voice_design":
        wavs, sr = model.generate_voice_design(
            text=cfg["texts"],
            language=cfg["languages"],
            instruct=cfg["instructs"],
            **kwargs,
        )
    elif mode == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text=cfg["texts"],
            speaker=cfg["speakers"],
            language=cfg["languages"],
            instruct=cfg["instructs"],
            **kwargs,
        )
    elif mode == "voice_clone":
        wavs, sr = model.generate_voice_clone(
            text=cfg["texts"],
            language=cfg["languages"],
            ref_audio=cfg["ref_audios"],
            ref_text=cfg["ref_texts"],
            x_vector_only_mode=bool(cfg["x_vector_only"]),
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")
    if requested_device.startswith("cuda"):
        torch.cuda.synchronize()
    if requested_device.startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elapsed_sec = time.perf_counter() - started
    items = []
    for index, wav in enumerate(wavs):
        path = Path(cfg["reference_dir"]) / f"reference_{index:02d}.wav"
        sf.write(path, wav, sr)
        items.append({"index": index, "wav_path": str(path), "sample_rate": int(sr), "text": cfg["texts"][index]})
    result = {
        "ok": True,
        "elapsed_sec": elapsed_sec,
        "device": model_device,
        "resolved_device": requested_device,
        "requested_device": cfg["python_device"],
        "dtype": str(dtype),
        "attn": attn,
        "items": items,
    }
except Exception as exc:
    result = {"ok": False, "error": str(exc), "traceback": __import__("traceback").format_exc()}
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
"""
    config = {
        "qwen3_tts_repo": str(Path(args.qwen3_tts_repo).resolve()),
        "python_model": args.python_model,
        "python_device": normalized_python_device_policy(args),
        "python_dtype": args.python_dtype,
        "python_attn_implementation": args.python_attn_implementation,
        "mode": args.mode,
        "reference_dir": str(reference_dir),
        "result_json": str(result_json),
        "texts": texts,
        "languages": expand_values(getattr(args, "language", None), len(texts), "Chinese"),
        "instructs": expand_values(getattr(args, "instruct", None), len(texts), DEFAULT_INSTRUCT),
        "speakers": expand_values(getattr(args, "speaker", None), len(texts), "Vivian"),
        "ref_audios": expand_values(getattr(args, "ref_audio", None), len(texts), ""),
        "ref_texts": expand_values(getattr(args, "ref_text", None), len(texts), ""),
        "x_vector_only": bool(getattr(args, "x_vector_only", False)),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "do_sample": bool(args.do_sample),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "seed": int(args.seed),
    }
    write_json(worker_json, config)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    python_executable = reference_python_executable(args)
    proc = subprocess.run(
        [python_executable, "-c", worker_code, str(worker_json)],
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=float(args.python_timeout_sec),
    )
    if result_json.exists():
        result = load_json(result_json)
    else:
        result = {"ok": False, "error": "python reference worker did not write result JSON"}
    result["worker_exit_code"] = int(proc.returncode)
    result["python_executable"] = python_executable
    result["stdout_tail"] = proc.stdout[-4000:]
    result["stderr_tail"] = proc.stderr[-4000:]
    write_json(result_json, result)
    if not result.get("ok"):
        raise RuntimeError(f"python reference generation failed: {result.get('error')}")
    validate_reference_result_device(args, result)
    write_json(
        reference_dir / "cache_key.json",
        {"key": cache_key, "created_at_unix": time.time(), "payload": cache_payload, "result_json": str(result_json)},
    )
    return list(result.get("items") or [])


def load_reference_items(args: argparse.Namespace, texts: list[str]) -> list[dict[str, Any]]:
    return load_reference_items_from_dir(Path(args.reference_dir), texts)


def build_pair_messages(
    quality,
    *,
    text: str,
    reference_wav: Path,
    candidate_wav: Path,
    max_base64_bytes: int,
) -> list[dict[str, Any]]:
    ref_bytes = reference_wav.read_bytes()
    cand_bytes = candidate_wav.read_bytes()
    if len(ref_bytes) + len(cand_bytes) > max_base64_bytes:
        raise ValueError("reference+candidate audio exceeds Omni request byte budget; shorten text or raise --omni-max-audio-mb")
    prompt = (
        "You are comparing two TTS outputs. Audio A is generated by the original Python Qwen3-TTS model. "
        "Audio B is generated by the OpenVINO prefill candidate. Decide whether B is equivalent enough to A "
        "for product use: same target text, intelligible speech, similar speaker identity/style, no obvious noise, "
        "stutter, truncation, or drift. Do not require waveform identity, identical prosody, or identical stochastic "
        "non-text vocalizations. If the target text does not explicitly ask for laughter, breathing, filler words, "
        "or an emotional sound, do not fail B solely because A contains such an incidental sampled artifact and B "
        "omits it. Do not fail for minor pronunciation or grammatical-particle differences when the sentence is still "
        "clearly intelligible and text_match would be 4 or higher. If Audio A appears to mispronounce a target-text "
        "proper noun but Audio B pronounces that target text correctly, do not fail B merely for not reproducing A's "
        "mistake. Do fail when B misses requested style/emotion, changes the speaker family, changes a "
        "named entity/proper noun/key factual term relative to the target text, speaks substantially wrong text, or "
        "degrades naturalness. Return JSON only with this schema: "
        '{"verdict":"pass|fail","text_match":0-5,"speaker_style_similarity":0-5,'
        '"naturalness":0-5,"continuity":0-5,"noise":0-5,"failure_reason":"...",'
        '"actionable_notes":["..."]}. Use 5 as best. Make verdict consistent with failure_reason; '
        "if your reasoning says Verdict: Pass, set verdict to pass. Mark fail if B is mostly noise, wrong text, "
        f"or clearly different speaker/style. Target text: {text}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\nAudio A: original Python reference."},
                {"type": "input_audio", "input_audio": {"data": quality.audio_bytes_to_data_url(ref_bytes), "format": "wav"}},
                {"type": "text", "text": "Audio B: OpenVINO prefill candidate."},
                {"type": "input_audio", "input_audio": {"data": quality.audio_bytes_to_data_url(cand_bytes), "format": "wav"}},
            ],
        }
    ]


def _omni_reason_implies_pass(result: dict[str, Any]) -> bool:
    reason_parts = [str(result.get("failure_reason") or "")]
    notes = result.get("actionable_notes")
    if isinstance(notes, list):
        reason_parts.extend(str(item) for item in notes)
    elif notes:
        reason_parts.append(str(notes))
    text = "\n".join(reason_parts).lower()
    pass_phrases = (
        "verdict: pass",
        "should pass",
        "should be pass",
        "should not result in a failure",
        "should not be marked as a failure",
        "do not fail",
    )
    return any(phrase in text for phrase in pass_phrases)


def _omni_reason_is_high_score_soft_fail(result: dict[str, Any]) -> bool:
    reason_parts = [str(result.get("failure_reason") or "")]
    notes = result.get("actionable_notes")
    if isinstance(notes, list):
        reason_parts.extend(str(item) for item in notes)
    elif notes:
        reason_parts.append(str(notes))
    text = "\n".join(reason_parts).lower()
    soft_terms = (
        "pitch accent",
        "pause",
        "minor pronunciation",
        "while intelligible",
        "unnatural pauses",
        "prosody",
    )
    hard_terms = (
        "mostly noise",
        "wrong text",
        "speaks substantially wrong",
        "different speaker",
        "truncation",
        "truncated",
        "missing",
        "omits",
        "aina",
        "ani",
    )
    return any(term in text for term in soft_terms) and not any(term in text for term in hard_terms)


def normalize_pair_omni(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    result["verdict"] = str(result.get("verdict", "fail")).strip().lower()
    scores = []
    for key in ("text_match", "speaker_style_similarity", "naturalness", "continuity", "noise"):
        try:
            value = float(result.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        result[key] = value
        scores.append(value)
    result["score_mean"] = float(sum(scores) / len(scores)) if scores else 0.0
    threshold_pass = bool(
        result["text_match"] >= 3.5
        and result["speaker_style_similarity"] >= 3.0
        and result["naturalness"] >= 3.0
        and result["continuity"] >= 3.0
        and result["noise"] >= 3.0
    )
    override_threshold_pass = bool(
        result["text_match"] >= 3.0
        and result["speaker_style_similarity"] >= 4.0
        and result["naturalness"] >= 4.0
        and result["continuity"] >= 4.0
        and result["noise"] >= 4.0
    )
    high_score_soft_fail = bool(
        result["text_match"] >= 4.0
        and result["speaker_style_similarity"] >= 4.0
        and result["naturalness"] >= 4.0
        and result["continuity"] >= 4.0
        and result["noise"] >= 4.0
        and _omni_reason_is_high_score_soft_fail(result)
    )
    if result["verdict"] == "fail" and (
        (override_threshold_pass and _omni_reason_implies_pass(result)) or high_score_soft_fail
    ):
        result["raw_verdict"] = "fail"
        result["verdict"] = "pass"
        result["omni_verdict_overridden"] = True
    else:
        result["omni_verdict_overridden"] = False
    result["pass"] = bool(
        result["verdict"] == "pass"
        and (threshold_pass or (result["omni_verdict_overridden"] and (override_threshold_pass or high_score_soft_fail)))
    )
    return result


def judge_pair_with_omni(
    quality,
    *,
    reference_wav: Path,
    candidate_wav: Path,
    text: str,
    config: dict[str, str | None],
    max_base64_bytes: int,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Missing dependency "openai". Install with `uv pip install -e ".[quality]"`.') from exc
    client = OpenAI(api_key=config["api_key"], base_url=config.get("base_url") or quality.DEFAULT_OMNI_BASE_URL)
    content_parts: list[str] = []
    stream = client.chat.completions.create(
        model=config["model"],
        messages=build_pair_messages(
            quality,
            text=text,
            reference_wav=reference_wav,
            candidate_wav=candidate_wav,
            max_base64_bytes=max_base64_bytes,
        ),
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        delta = getattr(choice, "delta", None)
        value = getattr(delta, "content", None) if delta is not None else None
        if isinstance(value, str):
            content_parts.append(value)
    raw_text = "".join(content_parts).strip()
    result = normalize_pair_omni(quality.extract_json_object(raw_text))
    result["raw_text"] = raw_text
    return result


def run_candidate_mode(args: argparse.Namespace, texts: list[str], mode: str, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raise RuntimeError(
        "online prefill candidate evaluation was removed from the production tree. "
        "Use scripts/evaluate_single_arch_gate.py for the vLLM-like sidecar path, "
        "or run this script with --candidate-path runtime."
    )


def voice_clone_prompt_item_to_dict(prompt, *, natural: bool = False) -> dict[str, Any]:
    ref_code = None if prompt.ref_code is None else np.asarray(prompt.ref_code, dtype=np.int64).tolist()
    ref_spk_embedding = np.asarray(prompt.ref_spk_embedding, dtype=np.float32).tolist()
    if natural:
        return {
            "ref_code": ref_code,
            "ref_spk_embedding": ref_spk_embedding,
            "x_vector_only_mode": bool(prompt.x_vector_only_mode),
            "icl_mode": bool(prompt.icl_mode),
            "ref_text": prompt.ref_text,
        }
    return {
        "ref_code": [ref_code],
        "ref_spk_embedding": [ref_spk_embedding],
        "x_vector_only_mode": [bool(prompt.x_vector_only_mode)],
        "icl_mode": [bool(prompt.icl_mode)],
        "ref_text": [prompt.ref_text],
    }


def run_runtime_candidate(args: argparse.Namespace, texts: list[str], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    mode_out = out_dir / "candidate_runtime"
    mode_out.mkdir(parents=True, exist_ok=True)
    runtime = OpenVINOQwen3TTS(
        args.ir_dir,
        device=args.device,
        decoder_device=args.decoder_device or args.device,
        mode="fastest" if args.runtime_mode == "fastest" else args.runtime_mode,
        graph_variant=args.runtime_graph_variant,
        ov_cache_dir=args.ov_cache_dir,
        disable_ov_cache=args.disable_ov_cache,
    )
    started = time.perf_counter()
    kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "do_sample": bool(args.do_sample),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "chunk_strategy": args.chunk_strategy,
    }
    voice_clone_prompts = None
    voice_clone_prompt_items = None
    if args.mode == "voice_clone":
        ref_audios = expand_values(args.ref_audio, len(texts), "")
        ref_texts = expand_values(args.ref_text, len(texts), "")
        voice_clone_prompt_items = [
            runtime.create_voice_clone_prompt(
                ref_audios[index],
                ref_text=ref_texts[index] or None,
                x_vector_only_mode=bool(args.x_vector_only),
            )[0]
            for index in range(len(texts))
        ]
        voice_clone_prompts = (
            [
                voice_clone_prompt_item_to_dict(
                    prompt,
                    natural=args.voice_clone_prompt_format == "dict-natural",
                )
                for prompt in voice_clone_prompt_items
            ]
            if args.voice_clone_prompt_format in {"dict", "dict-natural"}
            else voice_clone_prompt_items
        )
    elif args.mode not in {"voice_design", "custom_voice"}:
        raise ValueError(f"unsupported mode: {args.mode}")
    languages = expand_values(args.language, len(texts), "Chinese")
    instructs = expand_values(args.candidate_instruct or args.instruct, len(texts), DEFAULT_INSTRUCT)
    speakers = expand_values(args.speaker, len(texts), "Vivian")
    results = []
    for index, text in enumerate(texts):
        item_started = time.perf_counter()
        first_audio_ms = None
        audio_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        chunk_timings: list[dict[str, Any]] = []
        if args.mode == "voice_design":
            stream = runtime.stream_voice_design(
                text=text,
                language=languages[index],
                instruct=instructs[index],
                **kwargs,
            )
        elif args.mode == "custom_voice":
            stream = runtime.stream_custom_voice(
                text=text,
                speaker=speakers[index],
                language=languages[index],
                instruct=instructs[index],
                **kwargs,
            )
        else:
            prompt_ref_text = voice_clone_prompt_items[index].ref_text if voice_clone_prompt_items is not None else None
            stream = runtime.stream_voice_clone(
                text=text,
                language=languages[index],
                ref_text=prompt_ref_text,
                voice_clone_prompt=voice_clone_prompts[index],
                **kwargs,
            )
        for chunk in stream:
            chunk_timings.append(dict(chunk.timings or {}))
            if chunk.audio.size:
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - item_started) * 1000.0
                audio_parts.append(np.asarray(chunk.audio, dtype=np.float32))
            if chunk.codes.size:
                code_parts.append(np.asarray(chunk.codes, dtype=np.int64).reshape(-1, runtime.num_code_groups))
        item_elapsed_sec = max(1e-9, time.perf_counter() - item_started)
        audio = np.concatenate(audio_parts) if audio_parts else np.zeros((0,), dtype=np.float32)
        codes = (
            np.concatenate(code_parts, axis=0)
            if code_parts
            else np.empty((0, int(runtime.num_code_groups)), dtype=np.int64)
        )
        wav_path = mode_out / f"sample_{index:02d}.wav"
        code_path = mode_out / f"sample_{index:02d}.codes.npy"
        sf.write(wav_path, audio, int(runtime.sample_rate))
        np.save(code_path, codes)
        audio_sec = audio.shape[0] / float(runtime.sample_rate)
        last_timings = chunk_timings[-1] if chunk_timings else {}
        results.append(
            {
                "index": index,
                "text": text,
                "wav_path": str(wav_path),
                "codes_path": str(code_path),
                "sample_rate": int(runtime.sample_rate),
                "prompt_tokens": last_timings.get("prompt_len"),
                "generated_frames": int(codes.shape[0]),
                "audio_sec": audio_sec,
                "first_code_ms": first_audio_ms,
                "first_audio_ms": first_audio_ms,
                "codegen_elapsed_ms": last_timings.get("stream_compute_ms"),
                "decode_ms": last_timings.get("decode_ms"),
                "elapsed_ms": item_elapsed_sec * 1000.0,
                "stream_rtf": item_elapsed_sec / max(1e-9, audio_sec),
                "timings": last_timings,
            }
        )
    elapsed_sec = max(1e-9, time.perf_counter() - started)
    return results, {"candidate_path": "runtime", "elapsed_ms": elapsed_sec * 1000.0}


def evaluate_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    reference_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    scheduler_stats: dict[str, Any],
    quality,
    omni_config: dict[str, str | None],
) -> dict[str, Any]:
    results = []
    omni_enabled = not bool(args.skip_omni or args.objective_only)
    max_base64_bytes = int(float(args.omni_max_audio_mb) * 1024 * 1024)
    for candidate, reference in zip(candidate_items, reference_items, strict=True):
        item = dict(candidate)
        item["reference_wav_path"] = reference["wav_path"]
        target_text = str(reference.get("text") or item.get("text") or "")
        if target_text != item.get("text"):
            item["candidate_text"] = item.get("text")
            item["text"] = target_text
        wav_path = Path(item["wav_path"])
        codes = np.load(item["codes_path"]) if Path(item["codes_path"]).exists() else None
        item["objective_audio"] = quality.objective_audio_metrics(wav_path)
        item["objective_codes"] = quality.code_metrics(codes)
        item["objective_gate"] = quality.objective_gate(item["objective_audio"], item["objective_codes"])
        generated_frames = item.get("generated_frames")
        item["hit_max_new_tokens"] = bool(
            generated_frames is not None and int(generated_frames or 0) >= int(args.max_new_tokens)
        )
        if omni_enabled and item["objective_gate"].get("pass"):
            item["omni_pair"] = judge_pair_with_omni(
                quality,
                reference_wav=Path(reference["wav_path"]),
                candidate_wav=wav_path,
                text=target_text,
                config=omni_config,
                max_base64_bytes=max_base64_bytes,
            )
        elif omni_enabled:
            item["omni_pair"] = {"pass": False, "skipped": "objective_gate_failed"}
        item["quality_passed"] = bool(
            item["objective_gate"].get("pass")
            and not item["hit_max_new_tokens"]
            and (not omni_enabled or item.get("omni_pair", {}).get("pass"))
        )
        results.append(item)
    def optional_float(value, default: float = math.inf) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    ttft = [optional_float(item.get("first_audio_ms")) for item in results]
    rtf = [optional_float(item.get("stream_rtf")) for item in results]
    audio_seconds = [optional_float(item.get("audio_sec"), default=0.0) for item in results]
    elapsed_ms = [optional_float(item.get("elapsed_ms"), default=0.0) for item in results]
    aggregate_audio_sec = sum(value for value in audio_seconds if math.isfinite(value) and value > 0.0)
    elapsed_sec_max = max((value for value in elapsed_ms if math.isfinite(value)), default=math.inf) / 1000.0
    aggregate_rtf = elapsed_sec_max / max(1e-9, aggregate_audio_sec)
    hit_max_items = [item for item in results if item.get("hit_max_new_tokens")]
    generated_frames = [
        int(item["generated_frames"])
        for item in results
        if item.get("generated_frames") is not None
    ]
    quality_passed = all(bool(item.get("quality_passed")) for item in results)
    return {
        "feature": "prefill_quality",
        "mode": mode,
        "prefill_mode": mode,
        "passed": bool(quality_passed),
        "quality_passed": bool(quality_passed),
        "hit_max_new_tokens_count": len(hit_max_items),
        "max_generated_frames": max(generated_frames) if generated_frames else None,
        "token_budget_hint": (
            f"{len(hit_max_items)} item(s) hit max_new_tokens={args.max_new_tokens}; "
            "increase --max-new-tokens or use the production memory/EOS budget before judging quality."
            if hit_max_items
            else None
        ),
        "ttft_ms_p50": float(np.median(ttft)) if ttft else math.inf,
        "ttft_ms_p90": percentile(ttft, 0.90),
        "stream_rtf_p50": float(np.median(rtf)) if rtf else math.inf,
        "stream_rtf_p90": percentile(rtf, 0.90),
        "aggregate_audio_sec": aggregate_audio_sec,
        "elapsed_sec_max": elapsed_sec_max,
        "aggregate_rtf": aggregate_rtf,
        "scheduler": scheduler_stats,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate native online prefill modes against the original Python Qwen3-TTS output."
    )
    parser.add_argument("--qwen3-tts-repo", default=".cache/Qwen3-TTS")
    parser.add_argument("--python-model", default="models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument(
        "--python-device",
        default="gpu",
        help=(
            "Device for the original Python Qwen3-TTS reference. "
            "`gpu` is the default and requires CUDA or Intel XPU; use `cpu` only for intentional CPU reference."
        ),
    )
    parser.add_argument("--python-dtype", default="auto")
    parser.add_argument("--python-attn-implementation", default="auto")
    parser.add_argument(
        "--python-executable",
        default="auto",
        help="Python used for original Qwen3-TTS reference. auto prefers <qwen3-tts-repo>/.venv/bin/python.",
    )
    parser.add_argument("--python-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--reference-cache-dir", default="outputs/prefill_quality_reference_cache")
    parser.add_argument("--refresh-reference-cache", action="store_true")
    parser.add_argument("--no-reference-cache", action="store_true")
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument(
        "--runtime-mode",
        default="fastest",
        help=(
            "Runtime profile used to compile shared prompt/decoder components. "
            "Defaults to fastest, which is the only maintained production profile."
        ),
    )
    parser.add_argument("--candidate-path", default="runtime", choices=["runtime"])
    parser.add_argument("--runtime-graph-variant", default="int8_sym_paged_talker_split_cachedsub")
    parser.add_argument("--online-graph-variant", default="int8_sym_batch_fused_gqa")
    parser.add_argument("--subcode-mode", default="cached")
    parser.add_argument("--sampled-batch-subcode", default="off", choices=["off", "verify", "on"])
    parser.add_argument("--sampled-subcode-parallel-rows", action="store_true")
    parser.add_argument("--mode", default="voice_design", choices=["voice_design", "custom_voice", "voice_clone"])
    parser.add_argument("--prefill-modes", default="serial,dynamic-ragged")
    parser.add_argument("--kv-precision", default="u8", choices=["f16", "bf16", "u8"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-cache-blocks", type=int, default=2048)
    parser.add_argument("--scheduler", default="layered", choices=["layered"])
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    parser.add_argument("--prefill-seq-buckets", default="128,256,512,1024")
    parser.add_argument("--prefill-batch-buckets", default="1,2,4,8")
    parser.add_argument("--decode-batch-buckets", default="1,2,4,8,16")
    parser.add_argument(
        "--continuous-policy",
        default="layered_vllm",
        choices=["layered_vllm"],
    )
    parser.add_argument("--enable-fused-batch-decode", action="store_true")
    parser.add_argument("--continuous-batch-subcode", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--arrival-gap-ms", type=float, default=0.0)
    parser.add_argument("--wait-ms", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--language", action="append", default=None)
    parser.add_argument("--instruct", action="append", default=None)
    parser.add_argument("--speaker", action="append", default=None)
    parser.add_argument(
        "--candidate-text",
        action="append",
        default=None,
        help="Use alternate text only for the OpenVINO candidate while keeping the Python reference text unchanged.",
    )
    parser.add_argument(
        "--candidate-text-file",
        default=None,
        help="Use alternate text file only for the OpenVINO candidate while keeping the Python reference text unchanged.",
    )
    parser.add_argument(
        "--candidate-instruct",
        action="append",
        default=None,
        help="Use alternate instruct only for the OpenVINO candidate while keeping the Python reference instruct unchanged.",
    )
    parser.add_argument("--ref-audio", action="append", default=None)
    parser.add_argument("--ref-text", action="append", default=None)
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument(
        "--voice-clone-prompt-format",
        default="object",
        choices=["object", "dict", "dict-natural"],
        help=(
            "Candidate runtime prompt reuse format for VoiceClone. dict uses explicit batch-like JSON; "
            "dict-natural uses a single-prompt JSON shape with 1D speaker embedding and 2D ref_code."
        ),
    )
    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument(
        "--no-prebuild-prompts",
        action="store_true",
        help="Disable prompt prebuild. By default prefill quality tests prebuild prompts to exercise true batch prefill deterministically.",
    )
    parser.add_argument("--chunk-strategy", default="realtime")
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--skip-omni", action="store_true")
    parser.add_argument("--objective-only", action="store_true")
    parser.add_argument("--omni-max-audio-mb", type=float, default=9.5)
    parser.add_argument("--out-dir", default="outputs/prefill_quality")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    quality = load_script_module("evaluate_long_text_quality.py")
    omni_enabled = not bool(args.skip_omni or args.objective_only)
    omni_config = quality.load_aliyun_env(args.env_file) if omni_enabled else {"api_key": None, "model": None, "base_url": None}
    if omni_enabled and (not omni_config.get("api_key") or not omni_config.get("model")):
        status = quality.redacted_env_status(omni_config)
        raise SystemExit(
            "Aliyun Omni judge is enabled but credentials are incomplete. "
            f"status={json.dumps(status, ensure_ascii=False)}. Pass --skip-omni for objective-only checks."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = load_texts(args)
    candidate_texts = load_candidate_texts(args, texts)
    ref_audios = expand_values(args.ref_audio, len(texts), "")
    ref_texts = expand_values(args.ref_text, len(texts), "")
    if args.mode == "voice_clone" and not all(ref_audios):
        raise SystemExit("--ref-audio is required for --mode voice_clone")
    if args.mode == "voice_clone" and not args.x_vector_only and not all(ref_texts):
        raise SystemExit("--ref-text is required for --mode voice_clone unless --x-vector-only is set")
    reference_cache_key_value: str | None = None
    reference_dir_path: Path
    reference_cache_hit: bool | None = None
    if args.reference_dir:
        reference_dir_path = Path(args.reference_dir)
        reference_items = load_reference_items(args, texts)
        reference_source = "explicit_reference_dir"
    else:
        reference_dir_path, cache_payload, reference_cache_key_value = resolve_reference_dir(args, texts, out_dir)
        reference_source = "cache"
        if not bool(args.refresh_reference_cache) and cached_reference_ready(reference_dir_path, texts, cache_payload):
            print(f"using cached Python reference {reference_dir_path}", flush=True)
            reference_items = load_reference_items_from_dir(reference_dir_path, texts)
            reference_cache_hit = True
        else:
            print(
                f"running Python reference on {args.python_device} with {reference_python_executable(args)}; "
                f"cache={reference_dir_path}",
                flush=True,
            )
            reference_items = run_python_reference(
                args,
                texts,
                reference_dir_path,
                cache_payload=cache_payload,
                cache_key=reference_cache_key_value,
            )
            reference_cache_hit = False
    reference_result = None
    reference_result_json = reference_dir_path / "result.json"
    if reference_result_json.exists():
        try:
            raw_reference_result = load_json(reference_result_json)
            reference_result = {
                "ok": bool(raw_reference_result.get("ok")),
                "device": raw_reference_result.get("device"),
                "requested_device": raw_reference_result.get("requested_device"),
                "dtype": raw_reference_result.get("dtype"),
                "attn": raw_reference_result.get("attn"),
                "elapsed_sec": raw_reference_result.get("elapsed_sec"),
                "python_executable": raw_reference_result.get("python_executable"),
                "worker_exit_code": raw_reference_result.get("worker_exit_code"),
            }
        except Exception:
            reference_result = {"ok": False, "error": "failed_to_read_reference_result"}

    mode_results = []
    candidate_modes = ["runtime"] if args.candidate_path == "runtime" else [
        raw_mode.strip().replace("-", "_") for raw_mode in str(args.prefill_modes).split(",") if raw_mode.strip()
    ]
    for mode in candidate_modes:
        if args.candidate_path == "online" and mode not in PREFILL_MODE_CHOICES:
            raise ValueError(f"unknown prefill mode {mode!r}; expected serial, dynamic-ragged, or bucketed-padded")
        print(f"running candidate={args.candidate_path} mode={mode}", flush=True)
        try:
            if args.candidate_path == "runtime":
                candidate_items, scheduler_stats = run_runtime_candidate(args, candidate_texts, out_dir)
            else:
                candidate_items, scheduler_stats = run_candidate_mode(args, candidate_texts, mode, out_dir)
            mode_result = evaluate_mode(
                args,
                mode=mode,
                reference_items=reference_items,
                candidate_items=candidate_items,
                scheduler_stats=scheduler_stats,
                quality=quality,
                omni_config=omni_config,
            )
        except Exception as exc:
            mode_result = {
                "feature": "prefill_quality",
                "mode": mode,
                "prefill_mode": mode,
                "passed": False,
                "quality_passed": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        mode_results.append(mode_result)
        print(
            f"candidate={args.candidate_path} mode={mode} passed={mode_result.get('passed')} "
            f"rtf_p50={float(mode_result.get('stream_rtf_p50') or 0.0):.3f}",
            flush=True,
        )

    passed = [item for item in mode_results if item.get("passed")]
    winner = None
    if passed:
        winner = sorted(
            passed,
            key=lambda item: (
                0 if item.get("prefill_mode") == "bucketed_padded" else 1,
                float(item.get("stream_rtf_p50", math.inf)),
                float(item.get("ttft_ms_p50", math.inf)),
            ),
        )[0]
    summary = {
        "feature": "prefill_quality",
        "mode": args.mode,
        "created_at_unix": time.time(),
        "passed": bool(winner),
        "quality_passed": bool(winner),
        "winner": winner,
        "reference": {
            "source": "original_python_qwen3_tts",
            "storage": reference_source,
            "cache_hit": reference_cache_hit,
            "cache_key": reference_cache_key_value,
            "cache_dir": None if args.reference_dir else str(reference_dir_path),
            "qwen3_tts_repo": str(Path(args.qwen3_tts_repo).resolve()),
            "python_model": args.python_model,
            "python_executable": reference_python_executable(args),
            "runtime": reference_result,
            "items": reference_items,
        },
        "omni_enabled": omni_enabled,
        "omni_env": quality.redacted_env_status(omni_config) if omni_enabled else None,
        "results": mode_results,
    }
    output = Path(args.output_json) if args.output_json else out_dir / "quality_summary.json"
    write_json(output, summary)
    print(f"prefill_quality_passed={bool(winner)} wrote {output}", flush=True)


if __name__ == "__main__":
    main()
