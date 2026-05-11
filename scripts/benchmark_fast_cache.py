import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_TEXT = "你好，这是一次完全使用 OpenVINO 的 Qwen 三语音合成测试。"
DEFAULT_INSTRUCT = "A calm young female voice, natural Mandarin pronunciation."
QUALITY_RUNS = [
    ("no_cache_fp16", ["--mode", "no-cache"]),
    ("no_cache_int8", ["--mode", "no-cache", "--graph-variant", "int8"]),
    ("no_cache_int8_cachedsub", ["--mode", "no-cache", "--graph-variant", "int8_cachedsub"]),
    ("fast_cache_int8", ["--mode", "fast-cache"]),
]


def run_command(args: list[str]) -> None:
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def audio_metrics(path: Path) -> dict:
    audio, sample_rate = load_mono(path)
    abs_audio = np.abs(audio)
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "samples": int(audio.shape[0]),
        "duration_sec": float(audio.shape[0] / sample_rate) if sample_rate else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "peak": float(abs_audio.max()) if audio.size else 0.0,
        "silence_ratio": float(np.mean(abs_audio < 1e-4)) if audio.size else 1.0,
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        return np.array([], dtype=np.float32)
    count = 1 + (audio.size - frame) // hop
    values = np.empty(count, dtype=np.float32)
    for i in range(count):
        start = i * hop
        chunk = audio[start : start + frame]
        values[i] = np.sqrt(np.mean(np.square(chunk)))
    return values


def spectral_cosine(a: np.ndarray, b: np.ndarray, frame: int = 1024, hop: int = 512) -> float:
    if a.size < frame or b.size < frame:
        return 0.0
    frames = min(1 + (a.size - frame) // hop, 1 + (b.size - frame) // hop)
    if frames <= 0:
        return 0.0
    window = np.hanning(frame).astype(np.float32)
    sims = []
    for i in range(frames):
        start = i * hop
        spec_a = np.abs(np.fft.rfft(a[start : start + frame] * window))
        spec_b = np.abs(np.fft.rfft(b[start : start + frame] * window))
        denom = float(np.linalg.norm(spec_a) * np.linalg.norm(spec_b))
        if denom:
            sims.append(float(np.dot(spec_a, spec_b) / denom))
    return float(np.mean(sims)) if sims else 0.0


def audio_comparison(reference_path: Path, candidate_path: Path) -> dict:
    reference, ref_rate = load_mono(reference_path)
    candidate, cand_rate = load_mono(candidate_path)
    if ref_rate != cand_rate:
        raise ValueError(f"sample-rate mismatch: {reference_path}={ref_rate}, {candidate_path}={cand_rate}")

    overlap = min(reference.shape[0], candidate.shape[0])
    ref = reference[:overlap]
    cand = candidate[:overlap]
    ref_rms = float(np.sqrt(np.mean(np.square(reference)))) if reference.size else 0.0
    cand_rms = float(np.sqrt(np.mean(np.square(candidate)))) if candidate.size else 0.0

    frame = max(1, int(ref_rate * 0.02))
    hop = max(1, int(ref_rate * 0.01))
    ref_env = frame_rms(ref, frame, hop)
    cand_env = frame_rms(cand, frame, hop)
    env_len = min(ref_env.shape[0], cand_env.shape[0])

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "duration_delta_sec": float((candidate.shape[0] - reference.shape[0]) / ref_rate) if ref_rate else 0.0,
        "overlap_sec": float(overlap / ref_rate) if ref_rate else 0.0,
        "rms_ratio": float(cand_rms / ref_rms) if ref_rms else 0.0,
        "waveform_corr": correlation(ref, cand),
        "envelope_corr": correlation(ref_env[:env_len], cand_env[:env_len]) if env_len else 0.0,
        "spectral_cosine": spectral_cosine(ref, cand),
    }


def write_quality_metrics(out_dir: Path) -> None:
    metrics = {}
    for name, _ in QUALITY_RUNS:
        wav = out_dir / f"{name}.wav"
        if not wav.exists():
            raise FileNotFoundError(wav)
        metrics[name] = audio_metrics(wav)
    metrics["comparisons_to_no_cache_fp16"] = {
        name: audio_comparison(out_dir / "no_cache_fp16.wav", out_dir / f"{name}.wav")
        for name, _ in QUALITY_RUNS
        if name != "no_cache_fp16"
    }
    with open(out_dir / "audio_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_dir / 'audio_metrics.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--max-new-tokens", default="128")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--out-dir", default="outputs/fast_cache_bench")
    parser.add_argument("--quality", action="store_true", help="Also write paired WAV outputs and audio metrics.")
    parser.add_argument("--metrics-only", action="store_true", help="Only refresh audio_metrics.json from existing WAV files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_only:
        write_quality_metrics(out_dir)
        return

    base = [
        sys.executable,
        "-m",
        "qwen3_tts_ov",
        "voice-design",
        "--ir-dir",
        args.ir_dir,
        "--device",
        args.device,
        "--max-new-tokens",
        args.max_new_tokens,
        "--text",
        args.text,
        "--instruct",
        args.instruct,
        "--language",
        args.language,
        "--progress-interval",
        "16",
        "--profile",
    ]

    configs = [
        ("no_cache_fp16", ["--mode", "no-cache", "--skip-decode"]),
        ("no_cache_int8", ["--mode", "no-cache", "--graph-variant", "int8", "--skip-decode"]),
        ("no_cache_int8_cachedsub", ["--mode", "no-cache", "--graph-variant", "int8_cachedsub", "--skip-decode"]),
        ("cache_sdpa_fp16", ["--mode", "cache", "--cache-kernel", "sdpa", "--cache-step", "split", "--skip-decode"]),
        ("fast_cache_int8", ["--mode", "fast-cache", "--skip-decode"]),
    ]
    for name, extra in configs:
        run_command(base + extra)

    if args.quality:
        for name, extra in QUALITY_RUNS:
            wav = out_dir / f"{name}.wav"
            run_command(base + extra + ["--output", str(wav)])
        write_quality_metrics(out_dir)


if __name__ == "__main__":
    main()
