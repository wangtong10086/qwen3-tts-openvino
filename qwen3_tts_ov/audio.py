import base64
import io
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import soxr


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_probably_base64(value: str) -> bool:
    return value.startswith("data:audio") or ("/" not in value and "\\" not in value and len(value) > 256)


def load_audio(audio, target_sr: int, sr: int | None = None) -> np.ndarray:
    source_sr: int | None = sr
    if isinstance(audio, tuple):
        values, source_sr = audio
        values = np.asarray(values, dtype=np.float32)
        source_sr = int(source_sr)
    elif isinstance(audio, np.ndarray):
        if source_sr is None:
            raise ValueError("sr is required when audio is a numpy array")
        values = audio.astype(np.float32, copy=False)
    elif isinstance(audio, (str, Path)):
        item = str(audio)
        if _is_url(item):
            with urllib.request.urlopen(item) as response:
                payload = response.read()
            with io.BytesIO(payload) as handle:
                values, read_sr = sf.read(handle, dtype="float32", always_2d=False)
                source_sr = int(read_sr)
        elif _is_probably_base64(item):
            if item.startswith("data:") and "," in item:
                item = item.split(",", 1)[1]
            with io.BytesIO(base64.b64decode(item)) as handle:
                values, read_sr = sf.read(handle, dtype="float32", always_2d=False)
                source_sr = int(read_sr)
        else:
            try:
                values, read_sr = sf.read(item, dtype="float32", always_2d=False)
                source_sr = int(read_sr)
            except Exception as exc:
                try:
                    import librosa
                except Exception as import_exc:
                    raise RuntimeError(
                        f"failed to read audio with soundfile: {item}. "
                        "Install the optional audio-full extra for broader codec support."
                    ) from import_exc
                try:
                    values, read_sr = librosa.load(item, sr=None, mono=True)
                    source_sr = int(read_sr)
                except Exception:
                    raise exc
    else:
        raise TypeError(f"unsupported audio input type: {type(audio)!r}")
    if source_sr is None:
        raise ValueError("source sample rate could not be determined")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim > 1:
        values = np.mean(values, axis=-1)
    if source_sr != int(target_sr):
        values = soxr.resample(values, source_sr, int(target_sr))
    return values.astype(np.float32, copy=False)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def mel_filter_bank(sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = float(sr // 2 if fmax is None else fmax)
    fft_freqs = np.linspace(0.0, float(sr) / 2.0, 1 + n_fft // 2, dtype=np.float64)
    min_mel = _hz_to_mel(np.asarray([fmin], dtype=np.float64))[0]
    max_mel = _hz_to_mel(np.asarray([fmax], dtype=np.float64))[0]
    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.empty((n_mels, len(fft_freqs)), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / fdiff[index]
        upper = ramps[index + 2] / fdiff[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32, copy=False)


def stft_magnitude(audio: np.ndarray, n_fft: int = 1024, hop_size: int = 256, win_size: int = 1024) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        raise ValueError("audio must not be empty")
    padding = (n_fft - hop_size) // 2
    pad_mode = "reflect" if audio.size > 1 else "edge"
    padded = np.pad(audio, (padding, padding), mode=pad_mode)
    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size), mode="constant")
    frame_count = 1 + (padded.size - n_fft) // hop_size
    starts = np.arange(frame_count)[:, None] * hop_size
    offsets = np.arange(n_fft)[None, :]
    frames = padded[starts + offsets]
    window = np.hanning(win_size + 1)[:-1].astype(np.float32)
    if win_size < n_fft:
        padded_window = np.zeros(n_fft, dtype=np.float32)
        start = (n_fft - win_size) // 2
        padded_window[start : start + win_size] = window
        window = padded_window
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1).T
    magnitude = np.sqrt(np.square(spec.real) + np.square(spec.imag) + 1e-9)
    return magnitude.astype(np.float32, copy=False)


def speaker_mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    magnitude = stft_magnitude(audio, n_fft=n_fft, hop_size=hop_size, win_size=win_size)
    mel_basis = mel_filter_bank(sr=sr, n_fft=n_fft, n_mels=128, fmin=0, fmax=12000)
    mel = np.matmul(mel_basis, magnitude)
    return np.log(np.maximum(mel, 1e-5)).T.astype(np.float32, copy=False)
