import base64
import io
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_probably_base64(value: str) -> bool:
    return value.startswith("data:audio") or ("/" not in value and "\\" not in value and len(value) > 256)


def load_audio(audio, target_sr: int, sr: int | None = None) -> np.ndarray:
    if isinstance(audio, tuple):
        values, source_sr = audio
        values = np.asarray(values, dtype=np.float32)
        sr = int(source_sr)
    elif isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError("sr is required when audio is a numpy array")
        values = audio.astype(np.float32, copy=False)
    elif isinstance(audio, (str, Path)):
        item = str(audio)
        if _is_url(item):
            with urllib.request.urlopen(item) as response:
                payload = response.read()
            with io.BytesIO(payload) as handle:
                values, sr = sf.read(handle, dtype="float32", always_2d=False)
        elif _is_probably_base64(item):
            if item.startswith("data:") and "," in item:
                item = item.split(",", 1)[1]
            with io.BytesIO(base64.b64decode(item)) as handle:
                values, sr = sf.read(handle, dtype="float32", always_2d=False)
        else:
            values, sr = librosa.load(item, sr=None, mono=True)
    else:
        raise TypeError(f"unsupported audio input type: {type(audio)!r}")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim > 1:
        values = np.mean(values, axis=-1)
    if int(sr) != int(target_sr):
        values = librosa.resample(values, orig_sr=int(sr), target_sr=int(target_sr))
    return values.astype(np.float32, copy=False)


def speaker_mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    padding = (n_fft - hop_size) // 2
    padded = np.pad(audio.astype(np.float32, copy=False), (padding, padding), mode="reflect")
    spec = librosa.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window="hann",
        center=False,
        pad_mode="reflect",
    )
    magnitude = np.sqrt(np.square(spec.real) + np.square(spec.imag) + 1e-9)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128, fmin=0, fmax=12000)
    mel = np.matmul(mel_basis, magnitude)
    return np.log(np.maximum(mel, 1e-5)).T.astype(np.float32, copy=False)
