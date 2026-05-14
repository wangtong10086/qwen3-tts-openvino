import base64
import io

import numpy as np
import soundfile as sf

from qwen3_tts_ov.audio import load_audio, mel_filter_bank, speaker_mel_spectrogram, stft_magnitude


def test_load_audio_reads_soundfile_and_resamples(tmp_path):
    source_sr = 16_000
    target_sr = 24_000
    t = np.linspace(0, 0.1, int(source_sr * 0.1), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, wav, source_sr)

    values = load_audio(path, target_sr=target_sr)

    assert values.dtype == np.float32
    assert abs(values.shape[0] - int(target_sr * 0.1)) <= 2
    assert np.max(np.abs(values)) > 0


def test_load_audio_accepts_base64_data_url():
    payload = io.BytesIO()
    sf.write(payload, np.zeros(240, dtype=np.float32), 24_000, format="WAV")
    encoded = "data:audio/wav;base64," + base64.b64encode(payload.getvalue()).decode("ascii")

    values = load_audio(encoded, target_sr=24_000)

    assert values.shape == (240,)
    assert values.dtype == np.float32


def test_speaker_mel_spectrogram_shape_and_finiteness():
    sr = 24_000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220 * t)

    mel = speaker_mel_spectrogram(wav, sr=sr)

    assert mel.ndim == 2
    assert mel.shape[1] == 128
    assert mel.shape[0] > 0
    assert np.isfinite(mel).all()


def test_numpy_mel_helpers_are_deterministic():
    wav = np.linspace(-0.1, 0.1, 2048, dtype=np.float32)

    first = stft_magnitude(wav)
    second = stft_magnitude(wav)
    mel = mel_filter_bank(sr=24_000, n_fft=1024)

    np.testing.assert_allclose(first, second)
    assert mel.shape == (128, 513)
    assert np.count_nonzero(mel) > 0
