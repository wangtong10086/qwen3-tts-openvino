from qwen3_tts_ov.server import openai_speech_to_tts_request, stream_metadata


def test_openai_speech_maps_custom_voice_request():
    internal, response_format, stream_enabled = openai_speech_to_tts_request(
        {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice": "Vivian",
            "input": "Hello",
            "language": "English",
            "instructions": "Speak brightly.",
            "stream": True,
            "response_format": "pcm",
            "chunk_strategy": "stable",
            "max_new_tokens": 32,
        }
    )

    assert response_format == "pcm"
    assert stream_enabled is True
    assert internal["mode"] == "custom_voice"
    assert internal["speaker"] == "Vivian"
    assert internal["instruct"] == "Speak brightly."
    assert internal["generation"]["max_new_tokens"] == 32
    assert internal["stream"]["chunk_strategy"] == "stable"


def test_stream_metadata_uses_strategy_defaults():
    metadata = stream_metadata({"stream": {"chunk_strategy": "low_latency"}})

    assert metadata["chunk_strategy"] == "low_latency"
    assert metadata["initial_chunk_frames"] == 8
    assert metadata["chunk_frames"] == 12
    assert metadata["left_context_frames"] == 25
