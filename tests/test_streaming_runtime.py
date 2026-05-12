import types

import numpy as np

from qwen3_tts_ov.runtime import DEFAULT_STREAM_CHUNK_STRATEGIES, OpenVINOQwen3TTS


class FakeTimings:
    def snapshot(self, generated_tokens: int) -> dict:
        return {"generated_tokens": generated_tokens}


def make_runtime():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.num_code_groups = 2
    runtime.sample_rate = 24_000
    runtime.decode_upsample_rate = 2
    runtime.timings = FakeTimings()
    runtime.streaming_decoder_left_context = 25
    runtime.default_chunk_strategy = "low_latency"
    runtime.streaming_decoder_strategies = {name: dict(config) for name, config in DEFAULT_STREAM_CHUNK_STRATEGIES.items()}

    def decode_stream_window(self, window_codes, context_frames, new_frames, chunk_frames=12, left_context_frames=25):
        assert window_codes.shape[0] == context_frames + new_frames
        return np.arange(new_frames * self.decode_upsample_rate, dtype=np.float32)

    runtime.decode_stream_window = types.MethodType(decode_stream_window, runtime)
    return runtime


def test_stream_decode_codes_chunks_and_final_audio():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(5)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_frames=2, left_context_frames=1))

    assert [chunk.codes.shape[0] for chunk in chunks] == [2, 2, 1]
    assert [chunk.audio.shape[0] for chunk in chunks] == [4, 4, 2]
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    assert chunks[-1].timings["emitted_frames"] == 5
    assert chunks[-1].timings["stream_audio_ms"] > 0
    assert "stream_rtf" in chunks[-1].timings


def test_stream_decode_codes_emits_final_marker_on_exact_chunk_boundary():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(4)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_frames=2, left_context_frames=1))

    assert [chunk.codes.shape[0] for chunk in chunks] == [2, 2, 0]
    assert chunks[-1].audio.size == 0
    assert chunks[-1].is_final is True
    assert chunks[-1].timings["stream_audio_ms"] == chunks[-2].timings["stream_audio_ms"]


def test_stream_decode_codes_uses_prefix_as_context_without_emitting_it():
    runtime = make_runtime()
    prefix = np.asarray([[10, 11], [12, 13], [14, 15]], dtype=np.int64)
    codes = [np.asarray([20, 21], dtype=np.int64), np.asarray([22, 23], dtype=np.int64)]

    chunks = list(runtime.stream_decode_codes(codes, prefix_codes=prefix, chunk_frames=2, left_context_frames=2))

    assert len(chunks) == 2
    assert chunks[0].codes.tolist() == [[20, 21], [22, 23]]
    assert chunks[0].timings["prefix_frames"] == 3
    assert chunks[0].timings["emitted_frames"] == 2
    assert chunks[1].is_final is True


def test_stream_decode_codes_uses_initial_chunk_strategy_then_steady_chunk():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(5)]

    chunks = list(
        runtime.stream_decode_codes(
            codes,
            chunk_strategy="low_latency",
            initial_chunk_frames=1,
            chunk_frames=2,
            left_context_frames=1,
        )
    )

    assert [chunk.codes.shape[0] for chunk in chunks] == [1, 2, 2, 0]
    assert chunks[0].timings["strategy"] == "low_latency"
    assert chunks[0].timings["initial_chunk_frames"] == 1
    assert chunks[1].timings["configured_chunk_frames"] == 2


def test_smooth_strategy_uses_low_latency_first_chunk_and_larger_steady_chunk():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(32)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_strategy="smooth"))

    assert [chunk.codes.shape[0] for chunk in chunks] == [8, 24, 0]
    assert chunks[0].timings["strategy"] == "smooth"
    assert chunks[0].timings["initial_chunk_frames"] == 8
    assert chunks[1].timings["configured_chunk_frames"] == 24


def test_realtime_strategy_uses_eight_then_twelve_frame_chunks():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(32)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_strategy="realtime"))

    assert [chunk.codes.shape[0] for chunk in chunks] == [8, 12, 12, 0]
    assert chunks[0].timings["strategy"] == "realtime"
    assert chunks[0].timings["initial_chunk_frames"] == 8
    assert chunks[1].timings["configured_chunk_frames"] == 12


def test_stream_decoder_key_prefers_first_chunk_then_left_context_graph():
    runtime = make_runtime()
    runtime.streaming_decoder_graphs_by_context = {
        0: {12: "speech_decoder_stream_c0_t12.xml"},
        25: {
            12: "speech_decoder_stream_c25_t12.xml",
            24: "speech_decoder_stream_c25_t24.xml",
        },
    }

    assert runtime._stream_decoder_key(0, 12, 12, 25) == (0, 12)
    assert runtime._stream_decoder_key(12, 12, 12, 25) == (25, 12)
    assert runtime._stream_decoder_key(25, 18, 12, 25) == (25, 24)
