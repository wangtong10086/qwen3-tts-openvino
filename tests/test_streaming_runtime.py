import json
import types
from pathlib import Path

import numpy as np

from qwen3_tts_ov.runtime import (
    DEFAULT_STREAM_CHUNK_STRATEGIES,
    OpenVINOQwen3TTS,
    custom_voice_default_repetition_penalty,
    effective_paged_kv_unroll,
    env_flag_enabled,
    normalize_custom_voice_text,
    paged_kv_seed_uses_gqa,
    select_paged_kv_seed_key,
)
from qwen3_tts_ov.profiles import (
    FASTEST_GRAPH_VARIANT,
    FASTEST_PROFILE_NAME,
    effective_runtime_options,
    fastest_runtime_defaults,
)


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
    runtime.streaming_decoder_input_shape = "dynamic"
    runtime.default_chunk_strategy = "low_latency"
    runtime.streaming_decoder_strategies = {name: dict(config) for name, config in DEFAULT_STREAM_CHUNK_STRATEGIES.items()}

    def decode_stream_window(self, window_codes, context_frames, new_frames, chunk_frames=12, left_context_frames=25):
        assert window_codes.shape[0] == context_frames + new_frames
        return np.arange(new_frames * self.decode_upsample_rate, dtype=np.float32)

    runtime.decode_stream_window = types.MethodType(decode_stream_window, runtime)
    return runtime


def make_prompt_runtime(text_token_count=12, instruct_token_count=4, codec_prefill_count=4, speaker=False):
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.ids = {
        "tts_bos_token_id": 10,
        "tts_eos_token_id": 11,
        "tts_pad_token_id": 12,
        "codec_pad_id": 20,
        "codec_bos_id": 21,
    }
    runtime.num_code_groups = 16
    runtime._text_token_count = int(text_token_count)
    runtime._instruct_token_count = int(instruct_token_count)
    runtime._codec_prefill_count = int(codec_prefill_count)
    runtime._has_speaker = bool(speaker)

    class FakeTokenizer:
        def encode(self, text):
            return list(range(runtime._instruct_token_count if "\ninstruct" in text else runtime._text_token_count))

    runtime.tokenizer = FakeTokenizer()

    def embed_ids(ids):
        return np.zeros((1, len(ids), 3), dtype=np.float32)

    runtime.embed_text = lambda ids: embed_ids(ids)
    runtime.embed_text_cached = lambda _key, ids: embed_ids(ids)
    runtime.embed_codec_cached = lambda _key, ids: embed_ids(ids)
    runtime.language_codec_prefill = lambda language, speaker=None: list(range(runtime._codec_prefill_count))
    runtime.voice_clone_speaker_embed = lambda prompt_item: None
    runtime.speaker_token_embed = lambda speaker_name: (
        np.zeros((1, 1, 3), dtype=np.float32) if runtime._has_speaker and speaker_name else None
    )
    return runtime


def test_build_prompt_keeps_full_non_streaming_codec_prefill_without_speaker():
    runtime = make_prompt_runtime(text_token_count=12, instruct_token_count=4, codec_prefill_count=4, speaker=False)

    sequence, tts_pad_embed = runtime.build_prompt("text", "instruct", "Chinese", max_prompt_tokens=64)

    assert sequence.shape == (1, 18, 3)
    assert tts_pad_embed.shape == (1, 1, 3)


def test_build_prompt_keeps_full_non_streaming_codec_prefill_with_custom_speaker():
    runtime = make_prompt_runtime(text_token_count=12, instruct_token_count=4, codec_prefill_count=4, speaker=True)

    sequence, _ = runtime.build_prompt("text", "instruct", "Japanese", max_prompt_tokens=64, speaker="Ono_Anna")

    assert sequence.shape == (1, 19, 3)


def test_subtalker_sampling_auto_is_conservative_even_when_sampled_graph_exists():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.subtalker_sample_policy = "auto"
    runtime.variant_graphs = {}
    runtime.manifest = {"graphs": {"subcode_sampled_cached_next_embed": "subcode_sampled_cached_next_embed.xml"}}

    assert not runtime.should_sample_subcode(do_sample=False)
    assert not runtime.should_sample_subcode(do_sample=True)


def test_subtalker_sampling_auto_preserves_old_greedy_ir_without_sampled_graph():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.subtalker_sample_policy = "auto"
    runtime.variant_graphs = {}
    runtime.manifest = {"graphs": {"subcode_greedy_cached": "subcode_greedy_cached.xml"}}

    assert not runtime.should_sample_subcode(do_sample=False)
    assert not runtime.should_sample_subcode(do_sample=True)


def test_subtalker_sampling_on_uses_sampled_graph_when_available():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.subtalker_sample_policy = "on"
    runtime.variant_graphs = {}
    runtime.manifest = {"graphs": {"subcode_sampled_cached": "subcode_sampled_cached.xml"}}

    assert runtime.should_sample_subcode(do_sample=False)


def test_subtalker_sampling_require_reports_missing_graph():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.subtalker_sample_policy = "require"
    runtime.variant_graphs = {}
    runtime.manifest = {"graphs": {"subcode_greedy_cached": "subcode_greedy_cached.xml"}}

    try:
        runtime.should_sample_subcode(do_sample=False)
    except RuntimeError as exc:
        assert "subcode_sampled_cached" in str(exc)
    else:
        raise AssertionError("expected missing sampled subcode graph error")


def test_custom_voice_long_sample_default_repetition_penalty():
    long_text = "这是一段较长的自定义音色测试文本，用来验证长输出时不会进入重复或者静音循环。"

    assert custom_voice_default_repetition_penalty(
        long_text,
        do_sample=True,
        repetition_penalty=1.05,
        explicit_repetition_penalty=False,
    ) == 1.2
    assert custom_voice_default_repetition_penalty(
        long_text,
        do_sample=True,
        repetition_penalty=1.05,
        explicit_repetition_penalty=True,
    ) == 1.05
    assert custom_voice_default_repetition_penalty(
        long_text,
        do_sample=False,
        repetition_penalty=1.0,
        explicit_repetition_penalty=False,
    ) == 1.0


def test_custom_voice_ono_anna_japanese_pronunciation_override():
    assert (
        normalize_custom_voice_text("小野杏奈は自然に読みます。", "Ono_Anna", "Japanese")
        == "オノアンナは自然に読みます。"
    )
    assert (
        normalize_custom_voice_text("おの あんなは自然に読みます。", "ono_anna", "Auto")
        == "オノアンナは自然に読みます。"
    )
    assert normalize_custom_voice_text("小野杏奈は自然に読みます。", "Vivian", "Japanese") == "小野杏奈は自然に読みます。"
    assert normalize_custom_voice_text("小野杏奈は自然に読みます。", "Ono_Anna", "Chinese") == "小野杏奈は自然に読みます。"


def test_voice_clone_prompt_dict_preserves_ref_text_for_icl_prompt_reuse():
    runtime = object.__new__(OpenVINOQwen3TTS)
    prompts = runtime._normalize_voice_clone_prompt(
        {
            "ref_code": [
                np.ones((2, 16), dtype=np.int64),
                np.ones((3, 16), dtype=np.int64) * 2,
            ],
            "ref_spk_embedding": [
                np.ones(4, dtype=np.float32),
                np.ones(4, dtype=np.float32) * 2,
            ],
            "x_vector_only_mode": [False, False],
            "icl_mode": [True, True],
            "ref_text": ["第一条参考文本。", "Second reference text."],
        },
        2,
    )

    assert [prompt.ref_text for prompt in prompts] == ["第一条参考文本。", "Second reference text."]
    assert [prompt.x_vector_only_mode for prompt in prompts] == [False, False]
    assert [prompt.icl_mode for prompt in prompts] == [True, True]
    assert prompts[0].ref_code.shape == (2, 16)
    assert prompts[1].ref_code.shape == (3, 16)


def test_voice_clone_prompt_dict_accepts_scalar_ref_text_for_single_prompt_reuse():
    runtime = object.__new__(OpenVINOQwen3TTS)
    prompts = runtime._normalize_voice_clone_prompt(
        {
            "ref_code": [np.ones((2, 16), dtype=np.int64)],
            "ref_spk_embedding": [np.ones(4, dtype=np.float32)],
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "Shared reference text.",
        },
        1,
    )

    assert len(prompts) == 1
    assert prompts[0].ref_text == "Shared reference text."
    assert prompts[0].x_vector_only_mode is False
    assert prompts[0].icl_mode is True


def test_voice_clone_prompt_dict_accepts_natural_single_prompt_json_shape():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.num_code_groups = 16

    prompts = runtime._normalize_voice_clone_prompt(
        {
            "ref_code": np.ones((2, 16), dtype=np.int64).tolist(),
            "ref_spk_embedding": np.ones(4, dtype=np.float32).tolist(),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "Natural JSON reference text.",
        },
        1,
    )

    assert len(prompts) == 1
    assert prompts[0].ref_text == "Natural JSON reference text."
    assert prompts[0].ref_spk_embedding.shape == (4,)
    assert prompts[0].ref_code.shape == (2, 16)


def test_voice_clone_prompt_dict_broadcasts_natural_single_prompt_json_shape():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.num_code_groups = 16

    prompts = runtime._normalize_voice_clone_prompt(
        {
            "ref_code": np.ones((2, 16), dtype=np.int64).tolist(),
            "ref_spk_embedding": np.ones(4, dtype=np.float32).tolist(),
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "Broadcast natural JSON reference text.",
        },
        2,
    )

    assert len(prompts) == 2
    assert [prompt.ref_text for prompt in prompts] == [
        "Broadcast natural JSON reference text.",
        "Broadcast natural JSON reference text.",
    ]
    assert all(prompt.ref_code.shape == (2, 16) for prompt in prompts)


def test_voice_clone_prompt_dict_broadcasts_single_ref_text_and_flags_for_batch_reuse():
    runtime = object.__new__(OpenVINOQwen3TTS)
    prompts = runtime._normalize_voice_clone_prompt(
        {
            "ref_code": [
                np.ones((2, 16), dtype=np.int64),
                np.ones((3, 16), dtype=np.int64) * 2,
            ],
            "ref_spk_embedding": [
                np.ones(4, dtype=np.float32),
                np.ones(4, dtype=np.float32) * 2,
            ],
            "x_vector_only_mode": [False],
            "icl_mode": [True],
            "ref_text": ["Shared reference text."],
        },
        2,
    )

    assert [prompt.ref_text for prompt in prompts] == ["Shared reference text.", "Shared reference text."]
    assert [prompt.x_vector_only_mode for prompt in prompts] == [False, False]
    assert [prompt.icl_mode for prompt in prompts] == [True, True]


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


def test_decode_uses_streaming_decoder_when_full_decoder_missing():
    runtime = make_runtime()
    runtime.decoder_graphs = {}
    runtime.streaming_decoder_graphs_by_context = {0: {12: "first.xml"}, 25: {24: "steady.xml"}}

    audio = runtime.decode(np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64))

    assert audio.shape[0] == 6
    assert audio.dtype == np.float32


def test_smooth_strategy_uses_low_latency_first_chunk_and_larger_steady_chunk():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(32)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_strategy="smooth"))

    assert [chunk.codes.shape[0] for chunk in chunks] == [8, 24, 0]
    assert chunks[0].timings["strategy"] == "smooth"
    assert chunks[0].timings["initial_chunk_frames"] == 8
    assert chunks[1].timings["configured_chunk_frames"] == 24


def test_runtime_auto_chunk_strategy_is_available_for_direct_api():
    runtime = make_runtime()

    config = runtime._resolve_stream_chunk_config(chunk_strategy="auto")

    assert config["strategy"] == "auto"
    assert config["initial_chunk_frames"] == 8
    assert config["chunk_frames"] == 24


def test_realtime_strategy_uses_eight_then_twelve_frame_chunks():
    runtime = make_runtime()
    codes = [np.asarray([index, index + 10], dtype=np.int64) for index in range(32)]

    chunks = list(runtime.stream_decode_codes(codes, chunk_strategy="realtime"))

    assert [chunk.codes.shape[0] for chunk in chunks] == [8, 12, 12, 0]
    assert chunks[0].timings["strategy"] == "realtime"
    assert chunks[0].timings["initial_chunk_frames"] == 8
    assert chunks[1].timings["configured_chunk_frames"] == 12


def test_generate_voice_design_uses_native_stream_when_python_talker_absent():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.requested_mode = "fastest"
    runtime.talker_request = None
    runtime.sample_rate = 24_000
    calls = []

    def native_pipeline_mode(self):
        return "require"

    def stream_voice_design(self, **kwargs):
        calls.append(kwargs)
        yield types.SimpleNamespace(audio=np.asarray([0.1, 0.2], dtype=np.float32))
        yield types.SimpleNamespace(audio=np.asarray([0.3], dtype=np.float32))

    runtime._native_pipeline_mode = types.MethodType(native_pipeline_mode, runtime)
    runtime.stream_voice_design = types.MethodType(stream_voice_design, runtime)

    wavs, sample_rate = OpenVINOQwen3TTS.generate_voice_design(
        runtime,
        text="长文本走 native pipeline",
        instruct="自然朗读",
        language="Chinese",
        max_new_tokens=8,
        do_sample=True,
        repetition_penalty=1.05,
    )

    assert sample_rate == 24_000
    assert np.allclose(wavs[0], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert calls[0]["text"] == "长文本走 native pipeline"
    assert calls[0]["instruct"] == "自然朗读"
    assert calls[0]["language"] == "Chinese"
    assert calls[0]["do_sample"] is True


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


def test_streaming_decoder_graphs_are_discovered_from_ir_dir(tmp_path):
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.ir_dir = tmp_path
    (tmp_path / "speech_decoder_stream_c0_t8.xml").write_text("<xml/>")
    (tmp_path / "speech_decoder_stream_c25_t24.xml").write_text("<xml/>")

    graphs = runtime._load_streaming_decoder_graphs(
        {"left_context_frames": 25},
        {"streaming_decoder": {"12": "speech_decoder_stream_c25_t12.xml"}},
    )

    assert graphs == {
        0: {8: "speech_decoder_stream_c0_t8.xml"},
        25: {
            12: "speech_decoder_stream_c25_t12.xml",
            24: "speech_decoder_stream_c25_t24.xml",
        },
    }


def test_static_stream_decoder_window_pads_context_and_final_chunk():
    runtime = make_runtime()
    runtime.streaming_decoder_input_shape = "static"
    window = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.int64)

    padded, context, padded_frames = runtime._pad_static_stream_decoder_window(
        window,
        context_frames=1,
        target_context_frames=2,
        target_chunk_frames=4,
    )

    assert context == 2
    assert padded_frames == 2
    assert padded.shape == (6, 2)
    assert padded.tolist() == [[10, 11], [10, 11], [20, 21], [30, 31], [30, 31], [30, 31]]


def test_dynamic_stream_decoder_window_keeps_final_chunk_unpadded():
    runtime = make_runtime()
    runtime.streaming_decoder_input_shape = "dynamic"
    window = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.int64)

    padded, context, padded_frames = runtime._pad_static_stream_decoder_window(
        window,
        context_frames=1,
        target_context_frames=2,
        target_chunk_frames=4,
    )

    assert context == 2
    assert padded_frames == 0
    assert padded.tolist() == [[10, 11], [10, 11], [20, 21], [30, 31]]


def test_paged_kv_unroll_requires_explicit_experimental_flag():
    assert effective_paged_kv_unroll(4, experimental_enabled=False) == 1
    assert effective_paged_kv_unroll("4", experimental_enabled=True) == 4
    assert effective_paged_kv_unroll(None, experimental_enabled=True) == 1


def test_env_flag_enabled_accepts_common_truthy_values():
    assert env_flag_enabled("1") is True
    assert env_flag_enabled("true") is True
    assert env_flag_enabled("yes") is True
    assert env_flag_enabled("0") is False
    assert env_flag_enabled(None) is False


def test_paged_kv_seed_uses_gqa_for_subcode_variants():
    assert paged_kv_seed_uses_gqa("fused_cache_step_gqa")
    assert paged_kv_seed_uses_gqa("fused_cache_step_gqa_subcode_exact")
    assert paged_kv_seed_uses_gqa("fused_cache_step_unroll4_gqa_subcode_exact")
    assert not paged_kv_seed_uses_gqa("fused_cache_step_subcode_exact")


def test_paged_kv_seed_auto_prefers_sdpa_for_speed():
    graphs = {
        "fused_cache_step_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml",
        "fused_cache_step_gqa_subcode_exact": "fused_cache_step_sdpa_paged_gqa_subcode_exact_seed.xml",
    }

    key, subcode_attention, fallback = select_paged_kv_seed_key(graphs, prefer_gqa=True)

    assert key == "fused_cache_step_gqa"
    assert subcode_attention == "sdpa"
    assert fallback is False


def test_paged_kv_seed_auto_falls_back_to_sdpa_when_exact_missing():
    graphs = {"fused_cache_step_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml"}

    key, subcode_attention, fallback = select_paged_kv_seed_key(graphs, prefer_gqa=True)

    assert key == "fused_cache_step_gqa"
    assert subcode_attention == "sdpa"
    assert fallback is False


def test_paged_kv_seed_exact_reports_fallback_when_exact_missing():
    graphs = {"fused_cache_step_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml"}

    key, subcode_attention, fallback = select_paged_kv_seed_key(
        graphs,
        prefer_gqa=True,
        subcode_attention="exact",
    )

    assert key == "fused_cache_step_gqa"
    assert subcode_attention == "sdpa"
    assert fallback is True


def test_paged_kv_split_subcode_prefers_talker_stateful_gqa_seed():
    graphs = {
        "fused_cache_step_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml",
        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml",
    }

    key, subcode_attention, fallback = select_paged_kv_seed_key(
        graphs,
        prefer_gqa=True,
        split_subcode=True,
    )

    assert key == "talker_stateful_gqa"
    assert subcode_attention == "split"
    assert fallback is False


def test_paged_kv_split_subcode_top1_seed_prefers_talker_top1_gqa():
    graphs = {
        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml",
        "talker_top1_gqa": "talker_top1_sdpa_paged_gqa_seed.xml",
    }

    key, subcode_attention, fallback = select_paged_kv_seed_key(
        graphs,
        prefer_gqa=True,
        split_subcode=True,
        top1_seed=True,
    )

    assert key == "talker_top1_gqa"
    assert subcode_attention == "split"
    assert fallback is False


def test_fastest_profile_resolves_to_paged_kv_split_talker_variant():
    assert effective_runtime_options(FASTEST_PROFILE_NAME, "exact", "fused", "fp16") == (
        "no-cache",
        "exact",
        "fused",
        FASTEST_GRAPH_VARIANT,
    )

    defaults = fastest_runtime_defaults()
    assert defaults["graph_variant"] == "int8_sym_paged_talker_split"
    assert defaults["native_pipeline"] == "require"
    assert defaults["native_paged_kv"] == "require"
    assert defaults["native_paged_kv_gqa"] == "on"
    assert defaults["native_paged_kv_block_size"] == 16
    assert defaults["native_paged_kv_split_subcode"] == "on"


def test_runtime_resolves_relative_manifest_model_dir_from_ir_dir(monkeypatch, tmp_path):
    from qwen3_tts_ov import runtime as runtime_mod

    ir_dir = tmp_path / "voice_design"
    ir_dir.mkdir()
    manifest = {
        "model_dir": ".",
        "ids": {},
        "num_code_groups": 1,
        "sample_rate": 24000,
        "decode_upsample_rate": 80,
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "speech_decoder": {"256": "speech_decoder_t256.xml"},
        },
    }
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    seen = {}

    class FakeTokenizer:
        def __init__(self, model_dir):
            seen["model_dir"] = Path(model_dir)

    class FakeCore:
        available_devices = ["CPU"]

    class FakeCompiled:
        def create_infer_request(self):
            return object()

    def fake_compile_model(core, path, *args, **kwargs):
        seen.setdefault("compiled", []).append(Path(path))
        return FakeCompiled()

    monkeypatch.setattr(runtime_mod.ov, "Core", FakeCore)
    monkeypatch.setattr(runtime_mod, "Qwen2BPETokenizer", FakeTokenizer)
    monkeypatch.setattr(runtime_mod, "compile_model", fake_compile_model)

    runtime = OpenVINOQwen3TTS(
        ir_dir,
        "CPU",
        mode="cache",
        cache_step="fused",
        native_pipeline="require",
    )

    assert Path(runtime.model_dir) == ir_dir
    assert seen["model_dir"] == ir_dir
    assert [path.name for path in seen["compiled"]] == ["text_embedding.xml", "codec_embedding.xml"]
