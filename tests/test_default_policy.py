import json
from importlib import resources
from pathlib import Path

from qwen3_tts_ov.default_policy import resolve_generation_defaults, sampled_online_default_passed
from qwen3_tts_ov.server import online_batch_graph_capability


def test_packaged_default_policy_summary_is_available():
    summary = resources.files("qwen3_tts_ov").joinpath("default_policy_summary.json")

    assert summary.is_file()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["feature"] == "sampled_online_default"


def test_default_policy_is_conservative_without_summary(monkeypatch, tmp_path):
    monkeypatch.delenv("QWEN3_TTS_OV_DEFAULT_POLICY", raising=False)
    monkeypatch.setenv("QWEN3_TTS_OV_DEFAULT_POLICY_SUMMARY", str(tmp_path / "missing.json"))

    ok, source = sampled_online_default_passed()
    do_sample, repetition_penalty, metadata = resolve_generation_defaults(
        explicit_do_sample=None,
        explicit_repetition_penalty=None,
        fallback_do_sample=False,
        fallback_repetition_penalty=1.0,
        sampled_repetition_penalty=1.05,
    )

    assert ok is False
    assert source.startswith("missing:")
    assert do_sample is False
    assert repetition_penalty == 1.0
    assert metadata["default_policy_passed"] is False


def test_default_policy_summary_enables_sampled_defaults(monkeypatch, tmp_path):
    summary = tmp_path / "quality_summary.json"
    summary.write_text(
        json.dumps(
            {
                "feature": "sampled_online_default",
                "passed": True,
                "single_arch_gate": {
                    "passed": True,
                    "voice_design": {"passed": True},
                    "custom_voice": {"passed": True},
                    "voice_clone": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("QWEN3_TTS_OV_DEFAULT_POLICY", raising=False)
    monkeypatch.setenv("QWEN3_TTS_OV_DEFAULT_POLICY_SUMMARY", str(summary))

    do_sample, repetition_penalty, metadata = resolve_generation_defaults(
        explicit_do_sample=None,
        explicit_repetition_penalty=None,
        fallback_do_sample=False,
        fallback_repetition_penalty=1.0,
        sampled_repetition_penalty=1.05,
    )

    assert do_sample is True
    assert repetition_penalty == 1.05
    assert metadata["default_policy_passed"] is True


def test_default_policy_summary_requires_single_arch_gate(monkeypatch, tmp_path):
    summary = tmp_path / "quality_summary.json"
    summary.write_text(json.dumps({"feature": "sampled_online_default", "passed": True}), encoding="utf-8")
    monkeypatch.delenv("QWEN3_TTS_OV_DEFAULT_POLICY", raising=False)
    monkeypatch.setenv("QWEN3_TTS_OV_DEFAULT_POLICY_SUMMARY", str(summary))

    do_sample, repetition_penalty, metadata = resolve_generation_defaults(
        explicit_do_sample=None,
        explicit_repetition_penalty=None,
        fallback_do_sample=False,
        fallback_repetition_penalty=1.0,
        sampled_repetition_penalty=1.05,
    )

    assert do_sample is False
    assert repetition_penalty == 1.0
    assert metadata["default_policy_passed"] is False
    assert str(metadata["default_policy_source"]).startswith("summary_missing_single_arch_gate:")


def test_default_policy_env_override_respects_explicit_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN3_TTS_OV_DEFAULT_POLICY", "sampled-online")
    monkeypatch.setenv("QWEN3_TTS_OV_DEFAULT_POLICY_SUMMARY", str(tmp_path / "missing.json"))

    do_sample, repetition_penalty, metadata = resolve_generation_defaults(
        explicit_do_sample=False,
        explicit_repetition_penalty=1.2,
        fallback_do_sample=True,
        fallback_repetition_penalty=1.0,
        sampled_repetition_penalty=1.05,
    )

    assert do_sample is False
    assert repetition_penalty == 1.2
    assert metadata["default_policy_passed"] is True


def test_online_batching_no_longer_rejects_sampling_at_python_layer():
    source = Path("qwen3_tts_ov/online_batch.py").read_text(encoding="utf-8")

    assert "currently supports greedy do_sample=false only" not in source
    assert "currently requires repetition_penalty=1.0" not in source


def test_online_batch_graph_capability_requires_batch_safe_subcode_for_sampled_batch(tmp_path):
    ir_dir = tmp_path
    for name in (
        "talker_stateful_batch_gqa.xml",
        "fused_cache_step_batch_gqa.xml",
        "subcode_greedy_cached.xml",
    ):
        (ir_dir / name).write_text("<xml/>", encoding="utf-8")
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
        },
        "graph_variants": {
            "int8_sym_batch_fused_gqa": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_batch_gqa": "talker_stateful_batch_gqa.xml",
                        "fused_cache_step_batch_gqa": "fused_cache_step_batch_gqa.xml",
                    }
                }
            }
        },
    }

    row_by_row = online_batch_graph_capability(
        ir_dir,
        manifest,
        graph_variant="int8_sym_batch_fused_gqa",
        sampled_batch_subcode="off",
        max_batch_size=4,
    )
    sampled_batch = online_batch_graph_capability(
        ir_dir,
        manifest,
        graph_variant="int8_sym_batch_fused_gqa",
        sampled_batch_subcode="on",
        max_batch_size=4,
    )

    assert row_by_row["ok"] is True
    assert sampled_batch["ok"] is False
    assert sampled_batch["reason"] == "missing:subcode_greedy_cached_batch"


def test_online_batch_graph_capability_accepts_minimal_profile_without_fused_decode(tmp_path):
    ir_dir = tmp_path
    for name in (
        "talker_stateful_batch_gqa.xml",
        "subcode_greedy_cached.xml",
    ):
        (ir_dir / name).write_text("<xml/>", encoding="utf-8")
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
        },
        "graph_variants": {
            "int8_sym_batch_fused_gqa": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_batch_gqa": "talker_stateful_batch_gqa.xml",
                    }
                }
            }
        },
    }

    minimal = online_batch_graph_capability(
        ir_dir,
        manifest,
        graph_variant="int8_sym_batch_fused_gqa",
        sampled_batch_subcode="off",
        max_batch_size=4,
        require_fused_decode=False,
    )
    fused_required = online_batch_graph_capability(
        ir_dir,
        manifest,
        graph_variant="int8_sym_batch_fused_gqa",
        sampled_batch_subcode="off",
        max_batch_size=4,
        require_fused_decode=True,
    )

    assert minimal["ok"] is True
    assert minimal["fused_available"] is False
    assert minimal["fused_decode_required"] is False
    assert fused_required["ok"] is False
    assert fused_required["reason"] == "missing:paged_kv_seed.fused_cache_step_batch_gqa"


def test_online_batch_graph_capability_accepts_batch_safe_subcode(tmp_path):
    ir_dir = tmp_path
    for name in (
        "talker_stateful_batch_gqa.xml",
        "fused_cache_step_batch_gqa.xml",
        "subcode_greedy_cached.xml",
        "subcode_greedy_cached_batch.xml",
    ):
        (ir_dir / name).write_text("<xml/>", encoding="utf-8")
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "subcode_greedy_cached_batch": "subcode_greedy_cached_batch.xml",
        },
        "graph_variants": {
            "int8_sym_batch_fused_gqa": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_batch_gqa": "talker_stateful_batch_gqa.xml",
                        "fused_cache_step_batch_gqa": "fused_cache_step_batch_gqa.xml",
                    }
                }
            }
        },
    }

    capability = online_batch_graph_capability(
        ir_dir,
        manifest,
        graph_variant="int8_sym_batch_fused_gqa",
        sampled_batch_subcode="on",
        max_batch_size=4,
    )

    assert capability["ok"] is True
    assert capability["batch_subcode_key"] == "subcode_greedy_cached_batch"
