import argparse
import json

from qwen3_tts_ov.build_fastest import build_fastest_steps, manifest_has_fastest_variant


def make_args(**overrides):
    values = {
        "model_type": "voice_design",
        "model": None,
        "out_dir": None,
        "device": "GPU",
        "decoder_device": None,
        "encoder_device": None,
        "prompt_device": None,
        "npu_offload": "off",
        "ov_cache_dir": None,
        "disable_ov_cache": False,
        "preload_buckets": "warmup",
        "warmup_graphs": "core,stream,buckets",
        "warmup_strategy": "smooth",
        "graph_set": "production",
        "clean": False,
        "clean_native": False,
        "skip_submodule": True,
        "skip_native": False,
        "skip_export": False,
        "skip_compress": False,
        "skip_warmup": False,
        "force_native": False,
        "force_export": False,
        "force_compress": False,
        "dry_run": True,
        "output_json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_manifest_has_fastest_variant_requires_paged_talker_cached_subcode_and_streams():
    manifest = {
        "graphs": {"subcode_greedy_cached": "subcode_greedy_cached.xml"},
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed_int8_sym_paged_talker_split.xml"
                    }
                }
            }
        },
        "streaming_decoder": {
            "contexts": {
                "0": {"8": "speech_decoder_stream_c0_t8.xml"},
                "25": {"24": "speech_decoder_stream_c25_t24.xml"},
            }
        },
    }

    assert manifest_has_fastest_variant(manifest) is True

    manifest["streaming_decoder"]["contexts"]["25"] = {}
    assert manifest_has_fastest_variant(manifest) is False


def test_build_fastest_plans_default_voice_design_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(force_native=True)

    steps = build_fastest_steps(args)
    commands_by_name = {step.name: step.command for step in steps if step.command}

    assert commands_by_name["native"][-1] == "scripts/build_native_codegen.py"
    export = commands_by_name["export"]
    assert export[:3][-2:] == ["-m", "qwen3_tts_ov"]
    assert "export" in export
    assert export[export.index("--model-type") + 1] == "voice_design"
    assert export[export.index("--out-dir") + 1] == "openvino/voice_design"
    assert "--export-paged-kv-seed" in export
    assert "--paged-kv-subcode-attention-kernels" in export
    assert "--skip-fixed-cache-graphs" in export
    assert export[export.index("--cache-buckets") + 1] == "96"
    assert export[export.index("--fused-cache-unroll-steps") + 1] == ""
    assert export[export.index("--fused-cache-decode-unroll-steps") + 1] == ""
    assert export[export.index("--paged-kv-unroll-steps") + 1] == ""
    assert export[export.index("--stream-decoder-first-chunks") + 1] == "8,12"
    assert export[export.index("--stream-decoder-chunks") + 1] == "12,24"
    assert export[export.index("--stream-decoder-input-shape") + 1] == "static"
    compress = commands_by_name["compress"]
    assert compress[-4:] == ["--ir-dir", "openvino/voice_design", "--preset", "fastest"]
    warmup = commands_by_name["warmup"]
    assert warmup[warmup.index("--realtime-profile") + 1] == "fastest"
    assert warmup[warmup.index("--warmup-strategy") + 1] == "smooth"


def test_build_fastest_can_warm_gpu_npu_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(skip_native=True, skip_export=True, skip_compress=True, npu_offload="audio")

    steps = build_fastest_steps(args)
    warmup = next(step.command for step in steps if step.name == "warmup")

    assert warmup[warmup.index("--npu-offload") + 1] == "audio"


def test_build_fastest_skips_export_and_compress_when_fastest_manifest_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    manifest = {
        "graphs": {"subcode_greedy_cached": "subcode_greedy_cached.xml"},
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {"paged_kv_seed": {"talker_stateful_gqa": "talker_stateful.xml"}}
            }
        },
        "streaming_decoder": {
            "contexts": {
                "0": {"12": "speech_decoder_stream_c0_t12.xml"},
                "25": {"24": "speech_decoder_stream_c25_t24.xml"},
            }
        },
    }
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = make_args(skip_native=True)

    steps = build_fastest_steps(args)
    skipped = {step.name: step.skip_reason for step in steps if step.skip_reason}

    assert "manifest.json already exists" in skipped["export"]
    assert skipped["compress"] == "fastest graph variant already exists"


def test_build_fastest_base_export_includes_clone_graphs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_type="base", skip_native=True, skip_compress=True, skip_warmup=True)

    steps = build_fastest_steps(args)
    export = next(step.command for step in steps if step.name == "export")

    assert export[export.index("--model") + 1] == "models/Qwen3-TTS-12Hz-1.7B-Base"
    assert export[export.index("--out-dir") + 1] == "openvino/base"
    assert "--export-clone-graphs" in export


def test_build_fastest_compat_graph_set_keeps_diagnostic_unrolls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(graph_set="compat", skip_native=True, skip_compress=True, skip_warmup=True)

    steps = build_fastest_steps(args)
    export = next(step.command for step in steps if step.name == "export")

    assert "--skip-fixed-cache-graphs" not in export
    assert export[export.index("--cache-buckets") + 1] == "96,128,192,256,320,384"
    assert export[export.index("--fused-cache-unroll-steps") + 1] == "4,6,8,12"
    assert export[export.index("--fused-cache-decode-unroll-steps") + 1] == "4,8,12"
    assert export[export.index("--stream-decoder-input-shape") + 1] == "static"


def test_build_fastest_clean_ignores_existing_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    (ir_dir / "manifest.json").write_text("{}", encoding="utf-8")
    args = make_args(clean=True, skip_native=True, skip_compress=True, skip_warmup=True)

    steps = build_fastest_steps(args)
    skipped = {step.name for step in steps if step.skip_reason}
    export = next(step.command for step in steps if step.name == "export")

    assert "export" not in skipped
    assert export[export.index("--out-dir") + 1] == "openvino/voice_design"
