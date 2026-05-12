import json

import pytest

from qwen3_tts_ov.cache import build_ov_cache_config, default_ov_cache_root, normalize_ov_cache_mode, resolve_ov_cache_dir
from qwen3_tts_ov.cache_warmup import collect_warmup_tasks, select_buckets
from qwen3_tts_ov.manifest import resolve_ir_dir
from qwen3_tts_ov.profiles import (
    effective_codegen_unroll,
    effective_runtime_options,
    missing_graph_variant_message,
    scheduled_codegen_unrolls,
)
from qwen3_tts_ov.runtime import OpenVINOQwen3TTS


def test_default_ov_cache_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("QWEN3_TTS_OV_CACHE_DIR", raising=False)

    assert default_ov_cache_root() == tmp_path / "xdg-cache" / "qwen3-tts-ov" / "openvino-cache"


def test_build_ov_cache_config_can_disable_cache(tmp_path):
    assert build_ov_cache_config(tmp_path, ov_cache_mode="optimize_size") == {
        "CACHE_DIR": str(tmp_path),
        "CACHE_MODE": "OPTIMIZE_SIZE",
    }
    assert build_ov_cache_config(tmp_path, disable_ov_cache=True) == {}


def test_cache_warmup_prefers_common_low_latency_bucket():
    available = {80: "cache80.xml", 96: "cache96.xml", 112: "cache112.xml", 128: "cache128.xml"}

    assert select_buckets(available, "warmup") == {112: "cache112.xml"}


def test_resolve_cache_dir_is_namespaced(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN3_TTS_OV_CACHE_DIR", str(tmp_path / "cache-root"))
    manifest = {"tts_model_type": "voice_design", "graphs": {"text_embedding": "text_embedding.xml"}}

    first = resolve_ov_cache_dir(
        tmp_path / "ir",
        manifest,
        device="GPU",
        decoder_device="GPU",
        mode="cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="fp16",
        precision_hint="f16",
        compile_config={},
    )
    second = resolve_ov_cache_dir(
        tmp_path / "ir",
        manifest,
        device="GPU",
        decoder_device="CPU",
        mode="cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="fp16",
        precision_hint="f16",
        compile_config={},
    )

    assert str(first).startswith(str(tmp_path / "cache-root"))
    assert first != second


def test_normalize_ov_cache_mode_accepts_cli_values():
    assert normalize_ov_cache_mode("optimize_speed") == "OPTIMIZE_SPEED"
    assert normalize_ov_cache_mode("optimize-size") == "OPTIMIZE_SIZE"


def test_collect_warmup_tasks_uses_strategy_stream_decoders(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "code_frame_embedding": "code_frame_embedding.xml",
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
            "speech_decoder": {"64": "speech_decoder_t64.xml"},
        },
        "streaming_decoder": {
            "left_context_frames": 25,
            "strategies": {
                "low_latency": {
                    "initial_chunk_frames": 8,
                    "chunk_frames": 12,
                    "left_context_frames": 25,
                }
            },
            "contexts": {
                "0": {"8": "speech_decoder_stream_c0_t8.xml"},
                "25": {"12": "speech_decoder_stream_c25_t12.xml"},
            },
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(ir_dir, graphs="core,stream,buckets", preload_buckets="warmup")

    labels = [task.label for task in tasks]
    assert "core:text_embedding" in labels
    assert "stream:c0_t8" in labels
    assert "stream:c25_t12" in labels
    assert "bucket:fused_cache_step_buckets:128" in labels


def test_collect_warmup_tasks_reports_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="OpenVINO IR manifest not found"):
        collect_warmup_tasks(tmp_path / "missing-ir")


def test_default_voice_design_ir_falls_back_to_legacy_openvino_full(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "openvino_full"
    legacy.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "code_frame_embedding": "code_frame_embedding.xml",
        },
    }
    with open(legacy / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    assert resolve_ir_dir("openvino/voice_design", fallback_to_local_voice_design=True) == legacy.relative_to(tmp_path)
    assert resolve_ir_dir("auto", fallback_to_local_voice_design=True) == legacy.relative_to(tmp_path)
    tasks, _ = collect_warmup_tasks("openvino/voice_design", graphs="core")
    assert [task.label for task in tasks] == [
        "core:text_embedding",
        "core:codec_embedding",
        "core:code_frame_embedding",
    ]


def test_cache_warmup_default_auto_uses_local_voice_design(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    exported = tmp_path / "openvino" / "voice_design"
    exported.mkdir(parents=True)
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {"text_embedding": "text_embedding.xml"},
    }
    with open(exported / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks("auto", graphs="core")

    assert [task.label for task in tasks] == ["core:text_embedding"]


def test_realtime_int8_expands_to_fused_exact_int8_variant():
    assert effective_runtime_options("realtime-int8", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "int8_fused",
    )


def test_realtime_int8_sym_expands_to_fused_exact_sym_unroll4():
    mode, kernel, step, variant = effective_runtime_options("realtime-int8-sym", "sdpa", "split", "fp16")

    assert (mode, kernel, step, variant) == ("cache", "exact", "fused", "int8_sym_fused")
    assert effective_codegen_unroll("realtime-int8-sym", variant, "profile") == 4
    assert effective_codegen_unroll("realtime-int8-sym", variant, "1") == 1
    assert effective_codegen_unroll("realtime-int8-sym", variant, "12") == 12
    assert scheduled_codegen_unrolls("ll-v2", 4) == (4, 12)


def test_rms_realtime_modes_expand_to_fused_variants():
    assert effective_runtime_options("realtime-fp16-fused-rms", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "fp16_fused_rms",
    )
    assert effective_runtime_options("realtime-int8-sym-fused-rms", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "int8_sym_fused_rms",
    )
    assert effective_runtime_options("realtime-fp16-sdpa-fused-rms", "exact", "split", "fp16") == (
        "cache",
        "sdpa",
        "fused",
        "fp16_sdpa_fused_rms",
    )
    assert effective_codegen_unroll("cache", "fp16_fused_rms", "profile") == 4
    assert effective_codegen_unroll("cache", "int8_sym_fused_rms", "profile") == 4


def test_cachedsub_realtime_modes_expand_to_fused_variants():
    assert effective_runtime_options("realtime-fp16-fused-cachedsub", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "fp16_fused_cachedsub",
    )
    assert effective_runtime_options("realtime-int8-sym-fused-cachedsub", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "int8_sym_fused_cachedsub",
    )
    assert effective_runtime_options("realtime-int8-sym-fused-cachedsub-norepeat", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "int8_sym_fused_cachedsub",
    )
    assert effective_runtime_options("realtime-fp16-sdpa-fused-cachedsub", "exact", "split", "fp16") == (
        "cache",
        "sdpa",
        "fused",
        "fp16_sdpa_fused_cachedsub",
    )
    assert effective_runtime_options("realtime-int8-sym-sdpa-fused-cachedsub", "exact", "split", "fp16") == (
        "cache",
        "sdpa",
        "fused",
        "int8_sym_sdpa_fused_cachedsub",
    )
    assert effective_runtime_options("realtime-int8-sym-sdpa-fused-cachedsub-norepeat", "exact", "split", "fp16") == (
        "cache",
        "sdpa",
        "fused",
        "int8_sym_sdpa_fused_cachedsub",
    )
    assert effective_codegen_unroll("cache", "fp16_fused_cachedsub", "profile") == 4
    assert effective_codegen_unroll("cache", "int8_sym_fused_cachedsub", "profile") == 4
    assert effective_codegen_unroll("cache", "fp16_sdpa_fused_cachedsub", "profile") == 4
    assert effective_codegen_unroll("cache", "int8_sym_sdpa_fused_cachedsub", "profile") == 4
    assert effective_codegen_unroll("realtime-int8-sym-sdpa-fused-cachedsub-norepeat", "fp16", "profile") == 4
    assert effective_runtime_options("realtime-int8-sym-fused-cachedsub-rms", "sdpa", "split", "fp16") == (
        "cache",
        "exact",
        "fused",
        "int8_sym_fused_cachedsub_rms",
    )
    assert effective_codegen_unroll("cache", "int8_sym_fused_cachedsub_rms", "profile") == 4


def test_missing_rms_variant_message_reports_export_and_compression_hints():
    fp16_hint = missing_graph_variant_message("fp16_fused_rms", "none")
    int8_hint = missing_graph_variant_message("int8_sym_fused_rms", "fp16_fused_rms")

    assert "--rms-export-mode canonical" in fp16_hint
    assert "--source-variant fp16_fused_rms" in int8_hint
    assert "--fused-subcode-mode cached" in missing_graph_variant_message("fp16_fused_cachedsub", "none")
    assert "--source-variant fp16_fused_cachedsub" in missing_graph_variant_message(
        "int8_sym_fused_cachedsub",
        "fp16_fused_cachedsub",
    )
    assert "--fused-cache-kernels sdpa" in missing_graph_variant_message(
        "int8_sym_sdpa_fused_cachedsub",
        "fp16_sdpa_fused_cachedsub",
    )
    assert "--source-variant fp16_fused_cachedsub_rms" in missing_graph_variant_message(
        "int8_sym_fused_cachedsub_rms",
        "fp16_fused_cachedsub_rms",
    )


def test_realtime_int8_warmup_uses_int8_fused_bucket(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
        },
        "graph_variants": {
            "int8_fused": {
                "graphs": {
                    "fused_cache_step_buckets": {
                        "exact": {"128": "fused_cache_step_exact_cache128_int8_fused.xml"}
                    }
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(ir_dir, mode="realtime-int8", graphs="buckets")

    assert len(tasks) == 1
    assert tasks[0].graph == "fused_cache_step_exact_cache128_int8_fused.xml"


def test_realtime_int8_sym_warmup_uses_unroll_variant_bucket(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
            "fused_cache_step_unroll_buckets": {
                "exact": {
                    "4": {"128": "fused_cache_step_unroll4_exact_cache128.xml"}
                }
            },
        },
        "graph_variants": {
            "int8_sym_fused": {
                "graphs": {
                    "fused_cache_step_unroll_buckets": {
                        "exact": {
                            "4": {"128": "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"}
                        }
                    }
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(
        ir_dir,
        mode="realtime-int8-sym",
        graphs="buckets",
        codegen_decode_unroll="auto",
    )

    assert len(tasks) == 1
    assert tasks[0].label == "bucket:fused_cache_step_unroll4:128"
    assert tasks[0].graph == "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"


def test_realtime_int8_sym_ll_v2_warmup_includes_initial_and_steady_unrolls(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
            "fused_cache_step_unroll_buckets": {
                "exact": {
                    "4": {"128": "fused_cache_step_unroll4_exact_cache128.xml"},
                    "12": {"128": "fused_cache_step_unroll12_exact_cache128.xml"},
                }
            },
        },
        "graph_variants": {
            "int8_sym_fused": {
                "graphs": {
                    "fused_cache_step_unroll_buckets": {
                        "exact": {
                            "4": {"128": "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"},
                            "12": {"128": "fused_cache_step_unroll12_exact_cache128_int8_sym_fused.xml"},
                        }
                    }
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(
        ir_dir,
        mode="realtime-int8-sym",
        graphs="buckets",
        codegen_schedule="ll-v2",
    )

    assert [task.label for task in tasks] == [
        "bucket:fused_cache_step_unroll4:128",
        "bucket:fused_cache_step_unroll12:128",
    ]


def test_realtime_int8_sym_warmup_includes_decode_unroll_bucket(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
            "fused_cache_step_unroll_buckets": {
                "exact": {"4": {"128": "fused_cache_step_unroll4_exact_cache128.xml"}}
            },
            "fused_cache_decode_unroll_stateful_mask_buckets": {
                "exact": {"4": {"128": "fused_cache_decode_unroll4_exact_statefulmask_cache128.xml"}}
            },
        },
        "graph_variants": {
            "int8_sym_fused": {
                "graphs": {
                    "fused_cache_step_unroll_buckets": {
                        "exact": {"4": {"128": "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"}}
                    },
                    "fused_cache_decode_unroll_stateful_mask_buckets": {
                        "exact": {
                            "4": {
                                "128": "fused_cache_decode_unroll4_exact_statefulmask_cache128_int8_sym_fused.xml"
                            }
                        }
                    },
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    default_tasks, _ = collect_warmup_tasks(ir_dir, mode="realtime-int8-sym", graphs="buckets")
    assert [task.label for task in default_tasks] == ["bucket:fused_cache_step_unroll4:128"]

    tasks, _ = collect_warmup_tasks(
        ir_dir,
        mode="realtime-int8-sym",
        graphs="buckets",
        codegen_decode_unroll="auto",
    )

    assert [task.label for task in tasks] == [
        "bucket:fused_cache_step_unroll4:128",
        "bucket:fused_cache_decode_unroll4:128",
    ]
    assert tasks[1].graph == "fused_cache_decode_unroll4_exact_statefulmask_cache128_int8_sym_fused.xml"


def test_realtime_int8_sym_norepeat_warmup_uses_norepeat_unroll_and_decode(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_unroll_norepeat_buckets": {
                "exact": {"4": {"96": "fused_cache_step_unroll4_exact_norepeat_cache96.xml"}}
            },
            "fused_cache_decode_unroll_norepeat_buckets": {
                "exact": {"4": {"96": "fused_cache_decode_unroll4_exact_norepeat_cache96.xml"}}
            },
        },
        "graph_variants": {
            "int8_sym_fused": {
                "graphs": {
                    "fused_cache_step_unroll_norepeat_buckets": {
                        "exact": {"4": {"96": "fused_cache_step_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"}}
                    },
                    "fused_cache_decode_unroll_norepeat_buckets": {
                        "exact": {
                            "4": {"96": "fused_cache_decode_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"}
                        }
                    },
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(
        ir_dir,
        mode="realtime-int8-sym-norepeat",
        graphs="buckets",
        codegen_decode_unroll="auto",
        preferred_cache_bucket=96,
    )

    assert [task.label for task in tasks] == [
        "bucket:fused_cache_step_unroll4:96",
        "bucket:fused_cache_decode_unroll4:96",
    ]
    assert tasks[0].graph == "fused_cache_step_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"
    assert tasks[1].graph == "fused_cache_decode_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"


def test_realtime_int8_sym_warmup_includes_large_decode_unroll_bucket(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "fused_cache_step_unroll_buckets": {
                "exact": {"12": {"128": "fused_cache_step_unroll12_exact_cache128.xml"}}
            },
            "fused_cache_decode_unroll_stateful_mask_buckets": {
                "exact": {"12": {"128": "fused_cache_decode_unroll12_exact_statefulmask_cache128.xml"}}
            },
        },
        "graph_variants": {
            "int8_sym_fused": {
                "graphs": {
                    "fused_cache_step_unroll_buckets": {
                        "exact": {"12": {"128": "fused_cache_step_unroll12_exact_cache128_int8_sym_fused.xml"}}
                    },
                    "fused_cache_decode_unroll_stateful_mask_buckets": {
                        "exact": {
                            "12": {
                                "128": "fused_cache_decode_unroll12_exact_statefulmask_cache128_int8_sym_fused.xml"
                            }
                        }
                    },
                }
            }
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(
        ir_dir,
        mode="realtime-int8-sym",
        graphs="buckets",
        codegen_unroll=12,
        codegen_decode_unroll="auto",
    )

    assert [task.label for task in tasks] == [
        "bucket:fused_cache_step_unroll12:128",
        "bucket:fused_cache_decode_unroll12:128",
    ]
    assert tasks[1].graph == "fused_cache_decode_unroll12_exact_statefulmask_cache128_int8_sym_fused.xml"


def test_realtime_int8_missing_variant_reports_compression_hint(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design", "graphs": {}}, handle)

    with pytest.raises(ValueError, match="compress_openvino_weights.py"):
        collect_warmup_tasks(ir_dir, mode="realtime-int8", graphs="buckets")


def test_runtime_bucket_loader_prefers_int8_fused_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {"fused_cache_step_buckets": {"exact": {"128": "fp16.xml"}}}
    variant_graphs = {"fused_cache_step_buckets": {"exact": {"128": "int8.xml"}}}

    assert runtime._load_fused_cache_bucket_graphs(graphs, "exact", variant_graphs) == {128: "int8.xml"}


def test_runtime_bucket_selector_reuses_compiled_larger_bucket():
    available = {80: "cache80.xml", 112: "cache112.xml", 128: "cache128.xml"}

    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 97, [128]) == 128
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 97, []) == 112
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 72, [], preferred_min_bucket=80) == 80
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 72, [], preferred_min_bucket=None) == 80


def test_cache_warmup_select_buckets_uses_preferred_cache_bucket():
    available = {80: "cache80.xml", 96: "cache96.xml", 112: "cache112.xml"}

    assert select_buckets(available, "warmup", preferred_cache_bucket=96) == {96: "cache96.xml"}
    assert select_buckets(available, "warmup", preferred_cache_bucket="none") == {80: "cache80.xml"}


def test_runtime_codegen_schedule_selects_low_latency_then_steady_unroll():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.codegen_schedule = "ll-v2"
    runtime.codegen_unroll = 4
    runtime.fused_cache_unroll_bucket_graphs_by_step = {
        4: {128: "u4.xml"},
        12: {128: "u12.xml"},
    }

    assert runtime.select_codegen_unroll_for_step(0) == 4
    assert runtime.select_codegen_unroll_for_step(8) == 12


def test_runtime_unroll_bucket_loader_prefers_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {
        "fused_cache_step_unroll_buckets": {
            "exact": {"4": {"128": "fp16.xml"}}
        }
    }
    variant_graphs = {
        "fused_cache_step_unroll_buckets": {
            "exact": {"4": {"128": "int8-sym.xml"}}
        }
    }

    assert runtime._load_fused_cache_unroll_bucket_graphs_by_step(graphs, "exact", variant_graphs) == {
        4: {128: "int8-sym.xml"}
    }


def test_runtime_decode_unroll_bucket_loader_prefers_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {
        "fused_cache_decode_unroll_stateful_mask_buckets": {
            "exact": {"4": {"96": "fp16.xml", "128": "fp16-128.xml"}}
        }
    }
    variant_graphs = {
        "fused_cache_decode_unroll_stateful_mask_buckets": {
            "exact": {"4": {"96": "int8-sym.xml"}}
        }
    }

    assert runtime._load_fused_cache_decode_unroll_bucket_graphs_by_step(
        graphs,
        "exact",
        variant_graphs,
        "fused_cache_decode_unroll_stateful_mask_buckets",
    ) == {
        4: {96: "int8-sym.xml", 128: "fp16-128.xml"}
    }
