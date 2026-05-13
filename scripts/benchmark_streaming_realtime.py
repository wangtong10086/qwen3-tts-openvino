import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from qwen3_tts_ov.profiles import (
    REALTIME_BENCHMARK_PROFILE_OPTIONS,
    apply_realtime_profile,
)


DEFAULT_TEXT = "你好，这是一次用于测试实时流式合成性能的 OpenVINO 语音生成。"
DEFAULT_INSTRUCT = "用自然、清晰的中文女声朗读。"
NATIVE_COMPILE_ENV_KEYS = {
    "native_latency_high": "QWEN3_TTS_OV_NATIVE_LATENCY_HIGH",
    "native_precision_hint": "QWEN3_TTS_OV_NATIVE_PRECISION_HINT",
    "native_large_allocations": "QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS",
    "native_performance_hint": "QWEN3_TTS_OV_NATIVE_PERFORMANCE_HINT",
    "native_num_streams": "QWEN3_TTS_OV_NATIVE_NUM_STREAMS",
    "native_model_priority": "QWEN3_TTS_OV_NATIVE_MODEL_PRIORITY",
    "native_gpu_queue_priority": "QWEN3_TTS_OV_NATIVE_GPU_QUEUE_PRIORITY",
    "native_gpu_host_task_priority": "QWEN3_TTS_OV_NATIVE_GPU_HOST_TASK_PRIORITY",
    "native_gpu_queue_throttle": "QWEN3_TTS_OV_NATIVE_GPU_QUEUE_THROTTLE",
    "native_dynamic_quantization_group_size": "QWEN3_TTS_OV_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE",
    "native_activations_scale_factor": "QWEN3_TTS_OV_NATIVE_ACTIVATIONS_SCALE_FACTOR",
    "native_paged_kv_static_decode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE",
    "native_paged_kv_static_blocks": "QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_BLOCKS",
    "native_paged_kv_static_decode_mode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE",
    "native_paged_kv_score_aggregation": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION",
    "native_paged_kv_split_subcode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE",
    "native_paged_kv_split_subcode_mode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE",
    "native_subcode_device": "QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE",
}
PROFILE_SETS = {
    "fastest-gate": "fastest",
    "rtf-p90-gate": "fastest",
}
PROFILES = {
    "fastest": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_talker_split",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "require",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "16",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "preferred_cache_bucket": "0",
        "repetition_penalty": "1.0",
    },
    "fastest_large_alloc": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_talker_split",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "require",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "16",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "native_large_allocations": "1",
        "preferred_cache_bucket": "0",
        "repetition_penalty": "1.0",
    },
    "hybrid_fastest_paged_kv": {
        "mode": "fastest",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused_cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "native_pipeline": "require",
        "native_buffer_reuse": "1",
        "native_paged_kv_hybrid": "1",
        "native_paged_kv_hybrid_prefix_frames": "48",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "preferred_cache_bucket": "96",
        "repetition_penalty": "1.0",
    },
    "fastest_sdpa_int8_sym": {
        "mode": "cache",
        "cache_kernel": "sdpa",
        "cache_step": "fused",
        "graph_variant": "int8_sym_sdpa_fused_cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "native_pipeline": "require",
        "native_buffer_reuse": "1",
        "preferred_cache_bucket": "96",
        "repetition_penalty": "1.0",
    },
    "fastest_sdpa_int8_sym_async_decode": {
        "mode": "cache",
        "cache_kernel": "sdpa",
        "cache_step": "fused",
        "graph_variant": "int8_sym_sdpa_fused_cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "native_pipeline": "require",
        "native_async_decode": "1",
        "native_buffer_reuse": "1",
        "preferred_cache_bucket": "96",
        "repetition_penalty": "1.0",
    },
    "fastest_sdpa_fp16": {
        "mode": "cache",
        "cache_kernel": "sdpa",
        "cache_step": "fused",
        "graph_variant": "fp16_sdpa_fused_cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "native_pipeline": "require",
        "native_buffer_reuse": "1",
        "preferred_cache_bucket": "96",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_sdpa_subcode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_subcode_attention": "sdpa",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_recompute": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_recompute_async_decode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_async_decode": "1",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_recompute_subcode_cpu": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute",
        "native_codegen_device": "GPU",
        "native_subcode_device": "CPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_cached_exact": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "cached_exact",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_recompute_exact": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute_exact",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_subcode_int8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_talker_int8_subcode_int8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_talker_split",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_talker_int8_subcode_int8_no_score_aggregation": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_talker_split",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_score_aggregation": "0",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_split_talker_int8_subcode_int8_block16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_talker_split",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "16",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_large_alloc": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "native_large_allocations": "1",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_subcode_exact": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_subcode_attention": "exact",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_blocks": "128",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode_blocks16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_decode_mode": "full",
        "native_paged_kv_static_blocks": "16",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode_blocks32": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_decode_mode": "full",
        "native_paged_kv_static_blocks": "32",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode_minimal_blocks16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_decode_mode": "minimal",
        "native_paged_kv_static_blocks": "16",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode_minimal_blocks16_large_alloc": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_decode_mode": "minimal",
        "native_paged_kv_static_blocks": "16",
        "native_large_allocations": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_static_decode_minimal_blocks32": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_static_decode": "1",
        "native_paged_kv_static_decode_mode": "minimal",
        "native_paged_kv_static_blocks": "32",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_codegen_cpu": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "CPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_unroll4": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_unroll": "4",
        "native_paged_kv_experimental_unroll": "1",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_block16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "16",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_block4": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "4",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_kvcache_u8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_cacheinput_f16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_cacheinput_u8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_score_aggregation": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_paged_kv_score_aggregation": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_no_score_aggregation": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_paged_kv_score_aggregation": "0",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_score_aggregation_cacheinput_u8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_paged_kv_score_aggregation": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_kvcache_bf16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "bf16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_dq32": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_dynamic_quantization_group_size": "32",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_dq32_no_score_aggregation": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_dynamic_quantization_group_size": "32",
        "native_paged_kv_score_aggregation": "0",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_int8_subcode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_subcode",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_dq64": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_dynamic_quantization_group_size": "64",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_kvcache_u8_block16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "u8",
        "native_paged_kv_block_size": "16",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_remote_embed": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_remote_embed": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_host_embed": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_remote_embed": "0",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_native_prompt": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_prompt": "1",
        "native_prompt_device": "CPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_latency_high": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "native_latency_high": "1",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_num_streams_1": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_num_streams": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_latency_hint": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_performance_hint": "LATENCY",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_throughput_hint": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_performance_hint": "THROUGHPUT",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_async_decode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_async_decode": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_sync_decode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_async_decode": "0",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_precision_f32": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_precision_hint": "f32",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_precision_default": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_precision_hint": "default",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_latency_high": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "native_latency_high": "1",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_block16": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "16",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_block4": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "4",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_kvcache_u8": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_int8_sym": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_kv_seed",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_int8_sym": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_paged_kv_seed",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_gqa_int8_asym": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_asym_paged_kv_seed",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_async_decode": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_async_decode": "1",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
    "paged_kv_expanded_block32": {
        "mode": "no-cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "32",
        "native_codegen_device": "GPU",
        "native_buffer_reuse": "0",
        "repetition_penalty": "1.0",
    },
}


def realtime_benchmark_option_to_profile(option: dict) -> dict:
    realtime_profile = str(option.get("realtime_profile") or "fp16")
    mode, cache_kernel, cache_step, graph_variant = apply_realtime_profile(
        realtime_profile,
        "cache",
        "exact",
        "fused",
        "fp16",
    )
    requested_modes = {
        "fastest": "fastest",
        "auto": "fastest",
        "int8": "realtime-int8",
        "int8-sym": "realtime-int8-sym",
        "int8-sym-norepeat": "realtime-int8-sym-norepeat",
        "fp16-fused-rms": "realtime-fp16-fused-rms",
        "int8-sym-fused-rms": "realtime-int8-sym-fused-rms",
        "fp16-sdpa-fused-rms": "realtime-fp16-sdpa-fused-rms",
        "int8-sym-sdpa-fused-rms": "realtime-int8-sym-sdpa-fused-rms",
        "fp16-fused-cachedsub": "realtime-fp16-fused-cachedsub",
        "int8-sym-fused-cachedsub": "realtime-int8-sym-fused-cachedsub",
        "int8-sym-fused-cachedsub-norepeat": "realtime-int8-sym-fused-cachedsub-norepeat",
        "fp16-sdpa-fused-cachedsub": "realtime-fp16-sdpa-fused-cachedsub",
        "int8-sym-sdpa-fused-cachedsub": "realtime-int8-sym-sdpa-fused-cachedsub",
        "int8-sym-sdpa-fused-cachedsub-norepeat": "realtime-int8-sym-sdpa-fused-cachedsub-norepeat",
        "fp16-fused-cachedsub-rms": "realtime-fp16-fused-cachedsub-rms",
        "int8-sym-fused-cachedsub-rms": "realtime-int8-sym-fused-cachedsub-rms",
    }
    config = {
        "mode": requested_modes.get(realtime_profile, mode),
        "cache_kernel": cache_kernel,
        "cache_step": cache_step,
        "graph_variant": graph_variant,
        "codegen_unroll": str(option.get("codegen_unroll", "1")),
        "codegen_schedule": str(option.get("codegen_schedule", "current")),
        "codegen_decode_unroll": str(option.get("codegen_decode_unroll", "off")),
    }
    if "preferred_cache_bucket" in option:
        config["preferred_cache_bucket"] = str(option["preferred_cache_bucket"])
    if "repetition_penalty" in option:
        config["repetition_penalty"] = str(option["repetition_penalty"])
    if realtime_profile in {"fastest", "auto"} or str(option.get("codegen_decode_unroll", "off")) != "off":
        config.setdefault("native_pipeline", "require")
        config.setdefault("native_buffer_reuse", "1")
    return config


for _name, _option in REALTIME_BENCHMARK_PROFILE_OPTIONS.items():
    PROFILES.setdefault(_name, realtime_benchmark_option_to_profile(_option))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((pct / 100.0) * (len(ordered) - 1))
    index = min(len(ordered) - 1, max(0, int(index)))
    return float(ordered[index])


def summarize_results(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for item in results:
        if item.get("status") != "ok":
            continue
        key = (str(item.get("profile")), int(item.get("max_new_tokens") or 0))
        groups.setdefault(key, []).append(item)
    summaries = []
    for (profile, max_new_tokens), rows in sorted(groups.items()):
        rtfs = [float(row["stream_rtf"]) for row in rows if row.get("stream_rtf") is not None]
        first = [float(row["first_audio_ms"]) for row in rows if row.get("first_audio_ms") is not None]
        compute = [float(row["stream_compute_rtf"]) for row in rows if row.get("stream_compute_rtf") is not None]
        fallback = any(bool(row.get("fallback")) for row in rows)
        unroll_fallback = any(bool(row.get("unroll_fallback")) for row in rows)
        bad_exit = any(row.get("worker_exit_code") not in (None, 0) for row in rows)
        underrun_count = sum(1 for row in rows if row.get("underrun_risk"))
        p90_rtf = percentile(rtfs, 90)
        p90_first = percentile(first, 90)
        accepted = (
            bool(rows)
            and p90_rtf is not None
            and p90_rtf < 1.0
            and (p90_first is None or p90_first < 1000.0)
            and not fallback
            and not unroll_fallback
            and not bad_exit
            and underrun_count == 0
        )
        reasons = []
        if p90_rtf is None or p90_rtf >= 1.0:
            reasons.append("p90_stream_rtf>=1.0")
        if p90_first is not None and p90_first >= 1000.0:
            reasons.append("p90_first_audio_ms>=1000")
        if fallback:
            reasons.append("fallback")
        if unroll_fallback:
            reasons.append("unroll_fallback")
        if bad_exit:
            reasons.append("worker_exit")
        if underrun_count:
            reasons.append("underrun_risk")
        summaries.append(
            {
                "profile": profile,
                "max_new_tokens": max_new_tokens,
                "runs": len(rows),
                "p50_stream_rtf": percentile(rtfs, 50),
                "p90_stream_rtf": p90_rtf,
                "p50_stream_compute_rtf": percentile(compute, 50),
                "p90_stream_compute_rtf": percentile(compute, 90),
                "p50_first_audio_ms": percentile(first, 50),
                "p90_first_audio_ms": p90_first,
                "fallback": fallback,
                "unroll_fallback": unroll_fallback,
                "underrun_count": underrun_count,
                "accepted": accepted,
                "acceptance_reason": "ok" if accepted else ",".join(reasons),
            }
        )
    return summaries


def tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def run_profile(name: str, args: argparse.Namespace) -> dict:
    import gc

    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(PROFILES)}")

    import numpy as np

    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    config = PROFILES[name]
    previous_native_codegen = os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN")
    previous_native_pipeline = os.environ.get("QWEN3_TTS_OV_NATIVE_PIPELINE")
    previous_native_async_decode = os.environ.get("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE")
    previous_native_remote_embed = os.environ.get("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED")
    previous_native_buffer_reuse = os.environ.get("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE")
    previous_native_prompt = os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT")
    previous_native_prompt_device = os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE")
    previous_native_perf_count = os.environ.get("QWEN3_TTS_OV_NATIVE_PERF_COUNT")
    previous_native_paged_kv = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV")
    previous_native_paged_kv_gqa = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA")
    previous_native_paged_kv_precision = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION")
    previous_native_paged_kv_cache_input_precision = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION")
    previous_native_paged_kv_block_size = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE")
    previous_native_paged_kv_unroll = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL")
    previous_native_paged_kv_experimental_unroll = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL")
    previous_native_paged_kv_subcode_attention = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION")
    previous_native_paged_kv_hybrid = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID")
    previous_native_paged_kv_hybrid_prefix_frames = os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES")
    previous_native_codegen_device = os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE")
    previous_native_compile_env = {env_name: os.environ.get(env_name) for env_name in NATIVE_COMPILE_ENV_KEYS.values()}
    native_codegen = config.get("native_codegen")
    native_pipeline = config.get("native_pipeline")
    native_async_decode = config.get("native_async_decode")
    native_remote_embed = config.get("native_remote_embed")
    native_buffer_reuse = config.get("native_buffer_reuse")
    native_prompt = config.get("native_prompt")
    native_prompt_device = config.get("native_prompt_device")
    native_paged_kv = config.get("native_paged_kv")
    native_paged_kv_gqa = config.get("native_paged_kv_gqa")
    native_paged_kv_precision = config.get("native_paged_kv_precision")
    native_paged_kv_cache_input_precision = config.get("native_paged_kv_cache_input_precision")
    native_paged_kv_block_size = config.get("native_paged_kv_block_size")
    native_paged_kv_unroll = config.get("native_paged_kv_unroll")
    native_paged_kv_experimental_unroll = config.get("native_paged_kv_experimental_unroll")
    native_paged_kv_subcode_attention = config.get("native_paged_kv_subcode_attention")
    native_paged_kv_hybrid = config.get("native_paged_kv_hybrid")
    native_paged_kv_hybrid_prefix_frames = config.get("native_paged_kv_hybrid_prefix_frames")
    native_codegen_device = config.get("native_codegen_device")
    preferred_cache_bucket = config.get("preferred_cache_bucket", args.preferred_cache_bucket)
    repetition_penalty = float(config.get("repetition_penalty", args.repetition_penalty))
    worker_started = time.time()
    measured_started = None
    first_audio_ms = None
    audio_samples = 0
    chunks = 0
    final_timings = {}
    chunk_timings = []
    error = None
    try:
        if native_codegen:
            os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN"] = str(native_codegen)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_CODEGEN", None)
        if native_pipeline:
            os.environ["QWEN3_TTS_OV_NATIVE_PIPELINE"] = str(native_pipeline)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PIPELINE", None)
        if native_async_decode:
            os.environ["QWEN3_TTS_OV_NATIVE_ASYNC_DECODE"] = str(native_async_decode)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", None)
        if native_remote_embed:
            os.environ["QWEN3_TTS_OV_NATIVE_REMOTE_EMBED"] = str(native_remote_embed)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED", None)
        if native_buffer_reuse is not None:
            os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = str(native_buffer_reuse)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE", None)
        if native_prompt:
            os.environ["QWEN3_TTS_OV_NATIVE_PROMPT"] = str(native_prompt)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PROMPT", None)
        if native_prompt_device:
            os.environ["QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE"] = str(native_prompt_device)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE", None)
        if native_paged_kv:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = str(native_paged_kv)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV", None)
        if native_paged_kv_gqa is not None:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = str(native_paged_kv_gqa)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA", None)
        if native_paged_kv_precision:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION"] = str(native_paged_kv_precision)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION", None)
        if native_paged_kv_cache_input_precision:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION"] = str(
                native_paged_kv_cache_input_precision
            )
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION", None)
        if native_paged_kv_block_size:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE"] = str(native_paged_kv_block_size)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE", None)
        if native_paged_kv_unroll:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL"] = str(native_paged_kv_unroll)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL", None)
        if native_paged_kv_experimental_unroll:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL"] = str(native_paged_kv_experimental_unroll)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL", None)
        if native_paged_kv_subcode_attention:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION"] = str(native_paged_kv_subcode_attention)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION", None)
        if native_paged_kv_hybrid:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID"] = str(native_paged_kv_hybrid)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID", None)
        if native_paged_kv_hybrid_prefix_frames:
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES"] = str(native_paged_kv_hybrid_prefix_frames)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES", None)
        if native_codegen_device:
            os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE"] = str(native_codegen_device)
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE", None)
        if args.native_ov_profile:
            os.environ["QWEN3_TTS_OV_NATIVE_PERF_COUNT"] = "1"
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PERF_COUNT", None)
        for config_key, env_name in NATIVE_COMPILE_ENV_KEYS.items():
            value = config.get(config_key)
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = str(value)
        runtime = OpenVINOQwen3TTS(
            args.ir_dir,
            args.device,
            args.decoder_device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            mode=config["mode"],
            cache_kernel=config["cache_kernel"],
            cache_step=config["cache_step"],
            graph_variant=config["graph_variant"],
            codegen_unroll=config["codegen_unroll"],
            codegen_schedule=config.get("codegen_schedule", "current"),
            codegen_decode_unroll=config.get("codegen_decode_unroll", "off"),
            preferred_cache_bucket=preferred_cache_bucket,
            ov_cache_dir=args.ov_cache_dir,
            ov_cache_mode=args.ov_cache_mode,
            disable_ov_cache=args.disable_ov_cache,
        )
        for _ in range(int(args.warmup_generations or 0)):
            for chunk in runtime.stream_voice_design(
                text=args.text,
                instruct=args.instruct,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=args.max_prompt_tokens,
                progress_interval=0,
                chunk_strategy=args.chunk_strategy,
            ):
                if chunk.is_final:
                    break

        measured_started = time.time()
        for chunk in runtime.stream_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=args.max_prompt_tokens,
            progress_interval=0,
            chunk_strategy=args.chunk_strategy,
        ):
            final_timings = chunk.timings
            if chunk.audio.size:
                chunks += 1
                audio_samples += int(chunk.audio.shape[0])
                chunk_timings.append(dict(chunk.timings))
                if first_audio_ms is None:
                    first_audio_ms = (time.time() - measured_started) * 1000.0
            if chunk.is_final:
                break
        sample_rate = int(runtime.sample_rate)
        emitted_frames = int(final_timings.get("emitted_frames", 0) or 0)
        del runtime
        gc.collect()
    except Exception as exc:
        sample_rate = 24000
        emitted_frames = 0
        error = str(exc)
        gc.collect()
    finally:
        if previous_native_codegen is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_CODEGEN", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN"] = previous_native_codegen
        if previous_native_pipeline is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PIPELINE", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_PIPELINE"] = previous_native_pipeline
        if previous_native_async_decode is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_ASYNC_DECODE"] = previous_native_async_decode
        if previous_native_remote_embed is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_REMOTE_EMBED"] = previous_native_remote_embed
        if previous_native_buffer_reuse is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = previous_native_buffer_reuse
        if previous_native_prompt is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PROMPT", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_PROMPT"] = previous_native_prompt
        if previous_native_prompt_device is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE"] = previous_native_prompt_device
        if previous_native_perf_count is None:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_PERF_COUNT", None)
        else:
            os.environ["QWEN3_TTS_OV_NATIVE_PERF_COUNT"] = previous_native_perf_count
        for env_name, previous_value in (
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV", previous_native_paged_kv),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA", previous_native_paged_kv_gqa),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION", previous_native_paged_kv_precision),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION", previous_native_paged_kv_cache_input_precision),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE", previous_native_paged_kv_block_size),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL", previous_native_paged_kv_unroll),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL", previous_native_paged_kv_experimental_unroll),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION", previous_native_paged_kv_subcode_attention),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID", previous_native_paged_kv_hybrid),
            ("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES", previous_native_paged_kv_hybrid_prefix_frames),
            ("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE", previous_native_codegen_device),
        ):
            if previous_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_value
        for env_name, value in previous_native_compile_env.items():
            if value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = value

    stopped = time.time()
    elapsed_ms = (stopped - (measured_started or worker_started)) * 1000.0
    worker_elapsed_ms = (stopped - worker_started) * 1000.0
    audio_ms = (audio_samples / sample_rate * 1000.0) if audio_samples else 0.0
    audio_timings = chunk_timings[-1] if chunk_timings else final_timings
    native_timing = audio_timings.get("native_timing") or final_timings.get("native_timing") or {}
    if not isinstance(native_timing, dict):
        native_timing = {}
    stream_rtf = final_timings.get("stream_rtf")
    stream_compute_rtf = final_timings.get("stream_compute_rtf")
    if stream_rtf is None and audio_ms:
        stream_rtf = elapsed_ms / audio_ms
    return {
        "profile": name,
        "status": "error" if error else "ok",
        "error": error,
        "config": config,
        "first_audio_ms": first_audio_ms,
        "elapsed_ms": elapsed_ms,
        "worker_elapsed_ms": worker_elapsed_ms,
        "warmup_generations": int(args.warmup_generations or 0),
        "max_new_tokens": int(args.max_new_tokens),
        "repetition_penalty": repetition_penalty,
        "audio_ms": audio_ms,
        "chunks": chunks,
        "emitted_frames": emitted_frames,
        "tokens_per_second": (emitted_frames / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        "stream_rtf": stream_rtf,
        "stream_compute_rtf": stream_compute_rtf,
        "decode_path": audio_timings.get("decode_path"),
        "fallback": bool(audio_timings.get("fallback", False)),
        "codegen_unroll": audio_timings.get("codegen_unroll", int(config["codegen_unroll"])),
        "active_codegen_unroll": audio_timings.get("active_codegen_unroll"),
        "codegen_schedule": audio_timings.get("codegen_schedule", config.get("codegen_schedule", "current")),
        "preferred_cache_bucket": audio_timings.get("preferred_cache_bucket", preferred_cache_bucket),
        "selected_bucket": audio_timings.get("selected_bucket"),
        "native_codegen": bool(audio_timings.get("native_codegen", False)),
        "native_audio_pipeline": bool(audio_timings.get("native_audio_pipeline", False)),
        "native_prompt_pipeline": bool(audio_timings.get("native_prompt_pipeline", False)),
        "native_async_decode": bool(audio_timings.get("native_async_decode", bool(config.get("native_async_decode")))),
        "native_remote_embed": audio_timings.get("native_remote_embed"),
        "native_streaming_callbacks": bool(audio_timings.get("native_streaming_callbacks", False)),
        "codegen_no_repeat": bool(audio_timings.get("codegen_no_repeat", False)),
        "native_codegen_ms": final_timings.get("native_codegen_ms"),
        "native_pipeline_ms": final_timings.get("native_pipeline_ms"),
        "native_ov_profile": final_timings.get("native_ov_profile"),
        "native_timing": native_timing or None,
        "paged_kv": bool(audio_timings.get("paged_kv", False)),
        "paged_kv_backend": audio_timings.get("paged_kv_backend"),
        "paged_kv_seed_key": audio_timings.get("paged_kv_seed_key"),
        "paged_kv_gqa": audio_timings.get("paged_kv_gqa"),
        "paged_kv_unroll": audio_timings.get("paged_kv_unroll"),
        "paged_kv_heads": audio_timings.get("paged_kv_heads"),
        "paged_kv_block_size": audio_timings.get("paged_kv_block_size"),
        "paged_kv_precision": audio_timings.get("paged_kv_precision"),
        "paged_kv_cache_input_precision": audio_timings.get("paged_kv_cache_input_precision"),
        "paged_kv_score_aggregation": audio_timings.get("paged_kv_score_aggregation"),
        "paged_kv_split_subcode": audio_timings.get("paged_kv_split_subcode"),
        "selected_paged_split_subcode_graph": audio_timings.get("selected_paged_split_subcode_graph"),
        "hybrid_paged_kv": bool(audio_timings.get("hybrid_paged_kv", False)),
        "hybrid_phase": audio_timings.get("hybrid_phase"),
        "hybrid_prefix_frames": audio_timings.get("hybrid_prefix_frames"),
        "hybrid_prefix_actual_frames": audio_timings.get("hybrid_prefix_actual_frames"),
        "native_buffer_reuse": native_timing.get("buffer_reuse"),
        "native_kv_cache_tensor_reuse": native_timing.get("kv_cache_tensor_reuse"),
        "native_paged_static_decode_enabled": native_timing.get("paged_static_decode_enabled"),
        "native_paged_static_decode_mode": native_timing.get("paged_static_decode_mode"),
        "native_no_repeat_fast_path": native_timing.get("no_repeat_fast_path"),
        "host_prepare_ms": native_timing.get("host_prepare_ms"),
        "tensor_bind_ms": native_timing.get("tensor_bind_ms"),
        "codegen_infer_ms": native_timing.get("codegen_infer_ms"),
        "decode_infer_ms": native_timing.get("decode_infer_ms"),
        "native_callback_ms": native_timing.get("callback_ms"),
        "unroll_fallback": bool(audio_timings.get("unroll_fallback", False)),
        "underrun_risk": bool(stream_rtf is not None and stream_rtf >= 1.0),
        "audio_timings": audio_timings,
        "final_timings": final_timings,
        "chunk_timings": chunk_timings,
    }


def worker_command(args: argparse.Namespace, profile: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-profile",
        profile,
        "--ir-dir",
        str(args.ir_dir),
        "--device",
        args.device,
        "--ov-cache-mode",
        args.ov_cache_mode,
        "--profiles",
        profile,
        "--runs",
        "1",
        "--text",
        args.text,
        "--instruct",
        args.instruct,
        "--language",
        args.language,
        "--chunk-strategy",
        args.chunk_strategy,
        "--preferred-cache-bucket",
        str(args.preferred_cache_bucket),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-new-tokens",
        str(args.min_new_tokens),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--warmup-generations",
        str(args.warmup_generations),
        "--worker-timeout-sec",
        str(args.worker_timeout_sec),
        "--output-json",
        "",
    ]
    if args.native_ov_profile:
        cmd.append("--native-ov-profile")
    if args.decoder_device:
        cmd.extend(["--decoder-device", args.decoder_device])
    if args.ov_cache_dir:
        cmd.extend(["--ov-cache-dir", str(args.ov_cache_dir)])
    if args.disable_ov_cache:
        cmd.append("--disable-ov-cache")
    if args.allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    return cmd


def parse_worker_result(completed: subprocess.CompletedProcess, profile: str) -> dict:
    result = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("profile") == profile:
            result = candidate
            break
    if result is None:
        result = {
            "profile": profile,
            "status": "error",
            "error": "worker did not emit a profile JSON result",
        }
    result["worker_exit_code"] = completed.returncode
    result["stderr_tail"] = tail(completed.stderr.strip())
    if completed.returncode != 0 and result.get("status") != "ok":
        result["status"] = "error"
    return result


def run_isolated_profile(profile: str, args: argparse.Namespace) -> dict:
    try:
        completed = subprocess.run(
            worker_command(args, profile),
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=float(args.worker_timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "profile": profile,
            "status": "timeout",
            "error": f"worker exceeded --worker-timeout-sec={args.worker_timeout_sec}",
            "worker_exit_code": None,
            "stderr_tail": tail((exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""),
            "stdout_tail": tail((exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""),
        }
    return parse_worker_result(completed, profile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="auto")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument(
        "--profiles",
        default="fastest",
    )
    parser.add_argument("--profile-set", default=None, choices=sorted(PROFILE_SETS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--chunk-strategy", default="smooth", choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    parser.add_argument("--preferred-cache-bucket", default="96")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-new-tokens-set", default=None)
    parser.add_argument("--min-new-tokens", type=int, default=12)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--warmup-generations", type=int, default=0)
    parser.add_argument("--native-ov-profile", action="store_true")
    parser.add_argument("--worker-timeout-sec", type=float, default=300.0)
    parser.add_argument("--output-json", default="outputs/realtime_bench/streaming_profiles.json")
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        profile = args.worker_profile or parse_csv(args.profiles)[0]
        result = run_profile(profile, args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    if args.profile_set:
        args.profiles = PROFILE_SETS[args.profile_set]
    token_counts = [int(item) for item in parse_csv(args.max_new_tokens_set)] if args.max_new_tokens_set else [int(args.max_new_tokens)]
    results = []
    for max_new_tokens in token_counts:
        args.max_new_tokens = int(max_new_tokens)
        for run_index in range(args.runs):
            for profile in parse_csv(args.profiles):
                if args.no_isolate:
                    result = run_profile(profile, args)
                    result.setdefault("worker_exit_code", None)
                    result.setdefault("stderr_tail", "")
                else:
                    result = run_isolated_profile(profile, args)
                result["run"] = run_index + 1
                result["max_new_tokens"] = int(max_new_tokens)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    summaries = summarize_results(results)
    accepted = bool(summaries) and all(item.get("accepted") for item in summaries)

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"runs": results, "summaries": summaries, "accepted": accepted}, handle, ensure_ascii=False, indent=2)
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
