// Copyright (C) 2026 Qwen3-TTS OpenVINO contributors
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speech_generation/speech_generation_config.hpp>
#include <openvino/genai/speech_generation/speech_generation_perf_metrics.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr float NEG_INF = -3.4028234663852886e38f;

struct NamedTensor {
    std::string name;
    ov::Tensor tensor;
};

struct OnlineBatchSequence {
    int64_t id = 0;
    int64_t prompt_len = 0;
    int64_t hidden_size = 0;
    int64_t max_new_tokens = 0;
    int64_t min_new_tokens = 0;
    int64_t vocab_size = 0;
    int64_t num_code_groups = 0;
    int64_t eos_token_id = 0;
    int64_t generated = 0;
    int64_t block_offset = 0;
    int64_t block_capacity = 0;
    float repetition_penalty = 1.0f;
    bool do_sample = false;
    int64_t top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    uint64_t seed = 0;
    bool stop = false;
    bool pending_prefill = true;
    std::vector<float> prompt;
    std::vector<float> next_embed;
    std::vector<uint8_t> repeated_first_codes;
    std::mt19937_64 rng;
    double queued_at_ms = 0.0;
    double ttft_ms = -1.0;
    double last_token_ms = 0.0;
};

struct NativeCodegen {
    ov::Core core;
    ov::CompiledModel prefill_model;
    ov::CompiledModel decode_model;
    ov::CompiledModel text_embedding_model;
    ov::CompiledModel codec_embedding_model;
    ov::CompiledModel subcode_model;
    ov::CompiledModel batch_subcode_model;
    ov::CompiledModel first_stream_decoder_model;
    ov::CompiledModel steady_stream_decoder_model;
    ov::InferRequest prefill_request;
    ov::InferRequest decode_request;
    ov::InferRequest text_embedding_request;
    ov::InferRequest codec_embedding_request;
    ov::InferRequest subcode_request;
    ov::InferRequest batch_subcode_request;
    std::vector<ov::InferRequest> sampled_subcode_row_requests;
    ov::InferRequest online_logits_decode_request;
    ov::InferRequest first_stream_decoder_request;
    ov::InferRequest steady_stream_decoder_request;
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    int64_t bucket = 0;
    int64_t unroll = 4;
    bool stream_decoders_ready = false;
    bool voice_design_prompt_ready = false;
    bool prompt_embedding_cache_enabled = false;
    bool profile_enabled = false;
    bool paged_kv_enabled = false;
    bool paged_split_subcode = false;
    bool paged_seed_outputs_first_code = false;
    bool subcode_outputs_next_embed = false;
    bool paged_static_decode_enabled = false;
    bool paged_static_decode_requested = false;
    bool paged_fused_batch_decode_enabled = false;
    bool batch_subcode_enabled = false;
    std::string online_scheduler = "legacy";
    std::vector<int64_t> prefill_seq_buckets{128, 256, 512, 1024};
    std::vector<int64_t> prefill_batch_buckets{1, 2, 4, 8};
    std::vector<int64_t> decode_batch_buckets{1, 2, 4, 8, 16};
    std::string prefill_mode = "serial";
    int64_t max_num_batched_tokens = 16;
    std::string continuous_batch_policy = "raw_fused";
    int64_t batch_decode_unroll_requested = 1;
    int64_t batch_decode_unroll = 1;
    int64_t paged_kv_block_size = 8;
    int64_t paged_kv_heads = 16;
    int64_t paged_kv_head_dim = 128;
    int64_t paged_static_decode_block_capacity = 0;
    std::string paged_static_decode_mode = "dynamic";
    std::string paged_static_decode_failure;
    std::string paged_fused_batch_decode_graph;
    std::string paged_kv_precision = "f16";
    std::string paged_kv_cache_input_precision = "f32";
    int64_t paged_kv_cache_tensor_blocks = 0;
    std::vector<NamedTensor> paged_kv_cache_tensors;
    bool online_batch_ready = false;
    int64_t online_batch_next_id = 1;
    int64_t online_batch_hidden_size = 0;
    int64_t online_batch_vocab_size = 0;
    int64_t online_batch_num_code_groups = 0;
    int64_t online_batch_eos_token_id = 0;
    int64_t online_batch_cache_block_capacity = 0;
    int64_t online_batch_next_block_offset = 0;
    int64_t online_batch_total_requests = 0;
    int64_t online_batch_finished_requests = 0;
    std::vector<float> online_batch_tts_pad_embed;
    std::map<int64_t, OnlineBatchSequence> online_batch_sequences;
    std::deque<int64_t> online_batch_pending_ids;
    std::vector<std::pair<int64_t, int64_t>> online_batch_free_blocks;
    int64_t tts_bos_token_id = 0;
    int64_t tts_eos_token_id = 0;
    int64_t tts_pad_token_id = 0;
    int64_t codec_pad_id = 0;
    int64_t codec_bos_id = 0;
    int64_t first_context_frames = 0;
    int64_t first_chunk_frames = 8;
    bool first_stream_decoder_static_input = false;
    int64_t first_stream_decoder_input_frames = 8;
    int64_t steady_context_frames = 25;
    int64_t steady_chunk_frames = 12;
    bool steady_stream_decoder_static_input = false;
    int64_t steady_stream_decoder_input_frames = 37;
    int64_t stream_num_code_groups = 16;
    int64_t decode_upsample_rate = 2000;
    bool last_remote_embed_used = false;
    bool last_async_decode = false;
    struct ProfileEntry {
        std::string label;
        std::string node_name;
        std::string node_type;
        std::string exec_type;
        double real_time_ms = 0.0;
        double cpu_time_ms = 0.0;
        int64_t count = 0;
    };
    struct CodegenTraceFrame {
        int64_t frame = 0;
        std::string label;
        int64_t first_code = 0;
        std::vector<int64_t> codes;
        double hidden_l2 = 0.0;
        double hidden_max_abs = 0.0;
        double embed_l2 = 0.0;
        double embed_max_abs = 0.0;
    };
    struct RunTiming {
        bool buffer_reuse = true;
        bool no_repeat_fast_path = false;
        bool kv_cache_tensor_reuse = false;
        double host_prepare_ms = 0.0;
        double tensor_bind_ms = 0.0;
        double codegen_infer_ms = 0.0;
        double codegen_prefill_infer_ms = 0.0;
        double codegen_decode_infer_ms = 0.0;
        double codegen_subcode_infer_ms = 0.0;
        double subcode_bind_ms = 0.0;
        double subcode_output_read_ms = 0.0;
        double subcode_next_embed_ms = 0.0;
        uint64_t subcode_host_copy_bytes = 0;
        int64_t subcode_host_copy_fallback_count = 0;
        int64_t split_subcode_hidden_direct_bind_count = 0;
        int64_t split_subcode_hidden_bind_fallback_count = 0;
        uint64_t split_subcode_hidden_copy_bytes = 0;
        bool split_subcode_remote_next_embed_enabled = false;
        int64_t split_subcode_remote_next_embed_fallback_count = 0;
        int64_t split_subcode_next_embed_host_read_count = 0;
        bool decode_step_prebind_enabled = false;
        int64_t decode_step_prebind_fallback_count = 0;
        double decode_step_prebind_update_ms = 0.0;
        bool batch_prefill_enabled = false;
        std::string prefill_mode = "serial";
        bool batch_subcode_enabled = false;
        std::string online_scheduler = "legacy";
        int64_t max_num_batched_tokens = 0;
        int64_t prefill_seq_bucket = 0;
        int64_t prefill_batch_bucket = 0;
        int64_t decode_batch_bucket = 0;
        int64_t prefill_token_budget = 0;
        int64_t decode_token_budget = 0;
        std::vector<int64_t> prefill_seq_bucket_histogram;
        std::vector<int64_t> prefill_batch_bucket_histogram;
        std::vector<int64_t> decode_batch_bucket_histogram;
        std::string sampled_batch_subcode_policy = "off";
        bool sampled_batch_subcode_used = false;
        bool sampled_batch_subcode_verified = false;
        int64_t sampled_batch_subcode_fallback_count = 0;
        int64_t sampled_batch_subcode_mismatch_count = 0;
        int64_t sampled_batch_subcode_code_mismatch_count = 0;
        int64_t sampled_batch_subcode_embed_mismatch_count = 0;
        double sampled_batch_subcode_max_abs_diff = 0.0;
        std::string sampled_batch_subcode_fallback_reason;
        bool sampled_subcode_parallel_rows = false;
        int64_t sampled_subcode_parallel_row_count = 0;
        bool batch_buffer_reuse = false;
        int64_t batch_decode_unroll = 1;
        int64_t batch_decode_unroll_requested = 1;
        int64_t batch_decode_unroll_fallback_count = 0;
        int64_t batch_fused_decode_step_count = 0;
        int64_t batch_fused_decode_token_count = 0;
        int64_t batch_single_decode_step_count = 0;
        int64_t batch_single_decode_token_count = 0;
        int64_t batch_fused_decode_active1_bypass_count = 0;
        int64_t batch_fused_decode_logits_bypass_count = 0;
        std::vector<int64_t> active_batch_histogram;
        double sampling_ms = 0.0;
        double decode_infer_ms = 0.0;
        double callback_ms = 0.0;
        double callback_enqueue_ms = 0.0;
        double inline_decode_ms = 0.0;
        double codegen_callback_ms = 0.0;
        double decode_callback_ms = 0.0;
        double decode_queue_wait_ms = 0.0;
        int64_t decode_queue_depth_max = 0;
        int64_t decode_input_frames = 0;
        int64_t decode_padded_frames = 0;
        double ttft_ms = 0.0;
        double last_token_ms = 0.0;
        int64_t generated_frames = 0;
        double host_copy_ms = 0.0;
        uint64_t host_copy_bytes = 0;
        int64_t host_copy_fallback_count = 0;
        int64_t zero_copy_count = 0;
        int64_t codegen_prefill_count = 0;
        int64_t codegen_decode_count = 0;
        int64_t codegen_subcode_count = 0;
        std::vector<double> codegen_prefill_step_ms;
        std::vector<double> codegen_decode_step_ms;
        std::vector<double> codegen_bind_step_ms;
        std::vector<double> codegen_sampling_step_ms;
        std::vector<double> subcode_bind_step_ms;
        std::vector<double> subcode_infer_step_ms;
        std::vector<double> subcode_output_read_step_ms;
        std::vector<double> subcode_next_embed_step_ms;
        std::vector<double> decode_step_prebind_update_step_ms;
        double pcm_convert_ms = 0.0;
        double total_ms = 0.0;
    };
    struct ScratchBuffers {
        std::vector<int64_t> positions;
        std::vector<int64_t> decode_cache_position;
        std::vector<int64_t> subcode_first_code;
        std::vector<int64_t> subcode_frame_codes;
        std::vector<float> attention_mask;
        std::vector<float> repeated_mask;
        std::vector<float> allow_eos;
        std::vector<float> penalty;
        std::vector<float> next_embed;
        std::vector<float> subcode_past_hidden;
        std::vector<float> subcode_uniform_noise;
        std::vector<float> subcode_temperature;
    };
    std::unordered_map<std::string, ProfileEntry> profile_ops;
    std::unordered_map<std::string, std::vector<float>> prompt_embedding_cache;
    int64_t prompt_embedding_cache_hits = 0;
    int64_t prompt_embedding_cache_misses = 0;
    std::vector<CodegenTraceFrame> codegen_trace;
    RunTiming last_timing;
    ScratchBuffers scratch;
};

struct NativeSamplingConfig {
    bool do_sample = false;
    int64_t top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    uint64_t seed = 0;
};

using FrameCallback = int (*)(const int64_t* codes, int64_t num_frames, int64_t num_code_groups, void* user_data);
using AudioCallback = int (*)(
    const float* audio,
    int64_t num_samples,
    const int64_t* codes,
    int64_t num_frames,
    int64_t num_code_groups,
    int64_t is_final,
    double codegen_ms,
    double decode_ms,
    void* user_data);

char* dup_cstr(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

int fail(char** error, const std::string& message) {
    if (error) {
        *error = dup_cstr(message);
    }
    return 1;
}

template <typename Fn>
int guarded(char** error, Fn&& fn) {
    try {
        fn();
        return 0;
    } catch (const std::exception& exc) {
        return fail(error, exc.what());
    } catch (...) {
        return fail(error, "unknown native codegen error");
    }
}

int64_t parse_cache_bucket(const std::string& path) {
    std::smatch match;
    if (std::regex_search(path, match, std::regex("cache([0-9]+)"))) {
        return std::stoll(match[1].str());
    }
    return 0;
}

int64_t parse_unroll_steps(const std::string& path) {
    std::smatch match;
    if (std::regex_search(path, match, std::regex("unroll([0-9]+)"))) {
        return std::stoll(match[1].str());
    }
    return 1;
}

std::string lower_text(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool disabled_env_value(const char* value) {
    if (!value) {
        return false;
    }
    const std::string text = lower_text(value);
    return text == "0" || text == "false" || text == "off" || text == "no" || text == "none" || text == "default";
}

bool enabled_env(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    const std::string text = lower_text(value);
    if (text == "0" || text == "false" || text == "off" || text == "no") {
        return false;
    }
    if (text == "1" || text == "true" || text == "on" || text == "yes") {
        return true;
    }
    return default_value;
}

std::string normalize_continuous_batch_policy(std::string policy) {
    policy = lower_text(policy);
    std::replace(policy.begin(), policy.end(), '-', '_');
    if (policy.empty()) {
        return "raw_fused";
    }
    return policy;
}

bool valid_continuous_batch_policy(const std::string& policy) {
    return
        policy == "raw_fused" ||
        policy == "adaptive" ||
        policy == "adaptive_v2" ||
        policy == "context_bucketed" ||
        policy == "unified_vllm" ||
        policy == "layered_vllm" ||
        policy == "raw_fused_v2";
}

std::string continuous_batch_policy_error() {
    return "QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_POLICY must be raw_fused, adaptive, adaptive_v2, "
           "context_bucketed, unified_vllm, layered_vllm, or raw_fused_v2";
}

bool continuous_policy_uses_single_row_fast_path(const std::string& policy) {
    return
        policy == "adaptive" ||
        policy == "adaptive_v2" ||
        policy == "unified_vllm" ||
        policy == "layered_vllm" ||
        policy == "raw_fused_v2";
}

bool continuous_policy_uses_batched_prefill(const std::string& policy) {
    return policy == "adaptive_v2" || policy == "unified_vllm" || policy == "layered_vllm" || policy == "raw_fused_v2";
}

void apply_env_property(ov::AnyMap& config, const char* env_name, const char* property_name) {
    const char* value = std::getenv(env_name);
    if (value && !disabled_env_value(value)) {
        config[property_name] = std::string(value);
    }
}

ov::AnyMap compile_config(const char* cache_dir, const char* cache_mode) {
    ov::AnyMap config;
    const char* precision_hint = std::getenv("QWEN3_TTS_OV_NATIVE_PRECISION_HINT");
    if (!disabled_env_value(precision_hint)) {
        config["INFERENCE_PRECISION_HINT"] = std::string(precision_hint ? precision_hint : "f16");
    }
    if (cache_dir && std::strlen(cache_dir) > 0) {
        config[ov::cache_dir.name()] = std::string(cache_dir);
    }
    if (cache_mode && std::strlen(cache_mode) > 0) {
        config["CACHE_MODE"] = std::string(cache_mode);
    }
    if (enabled_env("QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS", false)) {
        config["GPU_ENABLE_LARGE_ALLOCATIONS"] = std::string("YES");
    }
    if (enabled_env("QWEN3_TTS_OV_NATIVE_LATENCY_HIGH", false)) {
        config["PERFORMANCE_HINT"] = std::string("LATENCY");
        config["NUM_STREAMS"] = std::string("1");
        config["MODEL_PRIORITY"] = std::string("HIGH");
        config["GPU_QUEUE_PRIORITY"] = std::string("HIGH");
        config["GPU_HOST_TASK_PRIORITY"] = std::string("HIGH");
        config["GPU_QUEUE_THROTTLE"] = std::string("LOW");
    }
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_PERFORMANCE_HINT", "PERFORMANCE_HINT");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_NUM_STREAMS", "NUM_STREAMS");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_MODEL_PRIORITY", "MODEL_PRIORITY");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_GPU_QUEUE_PRIORITY", "GPU_QUEUE_PRIORITY");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_GPU_HOST_TASK_PRIORITY", "GPU_HOST_TASK_PRIORITY");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_GPU_QUEUE_THROTTLE", "GPU_QUEUE_THROTTLE");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE", "DYNAMIC_QUANTIZATION_GROUP_SIZE");
    apply_env_property(config, "QWEN3_TTS_OV_NATIVE_ACTIVATIONS_SCALE_FACTOR", "ACTIVATIONS_SCALE_FACTOR");
    const char* perf_count = std::getenv("QWEN3_TTS_OV_NATIVE_PERF_COUNT");
    if (perf_count && std::strcmp(perf_count, "0") != 0 && std::strcmp(perf_count, "false") != 0 &&
        std::strcmp(perf_count, "off") != 0) {
        config["PERF_COUNT"] = std::string("YES");
    }
    return config;
}

bool is_gpu_device(const char* device) {
    if (!device) {
        return false;
    }
    return lower_text(std::string(device)).find("gpu") != std::string::npos;
}

ov::AnyMap compile_config_for_device(const char* cache_dir, const char* cache_mode, const char* device) {
    ov::AnyMap config = compile_config(cache_dir, cache_mode);
    if (!is_gpu_device(device)) {
        config.erase("GPU_ENABLE_LARGE_ALLOCATIONS");
        config.erase("GPU_QUEUE_PRIORITY");
        config.erase("GPU_HOST_TASK_PRIORITY");
        config.erase("GPU_QUEUE_THROTTLE");
    }
    return config;
}

ov::element::Type parse_element_type(const std::string& value) {
    const std::string text = lower_text(value);
    if (text == "f16" || text == "float16") {
        return ov::element::f16;
    }
    if (text == "bf16" || text == "bfloat16") {
        return ov::element::bf16;
    }
    if (text == "f32" || text == "float32") {
        return ov::element::f32;
    }
    if (text == "u8" || text == "uint8") {
        return ov::element::u8;
    }
    if (text == "i8" || text == "int8") {
        return ov::element::i8;
    }
    if (text == "u4" || text == "uint4") {
        return ov::element::u4;
    }
    if (text == "i4" || text == "int4") {
        return ov::element::i4;
    }
    throw std::runtime_error("unsupported KV cache precision: " + value);
}

ov::Shape concrete_shape(const ov::PartialShape& partial_shape, int64_t dynamic_value) {
    if (partial_shape.rank().is_dynamic()) {
        throw std::runtime_error("dynamic rank is not supported for paged KV cache tensor allocation");
    }
    ov::Shape shape;
    shape.reserve(partial_shape.rank().get_length());
    for (const auto& dim : partial_shape) {
        shape.push_back(static_cast<size_t>(dim.is_static() ? dim.get_length() : dynamic_value));
    }
    return shape;
}

size_t add_readvalue_initializers(const std::shared_ptr<ov::Model>& model) {
    size_t changed = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto read_value = std::dynamic_pointer_cast<ov::op::v6::ReadValue>(node);
        if (!read_value || read_value->get_input_size() != 0) {
            continue;
        }
        auto variable = read_value->get_variable();
        if (!variable) {
            continue;
        }
        const auto pshape = read_value->get_output_partial_shape(0);
        if (pshape.rank().is_dynamic()) {
            continue;
        }
        ov::Shape init_shape;
        init_shape.reserve(pshape.rank().get_length());
        for (const auto& dim : pshape) {
            init_shape.push_back(dim.is_static() ? static_cast<size_t>(dim.get_length()) : 0);
        }
        auto init = ov::op::v0::Constant::create(read_value->get_output_element_type(0), init_shape, std::vector<float>{});
        auto replacement = std::make_shared<ov::op::v6::ReadValue>(init, variable);
        replacement->set_friendly_name(read_value->get_friendly_name());
        ov::copy_runtime_info(read_value, replacement);
        read_value->output(0).replace(replacement->output(0));
        ++changed;
    }
    return changed;
}

size_t restore_unregistered_parameters(const std::shared_ptr<ov::Model>& model) {
    ov::ParameterVector missing;
    std::set<const ov::Node*> registered;
    for (const auto& param : model->get_parameters()) {
        registered.insert(param.get());
    }
    for (const auto& op : model->get_ops()) {
        auto param = ov::as_type_ptr<ov::op::v0::Parameter>(op);
        if (param && registered.count(param.get()) == 0) {
            missing.push_back(param);
            registered.insert(param.get());
        }
    }
    if (!missing.empty()) {
        model->add_parameters(missing);
    }
    return missing.size();
}

size_t specialize_kv_cache_parameters(
    const std::shared_ptr<ov::Model>& model,
    int64_t heads,
    int64_t block_size,
    int64_t head_dim,
    ov::element::Type cache_element_type) {
    size_t changed = 0;
    for (const auto& parameter : model->get_parameters()) {
        bool is_kv_cache = false;
        for (const auto& name : parameter->get_output_tensor(0).get_names()) {
            if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
                is_kv_cache = true;
                break;
            }
        }
        if (!is_kv_cache) {
            continue;
        }
        parameter->set_element_type(cache_element_type);
        parameter->set_partial_shape(
            ov::PartialShape{ov::Dimension::dynamic(), heads, head_dim, block_size});
        parameter->validate_and_infer_types();
        ++changed;
    }
    return changed;
}

bool flatten_frame_embed_result(const std::shared_ptr<ov::Model>& model) {
    auto results = model->get_results();
    if (results.size() < 3) {
        return false;
    }
    auto result = results[2];
    if (!result || result->get_input_size() == 0) {
        return false;
    }
    const auto source = result->input_value(0);
    const auto rank = source.get_partial_shape().rank();
    if (rank.is_dynamic() || rank.get_length() != 3) {
        return false;
    }
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, -1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(source, target_shape, false);
    reshape->set_friendly_name("frame_embed_flatten");
    ov::copy_runtime_info(result, reshape);
    result->input(0).replace_source_output(reshape->output(0));
    return true;
}

std::string model_input_summary(const std::shared_ptr<ov::Model>& model);

int64_t infer_paged_hidden_size(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return 0;
    }
    auto static_last_dim = [](const ov::PartialShape& shape) -> int64_t {
        if (shape.rank().is_dynamic() || shape.rank().get_length() == 0) {
            return 0;
        }
        const auto dim = shape[shape.rank().get_length() - 1];
        return dim.is_static() ? dim.get_length() : 0;
    };
    for (const auto& input : model->inputs()) {
        std::string name;
        try {
            name = input.get_any_name();
        } catch (...) {
            continue;
        }
        if (name != "tts_pad_embed" && name != "inputs_embeds") {
            continue;
        }
        const int64_t hidden = static_last_dim(input.get_partial_shape());
        if (hidden > 0) {
            return hidden;
        }
    }
    for (const auto& output : model->outputs()) {
        std::string name;
        try {
            name = output.get_any_name();
        } catch (...) {
            continue;
        }
        if (name != "last_hidden") {
            continue;
        }
        const int64_t hidden = static_last_dim(output.get_partial_shape());
        if (hidden > 0) {
            return hidden;
        }
    }
    return 0;
}

bool reshape_paged_decode_model(
    std::shared_ptr<ov::Model>& model,
    int64_t heads,
    int64_t block_size,
    int64_t head_dim,
    int64_t block_capacity,
    const std::string& static_mode) {
    if (!model) {
        return false;
    }
    const int64_t hidden_size = infer_paged_hidden_size(model);
    if (hidden_size <= 0 || heads <= 0 || block_size <= 0 || head_dim <= 0 || block_capacity <= 0) {
        return false;
    }
    std::map<ov::Output<ov::Node>, ov::PartialShape> shapes;
    bool changed = false;
    const bool reshape_cache_buffers = static_mode == "full";
    for (const auto& input : model->inputs()) {
        std::string name;
        try {
            name = input.get_any_name();
        } catch (...) {
            continue;
        }
        const auto pshape = input.get_partial_shape();
        const auto rank = pshape.rank();
        if (rank.is_dynamic()) {
            continue;
        }
        const auto rank_len = rank.get_length();
        const auto set_shape = [&](const ov::PartialShape& shape) {
            shapes[input] = shape;
            changed = true;
        };
        if (name == "inputs_embeds") {
            if (rank_len == 2) {
                set_shape(ov::PartialShape{1, hidden_size});
            } else if (rank_len == 3) {
                set_shape(ov::PartialShape{1, 1, hidden_size});
            }
        } else if (name == "position_ids") {
            if (rank_len == 2) {
                set_shape(ov::PartialShape{3, 1});
            } else if (rank_len == 3) {
                set_shape(ov::PartialShape{3, 1, 1});
            }
        } else if (name == "attention_mask") {
            if (rank_len == 4) {
                set_shape(ov::PartialShape{1, 1, 1, ov::Dimension::dynamic()});
            }
        } else if (name == "allow_eos" || name == "allow_eos_steps" || name == "beam_idx" ||
                   name == "past_lens" || name == "score_aggregation_window") {
            set_shape(ov::PartialShape{1});
        } else if (name == "subsequence_begins" || name == "block_indices_begins") {
            set_shape(ov::PartialShape{2});
        } else if (name == "block_indices" && reshape_cache_buffers) {
            set_shape(ov::PartialShape{block_capacity});
        } else if (name == "max_context_len") {
            set_shape(ov::PartialShape{});
        } else if (reshape_cache_buffers && (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0)) {
            set_shape(ov::PartialShape{block_capacity, heads, head_dim, block_size});
        }
    }
    if (!changed) {
        return false;
    }
    model->reshape(shapes);
    model->validate_nodes_and_infer_types();
    if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
        std::cerr << "paged_kv_static_decode mode=" << static_mode
                  << " inputs=[" << model_input_summary(model) << "]" << std::endl;
    }
    return true;
}

std::shared_ptr<ov::Model> convert_paged_kv_seed_model(
    ov::Core& core,
    const char* seed_xml,
    int64_t heads,
    int64_t block_size,
    int64_t head_dim,
    ov::element::Type cache_element_type) {
    auto model = core.read_model(seed_xml);
    add_readvalue_initializers(model);
    const bool allow_score_aggregation = enabled_env("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION", true);
    try {
        ov::pass::SDPAToPagedAttention(
            false,
            false,
            allow_score_aggregation,
            false,
            false,
            false)
            .run_on_model(model);
    } catch (const std::exception& exc) {
        if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
            std::cerr << "SDPAToPagedAttention reported non-fatal error: " << exc.what() << std::endl;
        }
    }
    const size_t restored_parameters = restore_unregistered_parameters(model);
    specialize_kv_cache_parameters(model, heads, block_size, head_dim, cache_element_type);
    if (restored_parameters == 0) {
        flatten_frame_embed_result(model);
    }
    if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
        std::cerr << "paged_kv_seed restored_parameters=" << restored_parameters
                  << " inputs=[" << model_input_summary(model) << "]" << std::endl;
    }
    try {
        model->validate_nodes_and_infer_types();
    } catch (const std::exception&) {
        // SDPAToPagedAttention may leave beam_idx in a state where model
        // validation still reports it as undeclared, while the converted graph
        // is accepted by compile_model and beam_idx is bound as a normal input.
        // Treat compile_model as the authoritative check for this experimental
        // paged-KV path.
    }
    return model;
}

std::vector<float> make_attention_mask(int64_t prompt_len, int64_t bucket) {
    if (prompt_len <= 0 || bucket <= 0) {
        throw std::runtime_error("prompt_len and bucket must be positive");
    }
    std::vector<float> mask(static_cast<size_t>(prompt_len * bucket), NEG_INF);
    for (int64_t row = 0; row < prompt_len; ++row) {
        const int64_t allowed_end = std::min<int64_t>(row, bucket - 1);
        for (int64_t col = 0; col <= allowed_end; ++col) {
            mask[static_cast<size_t>(row * bucket + col)] = 0.0f;
        }
    }
    return mask;
}

std::vector<int64_t> make_positions(int64_t start, int64_t count) {
    std::vector<int64_t> positions(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        positions[static_cast<size_t>(i)] = start + i;
    }
    return positions;
}

void fill_positions(std::vector<int64_t>& positions, int64_t start, int64_t count) {
    positions.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        positions[static_cast<size_t>(i)] = start + i;
    }
}

void fill_attention_mask(std::vector<float>& mask, int64_t prompt_len, int64_t bucket) {
    if (prompt_len <= 0 || bucket <= 0) {
        throw std::runtime_error("prompt_len and bucket must be positive");
    }
    mask.assign(static_cast<size_t>(prompt_len * bucket), NEG_INF);
    for (int64_t row = 0; row < prompt_len; ++row) {
        const int64_t allowed_end = std::min<int64_t>(row, bucket - 1);
        for (int64_t col = 0; col <= allowed_end; ++col) {
            mask[static_cast<size_t>(row * bucket + col)] = 0.0f;
        }
    }
}

std::vector<float> make_allow_eos(int64_t step, int64_t min_new_tokens, int64_t unroll) {
    std::vector<float> allow(static_cast<size_t>(unroll), 0.0f);
    for (int64_t i = 0; i < unroll; ++i) {
        allow[static_cast<size_t>(i)] = (step + i >= min_new_tokens) ? 1.0f : 0.0f;
    }
    return allow;
}

void fill_allow_eos(std::vector<float>& allow, int64_t step, int64_t min_new_tokens, int64_t unroll) {
    allow.resize(static_cast<size_t>(unroll));
    for (int64_t i = 0; i < unroll; ++i) {
        allow[static_cast<size_t>(i)] = (step + i >= min_new_tokens) ? 1.0f : 0.0f;
    }
}

double elapsed_ms_since(std::chrono::steady_clock::time_point started) {
    return ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - started) / 1000.0;
}

template <typename Fn>
void measure_ms(double& target, Fn&& fn) {
    const auto started = std::chrono::steady_clock::now();
    fn();
    target += elapsed_ms_since(started);
}

void copy_matching_states(ov::InferRequest& source, ov::InferRequest& target) {
    std::unordered_map<std::string, ov::Tensor> states;
    for (const auto& state : source.query_state()) {
        states.emplace(state.get_name(), state.get_state());
    }
    for (auto& state : target.query_state()) {
        auto found = states.find(state.get_name());
        if (found != states.end()) {
            state.set_state(found->second);
        }
    }
}

bool set_repeated_mask_state(ov::InferRequest& request, const std::vector<float>& repeated_mask) {
    for (auto& state : request.query_state()) {
        const auto name = state.get_name();
        if (name == "repeated_mask" || name.rfind("repeated_mask", 0) == 0) {
            ov::Tensor tensor(ov::element::f32, ov::Shape{1, repeated_mask.size()}, repeated_mask.data());
            state.set_state(tensor);
            return true;
        }
    }
    return false;
}

bool compiled_model_has_input(const ov::CompiledModel& model, const std::string& name) {
    for (const auto& input : model.inputs()) {
        const auto names = input.get_names();
        if (names.find(name) != names.end()) {
            return true;
        }
    }
    return false;
}

bool compiled_model_output_has_name(const ov::CompiledModel& model, size_t index, const std::string& name) {
    if (index >= model.outputs().size()) {
        return false;
    }
    const auto names = model.output(index).get_names();
    return names.find(name) != names.end();
}

bool compiled_model_first_output_is_first_code(const ov::CompiledModel& model) {
    if (model.outputs().empty()) {
        return false;
    }
    return compiled_model_output_has_name(model, 0, "first_code") ||
           model.output(0).get_element_type() == ov::element::i64;
}

int64_t compiled_model_static_input_size(const ov::CompiledModel& model, const std::string& name) {
    for (const auto& input : model.inputs()) {
        const auto names = input.get_names();
        if (names.find(name) == names.end()) {
            continue;
        }
        const auto pshape = input.get_partial_shape();
        if (pshape.rank().is_static() && pshape.rank().get_length() == 1 && pshape[0].is_static()) {
            return pshape[0].get_length();
        }
        return 0;
    }
    return 0;
}

int64_t compiled_model_input_rank(const ov::CompiledModel& model, const std::string& name) {
    for (const auto& input : model.inputs()) {
        const auto names = input.get_names();
        if (names.find(name) == names.end()) {
            continue;
        }
        const auto rank = input.get_partial_shape().rank();
        if (rank.is_static()) {
            return rank.get_length();
        }
        return 0;
    }
    return 0;
}

bool request_has_repeated_mask_state(ov::InferRequest& request) {
    for (const auto& state : request.query_state()) {
        const auto name = state.get_name();
        if (name == "repeated_mask" || name.rfind("repeated_mask", 0) == 0) {
            return true;
        }
    }
    return false;
}

template <typename T>
const T* tensor_data(const ov::Tensor& tensor) {
    return tensor.data<const T>();
}

std::string tensor_debug_string(const ov::Tensor& tensor) {
    std::ostringstream out;
    out << "type=" << tensor.get_element_type() << ", shape=" << tensor.get_shape() << ", size=" << tensor.get_size();
    return out.str();
}

template <typename T>
struct HostTensorView {
    const T* data = nullptr;
    size_t size = 0;
    ov::Tensor host_tensor;
    bool copied = false;
};

template <typename T>
HostTensorView<T> tensor_to_host_view(NativeCodegen* runner, const ov::Tensor& tensor, const char* label) {
    HostTensorView<T> view;
    view.size = tensor.get_size();
    if (view.size == 0) {
        return view;
    }
    view.data = tensor.data<const T>();
    if (view.data) {
        if (runner) {
            runner->last_timing.zero_copy_count += 1;
        }
        return view;
    }
    const auto copy_started = std::chrono::steady_clock::now();
    view.host_tensor = ov::Tensor(tensor.get_element_type(), tensor.get_shape());
    tensor.copy_to(view.host_tensor);
    if (runner) {
        runner->last_timing.host_copy_ms += elapsed_ms_since(copy_started);
        runner->last_timing.host_copy_bytes += static_cast<uint64_t>(tensor.get_byte_size());
        runner->last_timing.host_copy_fallback_count += 1;
    }
    view.data = view.host_tensor.template data<const T>();
    view.copied = true;
    if (!view.data) {
        std::ostringstream msg;
        msg << "failed to copy " << (label ? label : "tensor") << " to host-readable memory ("
            << tensor_debug_string(tensor) << ")";
        throw std::runtime_error(msg.str());
    }
    return view;
}

template <typename T>
std::vector<T> tensor_to_host_vector(NativeCodegen* runner, const ov::Tensor& tensor, const char* label) {
    const size_t size = tensor.get_size();
    if (size == 0) {
        return {};
    }
    const auto view = tensor_to_host_view<T>(runner, tensor, label);
    return std::vector<T>(view.data, view.data + view.size);
}

struct FloatTensorView {
    const float* data = nullptr;
    size_t size = 0;
    ov::Tensor host_tensor;
    std::vector<float> converted_storage;
    bool copied = false;
    bool converted = false;
};

FloatTensorView tensor_to_host_f32_view(NativeCodegen* runner, const ov::Tensor& tensor, const char* label) {
    FloatTensorView view;
    view.size = tensor.get_size();
    if (view.size == 0) {
        return view;
    }
    const auto element_type = tensor.get_element_type();
    if (element_type == ov::element::f32) {
        auto base = tensor_to_host_view<float>(runner, tensor, label);
        view.data = base.data;
        view.size = base.size;
        view.host_tensor = std::move(base.host_tensor);
        view.copied = base.copied;
        return view;
    }
    if (element_type == ov::element::f16) {
        const auto values = tensor_to_host_view<ov::float16>(runner, tensor, label);
        view.converted_storage.reserve(values.size);
        for (size_t i = 0; i < values.size; ++i) {
            view.converted_storage.push_back(static_cast<float>(values.data[i]));
        }
        view.data = view.converted_storage.data();
        view.size = view.converted_storage.size();
        view.copied = values.copied;
        view.converted = true;
        return view;
    }
    if (element_type == ov::element::bf16) {
        const auto values = tensor_to_host_view<ov::bfloat16>(runner, tensor, label);
        view.converted_storage.reserve(values.size);
        for (size_t i = 0; i < values.size; ++i) {
            view.converted_storage.push_back(static_cast<float>(values.data[i]));
        }
        view.data = view.converted_storage.data();
        view.size = view.converted_storage.size();
        view.copied = values.copied;
        view.converted = true;
        return view;
    }
    std::ostringstream msg;
    msg << (label ? label : "tensor") << " must be f32/f16/bf16, got " << tensor_debug_string(tensor);
    throw std::runtime_error(msg.str());
}

std::vector<float> tensor_to_host_f32_vector(NativeCodegen* runner, const ov::Tensor& tensor, const char* label) {
    const auto view = tensor_to_host_f32_view(runner, tensor, label);
    if (!view.data || view.size == 0) {
        return {};
    }
    return std::vector<float>(view.data, view.data + view.size);
}

int64_t tensor_to_host_i64_scalar(NativeCodegen* runner, const ov::Tensor& tensor, const char* label) {
    const auto view = tensor_to_host_view<int64_t>(runner, tensor, label);
    if (!view.data || view.size == 0) {
        std::ostringstream msg;
        msg << (label ? label : "tensor") << " must contain at least one i64 value ("
            << tensor_debug_string(tensor) << ")";
        throw std::runtime_error(msg.str());
    }
    return view.data[0];
}

bool env_enabled(const char* name, bool default_value) {
    return enabled_env(name, default_value);
}

int64_t env_int64(const char* name, int64_t default_value) {
    const char* value = std::getenv(name);
    if (!value || std::strlen(value) == 0 || disabled_env_value(value)) {
        return default_value;
    }
    try {
        return std::stoll(value);
    } catch (...) {
        return default_value;
    }
}

std::vector<int64_t> parse_int64_list(const std::string& text, const std::vector<int64_t>& defaults) {
    std::vector<int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if (item.empty()) {
            continue;
        }
        try {
            const int64_t parsed = std::stoll(item);
            if (parsed > 0) {
                values.push_back(parsed);
            }
        } catch (...) {
        }
    }
    if (values.empty()) {
        values = defaults;
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<int64_t> env_int64_list(const char* name, const std::vector<int64_t>& defaults) {
    const char* value = std::getenv(name);
    if (!value || std::strlen(value) == 0 || disabled_env_value(value)) {
        return defaults;
    }
    return parse_int64_list(value, defaults);
}

int64_t bucket_ge(const std::vector<int64_t>& buckets, int64_t value) {
    if (buckets.empty()) {
        return std::max<int64_t>(1, value);
    }
    for (const int64_t bucket : buckets) {
        if (bucket >= value) {
            return bucket;
        }
    }
    return buckets.back();
}

int64_t bucket_limit_le(const std::vector<int64_t>& buckets, int64_t limit) {
    if (limit <= 0) {
        return 0;
    }
    int64_t selected = 0;
    for (const int64_t bucket : buckets) {
        if (bucket <= limit) {
            selected = bucket;
        }
    }
    return selected > 0 ? selected : std::min<int64_t>(limit, buckets.empty() ? limit : buckets.front());
}

void increment_histogram(std::vector<int64_t>& histogram, int64_t index) {
    if (index < 0) {
        return;
    }
    if (histogram.size() <= static_cast<size_t>(index)) {
        histogram.resize(static_cast<size_t>(index + 1), 0);
    }
    histogram[static_cast<size_t>(index)] += 1;
}

std::string env_string(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    if (!value || std::strlen(value) == 0) {
        return default_value;
    }
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::replace(out.begin(), out.end(), '-', '_');
    return out;
}

std::string sampled_batch_subcode_policy_from_env() {
    std::string policy = env_string("QWEN3_TTS_OV_NATIVE_SAMPLED_BATCH_SUBCODE", "off");
    if (policy == "0" || policy == "false" || policy == "disabled") {
        return "off";
    }
    if (policy == "1" || policy == "true" || policy == "enabled") {
        return "on";
    }
    if (policy == "off" || policy == "verify" || policy == "on") {
        return policy;
    }
    return "off";
}

double micros_to_ms(std::chrono::microseconds value) {
    return std::chrono::duration<double, std::milli>(value).count();
}

std::string json_escape_native(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

std::string profile_scope(const NativeCodegen::ProfileEntry& item) {
    const std::string name = lower_text(item.node_name);
    if (name.find("code_predictor") != std::string::npos ||
        name.find("subcode") != std::string::npos ||
        name.find("mtp") != std::string::npos) {
        return "subcode_predictor";
    }
    if (item.node_type == "PagedAttentionExtension") {
        return "talker_attention";
    }
    if ((item.node_type == "SDPA" ||
         item.node_type == "ScaledDotProductAttention") &&
        name.find("talker") != std::string::npos) {
        return "talker_attention";
    }
    if (name.find("__module.talker.model.layers") != std::string::npos) {
        if (name.find(".mlp.") != std::string::npos) {
            return "talker_mlp";
        }
        if (name.find("self_attn") != std::string::npos ||
            item.node_type == "SDPA" ||
            item.node_type == "PagedAttentionExtension") {
            return "talker_attention";
        }
        if (item.node_type == "RMS") {
            return "talker_norm";
        }
        return "talker_layer_other";
    }
    if (name.find("__module.talker.codec_head") != std::string::npos) {
        return "talker_codec_head";
    }
    if (name.find("stream_decoder") != std::string::npos ||
        name.find("speech_decoder") != std::string::npos ||
        name.find("__module.decoder") != std::string::npos) {
        return "audio_decoder";
    }
    return "other";
}

std::string model_input_summary(const std::shared_ptr<ov::Model>& model) {
    std::ostringstream out;
    bool first = true;
    for (const auto& input : model->inputs()) {
        if (!first) {
            out << ", ";
        }
        first = false;
        try {
            out << input.get_any_name();
        } catch (...) {
            out << "<unnamed>";
        }
        out << ":" << input.get_partial_shape().to_string() << ":" << input.get_element_type();
    }
    return out.str();
}

void record_request_profile(NativeCodegen* runner, const std::string& label, const ov::InferRequest& request) {
    if (!runner || !runner->profile_enabled) {
        return;
    }
    for (const auto& info : request.get_profiling_info()) {
        if (info.status != ov::ProfilingInfo::Status::EXECUTED || info.real_time.count() <= 0) {
            continue;
        }
        const std::string key = label + "\x1f" + info.node_name + "\x1f" + info.node_type + "\x1f" + info.exec_type;
        auto& item = runner->profile_ops[key];
        item.label = label;
        item.node_name = info.node_name;
        item.node_type = info.node_type;
        item.exec_type = info.exec_type;
        item.real_time_ms += micros_to_ms(info.real_time);
        item.cpu_time_ms += micros_to_ms(info.cpu_time);
        item.count += 1;
    }
}

std::string native_profile_json(const NativeCodegen& runner) {
    if (!runner.profile_enabled) {
        return "null";
    }
    std::vector<NativeCodegen::ProfileEntry> entries;
    entries.reserve(runner.profile_ops.size());
    for (const auto& kv : runner.profile_ops) {
        entries.push_back(kv.second);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.real_time_ms > rhs.real_time_ms;
    });

    auto aggregate = [&](const std::string& field) {
        std::unordered_map<std::string, NativeCodegen::ProfileEntry> totals;
        for (const auto& item : entries) {
            const std::string key = field == "label" ? item.label : (field == "scope" ? profile_scope(item) : item.node_type);
            auto& total = totals[key];
            total.label = key;
            total.node_type = key;
            total.real_time_ms += item.real_time_ms;
            total.cpu_time_ms += item.cpu_time_ms;
            total.count += item.count;
        }
        std::vector<NativeCodegen::ProfileEntry> out;
        out.reserve(totals.size());
        for (const auto& kv : totals) {
            out.push_back(kv.second);
        }
        std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.real_time_ms > rhs.real_time_ms;
        });
        return out;
    };

    auto append_totals = [](std::string& json, const std::vector<NativeCodegen::ProfileEntry>& items, const char* name) {
        json += "  \"" + std::string(name) + "\": [\n";
        for (size_t i = 0; i < items.size(); ++i) {
            const auto& item = items[i];
            const std::string label = std::string(name) == "by_label" ? item.label : item.node_type;
            json += "    {\"name\": \"" + json_escape_native(label) +
                    "\", \"real_time_ms\": " + std::to_string(item.real_time_ms) +
                    ", \"cpu_time_ms\": " + std::to_string(item.cpu_time_ms) +
                    ", \"count\": " + std::to_string(item.count) + "}";
            json += (i + 1 == items.size()) ? "\n" : ",\n";
        }
        json += "  ],\n";
    };

    std::string json = "{\n";
    json += "  \"enabled\": true,\n";
    append_totals(json, aggregate("label"), "by_label");
    append_totals(json, aggregate("node_type"), "by_type");
    append_totals(json, aggregate("scope"), "by_scope");
    json += "  \"top\": [\n";
    const size_t limit = std::min<size_t>(50, entries.size());
    for (size_t i = 0; i < limit; ++i) {
        const auto& item = entries[i];
        json += "    {\"label\": \"" + json_escape_native(item.label) +
                "\", \"node_name\": \"" + json_escape_native(item.node_name) +
                "\", \"node_type\": \"" + json_escape_native(item.node_type) +
                "\", \"exec_type\": \"" + json_escape_native(item.exec_type) +
                "\", \"real_time_ms\": " + std::to_string(item.real_time_ms) +
                ", \"cpu_time_ms\": " + std::to_string(item.cpu_time_ms) +
                ", \"count\": " + std::to_string(item.count) + "}";
        json += (i + 1 == limit) ? "\n" : ",\n";
    }
    json += "  ],\n";
    std::map<std::string, std::vector<NativeCodegen::ProfileEntry>> entries_by_label;
    for (const auto& item : entries) {
        entries_by_label[item.label].push_back(item);
    }
    json += "  \"top_by_label\": {\n";
    size_t label_index = 0;
    for (auto& kv : entries_by_label) {
        auto& items = kv.second;
        std::sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.real_time_ms > rhs.real_time_ms;
        });
        json += "    \"" + json_escape_native(kv.first) + "\": [\n";
        const size_t label_limit = std::min<size_t>(20, items.size());
        for (size_t i = 0; i < label_limit; ++i) {
            const auto& item = items[i];
            json += "      {\"node_name\": \"" + json_escape_native(item.node_name) +
                    "\", \"node_type\": \"" + json_escape_native(item.node_type) +
                    "\", \"exec_type\": \"" + json_escape_native(item.exec_type) +
                    "\", \"real_time_ms\": " + std::to_string(item.real_time_ms) +
                    ", \"cpu_time_ms\": " + std::to_string(item.cpu_time_ms) +
                    ", \"count\": " + std::to_string(item.count) + "}";
            json += (i + 1 == label_limit) ? "\n" : ",\n";
        }
        json += "    ]";
        json += (++label_index == entries_by_label.size()) ? "\n" : ",\n";
    }
    json += "  }\n";
    json += "}";
    return json;
}

std::pair<double, double> f32_l2_and_max_abs(const float* data, size_t size) {
    if (!data || size == 0) {
        return {0.0, 0.0};
    }
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double value = static_cast<double>(data[i]);
        sum_sq += value * value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    return {std::sqrt(sum_sq), max_abs};
}

void append_codegen_trace_json(std::string& json, const NativeCodegen& runner) {
    json += ", \"codegen_trace\": [";
    for (size_t i = 0; i < runner.codegen_trace.size(); ++i) {
        const auto& item = runner.codegen_trace[i];
        json += "{";
        json += "\"frame\": " + std::to_string(item.frame);
        json += ", \"label\": \"" + json_escape_native(item.label) + "\"";
        json += ", \"first_code\": " + std::to_string(item.first_code);
        json += ", \"codes\": [";
        for (size_t j = 0; j < item.codes.size(); ++j) {
            json += std::to_string(item.codes[j]);
            json += (j + 1 == item.codes.size()) ? "" : ",";
        }
        json += "]";
        json += ", \"hidden_l2\": " + std::to_string(item.hidden_l2);
        json += ", \"hidden_max_abs\": " + std::to_string(item.hidden_max_abs);
        json += ", \"embed_l2\": " + std::to_string(item.embed_l2);
        json += ", \"embed_max_abs\": " + std::to_string(item.embed_max_abs);
        json += "}";
        json += (i + 1 == runner.codegen_trace.size()) ? "" : ",";
    }
    json += "]";
}

void append_timing_distribution_json(std::string& json, const char* name, const std::vector<double>& values) {
    json += ", \"" + std::string(name) + "\": {";
    json += "\"count\": " + std::to_string(values.size());
    if (values.empty()) {
        json += ", \"total_ms\": 0.000000, \"mean_ms\": 0.000000, \"p50_ms\": 0.000000, \"p90_ms\": 0.000000, \"max_ms\": 0.000000}";
        return;
    }
    std::vector<double> ordered;
    ordered.reserve(values.size());
    double total = 0.0;
    double max_value = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) {
            continue;
        }
        ordered.push_back(value);
        total += value;
        max_value = std::max(max_value, value);
    }
    if (ordered.empty()) {
        json += ", \"total_ms\": 0.000000, \"mean_ms\": 0.000000, \"p50_ms\": 0.000000, \"p90_ms\": 0.000000, \"max_ms\": 0.000000}";
        return;
    }
    std::sort(ordered.begin(), ordered.end());
    auto pct = [&](double percent) {
        const double raw = (percent / 100.0) * static_cast<double>(ordered.size() - 1);
        const size_t index = static_cast<size_t>(std::max<double>(0.0, std::min<double>(ordered.size() - 1, std::round(raw))));
        return ordered[index];
    };
    json += ", \"total_ms\": " + std::to_string(total);
    json += ", \"mean_ms\": " + std::to_string(total / static_cast<double>(ordered.size()));
    json += ", \"p50_ms\": " + std::to_string(pct(50.0));
    json += ", \"p90_ms\": " + std::to_string(pct(90.0));
    json += ", \"max_ms\": " + std::to_string(max_value);
    json += "}";
}

void append_int_vector_json(std::string& json, const char* name, const std::vector<int64_t>& values) {
    json += ", \"" + std::string(name) + "\": [";
    for (size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            json += ", ";
        }
        json += std::to_string(values[index]);
    }
    json += "]";
}

std::string native_timing_json(const NativeCodegen& runner) {
    const auto& item = runner.last_timing;
    std::string json = "{";
    json += "\"buffer_reuse\": ";
    json += item.buffer_reuse ? "true" : "false";
    json += ", \"no_repeat_fast_path\": ";
    json += item.no_repeat_fast_path ? "true" : "false";
    json += ", \"kv_cache_tensor_reuse\": ";
    json += item.kv_cache_tensor_reuse ? "true" : "false";
    json += ", \"paged_kv_precision\": \"" + json_escape_native(runner.paged_kv_precision) + "\"";
    json += ", \"paged_kv_cache_input_precision\": \"" + json_escape_native(runner.paged_kv_cache_input_precision) + "\"";
    json += ", \"paged_split_subcode\": ";
    json += runner.paged_split_subcode ? "true" : "false";
    json += ", \"paged_seed_outputs_first_code\": ";
    json += runner.paged_seed_outputs_first_code ? "true" : "false";
    json += ", \"prompt_embedding_cache_enabled\": ";
    json += runner.prompt_embedding_cache_enabled ? "true" : "false";
    json += ", \"prompt_embedding_cache_size\": " + std::to_string(runner.prompt_embedding_cache.size());
    json += ", \"prompt_embedding_cache_hits\": " + std::to_string(runner.prompt_embedding_cache_hits);
    json += ", \"prompt_embedding_cache_misses\": " + std::to_string(runner.prompt_embedding_cache_misses);
    json += ", \"async_decode\": ";
    json += runner.last_async_decode ? "true" : "false";
    json += ", \"paged_static_decode_enabled\": ";
    json += runner.paged_static_decode_enabled ? "true" : "false";
    json += ", \"paged_static_decode_requested\": ";
    json += runner.paged_static_decode_requested ? "true" : "false";
    json += ", \"paged_static_decode_mode\": \"" + json_escape_native(runner.paged_static_decode_mode) + "\"";
    json += ", \"paged_static_decode_failure\": \"" + json_escape_native(runner.paged_static_decode_failure) + "\"";
    json += ", \"paged_fused_batch_decode_enabled\": ";
    json += runner.paged_fused_batch_decode_enabled ? "true" : "false";
    json += ", \"paged_fused_batch_decode_graph\": \"" + json_escape_native(runner.paged_fused_batch_decode_graph) + "\"";
    json += ", \"continuous_batch_policy\": \"" + json_escape_native(runner.continuous_batch_policy) + "\"";
    json += ", \"online_scheduler\": \"" + json_escape_native(runner.online_scheduler) + "\"";
    json += ", \"prefill_mode\": \"" + json_escape_native(runner.prefill_mode) + "\"";
    json += ", \"max_num_batched_tokens\": " + std::to_string(runner.max_num_batched_tokens);
    json += ", \"batch_subcode_compiled\": ";
    json += runner.batch_subcode_enabled ? "true" : "false";
    json += ", \"batch_decode_unroll_requested\": " + std::to_string(runner.batch_decode_unroll_requested);
    json += ", \"batch_decode_unroll\": " + std::to_string(runner.batch_decode_unroll);
    json += ", \"host_prepare_ms\": " + std::to_string(item.host_prepare_ms);
    json += ", \"tensor_bind_ms\": " + std::to_string(item.tensor_bind_ms);
    json += ", \"codegen_infer_ms\": " + std::to_string(item.codegen_infer_ms);
    json += ", \"codegen_prefill_infer_ms\": " + std::to_string(item.codegen_prefill_infer_ms);
    json += ", \"codegen_decode_infer_ms\": " + std::to_string(item.codegen_decode_infer_ms);
    json += ", \"codegen_subcode_infer_ms\": " + std::to_string(item.codegen_subcode_infer_ms);
    json += ", \"subcode_bind_ms\": " + std::to_string(item.subcode_bind_ms);
    json += ", \"subcode_output_read_ms\": " + std::to_string(item.subcode_output_read_ms);
    json += ", \"subcode_next_embed_ms\": " + std::to_string(item.subcode_next_embed_ms);
    json += ", \"subcode_host_copy_bytes\": " + std::to_string(item.subcode_host_copy_bytes);
    json += ", \"subcode_host_copy_fallback_count\": " + std::to_string(item.subcode_host_copy_fallback_count);
    json += ", \"split_subcode_hidden_direct_bind_count\": " + std::to_string(item.split_subcode_hidden_direct_bind_count);
    json += ", \"split_subcode_hidden_bind_fallback_count\": " + std::to_string(item.split_subcode_hidden_bind_fallback_count);
    json += ", \"split_subcode_hidden_copy_bytes\": " + std::to_string(item.split_subcode_hidden_copy_bytes);
    json += ", \"split_subcode_remote_next_embed_enabled\": ";
    json += item.split_subcode_remote_next_embed_enabled ? "true" : "false";
    json += ", \"split_subcode_remote_next_embed_fallback_count\": " +
            std::to_string(item.split_subcode_remote_next_embed_fallback_count);
    json += ", \"split_subcode_next_embed_host_read_count\": " +
            std::to_string(item.split_subcode_next_embed_host_read_count);
    json += ", \"decode_step_prebind_enabled\": ";
    json += item.decode_step_prebind_enabled ? "true" : "false";
    json += ", \"decode_step_prebind_fallback_count\": " +
            std::to_string(item.decode_step_prebind_fallback_count);
    json += ", \"decode_step_prebind_update_ms\": " + std::to_string(item.decode_step_prebind_update_ms);
    json += ", \"batch_prefill_enabled\": ";
    json += item.batch_prefill_enabled ? "true" : "false";
    json += ", \"prefill_mode\": \"" + json_escape_native(item.prefill_mode) + "\"";
    json += ", \"batch_subcode_enabled\": ";
    json += item.batch_subcode_enabled ? "true" : "false";
    json += ", \"scheduler_kind\": \"" + json_escape_native(item.online_scheduler) + "\"";
    json += ", \"prefill_seq_bucket\": " + std::to_string(item.prefill_seq_bucket);
    json += ", \"prefill_batch_bucket\": " + std::to_string(item.prefill_batch_bucket);
    json += ", \"decode_batch_bucket\": " + std::to_string(item.decode_batch_bucket);
    json += ", \"prefill_token_budget\": " + std::to_string(item.prefill_token_budget);
    json += ", \"decode_token_budget\": " + std::to_string(item.decode_token_budget);
    json += ", \"max_num_batched_tokens_effective\": " + std::to_string(item.max_num_batched_tokens);
    json += ", \"sampled_batch_subcode_policy\": \"" + json_escape_native(item.sampled_batch_subcode_policy) + "\"";
    json += ", \"sampled_batch_subcode_used\": ";
    json += item.sampled_batch_subcode_used ? "true" : "false";
    json += ", \"sampled_batch_subcode_verified\": ";
    json += item.sampled_batch_subcode_verified ? "true" : "false";
    json += ", \"sampled_batch_subcode_fallback_count\": " +
            std::to_string(item.sampled_batch_subcode_fallback_count);
    json += ", \"sampled_batch_subcode_mismatch_count\": " +
            std::to_string(item.sampled_batch_subcode_mismatch_count);
    json += ", \"sampled_batch_subcode_code_mismatch_count\": " +
            std::to_string(item.sampled_batch_subcode_code_mismatch_count);
    json += ", \"sampled_batch_subcode_embed_mismatch_count\": " +
            std::to_string(item.sampled_batch_subcode_embed_mismatch_count);
    json += ", \"sampled_batch_subcode_max_abs_diff\": " +
            std::to_string(item.sampled_batch_subcode_max_abs_diff);
    json += ", \"sampled_batch_subcode_fallback_reason\": \"" +
            json_escape_native(item.sampled_batch_subcode_fallback_reason) + "\"";
    json += ", \"sampled_subcode_parallel_rows\": ";
    json += item.sampled_subcode_parallel_rows ? "true" : "false";
    json += ", \"sampled_subcode_parallel_row_count\": " +
            std::to_string(item.sampled_subcode_parallel_row_count);
    json += ", \"batch_buffer_reuse\": ";
    json += item.batch_buffer_reuse ? "true" : "false";
    json += ", \"batch_decode_unroll_fallback_count\": " + std::to_string(item.batch_decode_unroll_fallback_count);
    json += ", \"batch_fused_decode_step_count\": " + std::to_string(item.batch_fused_decode_step_count);
    json += ", \"batch_fused_decode_token_count\": " + std::to_string(item.batch_fused_decode_token_count);
    json += ", \"batch_single_decode_step_count\": " + std::to_string(item.batch_single_decode_step_count);
    json += ", \"batch_single_decode_token_count\": " + std::to_string(item.batch_single_decode_token_count);
    json += ", \"batch_fused_decode_active1_bypass_count\": " +
            std::to_string(item.batch_fused_decode_active1_bypass_count);
    json += ", \"batch_fused_decode_logits_bypass_count\": " +
            std::to_string(item.batch_fused_decode_logits_bypass_count);
    json += ", \"subcode_outputs_next_embed\": ";
    json += runner.subcode_outputs_next_embed ? "true" : "false";
    json += ", \"sampling_ms\": " + std::to_string(item.sampling_ms);
    json += ", \"decode_infer_ms\": " + std::to_string(item.decode_infer_ms);
    json += ", \"callback_ms\": " + std::to_string(item.callback_ms);
    json += ", \"callback_enqueue_ms\": " + std::to_string(item.callback_enqueue_ms);
    json += ", \"inline_decode_ms\": " + std::to_string(item.inline_decode_ms);
    json += ", \"codegen_callback_ms\": " + std::to_string(item.codegen_callback_ms);
    json += ", \"decode_callback_ms\": " + std::to_string(item.decode_callback_ms);
    json += ", \"decode_queue_wait_ms\": " + std::to_string(item.decode_queue_wait_ms);
    json += ", \"decode_queue_depth_max\": " + std::to_string(item.decode_queue_depth_max);
    json += ", \"decode_input_frames\": " + std::to_string(item.decode_input_frames);
    json += ", \"decode_padded_frames\": " + std::to_string(item.decode_padded_frames);
    json += ", \"ttft_ms\": " + std::to_string(item.ttft_ms);
    json += ", \"last_token_ms\": " + std::to_string(item.last_token_ms);
    json += ", \"generated_frames\": " + std::to_string(item.generated_frames);
    json += ", \"host_copy_ms\": " + std::to_string(item.host_copy_ms);
    json += ", \"host_copy_bytes\": " + std::to_string(item.host_copy_bytes);
    json += ", \"host_copy_fallback_count\": " + std::to_string(item.host_copy_fallback_count);
    json += ", \"zero_copy_count\": " + std::to_string(item.zero_copy_count);
    json += ", \"codegen_prefill_count\": " + std::to_string(item.codegen_prefill_count);
    json += ", \"codegen_decode_count\": " + std::to_string(item.codegen_decode_count);
    json += ", \"codegen_subcode_count\": " + std::to_string(item.codegen_subcode_count);
    json += ", \"pcm_convert_ms\": " + std::to_string(item.pcm_convert_ms);
    json += ", \"total_ms\": " + std::to_string(item.total_ms);
    append_timing_distribution_json(json, "codegen_prefill_step_stats", item.codegen_prefill_step_ms);
    append_timing_distribution_json(json, "codegen_decode_step_stats", item.codegen_decode_step_ms);
    append_timing_distribution_json(json, "codegen_bind_step_stats", item.codegen_bind_step_ms);
    append_timing_distribution_json(json, "codegen_sampling_step_stats", item.codegen_sampling_step_ms);
    append_timing_distribution_json(json, "subcode_bind_step_stats", item.subcode_bind_step_ms);
    append_timing_distribution_json(json, "subcode_infer_step_stats", item.subcode_infer_step_ms);
    append_timing_distribution_json(json, "subcode_output_read_step_stats", item.subcode_output_read_step_ms);
    append_timing_distribution_json(json, "subcode_next_embed_step_stats", item.subcode_next_embed_step_ms);
    append_timing_distribution_json(json, "decode_step_prebind_update_step_stats", item.decode_step_prebind_update_step_ms);
    append_int_vector_json(json, "active_batch_histogram", item.active_batch_histogram);
    append_int_vector_json(json, "prefill_seq_bucket_histogram", item.prefill_seq_bucket_histogram);
    append_int_vector_json(json, "prefill_batch_bucket_histogram", item.prefill_batch_bucket_histogram);
    append_int_vector_json(json, "decode_batch_bucket_histogram", item.decode_batch_bucket_histogram);
    append_codegen_trace_json(json, runner);
    json += "}";
    return json;
}

ov::Tensor try_create_remote_tensor(ov::CompiledModel& model, const ov::element::Type& element_type, const ov::Shape& shape) {
    try {
        ov::RemoteContext context = model.get_context();
        return context.create_tensor(element_type, shape);
    } catch (const ov::Exception&) {
        return ov::Tensor();
    } catch (const std::exception&) {
        return ov::Tensor();
    }
}

struct RemoteEmbedChain {
    bool enabled = false;
    ov::Tensor prefill_output;
    std::array<ov::Tensor, 2> decode_outputs;
    size_t next_decode_output = 0;
    ov::Tensor next_input;
};

struct SplitSubcodeRemoteNextEmbedChain {
    bool enabled = false;
    bool requested = false;
    bool required = false;
    bool verified_decode_input = false;
    std::array<ov::Tensor, 2> outputs;
    size_t next_output = 0;
    std::string failure;
};

struct PagedDecodeStepBindings {
    bool enabled = false;
    bool attempted = false;
    bool has_tts_pad_embed = false;
    bool has_allow_eos = false;
    bool has_allow_eos_steps = false;
    bool has_beam_idx = false;
    bool has_score_aggregation_window = false;
    int64_t hidden_size = 0;
    int64_t block_size = 0;
    int64_t position_rank = 0;
    int64_t inputs_rank = 0;
    int64_t block_indices_len = 0;
    std::vector<int64_t> position_ids;
    std::vector<int32_t> block_indices;
    std::vector<float> allow_eos;
    std::vector<int64_t> beam_idx;
    std::vector<int32_t> score_aggregation_window;
    std::vector<int32_t> past_lens;
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices_begins;
    std::vector<int32_t> max_context_len;
    ov::Tensor inputs_embeds_tensor;
    ov::Tensor position_ids_tensor;
    ov::Tensor tts_pad_embed_tensor;
    ov::Tensor allow_eos_tensor;
    ov::Tensor allow_eos_steps_tensor;
    ov::Tensor beam_idx_tensor;
    ov::Tensor score_aggregation_window_tensor;
    ov::Tensor past_lens_tensor;
    ov::Tensor subsequence_begins_tensor;
    ov::Tensor block_indices_tensor;
    ov::Tensor block_indices_begins_tensor;
    ov::Tensor max_context_len_tensor;
    std::string failure;
};

struct PromptEmbeds {
    std::vector<float> sequence;
    std::vector<float> tts_pad_embed;
    int64_t prompt_len = 0;
    int64_t hidden_size = 0;
};

RemoteEmbedChain make_remote_embed_chain(NativeCodegen* runner, int64_t hidden_size) {
    RemoteEmbedChain chain;
    if (!runner || hidden_size <= 0 || !env_enabled("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED", false)) {
        return chain;
    }

    const ov::Shape frame_embed_shape{1, 1, static_cast<size_t>(hidden_size)};
    chain.prefill_output = try_create_remote_tensor(runner->prefill_model, ov::element::f32, frame_embed_shape);
    chain.decode_outputs[0] = try_create_remote_tensor(runner->decode_model, ov::element::f32, frame_embed_shape);
    chain.decode_outputs[1] = try_create_remote_tensor(runner->decode_model, ov::element::f32, frame_embed_shape);
    chain.enabled = static_cast<bool>(chain.prefill_output) &&
                    static_cast<bool>(chain.decode_outputs[0]) &&
                    static_cast<bool>(chain.decode_outputs[1]);
    return chain;
}

RemoteEmbedChain make_paged_remote_embed_chain(NativeCodegen* runner, int64_t hidden_size) {
    RemoteEmbedChain chain;
    const char* remote_embed_value = std::getenv("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED");
    const std::string remote_embed_env = lower_text(remote_embed_value ? remote_embed_value : "1");
    if (!runner || hidden_size <= 0 ||
        remote_embed_env == "0" || remote_embed_env == "false" ||
        remote_embed_env == "off" || remote_embed_env == "no") {
        return chain;
    }
    try {
        const auto input_rank = runner->prefill_model.input("inputs_embeds").get_partial_shape().rank();
        const auto output_rank = runner->prefill_model.output(2).get_partial_shape().rank();
        if (input_rank.is_dynamic() || output_rank.is_dynamic() ||
            input_rank.get_length() != 2 || output_rank.get_length() != 2) {
            return chain;
        }
    } catch (...) {
        return chain;
    }

    const ov::Shape frame_embed_shape{1, static_cast<size_t>(hidden_size)};
    chain.decode_outputs[0] = try_create_remote_tensor(runner->prefill_model, ov::element::f32, frame_embed_shape);
    chain.decode_outputs[1] = try_create_remote_tensor(runner->prefill_model, ov::element::f32, frame_embed_shape);
    chain.enabled = static_cast<bool>(chain.decode_outputs[0]) && static_cast<bool>(chain.decode_outputs[1]);
    return chain;
}

ov::Shape subcode_next_embed_output_shape(NativeCodegen* runner, int64_t hidden_size) {
    if (runner && runner->subcode_model.outputs().size() > 1) {
        try {
            const auto pshape = runner->subcode_model.output(1).get_partial_shape();
            if (!pshape.rank().is_dynamic()) {
                ov::Shape shape;
                shape.reserve(pshape.rank().get_length());
                for (const auto& dim : pshape) {
                    if (dim.is_static()) {
                        shape.push_back(static_cast<size_t>(dim.get_length()));
                    } else {
                        shape.push_back(1);
                    }
                }
                if (!shape.empty() && shape.back() == static_cast<size_t>(hidden_size)) {
                    return shape;
                }
            }
        } catch (...) {
        }
    }
    return ov::Shape{1, 1, static_cast<size_t>(hidden_size)};
}

ov::Shape paged_single_embed_input_shape(NativeCodegen* runner, int64_t hidden_size) {
    if (runner) {
        try {
            const int64_t rank = compiled_model_input_rank(runner->prefill_model, "inputs_embeds");
            if (rank == 2) {
                return ov::Shape{1, static_cast<size_t>(hidden_size)};
            }
        } catch (...) {
        }
    }
    return ov::Shape{1, 1, static_cast<size_t>(hidden_size)};
}

size_t shape_element_count(const ov::Shape& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }
    return count;
}

SplitSubcodeRemoteNextEmbedChain make_split_subcode_remote_next_embed_chain(
    NativeCodegen* runner,
    int64_t hidden_size) {
    SplitSubcodeRemoteNextEmbedChain chain;
    chain.requested = env_enabled("QWEN3_TTS_OV_NATIVE_SPLIT_SUBCODE_REMOTE_NEXT_EMBED", false);
    chain.required = env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_SPLIT_SUBCODE_REMOTE_NEXT_EMBED", false);
    if (!chain.requested || !runner || hidden_size <= 0) {
        return chain;
    }
    if (!runner->paged_split_subcode) {
        chain.failure = "paged split-subcode is disabled";
        return chain;
    }
    if (!runner->subcode_outputs_next_embed) {
        chain.failure = "subcode graph does not output next_embed";
        return chain;
    }
    const ov::Shape shape = subcode_next_embed_output_shape(runner, hidden_size);
    const ov::Shape input_shape = paged_single_embed_input_shape(runner, hidden_size);
    if (shape_element_count(shape) != shape_element_count(input_shape)) {
        std::ostringstream msg;
        msg << "subcode next_embed output shape " << shape
            << " has a different element count than paged inputs_embeds shape " << input_shape;
        chain.failure = msg.str();
        return chain;
    }
    chain.outputs[0] = try_create_remote_tensor(runner->subcode_model, ov::element::f32, input_shape);
    chain.outputs[1] = try_create_remote_tensor(runner->subcode_model, ov::element::f32, input_shape);
    chain.enabled = static_cast<bool>(chain.outputs[0]) && static_cast<bool>(chain.outputs[1]);
    if (!chain.enabled) {
        chain.failure = "failed to allocate split-subcode next_embed remote tensors";
    }
    return chain;
}

std::string build_assistant_text(const std::string& text) {
    return std::string("<|im_start|>assistant\n") + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string build_instruct_text(const std::string& instruct) {
    return std::string("<|im_start|>user\n") + instruct + "<|im_end|>\n";
}

std::vector<int64_t> tensor_to_i64_vector(const ov::Tensor& tensor) {
    const int64_t* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
}

std::vector<int64_t> tokenize_to_ids(ov::genai::Tokenizer& tokenizer, const std::string& text) {
    auto tokenized = tokenizer.encode(text);
    return tensor_to_i64_vector(tokenized.input_ids);
}

std::vector<float> embed_ids(
    NativeCodegen* runner,
    ov::InferRequest& request,
    const std::string& profile_label,
    const std::vector<int64_t>& ids,
    int64_t* hidden_size) {
    if (ids.empty()) {
        throw std::runtime_error("cannot embed an empty token id sequence");
    }
    request.set_tensor(
        "input_ids",
        ov::Tensor(ov::element::i64, ov::Shape{1, ids.size()}, const_cast<int64_t*>(ids.data())));
    request.infer();
    record_request_profile(runner, profile_label, request);
    auto output = request.get_output_tensor(0);
    const auto shape = output.get_shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != ids.size()) {
        throw std::runtime_error("unexpected embedding output shape");
    }
    const int64_t hidden = static_cast<int64_t>(shape[2]);
    if (hidden_size) {
        if (*hidden_size != 0 && *hidden_size != hidden) {
            throw std::runtime_error("embedding hidden sizes do not match");
        }
        *hidden_size = hidden;
    }
    const float* data = output.data<const float>();
    return std::vector<float>(data, data + output.get_size());
}

std::string embedding_cache_key(const std::string& label, const std::vector<int64_t>& ids) {
    std::string key = label;
    key.reserve(key.size() + ids.size() * 6);
    key.push_back(':');
    for (int64_t id : ids) {
        key += std::to_string(id);
        key.push_back(',');
    }
    return key;
}

std::vector<float> embed_ids_cached(
    NativeCodegen* runner,
    ov::InferRequest& request,
    const std::string& profile_label,
    const std::string& cache_label,
    const std::vector<int64_t>& ids,
    int64_t* hidden_size) {
    if (!runner || !env_enabled("QWEN3_TTS_OV_NATIVE_PROMPT_EMBED_CACHE", true)) {
        if (runner) {
            runner->prompt_embedding_cache_enabled = false;
        }
        return embed_ids(runner, request, profile_label, ids, hidden_size);
    }
    runner->prompt_embedding_cache_enabled = true;
    const std::string key = embedding_cache_key(cache_label, ids);
    auto it = runner->prompt_embedding_cache.find(key);
    if (it != runner->prompt_embedding_cache.end()) {
        const int64_t hidden = ids.empty() ? 0 : static_cast<int64_t>(it->second.size() / ids.size());
        if (hidden > 0 && static_cast<size_t>(hidden * ids.size()) == it->second.size()) {
            if (hidden_size) {
                if (*hidden_size != 0 && *hidden_size != hidden) {
                    throw std::runtime_error("cached embedding hidden sizes do not match");
                }
                *hidden_size = hidden;
            }
            runner->prompt_embedding_cache_hits += 1;
            return it->second;
        }
        runner->prompt_embedding_cache.erase(it);
    }
    runner->prompt_embedding_cache_misses += 1;
    auto embedded = embed_ids(runner, request, profile_label, ids, hidden_size);
    runner->prompt_embedding_cache.emplace(key, embedded);
    return embedded;
}

void append_embedding_slice(
    std::vector<float>& output,
    const std::vector<float>& embedding,
    int64_t start,
    int64_t count,
    int64_t hidden_size) {
    if (start < 0 || count < 0 || hidden_size <= 0) {
        throw std::runtime_error("invalid embedding slice");
    }
    const int64_t total_tokens = static_cast<int64_t>(embedding.size()) / hidden_size;
    if (start + count > total_tokens) {
        throw std::runtime_error("embedding slice out of range");
    }
    const float* begin = embedding.data() + static_cast<size_t>(start * hidden_size);
    const float* end = begin + static_cast<size_t>(count * hidden_size);
    output.insert(output.end(), begin, end);
}

void append_sum_token(
    std::vector<float>& output,
    const float* left,
    const float* right,
    int64_t hidden_size) {
    for (int64_t i = 0; i < hidden_size; ++i) {
        output.push_back(left[static_cast<size_t>(i)] + right[static_cast<size_t>(i)]);
    }
}

std::pair<bool, int64_t> stream_decoder_input_spec(const ov::CompiledModel& model, int64_t fallback_frames) {
    try {
        const auto pshape = model.input(0).get_partial_shape();
        if (pshape.rank().is_static() && pshape.size() >= 2 && pshape[1].is_static()) {
            return {true, static_cast<int64_t>(pshape[1].get_length())};
        }
    } catch (...) {
    }
    return {false, fallback_frames};
}

struct NativeDecodeTask {
    std::vector<int64_t> window;
    std::vector<int64_t> new_codes;
    int64_t new_frames = 0;
    int64_t num_code_groups = 0;
    int64_t chunk_index = 0;
    int64_t decode_input_frames = 0;
    int64_t decode_padded_frames = 0;
    bool is_final = false;
    double codegen_ms = 0.0;
    std::chrono::steady_clock::time_point queued_at;
};

struct NativeAudioStreamState {
    NativeCodegen* runner = nullptr;
    AudioCallback callback = nullptr;
    void* user_data = nullptr;
    std::vector<int64_t> all_codes;
    int64_t emitted_frames = 0;
    int64_t pending_frames = 0;
    int64_t chunk_index = 0;
    std::chrono::steady_clock::time_point codegen_started;
    bool async_decode = false;
    bool stop_decoder = false;
    std::deque<NativeDecodeTask> decode_queue;
    std::exception_ptr worker_error;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread decoder_thread;
};

int64_t frame_count(const NativeAudioStreamState& state) {
    return static_cast<int64_t>(state.all_codes.size() / static_cast<size_t>(state.runner->stream_num_code_groups));
}

void append_codes(NativeAudioStreamState& state, const int64_t* codes, int64_t num_frames, int64_t num_code_groups) {
    if (num_frames <= 0) {
        return;
    }
    if (num_code_groups != state.runner->stream_num_code_groups) {
        throw std::runtime_error("native audio stream received an unexpected code group count");
    }
    const size_t count = static_cast<size_t>(num_frames * num_code_groups);
    state.all_codes.insert(state.all_codes.end(), codes, codes + count);
    state.pending_frames += num_frames;
}

NativeDecodeTask make_decode_task(NativeAudioStreamState& state, bool is_final) {
    auto* runner = state.runner;
    const int64_t num_code_groups = runner->stream_num_code_groups;
    const int64_t total_frames = frame_count(state);
    const int64_t new_frames = total_frames - state.emitted_frames;
    const auto decode_submit_time = std::chrono::steady_clock::now();
    const double codegen_ms = ov::genai::PerfMetrics::get_microsec(decode_submit_time - state.codegen_started) / 1000.0;

    NativeDecodeTask task;
    task.num_code_groups = num_code_groups;
    task.chunk_index = state.chunk_index;
    task.is_final = is_final;
    task.codegen_ms = codegen_ms;
    task.queued_at = decode_submit_time;

    if (new_frames <= 0) {
        state.chunk_index += 1;
        state.codegen_started = std::chrono::steady_clock::now();
        return task;
    }

    const bool first_chunk = state.chunk_index == 0;
    const int64_t target_context = first_chunk ? runner->first_context_frames : runner->steady_context_frames;

    const int64_t context_start = std::max<int64_t>(0, state.emitted_frames - target_context);
    const int64_t context_frames = state.emitted_frames - context_start;
    if (target_context > context_frames && context_frames > 0) {
        const int64_t pad_count = target_context - context_frames;
        const int64_t* pad_frame = state.all_codes.data() + static_cast<size_t>(context_start * num_code_groups);
        task.window.reserve(static_cast<size_t>((pad_count + total_frames - context_start) * num_code_groups));
        for (int64_t i = 0; i < pad_count; ++i) {
            task.window.insert(task.window.end(), pad_frame, pad_frame + num_code_groups);
        }
    } else {
        task.window.reserve(static_cast<size_t>((total_frames - context_start) * num_code_groups));
    }
    const int64_t* window_begin = state.all_codes.data() + static_cast<size_t>(context_start * num_code_groups);
    const int64_t* window_end = state.all_codes.data() + static_cast<size_t>(total_frames * num_code_groups);
    task.window.insert(task.window.end(), window_begin, window_end);

    const bool static_decoder_input =
        first_chunk ? runner->first_stream_decoder_static_input : runner->steady_stream_decoder_static_input;
    const int64_t target_input_frames =
        first_chunk ? runner->first_stream_decoder_input_frames : runner->steady_stream_decoder_input_frames;
    int64_t current_input_frames = static_cast<int64_t>(task.window.size() / static_cast<size_t>(num_code_groups));
    if (static_decoder_input) {
        if (current_input_frames > target_input_frames) {
            throw std::runtime_error(
                "static streaming decoder input is too small for the requested chunk; export a larger chunk graph");
        }
        if (current_input_frames < target_input_frames) {
            const int64_t pad_count = target_input_frames - current_input_frames;
            std::vector<int64_t> pad_frame(static_cast<size_t>(num_code_groups), 0);
            if (!task.window.empty()) {
                const auto begin = task.window.begin() + static_cast<std::ptrdiff_t>((current_input_frames - 1) * num_code_groups);
                std::copy(begin, begin + static_cast<std::ptrdiff_t>(num_code_groups), pad_frame.begin());
            }
            for (int64_t i = 0; i < pad_count; ++i) {
                task.window.insert(task.window.end(), pad_frame.begin(), pad_frame.end());
            }
            task.decode_padded_frames = pad_count;
            current_input_frames = target_input_frames;
        }
    }
    task.decode_input_frames = current_input_frames;

    const int64_t* new_codes = state.all_codes.data() + static_cast<size_t>(state.emitted_frames * num_code_groups);
    task.new_codes.assign(new_codes, new_codes + static_cast<size_t>(new_frames * num_code_groups));
    task.new_frames = new_frames;
    state.emitted_frames = total_frames;
    state.pending_frames = 0;
    state.chunk_index += 1;
    state.codegen_started = std::chrono::steady_clock::now();
    return task;
}

void execute_decode_task(NativeAudioStreamState& state, const NativeDecodeTask& task) {
    auto* runner = state.runner;
    const int64_t num_code_groups = task.num_code_groups;
    runner->last_timing.decode_queue_wait_ms += std::max(0.0, elapsed_ms_since(task.queued_at));

    if (task.new_frames <= 0) {
        if (state.callback) {
            int callback_rc = 0;
            double callback_ms = 0.0;
            measure_ms(callback_ms, [&]() {
                callback_rc = state.callback(
                    nullptr, 0, nullptr, 0, num_code_groups, task.is_final ? 1 : 0, task.codegen_ms, 0.0, state.user_data);
            });
            runner->last_timing.decode_callback_ms += callback_ms;
            runner->last_timing.callback_ms += callback_ms;
            if (callback_rc != 0) {
                throw std::runtime_error("native audio callback requested stop");
            }
        }
        return;
    }

    runner->last_timing.decode_input_frames += task.decode_input_frames;
    runner->last_timing.decode_padded_frames += task.decode_padded_frames;

    ov::InferRequest& request = task.chunk_index == 0 ? runner->first_stream_decoder_request : runner->steady_stream_decoder_request;

    const auto decode_started = std::chrono::steady_clock::now();
    measure_ms(runner->last_timing.tensor_bind_ms, [&]() {
        request.set_tensor(
            "audio_codes",
            ov::Tensor(
                ov::element::i64,
                ov::Shape{
                    1,
                    task.window.size() / static_cast<size_t>(num_code_groups),
                    static_cast<size_t>(num_code_groups)},
                const_cast<int64_t*>(task.window.data())));
    });
    measure_ms(runner->last_timing.decode_infer_ms, [&]() {
        request.infer();
    });
    record_request_profile(
        runner,
        task.chunk_index == 0 ? std::string("stream_decoder_first") : std::string("stream_decoder_steady"),
        request);
    auto audio_tensor = request.get_output_tensor(0);
    const float* audio_data = audio_tensor.data<const float>();
    const int64_t requested_samples = task.new_frames * runner->decode_upsample_rate;
    const int64_t available_samples = static_cast<int64_t>(audio_tensor.get_size());
    const int64_t emit_samples = std::min<int64_t>(requested_samples, available_samples);
    const auto decode_stopped = std::chrono::steady_clock::now();
    const double decode_ms = ov::genai::PerfMetrics::get_microsec(decode_stopped - decode_started) / 1000.0;

    if (state.callback) {
        int callback_rc = 0;
        double callback_ms = 0.0;
        measure_ms(callback_ms, [&]() {
            callback_rc = state.callback(
                audio_data,
                emit_samples,
                task.new_codes.data(),
                task.new_frames,
                num_code_groups,
                task.is_final ? 1 : 0,
                task.codegen_ms,
                decode_ms,
                state.user_data);
        });
        runner->last_timing.decode_callback_ms += callback_ms;
        runner->last_timing.callback_ms += callback_ms;
        if (callback_rc != 0) {
            throw std::runtime_error("native audio callback requested stop");
        }
    }
}

void rethrow_decode_worker_error(NativeAudioStreamState& state) {
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        error = state.worker_error;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void decode_worker_loop(NativeAudioStreamState& state) {
    while (true) {
        NativeDecodeTask task;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&]() {
                return state.stop_decoder || !state.decode_queue.empty();
            });
            if (state.decode_queue.empty()) {
                if (state.stop_decoder) {
                    return;
                }
                continue;
            }
            task = std::move(state.decode_queue.front());
            state.decode_queue.pop_front();
        }
        try {
            execute_decode_task(state, task);
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.worker_error = std::current_exception();
            state.stop_decoder = true;
            state.decode_queue.clear();
            state.cv.notify_all();
            return;
        }
    }
}

void start_decode_worker(NativeAudioStreamState& state) {
    if (!state.async_decode) {
        return;
    }
    state.decoder_thread = std::thread([&state]() {
        decode_worker_loop(state);
    });
}

void finish_decode_worker(NativeAudioStreamState& state) {
    if (!state.async_decode) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stop_decoder = true;
    }
    state.cv.notify_all();
    if (state.decoder_thread.joinable()) {
        state.decoder_thread.join();
    }
    rethrow_decode_worker_error(state);
}

void cancel_decode_worker(NativeAudioStreamState& state) {
    if (!state.async_decode) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.stop_decoder = true;
        state.decode_queue.clear();
    }
    state.cv.notify_all();
    if (state.decoder_thread.joinable()) {
        state.decoder_thread.join();
    }
}

void decode_and_emit_audio(NativeAudioStreamState& state, bool is_final) {
    rethrow_decode_worker_error(state);
    NativeDecodeTask task = make_decode_task(state, is_final);
    if (!state.async_decode) {
        double inline_ms = 0.0;
        measure_ms(inline_ms, [&]() {
            execute_decode_task(state, task);
        });
        state.runner->last_timing.inline_decode_ms += inline_ms;
        return;
    }
    double enqueue_ms = 0.0;
    {
        const auto enqueue_started = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.worker_error) {
            std::rethrow_exception(state.worker_error);
        }
        state.decode_queue.push_back(std::move(task));
        state.runner->last_timing.decode_queue_depth_max = std::max<int64_t>(
            state.runner->last_timing.decode_queue_depth_max,
            static_cast<int64_t>(state.decode_queue.size()));
        enqueue_ms = elapsed_ms_since(enqueue_started);
    }
    state.runner->last_timing.callback_enqueue_ms += enqueue_ms;
    state.cv.notify_one();
}

int native_audio_frame_callback(const int64_t* codes, int64_t num_frames, int64_t num_code_groups, void* user_data) {
    auto* state = static_cast<NativeAudioStreamState*>(user_data);
    rethrow_decode_worker_error(*state);
    append_codes(*state, codes, num_frames, num_code_groups);
    const int64_t target_frames = state->chunk_index == 0 ? state->runner->first_chunk_frames : state->runner->steady_chunk_frames;
    if (state->pending_frames >= target_frames) {
        decode_and_emit_audio(*state, false);
    }
    rethrow_decode_worker_error(*state);
    return 0;
}

void set_i32_tensor(ov::InferRequest& request, const std::string& name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void set_scalar_i32_tensor(ov::InferRequest& request, const std::string& name, int32_t value) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{});
    tensor.data<int32_t>()[0] = value;
    request.set_tensor(name, tensor);
}

std::vector<NamedTensor> make_paged_kv_cache_tensors(ov::CompiledModel& model, int64_t num_blocks) {
    bool all_gpu_device = false;
    ov::RemoteContext remote_context;
    try {
        std::vector<std::string> execution_devices = model.get_property(ov::execution_devices);
        all_gpu_device = !execution_devices.empty() &&
            std::all_of(execution_devices.begin(), execution_devices.end(), [](const std::string& device) {
                return device.find("GPU") != std::string::npos;
            });
        if (all_gpu_device) {
            remote_context = model.get_context();
        }
    } catch (...) {
        all_gpu_device = false;
    }
    std::vector<NamedTensor> tensors;
    for (const auto& input : model.inputs()) {
        std::string name;
        try {
            name = input.get_any_name();
        } catch (...) {
            continue;
        }
        if (name.rfind("key_cache.", 0) != 0 && name.rfind("value_cache.", 0) != 0) {
            continue;
        }
        const auto shape = concrete_shape(input.get_partial_shape(), num_blocks);
        ov::Tensor tensor = all_gpu_device
            ? remote_context.create_tensor(input.get_element_type(), shape)
            : ov::Tensor(input.get_element_type(), shape);
        if (!all_gpu_device) {
            std::memset(tensor.data(), 0, tensor.get_byte_size());
        }
        tensors.push_back(NamedTensor{name, tensor});
    }
    return tensors;
}

const std::vector<NamedTensor>& get_paged_kv_cache_tensors(NativeCodegen* runner, ov::CompiledModel& model, int64_t num_blocks) {
    if (!runner) {
        throw std::runtime_error("paged KV cache tensor reuse requires a runner");
    }
    const bool reuse_enabled = env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_TENSOR_REUSE", true);
    const bool can_reuse =
        reuse_enabled &&
        !runner->paged_kv_cache_tensors.empty() &&
        runner->paged_kv_cache_tensor_blocks == num_blocks;
    runner->last_timing.kv_cache_tensor_reuse = can_reuse;
    if (!reuse_enabled ||
        runner->paged_kv_cache_tensors.empty() ||
        runner->paged_kv_cache_tensor_blocks != num_blocks) {
        runner->paged_kv_cache_tensors = make_paged_kv_cache_tensors(model, num_blocks);
        runner->paged_kv_cache_tensor_blocks = num_blocks;
    }
    return runner->paged_kv_cache_tensors;
}

void bind_named_tensors(ov::InferRequest& request, const std::vector<NamedTensor>& tensors) {
    for (const auto& item : tensors) {
        request.set_tensor(item.name, item.tensor);
    }
}

void bind_paged_kv_cache_tensors(ov::CompiledModel& model, ov::InferRequest& request, int64_t num_blocks) {
    bind_named_tensors(request, make_paged_kv_cache_tensors(model, num_blocks));
}

void bind_paged_step_inputs(
    ov::InferRequest& request,
    const float* embeds,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t position_start,
    int64_t past_len,
    int64_t block_size,
    const float* tts_pad_embed,
    const std::vector<float>& allow_eos_values,
    std::vector<int64_t>& position_ids,
    std::vector<int32_t>& block_indices,
    std::vector<float>& allow_eos_buffer,
    std::vector<int64_t>& beam_idx_buffer,
    bool bind_inputs_embeds = true) {
    if (seq_len <= 0 || hidden_size <= 0 || block_size <= 0) {
        throw std::runtime_error("invalid paged KV step shape");
    }
    position_ids.resize(static_cast<size_t>(3 * seq_len));
    for (int64_t row = 0; row < 3; ++row) {
        for (int64_t i = 0; i < seq_len; ++i) {
            position_ids[static_cast<size_t>(row * seq_len + i)] = position_start + i;
        }
    }

    const int64_t total_len = past_len + seq_len;
    const int64_t blocks_used = std::max<int64_t>(1, (total_len + block_size - 1) / block_size);
    const int64_t block_indices_len = std::max<int64_t>(
        blocks_used,
        compiled_model_static_input_size(request.get_compiled_model(), "block_indices"));
    block_indices.assign(static_cast<size_t>(block_indices_len), 0);
    for (int64_t i = 0; i < blocks_used; ++i) {
        block_indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    const auto& compiled = request.get_compiled_model();
    const int64_t inputs_rank = compiled_model_input_rank(compiled, "inputs_embeds");
    if (bind_inputs_embeds) {
        if (!embeds) {
            throw std::runtime_error("inputs_embeds host buffer is null");
        }
        if (inputs_rank == 3) {
            request.set_tensor(
                "inputs_embeds",
                ov::Tensor(
                    ov::element::f32,
                    ov::Shape{static_cast<size_t>(seq_len), 1, static_cast<size_t>(hidden_size)},
                    const_cast<float*>(embeds)));
        } else {
            request.set_tensor(
                "inputs_embeds",
                ov::Tensor(
                    ov::element::f32,
                    ov::Shape{static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)},
                    const_cast<float*>(embeds)));
        }
    }
    const int64_t position_rank = compiled_model_input_rank(compiled, "position_ids");
    if (position_rank == 3) {
        request.set_tensor(
            "position_ids",
            ov::Tensor(ov::element::i64, ov::Shape{3, static_cast<size_t>(seq_len), 1}, position_ids.data()));
    } else {
        request.set_tensor(
            "position_ids",
            ov::Tensor(ov::element::i64, ov::Shape{3, static_cast<size_t>(seq_len)}, position_ids.data()));
    }
    if (compiled_model_has_input(compiled, "tts_pad_embed")) {
        request.set_tensor(
            "tts_pad_embed",
            ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, const_cast<float*>(tts_pad_embed)));
    }
    allow_eos_buffer = allow_eos_values;
    if (allow_eos_buffer.empty()) {
        allow_eos_buffer.assign(1, 0.0f);
    }
    if (compiled_model_has_input(request.get_compiled_model(), "allow_eos")) {
        request.set_tensor("allow_eos", ov::Tensor(ov::element::f32, ov::Shape{1}, allow_eos_buffer.data()));
    }
    if (compiled_model_has_input(request.get_compiled_model(), "allow_eos_steps")) {
        request.set_tensor(
            "allow_eos_steps",
            ov::Tensor(ov::element::f32, ov::Shape{allow_eos_buffer.size()}, allow_eos_buffer.data()));
    }
    if (compiled_model_has_input(request.get_compiled_model(), "beam_idx")) {
        beam_idx_buffer.assign(static_cast<size_t>(seq_len), 0);
        request.set_tensor("beam_idx", ov::Tensor(ov::element::i64, ov::Shape{beam_idx_buffer.size()}, beam_idx_buffer.data()));
    }
    if (compiled_model_has_input(request.get_compiled_model(), "score_aggregation_window")) {
        set_i32_tensor(request, "score_aggregation_window", {1});
    }
    set_i32_tensor(request, "past_lens", {static_cast<int32_t>(past_len)});
    set_i32_tensor(request, "subsequence_begins", {0, static_cast<int32_t>(seq_len)});
    request.set_tensor("block_indices", ov::Tensor(ov::element::i32, ov::Shape{block_indices.size()}, block_indices.data()));
    set_i32_tensor(request, "block_indices_begins", {0, static_cast<int32_t>(blocks_used)});
    set_scalar_i32_tensor(request, "max_context_len", static_cast<int32_t>(total_len));
}

struct PagedBatchSequenceState {
    int64_t prompt_len = 0;
    int64_t generated = 0;
    bool stop = false;
    int64_t block_offset = 0;
    int64_t block_capacity = 0;
    std::vector<float> next_embed;
    double ttft_ms = -1.0;
    double last_token_ms = 0.0;
};

void bind_paged_batch_step_inputs(
    ov::InferRequest& request,
    const float* embeds,
    const std::vector<int64_t>& seq_lens,
    int64_t hidden_size,
    const std::vector<int64_t>& position_starts,
    const std::vector<int64_t>& past_lens,
    int64_t block_size,
    const float* tts_pad_embed,
    const std::vector<float>& allow_eos_values,
    const std::vector<int64_t>& block_offsets,
    const std::vector<int64_t>& block_counts,
    std::vector<int64_t>& position_ids,
    std::vector<int32_t>& past_lens_buffer,
    std::vector<int32_t>& subsequence_begins,
    std::vector<int32_t>& block_indices,
    std::vector<int32_t>& block_indices_begins,
    std::vector<float>& allow_eos_buffer,
    std::vector<int64_t>& beam_idx_buffer,
    std::vector<int32_t>& score_aggregation_window_buffer) {
    const size_t batch = seq_lens.size();
    if (
        batch == 0 ||
        !embeds ||
        !tts_pad_embed ||
        position_starts.size() != batch ||
        past_lens.size() != batch ||
        block_offsets.size() != batch ||
        block_counts.size() != batch ||
        hidden_size <= 0 ||
        block_size <= 0) {
        throw std::runtime_error("invalid paged KV batch step arguments");
    }
    int64_t total_tokens = 0;
    int64_t max_total_len = 0;
    for (size_t i = 0; i < batch; ++i) {
        if (seq_lens[i] <= 0 || block_counts[i] <= 0) {
            throw std::runtime_error("invalid paged KV batch sequence shape");
        }
        total_tokens += seq_lens[i];
        max_total_len = std::max<int64_t>(max_total_len, past_lens[i] + seq_lens[i]);
    }
    position_ids.resize(static_cast<size_t>(3 * total_tokens));
    subsequence_begins.resize(batch + 1);
    past_lens_buffer.resize(batch);
    block_indices.clear();
    block_indices_begins.resize(batch + 1);
    beam_idx_buffer.resize(static_cast<size_t>(total_tokens));

    int64_t token_offset = 0;
    int64_t block_offset = 0;
    subsequence_begins[0] = 0;
    block_indices_begins[0] = 0;
    for (size_t seq = 0; seq < batch; ++seq) {
        past_lens_buffer[seq] = static_cast<int32_t>(past_lens[seq]);
        for (int64_t i = 0; i < seq_lens[seq]; ++i) {
            beam_idx_buffer[static_cast<size_t>(token_offset + i)] = static_cast<int64_t>(seq);
            for (int64_t row = 0; row < 3; ++row) {
                position_ids[static_cast<size_t>(row * total_tokens + token_offset + i)] = position_starts[seq] + i;
            }
        }
        token_offset += seq_lens[seq];
        subsequence_begins[seq + 1] = static_cast<int32_t>(token_offset);
        for (int64_t block = 0; block < block_counts[seq]; ++block) {
            block_indices.push_back(static_cast<int32_t>(block_offsets[seq] + block));
        }
        block_offset += block_counts[seq];
        block_indices_begins[seq + 1] = static_cast<int32_t>(block_offset);
    }

    const auto& compiled = request.get_compiled_model();
    const int64_t inputs_rank = compiled_model_input_rank(compiled, "inputs_embeds");
    if (inputs_rank == 3) {
        request.set_tensor(
            "inputs_embeds",
            ov::Tensor(
                ov::element::f32,
                ov::Shape{static_cast<size_t>(total_tokens), 1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(embeds)));
    } else {
        request.set_tensor(
            "inputs_embeds",
            ov::Tensor(
                ov::element::f32,
                ov::Shape{static_cast<size_t>(total_tokens), static_cast<size_t>(hidden_size)},
                const_cast<float*>(embeds)));
    }
    const int64_t position_rank = compiled_model_input_rank(compiled, "position_ids");
    if (position_rank == 3) {
        request.set_tensor(
            "position_ids",
            ov::Tensor(ov::element::i64, ov::Shape{3, static_cast<size_t>(total_tokens), 1}, position_ids.data()));
    } else {
        request.set_tensor(
            "position_ids",
            ov::Tensor(ov::element::i64, ov::Shape{3, static_cast<size_t>(total_tokens)}, position_ids.data()));
    }
    if (compiled_model_has_input(compiled, "tts_pad_embed")) {
        request.set_tensor(
            "tts_pad_embed",
            ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, const_cast<float*>(tts_pad_embed)));
    }
    allow_eos_buffer = allow_eos_values;
    if (allow_eos_buffer.size() != batch) {
        allow_eos_buffer.assign(batch, 0.0f);
    }
    if (compiled_model_has_input(compiled, "allow_eos")) {
        request.set_tensor("allow_eos", ov::Tensor(ov::element::f32, ov::Shape{allow_eos_buffer.size()}, allow_eos_buffer.data()));
    }
    if (compiled_model_has_input(compiled, "allow_eos_steps")) {
        request.set_tensor(
            "allow_eos_steps",
            ov::Tensor(ov::element::f32, ov::Shape{allow_eos_buffer.size()}, allow_eos_buffer.data()));
    }
    if (compiled_model_has_input(compiled, "beam_idx")) {
        request.set_tensor("beam_idx", ov::Tensor(ov::element::i64, ov::Shape{beam_idx_buffer.size()}, beam_idx_buffer.data()));
    }
    if (compiled_model_has_input(compiled, "score_aggregation_window")) {
        score_aggregation_window_buffer.assign(batch, 1);
        request.set_tensor(
            "score_aggregation_window",
            ov::Tensor(
                ov::element::i32,
                ov::Shape{score_aggregation_window_buffer.size()},
                score_aggregation_window_buffer.data()));
    }
    request.set_tensor("past_lens", ov::Tensor(ov::element::i32, ov::Shape{past_lens_buffer.size()}, past_lens_buffer.data()));
    request.set_tensor(
        "subsequence_begins",
        ov::Tensor(ov::element::i32, ov::Shape{subsequence_begins.size()}, subsequence_begins.data()));
    request.set_tensor("block_indices", ov::Tensor(ov::element::i32, ov::Shape{block_indices.size()}, block_indices.data()));
    request.set_tensor(
        "block_indices_begins",
        ov::Tensor(ov::element::i32, ov::Shape{block_indices_begins.size()}, block_indices_begins.data()));
    set_scalar_i32_tensor(request, "max_context_len", static_cast<int32_t>(max_total_len));
}

int64_t select_first_code_from_logits_row(
    const float* logits,
    int64_t vocab_size,
    int64_t generated,
    int64_t min_new_tokens,
    int64_t eos_token_id,
    const std::vector<uint8_t>* repeated_mask = nullptr,
    float repetition_penalty = 1.0f,
    const NativeSamplingConfig* sampling = nullptr,
    std::mt19937_64* rng = nullptr) {
    if (!logits || vocab_size <= 0) {
        throw std::runtime_error("invalid logits row for batched first-code selection");
    }
    const int64_t suppress_from = std::max<int64_t>(0, vocab_size - 1024);
    std::vector<std::pair<int64_t, float>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    for (int64_t token_id = 0; token_id < vocab_size; ++token_id) {
        float score = logits[static_cast<size_t>(token_id)];
        if (token_id >= suppress_from && token_id != eos_token_id) {
            score = NEG_INF;
        }
        if (token_id == eos_token_id && generated < min_new_tokens) {
            score = NEG_INF;
        }
        if (
            repeated_mask &&
            repetition_penalty != 1.0f &&
            token_id >= 0 &&
            static_cast<size_t>(token_id) < repeated_mask->size() &&
            (*repeated_mask)[static_cast<size_t>(token_id)] != 0 &&
            score > NEG_INF / 2.0f) {
            score = score < 0.0f ? score * repetition_penalty : score / repetition_penalty;
        }
        if (score > NEG_INF / 2.0f && std::isfinite(score)) {
            candidates.emplace_back(token_id, score);
        }
    }
    if (candidates.empty()) {
        return eos_token_id;
    }

    const bool do_sample = sampling && sampling->do_sample;
    if (!do_sample) {
        return std::max_element(
                   candidates.begin(),
                   candidates.end(),
                   [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })
            ->first;
    }
    if (!rng) {
        throw std::runtime_error("sampling requested without RNG");
    }

    const float temperature = std::max<float>(1.0e-6f, sampling->temperature);
    for (auto& item : candidates) {
        item.second /= temperature;
    }

    const int64_t top_k = sampling->top_k;
    if (top_k > 0 && static_cast<size_t>(top_k) < candidates.size()) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
        candidates.resize(static_cast<size_t>(top_k));
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    const double max_score = static_cast<double>(candidates.front().second);
    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (const auto& item : candidates) {
        const double weight = std::exp(static_cast<double>(item.second) - max_score);
        weights.push_back(weight);
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return candidates.front().first;
    }

    const float top_p = sampling->top_p;
    if (top_p > 0.0f && top_p < 1.0f && candidates.size() > 1) {
        double kept = 0.0;
        size_t keep_count = 0;
        for (; keep_count < weights.size(); ++keep_count) {
            kept += weights[keep_count];
            if (kept / total >= static_cast<double>(top_p)) {
                ++keep_count;
                break;
            }
        }
        keep_count = std::max<size_t>(1, std::min(keep_count, candidates.size()));
        candidates.resize(keep_count);
        weights.resize(keep_count);
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(*rng)].first;
}

int64_t select_first_code_greedy_from_logits_row(
    const float* logits,
    int64_t vocab_size,
    int64_t generated,
    int64_t min_new_tokens,
    int64_t eos_token_id) {
    return select_first_code_from_logits_row(
        logits,
        vocab_size,
        generated,
        min_new_tokens,
        eos_token_id,
        nullptr,
        1.0f,
        nullptr,
        nullptr);
}

void run_paged_split_subcode_batch(
    NativeCodegen* runner,
    const ov::Tensor& last_hidden_tensor,
    const std::vector<int64_t>& first_codes,
    const float* tts_pad_embed,
    int64_t hidden_size,
    int64_t num_code_groups,
    std::vector<int64_t>& frame_codes,
    std::vector<float>& next_embeds) {
    if (!runner || !tts_pad_embed || hidden_size <= 0 || num_code_groups <= 0 || first_codes.empty()) {
        throw std::runtime_error("invalid paged split subcode batch arguments");
    }
    const size_t batch = first_codes.size();
    const bool use_batch_subcode = batch > 1 && runner->batch_subcode_enabled;
    if (batch > 1 && !use_batch_subcode) {
        const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, "paged split subcode fallback hidden");
        if (!hidden_view.data || hidden_view.size < batch * static_cast<size_t>(hidden_size)) {
            throw std::runtime_error("paged split subcode fallback hidden tensor is too small");
        }
        frame_codes.clear();
        next_embeds.clear();
        frame_codes.reserve(batch * static_cast<size_t>(num_code_groups));
        next_embeds.reserve(batch * static_cast<size_t>(hidden_size));
        for (size_t row = 0; row < batch; ++row) {
            ov::Tensor row_hidden_tensor(
                ov::element::f32,
                ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(hidden_view.data + row * static_cast<size_t>(hidden_size)));
            std::vector<int64_t> row_first_code{first_codes[row]};
            std::vector<int64_t> row_codes;
            std::vector<float> row_next_embed;
            run_paged_split_subcode_batch(
                runner,
                row_hidden_tensor,
                row_first_code,
                tts_pad_embed,
                hidden_size,
                num_code_groups,
                row_codes,
                row_next_embed);
            frame_codes.insert(frame_codes.end(), row_codes.begin(), row_codes.end());
            next_embeds.insert(next_embeds.end(), row_next_embed.begin(), row_next_embed.end());
        }
        return;
    }
    auto& request = use_batch_subcode ? runner->batch_subcode_request : runner->subcode_request;
    auto& compiled_model = use_batch_subcode ? runner->batch_subcode_model : runner->subcode_model;
    runner->last_timing.batch_subcode_enabled = runner->last_timing.batch_subcode_enabled || use_batch_subcode;
    const uint64_t host_copy_bytes_before = runner->last_timing.host_copy_bytes;
    const int64_t host_copy_fallbacks_before = runner->last_timing.host_copy_fallback_count;
    double bind_ms = 0.0;
    measure_ms(bind_ms, [&]() {
        request.set_tensor("past_hidden", last_hidden_tensor);
        request.set_tensor(
            "first_code",
            ov::Tensor(ov::element::i64, ov::Shape{batch, 1}, const_cast<int64_t*>(first_codes.data())));
        if (compiled_model_has_input(compiled_model, "tts_pad_embed")) {
            request.set_tensor(
                "tts_pad_embed",
                ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, const_cast<float*>(tts_pad_embed)));
        }
    });
    runner->last_timing.subcode_bind_ms += bind_ms;
    runner->last_timing.subcode_bind_step_ms.push_back(bind_ms);
    double subcode_infer_ms = 0.0;
    measure_ms(subcode_infer_ms, [&]() {
        request.infer();
    });
    runner->last_timing.codegen_infer_ms += subcode_infer_ms;
    runner->last_timing.codegen_subcode_infer_ms += subcode_infer_ms;
    runner->last_timing.codegen_subcode_count += 1;
    runner->last_timing.subcode_infer_step_ms.push_back(subcode_infer_ms);
    record_request_profile(
        runner,
        use_batch_subcode ? "codegen_paged_kv_subcode_batch" : "codegen_paged_kv_subcode_single",
        request);

    double output_read_ms = 0.0;
    HostTensorView<int64_t> codes;
    FloatTensorView embed;
    measure_ms(output_read_ms, [&]() {
        auto codes_tensor = request.get_output_tensor(0);
        codes = tensor_to_host_view<int64_t>(runner, codes_tensor, "paged split subcode batch codes");
        auto embed_tensor = request.get_output_tensor(1);
        embed = tensor_to_host_f32_view(runner, embed_tensor, "paged split subcode batch embed");
    });
    runner->last_timing.subcode_output_read_ms += output_read_ms;
    runner->last_timing.subcode_output_read_step_ms.push_back(output_read_ms);
    runner->last_timing.subcode_host_copy_bytes += runner->last_timing.host_copy_bytes - host_copy_bytes_before;
    runner->last_timing.subcode_host_copy_fallback_count +=
        runner->last_timing.host_copy_fallback_count - host_copy_fallbacks_before;
    if (!codes.data || codes.size < batch * static_cast<size_t>(num_code_groups)) {
        throw std::runtime_error("paged split subcode batch returned too few codec groups");
    }
    if (!embed.data || embed.size < batch * static_cast<size_t>(hidden_size)) {
        throw std::runtime_error("paged split subcode batch returned too small next_embed");
    }

    double next_embed_ms = 0.0;
    measure_ms(next_embed_ms, [&]() {
        frame_codes.assign(codes.data, codes.data + batch * static_cast<size_t>(num_code_groups));
        next_embeds.resize(batch * static_cast<size_t>(hidden_size));
        for (size_t seq = 0; seq < batch; ++seq) {
            const float* src = embed.data + seq * static_cast<size_t>(hidden_size);
            float* dst = next_embeds.data() + seq * static_cast<size_t>(hidden_size);
            if (runner->subcode_outputs_next_embed) {
                std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
            } else {
                for (int64_t i = 0; i < hidden_size; ++i) {
                    dst[static_cast<size_t>(i)] =
                        src[static_cast<size_t>(i)] + tts_pad_embed[static_cast<size_t>(i)];
                }
            }
        }
    });
    runner->last_timing.subcode_next_embed_ms += next_embed_ms;
    runner->last_timing.subcode_next_embed_step_ms.push_back(next_embed_ms);
}

void run_paged_split_subcode_rows(
    NativeCodegen* runner,
    const ov::Tensor& last_hidden_tensor,
    const std::vector<int64_t>& first_codes,
    const float* tts_pad_embed,
    int64_t hidden_size,
    int64_t num_code_groups,
    std::vector<int64_t>& frame_codes,
    std::vector<float>& next_embeds,
    const char* hidden_label = "paged split subcode row fallback hidden") {
    if (!runner || first_codes.empty() || hidden_size <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid paged split subcode row arguments");
    }
    const size_t batch = first_codes.size();
    const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, hidden_label);
    if (!hidden_view.data || hidden_view.size < batch * static_cast<size_t>(hidden_size)) {
        throw std::runtime_error("paged split subcode row hidden tensor is too small");
    }
    const bool parallel_rows =
        batch > 1 &&
        env_enabled("QWEN3_TTS_OV_NATIVE_SAMPLED_SUBCODE_PARALLEL_ROWS", false) &&
        static_cast<bool>(runner->subcode_model);
    if (parallel_rows) {
        runner->last_timing.sampled_subcode_parallel_rows = true;
        runner->last_timing.sampled_subcode_parallel_row_count += static_cast<int64_t>(batch);
        while (runner->sampled_subcode_row_requests.size() < batch) {
            runner->sampled_subcode_row_requests.push_back(runner->subcode_model.create_infer_request());
        }
        std::vector<int64_t> first_code_storage(batch, 0);
        std::vector<ov::Tensor> hidden_tensors;
        std::vector<ov::Tensor> first_code_tensors;
        hidden_tensors.reserve(batch);
        first_code_tensors.reserve(batch);
        double bind_ms = 0.0;
        measure_ms(bind_ms, [&]() {
            for (size_t row = 0; row < batch; ++row) {
                first_code_storage[row] = first_codes[row];
                hidden_tensors.emplace_back(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    const_cast<float*>(hidden_view.data + row * static_cast<size_t>(hidden_size)));
                first_code_tensors.emplace_back(
                    ov::element::i64,
                    ov::Shape{1, 1},
                    first_code_storage.data() + row);
                auto& request = runner->sampled_subcode_row_requests[row];
                request.set_tensor("past_hidden", hidden_tensors.back());
                request.set_tensor("first_code", first_code_tensors.back());
                if (compiled_model_has_input(runner->subcode_model, "tts_pad_embed")) {
                    request.set_tensor(
                        "tts_pad_embed",
                        ov::Tensor(
                            ov::element::f32,
                            ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                            const_cast<float*>(tts_pad_embed)));
                }
            }
        });
        runner->last_timing.subcode_bind_ms += bind_ms;
        runner->last_timing.subcode_bind_step_ms.push_back(bind_ms);
        double infer_ms = 0.0;
        measure_ms(infer_ms, [&]() {
            for (size_t row = 0; row < batch; ++row) {
                runner->sampled_subcode_row_requests[row].start_async();
            }
            for (size_t row = 0; row < batch; ++row) {
                runner->sampled_subcode_row_requests[row].wait();
            }
        });
        runner->last_timing.codegen_infer_ms += infer_ms;
        runner->last_timing.codegen_subcode_infer_ms += infer_ms;
        runner->last_timing.codegen_subcode_count += 1;
        runner->last_timing.subcode_infer_step_ms.push_back(infer_ms);
        frame_codes.clear();
        next_embeds.resize(batch * static_cast<size_t>(hidden_size));
        frame_codes.reserve(batch * static_cast<size_t>(num_code_groups));
        double output_read_ms = 0.0;
        measure_ms(output_read_ms, [&]() {
            for (size_t row = 0; row < batch; ++row) {
                auto& request = runner->sampled_subcode_row_requests[row];
                record_request_profile(runner, "codegen_paged_kv_subcode_parallel_row", request);
                auto codes_tensor = request.get_output_tensor(0);
                const auto codes = tensor_to_host_view<int64_t>(runner, codes_tensor, "paged split subcode parallel row codes");
                auto embed_tensor = request.get_output_tensor(1);
                const auto embed = tensor_to_host_f32_view(runner, embed_tensor, "paged split subcode parallel row embed");
                if (!codes.data || codes.size < static_cast<size_t>(num_code_groups) ||
                    !embed.data || embed.size < static_cast<size_t>(hidden_size)) {
                    throw std::runtime_error("paged split subcode parallel row output is too small");
                }
                frame_codes.insert(frame_codes.end(), codes.data, codes.data + static_cast<size_t>(num_code_groups));
                float* dst = next_embeds.data() + row * static_cast<size_t>(hidden_size);
                if (runner->subcode_outputs_next_embed) {
                    std::memcpy(dst, embed.data, static_cast<size_t>(hidden_size) * sizeof(float));
                } else {
                    for (int64_t i = 0; i < hidden_size; ++i) {
                        dst[static_cast<size_t>(i)] =
                            embed.data[static_cast<size_t>(i)] + tts_pad_embed[static_cast<size_t>(i)];
                    }
                }
            }
        });
        runner->last_timing.subcode_output_read_ms += output_read_ms;
        runner->last_timing.subcode_output_read_step_ms.push_back(output_read_ms);
        return;
    }
    frame_codes.clear();
    next_embeds.clear();
    frame_codes.reserve(batch * static_cast<size_t>(num_code_groups));
    next_embeds.reserve(batch * static_cast<size_t>(hidden_size));
    for (size_t row = 0; row < batch; ++row) {
        ov::Tensor row_hidden_tensor(
            ov::element::f32,
            ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
            const_cast<float*>(hidden_view.data + row * static_cast<size_t>(hidden_size)));
        std::vector<int64_t> row_first_code{first_codes[row]};
        std::vector<int64_t> row_codes;
        std::vector<float> row_next_embed;
        run_paged_split_subcode_batch(
            runner,
            row_hidden_tensor,
            row_first_code,
            tts_pad_embed,
            hidden_size,
            num_code_groups,
            row_codes,
            row_next_embed);
        frame_codes.insert(frame_codes.end(), row_codes.begin(), row_codes.end());
        next_embeds.insert(next_embeds.end(), row_next_embed.begin(), row_next_embed.end());
    }
}

double max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double max_diff = 0.0;
    for (size_t index = 0; index < count; ++index) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(lhs[index] - rhs[index])));
    }
    if (lhs.size() != rhs.size()) {
        return std::numeric_limits<double>::infinity();
    }
    return max_diff;
}

void run_online_sampled_subcode(
    NativeCodegen* runner,
    const ov::Tensor& hidden_tensor,
    const std::vector<int64_t>& first_codes,
    const float* tts_pad_embed,
    int64_t hidden_size,
    int64_t num_code_groups,
    std::vector<int64_t>& frame_codes,
    std::vector<float>& next_embeds) {
    const std::string policy = sampled_batch_subcode_policy_from_env();
    runner->last_timing.sampled_batch_subcode_policy = policy;
    const size_t batch = first_codes.size();
    const bool can_batch = batch > 1 && runner->batch_subcode_enabled;
    if (policy == "off" || !can_batch) {
        if (!can_batch && policy != "off") {
            runner->last_timing.sampled_batch_subcode_fallback_count += 1;
            runner->last_timing.sampled_batch_subcode_fallback_reason = "batch_subcode_unavailable";
        }
        run_paged_split_subcode_rows(
            runner,
            hidden_tensor,
            first_codes,
            tts_pad_embed,
            hidden_size,
            num_code_groups,
            frame_codes,
            next_embeds,
            "online sampled row subcode hidden");
        return;
    }

    if (policy == "on") {
        run_paged_split_subcode_batch(
            runner,
            hidden_tensor,
            first_codes,
            tts_pad_embed,
            hidden_size,
            num_code_groups,
            frame_codes,
            next_embeds);
        runner->last_timing.sampled_batch_subcode_used = true;
        return;
    }

    std::vector<int64_t> row_codes;
    std::vector<float> row_embeds;
    run_paged_split_subcode_rows(
        runner,
        hidden_tensor,
        first_codes,
        tts_pad_embed,
        hidden_size,
        num_code_groups,
        row_codes,
        row_embeds,
        "online sampled verify row hidden");

    std::vector<int64_t> batch_codes;
    std::vector<float> batch_embeds;
    try {
        run_paged_split_subcode_batch(
            runner,
            hidden_tensor,
            first_codes,
            tts_pad_embed,
            hidden_size,
            num_code_groups,
            batch_codes,
            batch_embeds);
        const double embed_diff = max_abs_diff(row_embeds, batch_embeds);
        runner->last_timing.sampled_batch_subcode_max_abs_diff =
            std::max(runner->last_timing.sampled_batch_subcode_max_abs_diff, embed_diff);
        const double tolerance =
            static_cast<double>(env_int64("QWEN3_TTS_OV_NATIVE_SAMPLED_BATCH_SUBCODE_EMBED_TOLERANCE_MICRO", 1000)) /
            1000000.0;
        const bool codes_match = batch_codes == row_codes;
        const bool embeds_match = embed_diff <= tolerance;
        if (codes_match && embeds_match) {
            frame_codes.swap(batch_codes);
            next_embeds.swap(batch_embeds);
            runner->last_timing.sampled_batch_subcode_used = true;
            runner->last_timing.sampled_batch_subcode_verified = true;
            return;
        }
        runner->last_timing.sampled_batch_subcode_mismatch_count += 1;
        if (!codes_match) {
            runner->last_timing.sampled_batch_subcode_code_mismatch_count += 1;
        }
        if (!embeds_match) {
            runner->last_timing.sampled_batch_subcode_embed_mismatch_count += 1;
        }
        runner->last_timing.sampled_batch_subcode_fallback_count += 1;
        runner->last_timing.sampled_batch_subcode_fallback_reason = "verify_mismatch";
    } catch (const std::exception& exc) {
        runner->last_timing.sampled_batch_subcode_fallback_count += 1;
        runner->last_timing.sampled_batch_subcode_fallback_reason = std::string("verify_error:") + exc.what();
    }
    frame_codes.swap(row_codes);
    next_embeds.swap(row_embeds);
}

bool prepare_decode_step_prebind(
    PagedDecodeStepBindings& binding,
    ov::InferRequest& request,
    const float* embeds,
    int64_t hidden_size,
    int64_t block_size,
    int64_t block_indices_capacity,
    const float* tts_pad_embed,
    size_t allow_eos_size,
    std::string* error) {
    binding.attempted = true;
    binding.enabled = false;
    binding.failure.clear();
    if (!env_enabled("QWEN3_TTS_OV_NATIVE_DECODE_STEP_PREBIND", false)) {
        binding.failure = "QWEN3_TTS_OV_NATIVE_DECODE_STEP_PREBIND=0";
        return false;
    }
    if (!embeds || !tts_pad_embed || hidden_size <= 0 || block_size <= 0) {
        binding.failure = "invalid prebind buffers";
        if (error) {
            *error = binding.failure;
        }
        return false;
    }
    try {
        const auto& compiled = request.get_compiled_model();
        binding.hidden_size = hidden_size;
        binding.block_size = block_size;
        binding.inputs_rank = compiled_model_input_rank(compiled, "inputs_embeds");
        binding.position_rank = compiled_model_input_rank(compiled, "position_ids");
        binding.block_indices_len = std::max<int64_t>(
            1,
            std::max<int64_t>(
                compiled_model_static_input_size(compiled, "block_indices"),
                block_indices_capacity));
        binding.has_tts_pad_embed = compiled_model_has_input(compiled, "tts_pad_embed");
        binding.has_allow_eos = compiled_model_has_input(compiled, "allow_eos");
        binding.has_allow_eos_steps = compiled_model_has_input(compiled, "allow_eos_steps");
        binding.has_beam_idx = compiled_model_has_input(compiled, "beam_idx");
        binding.has_score_aggregation_window = compiled_model_has_input(compiled, "score_aggregation_window");

        binding.position_ids.assign(3, 0);
        binding.block_indices.assign(static_cast<size_t>(binding.block_indices_len), 0);
        binding.allow_eos.assign(std::max<size_t>(1, allow_eos_size), 0.0f);
        binding.beam_idx.assign(1, 0);
        binding.score_aggregation_window.assign(1, 1);
        binding.past_lens.assign(1, 0);
        binding.subsequence_begins.assign({0, 1});
        binding.block_indices_begins.assign({0, 1});
        binding.max_context_len.assign(1, 1);

        if (binding.inputs_rank == 3) {
            binding.inputs_embeds_tensor = ov::Tensor(
                ov::element::f32,
                ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(embeds));
        } else {
            binding.inputs_embeds_tensor = ov::Tensor(
                ov::element::f32,
                ov::Shape{1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(embeds));
        }
        if (binding.position_rank == 3) {
            binding.position_ids_tensor = ov::Tensor(ov::element::i64, ov::Shape{3, 1, 1}, binding.position_ids.data());
        } else {
            binding.position_ids_tensor = ov::Tensor(ov::element::i64, ov::Shape{3, 1}, binding.position_ids.data());
        }
        binding.block_indices_tensor = ov::Tensor(
            ov::element::i32,
            ov::Shape{binding.block_indices.size()},
            binding.block_indices.data());
        binding.past_lens_tensor = ov::Tensor(ov::element::i32, ov::Shape{1}, binding.past_lens.data());
        binding.subsequence_begins_tensor = ov::Tensor(
            ov::element::i32,
            ov::Shape{binding.subsequence_begins.size()},
            binding.subsequence_begins.data());
        binding.block_indices_begins_tensor = ov::Tensor(
            ov::element::i32,
            ov::Shape{binding.block_indices_begins.size()},
            binding.block_indices_begins.data());
        binding.max_context_len_tensor = ov::Tensor(ov::element::i32, ov::Shape{}, binding.max_context_len.data());
        if (binding.has_tts_pad_embed) {
            binding.tts_pad_embed_tensor = ov::Tensor(
                ov::element::f32,
                ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(tts_pad_embed));
        }
        if (binding.has_allow_eos) {
            binding.allow_eos_tensor = ov::Tensor(ov::element::f32, ov::Shape{1}, binding.allow_eos.data());
        }
        if (binding.has_allow_eos_steps) {
            binding.allow_eos_steps_tensor = ov::Tensor(
                ov::element::f32,
                ov::Shape{binding.allow_eos.size()},
                binding.allow_eos.data());
        }
        if (binding.has_beam_idx) {
            binding.beam_idx_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, binding.beam_idx.data());
        }
        if (binding.has_score_aggregation_window) {
            binding.score_aggregation_window_tensor = ov::Tensor(
                ov::element::i32,
                ov::Shape{1},
                binding.score_aggregation_window.data());
        }

        request.set_tensor("inputs_embeds", binding.inputs_embeds_tensor);
        request.set_tensor("position_ids", binding.position_ids_tensor);
        if (binding.has_tts_pad_embed) {
            request.set_tensor("tts_pad_embed", binding.tts_pad_embed_tensor);
        }
        if (binding.has_allow_eos) {
            request.set_tensor("allow_eos", binding.allow_eos_tensor);
        }
        if (binding.has_allow_eos_steps) {
            request.set_tensor("allow_eos_steps", binding.allow_eos_steps_tensor);
        }
        if (binding.has_beam_idx) {
            request.set_tensor("beam_idx", binding.beam_idx_tensor);
        }
        if (binding.has_score_aggregation_window) {
            request.set_tensor("score_aggregation_window", binding.score_aggregation_window_tensor);
        }
        request.set_tensor("past_lens", binding.past_lens_tensor);
        request.set_tensor("subsequence_begins", binding.subsequence_begins_tensor);
        request.set_tensor("block_indices", binding.block_indices_tensor);
        request.set_tensor("block_indices_begins", binding.block_indices_begins_tensor);
        request.set_tensor("max_context_len", binding.max_context_len_tensor);
        binding.enabled = true;
        return true;
    } catch (const std::exception& exc) {
        binding.failure = exc.what();
    } catch (...) {
        binding.failure = "unknown decode step prebind failure";
    }
    if (error) {
        *error = binding.failure;
    }
    return false;
}

void update_prebound_decode_step_inputs(
    PagedDecodeStepBindings& binding,
    ov::InferRequest& request,
    const float* embeds,
    int64_t position_start,
    int64_t past_len,
    const std::vector<float>& allow_eos_values,
    const ov::Tensor* remote_input) {
    if (!binding.enabled) {
        throw std::runtime_error("decode step prebind is not enabled");
    }
    const int64_t total_len = past_len + 1;
    const int64_t blocks_used = std::max<int64_t>(1, (total_len + binding.block_size - 1) / binding.block_size);
    if (blocks_used > binding.block_indices_len) {
        throw std::runtime_error("decode step prebind block_indices buffer is too small");
    }
    for (int64_t row = 0; row < 3; ++row) {
        binding.position_ids[static_cast<size_t>(row)] = position_start;
    }
    std::fill(binding.block_indices.begin(), binding.block_indices.end(), 0);
    for (int64_t i = 0; i < blocks_used; ++i) {
        binding.block_indices[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    std::fill(binding.allow_eos.begin(), binding.allow_eos.end(), 0.0f);
    for (size_t i = 0; i < binding.allow_eos.size() && i < allow_eos_values.size(); ++i) {
        binding.allow_eos[i] = allow_eos_values[i];
    }
    binding.past_lens[0] = static_cast<int32_t>(past_len);
    binding.subsequence_begins[0] = 0;
    binding.subsequence_begins[1] = 1;
    binding.block_indices_begins[0] = 0;
    binding.block_indices_begins[1] = static_cast<int32_t>(blocks_used);
    binding.max_context_len[0] = static_cast<int32_t>(total_len);
    if (binding.has_beam_idx) {
        binding.beam_idx[0] = 0;
    }
    if (remote_input && static_cast<bool>(*remote_input)) {
        request.set_tensor("inputs_embeds", *remote_input);
    } else if (embeds) {
        request.set_tensor("inputs_embeds", binding.inputs_embeds_tensor);
    }
}

int64_t select_first_code_from_logits(
    NativeCodegen* runner,
    const ov::Tensor& logits_tensor,
    int64_t generated,
    int64_t min_new_tokens,
    int64_t vocab_size,
    int64_t eos_token_id,
    const std::vector<uint8_t>* repeated_mask = nullptr,
    float repetition_penalty = 1.0f,
    const NativeSamplingConfig* sampling = nullptr,
    std::mt19937_64* rng = nullptr) {
    const size_t size = logits_tensor.get_size();
    const auto logits_host = tensor_to_host_f32_view(runner, logits_tensor, "paged split logits");
    int64_t effective_vocab_size = vocab_size;
    const auto shape = logits_tensor.get_shape();
    if (!shape.empty() && shape.back() > 0) {
        const int64_t last_dim = static_cast<int64_t>(shape.back());
        if (
            effective_vocab_size <= 0 ||
            static_cast<size_t>(effective_vocab_size) > size ||
            size % static_cast<size_t>(effective_vocab_size) != 0) {
            effective_vocab_size = last_dim;
        }
    }
    if (
        !logits_host.data ||
        logits_host.size == 0 ||
        size == 0 ||
        effective_vocab_size <= 0 ||
        static_cast<size_t>(effective_vocab_size) > logits_host.size ||
        logits_host.size % static_cast<size_t>(effective_vocab_size) != 0) {
        std::ostringstream msg;
        msg << "invalid paged split logits tensor (" << tensor_debug_string(logits_tensor)
            << ", requested_vocab_size=" << vocab_size
            << ", effective_vocab_size=" << effective_vocab_size << ")";
        throw std::runtime_error(msg.str());
    }
    vocab_size = effective_vocab_size;
    const size_t rows = std::max<size_t>(1, logits_host.size / static_cast<size_t>(vocab_size));
    const float* logits = logits_host.data + (rows - 1) * static_cast<size_t>(vocab_size);
    const int64_t suppress_from = std::max<int64_t>(0, vocab_size - 1024);
    std::vector<std::pair<int64_t, float>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    for (int64_t token_id = 0; token_id < vocab_size; ++token_id) {
        float score = logits[static_cast<size_t>(token_id)];
        if (token_id >= suppress_from && token_id != eos_token_id) {
            score = NEG_INF;
        }
        if (token_id == eos_token_id && generated < min_new_tokens) {
            score = NEG_INF;
        }
        if (
            repeated_mask &&
            repetition_penalty != 1.0f &&
            token_id >= 0 &&
            static_cast<size_t>(token_id) < repeated_mask->size() &&
            (*repeated_mask)[static_cast<size_t>(token_id)] != 0 &&
            score > NEG_INF / 2.0f) {
            score = score < 0.0f ? score * repetition_penalty : score / repetition_penalty;
        }
        if (score > NEG_INF / 2.0f && std::isfinite(score)) {
            candidates.emplace_back(token_id, score);
        }
    }
    if (candidates.empty()) {
        return eos_token_id;
    }

    const bool do_sample = sampling && sampling->do_sample;
    if (!do_sample) {
        return std::max_element(
            candidates.begin(),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; })->first;
    }
    if (!rng) {
        throw std::runtime_error("sampling requested without RNG");
    }

    const float temperature = std::max<float>(1.0e-6f, sampling->temperature);
    for (auto& item : candidates) {
        item.second /= temperature;
    }

    const int64_t top_k = sampling->top_k;
    if (top_k > 0 && static_cast<size_t>(top_k) < candidates.size()) {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(top_k),
            candidates.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
        candidates.resize(static_cast<size_t>(top_k));
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    const double max_score = static_cast<double>(candidates.front().second);
    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (const auto& item : candidates) {
        const double weight = std::exp(static_cast<double>(item.second) - max_score);
        weights.push_back(weight);
        total += weight;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return candidates.front().first;
    }

    const float top_p = sampling->top_p;
    if (top_p > 0.0f && top_p < 1.0f && candidates.size() > 1) {
        double kept = 0.0;
        size_t keep_count = 0;
        for (; keep_count < weights.size(); ++keep_count) {
            kept += weights[keep_count];
            if (kept / total >= static_cast<double>(top_p)) {
                ++keep_count;
                break;
            }
        }
        keep_count = std::max<size_t>(1, std::min(keep_count, candidates.size()));
        candidates.resize(keep_count);
        weights.resize(keep_count);
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(*rng)].first;
}

void copy_last_hidden_vector(
    NativeCodegen* runner,
    const ov::Tensor& last_hidden_tensor,
    int64_t hidden_size,
    std::vector<float>& out) {
    const auto host = tensor_to_host_f32_view(runner, last_hidden_tensor, "paged split last_hidden");
    const size_t size = last_hidden_tensor.get_size();
    if (!host.data || host.size == 0 || hidden_size <= 0 || size < static_cast<size_t>(hidden_size)) {
        std::ostringstream msg;
        msg << "invalid paged split last_hidden tensor (" << tensor_debug_string(last_hidden_tensor)
            << ", hidden_size=" << hidden_size << ")";
        throw std::runtime_error(msg.str());
    }
    const size_t rows = std::max<size_t>(1, host.size / static_cast<size_t>(hidden_size));
    const float* last_row = host.data + (rows - 1) * static_cast<size_t>(hidden_size);
    out.resize(static_cast<size_t>(hidden_size));
    std::memcpy(out.data(), last_row, static_cast<size_t>(hidden_size) * sizeof(float));
}

void run_paged_split_subcode_step(
    NativeCodegen* runner,
    const ov::Tensor& last_hidden_tensor,
    int64_t first_code,
    const ov::Tensor* first_code_tensor,
    const float* tts_pad_embed,
    int64_t hidden_size,
    int64_t num_code_groups,
    std::vector<int64_t>& frame_codes,
    std::vector<float>& next_embed,
    ov::Tensor* remote_next_embed_output,
    bool need_host_next_embed,
    bool* remote_next_embed_written,
    const NativeSamplingConfig* sampling = nullptr,
    std::mt19937_64* rng = nullptr) {
    if (!runner || !tts_pad_embed || hidden_size <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid paged split subcode arguments");
    }
    if (!runner->paged_split_subcode) {
        throw std::runtime_error("paged split subcode graph is not configured");
    }
    auto& request = runner->subcode_request;
    if (remote_next_embed_written) {
        *remote_next_embed_written = false;
    }
    runner->scratch.subcode_first_code.resize(1);
    runner->scratch.subcode_first_code[0] = first_code;
    const uint64_t host_copy_bytes_before = runner->last_timing.host_copy_bytes;
    const int64_t host_copy_fallback_before = runner->last_timing.host_copy_fallback_count;
    double bind_ms = 0.0;
    measure_ms(bind_ms, [&]() {
        bool direct_hidden_bound = false;
        const auto hidden_shape = last_hidden_tensor.get_shape();
        const bool direct_hidden_enabled = env_enabled("QWEN3_TTS_OV_NATIVE_SPLIT_SUBCODE_REMOTE_HIDDEN", true);
        const bool require_direct_hidden = env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_SPLIT_SUBCODE_REMOTE_HIDDEN", false);
        std::string direct_hidden_failure;
        if (
            direct_hidden_enabled &&
            last_hidden_tensor.get_element_type() == ov::element::f32 &&
            hidden_shape.size() == 3 &&
            hidden_shape[0] == 1 &&
            hidden_shape[1] == 1 &&
            hidden_shape[2] == static_cast<size_t>(hidden_size)) {
            try {
                request.set_tensor("past_hidden", last_hidden_tensor);
                direct_hidden_bound = true;
                runner->last_timing.split_subcode_hidden_direct_bind_count += 1;
                runner->last_timing.zero_copy_count += 1;
            } catch (const std::exception& exc) {
                direct_hidden_bound = false;
                direct_hidden_failure = exc.what();
            } catch (...) {
                direct_hidden_bound = false;
                direct_hidden_failure = "unknown set_tensor failure";
            }
        } else if (!direct_hidden_enabled) {
            direct_hidden_failure = "QWEN3_TTS_OV_NATIVE_SPLIT_SUBCODE_REMOTE_HIDDEN=0";
        } else {
            direct_hidden_failure = "last_hidden tensor is not directly bindable: " + tensor_debug_string(last_hidden_tensor);
        }
        if (!direct_hidden_bound) {
            runner->last_timing.split_subcode_hidden_bind_fallback_count += 1;
            runner->last_timing.split_subcode_hidden_copy_bytes +=
                static_cast<uint64_t>(std::max<int64_t>(0, hidden_size)) * sizeof(float);
            if (require_direct_hidden) {
                throw std::runtime_error(
                    "required split-subcode remote hidden bind failed: " + direct_hidden_failure);
            }
            copy_last_hidden_vector(runner, last_hidden_tensor, hidden_size, runner->scratch.subcode_past_hidden);
            request.set_tensor(
                "past_hidden",
                ov::Tensor(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    runner->scratch.subcode_past_hidden.data()));
        }
        bool first_code_bound = false;
        if (
            first_code_tensor &&
            first_code_tensor->get_element_type() == ov::element::i64 &&
            first_code_tensor->get_size() >= 1) {
            try {
                request.set_tensor("first_code", *first_code_tensor);
                first_code_bound = true;
                runner->last_timing.zero_copy_count += 1;
            } catch (...) {
                first_code_bound = false;
            }
        }
        if (!first_code_bound) {
            request.set_tensor(
                "first_code",
                ov::Tensor(ov::element::i64, ov::Shape{1, 1}, runner->scratch.subcode_first_code.data()));
        }
        if (compiled_model_has_input(runner->subcode_model, "uniform_noise")) {
            int64_t subcode_groups = std::max<int64_t>(1, num_code_groups - 1);
            int64_t subcode_vocab = 2048;
            try {
                const auto pshape = runner->subcode_model.input("uniform_noise").get_partial_shape();
                if (pshape.rank().is_static() && pshape.rank().get_length() >= 3) {
                    const auto groups_dim = pshape[1];
                    const auto vocab_dim = pshape[2];
                    if (groups_dim.is_static()) {
                        subcode_groups = groups_dim.get_length();
                    }
                    if (vocab_dim.is_static()) {
                        subcode_vocab = vocab_dim.get_length();
                    }
                }
            } catch (...) {
                subcode_groups = std::max<int64_t>(1, num_code_groups - 1);
                subcode_vocab = 2048;
            }
            runner->scratch.subcode_uniform_noise.resize(
                static_cast<size_t>(subcode_groups * subcode_vocab));
            if (!rng) {
                static thread_local std::mt19937_64 fallback_rng{
                    static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
                rng = &fallback_rng;
            }
            std::uniform_real_distribution<float> dist(1.0e-6f, 0.999999f);
            for (float& value : runner->scratch.subcode_uniform_noise) {
                value = dist(*rng);
            }
            request.set_tensor(
                "uniform_noise",
                ov::Tensor(
                    ov::element::f32,
                    ov::Shape{1, static_cast<size_t>(subcode_groups), static_cast<size_t>(subcode_vocab)},
                    runner->scratch.subcode_uniform_noise.data()));
        }
        if (compiled_model_has_input(runner->subcode_model, "temperature")) {
            runner->scratch.subcode_temperature.assign(
                1,
                sampling ? std::max<float>(1.0e-6f, sampling->temperature) : 0.9f);
            request.set_tensor(
                "temperature",
                ov::Tensor(ov::element::f32, ov::Shape{1}, runner->scratch.subcode_temperature.data()));
        }
        if (runner->subcode_outputs_next_embed && compiled_model_has_input(runner->subcode_model, "tts_pad_embed")) {
            request.set_tensor(
                "tts_pad_embed",
                ov::Tensor(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    const_cast<float*>(tts_pad_embed)));
        }
        if (remote_next_embed_output && static_cast<bool>(*remote_next_embed_output)) {
            try {
                request.set_output_tensor(1, *remote_next_embed_output);
                if (remote_next_embed_written) {
                    *remote_next_embed_written = true;
                }
                runner->last_timing.split_subcode_remote_next_embed_enabled = true;
                runner->last_timing.zero_copy_count += 1;
            } catch (const std::exception& exc) {
                runner->last_timing.split_subcode_remote_next_embed_fallback_count += 1;
                if (env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_SPLIT_SUBCODE_REMOTE_NEXT_EMBED", false)) {
                    throw std::runtime_error(
                        std::string("required split-subcode remote next_embed output bind failed: ") + exc.what());
                }
            } catch (...) {
                runner->last_timing.split_subcode_remote_next_embed_fallback_count += 1;
                if (env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_SPLIT_SUBCODE_REMOTE_NEXT_EMBED", false)) {
                    throw std::runtime_error("required split-subcode remote next_embed output bind failed");
                }
            }
        }
    });
    runner->last_timing.tensor_bind_ms += bind_ms;
    runner->last_timing.subcode_bind_ms += bind_ms;
    runner->last_timing.subcode_bind_step_ms.push_back(bind_ms);
    double subcode_infer_ms = 0.0;
    measure_ms(subcode_infer_ms, [&]() {
        request.infer();
    });
    runner->last_timing.codegen_infer_ms += subcode_infer_ms;
    runner->last_timing.codegen_subcode_infer_ms += subcode_infer_ms;
    runner->last_timing.codegen_subcode_count += 1;
    runner->last_timing.subcode_infer_step_ms.push_back(subcode_infer_ms);
    record_request_profile(runner, "codegen_paged_kv_subcode", request);
    ov::Tensor codes_tensor;
    ov::Tensor embed_tensor;
    HostTensorView<int64_t> codes;
    FloatTensorView embed;
    const bool read_host_next_embed =
        need_host_next_embed || !(remote_next_embed_written && *remote_next_embed_written);
    double output_read_ms = 0.0;
    measure_ms(output_read_ms, [&]() {
        codes_tensor = request.get_output_tensor(0);
        codes = tensor_to_host_view<int64_t>(runner, codes_tensor, "paged split subcode codes");
        if (read_host_next_embed) {
            embed_tensor = request.get_output_tensor(1);
            embed = tensor_to_host_f32_view(runner, embed_tensor, "paged split subcode embed");
            runner->last_timing.split_subcode_next_embed_host_read_count += 1;
        }
    });
    runner->last_timing.subcode_output_read_ms += output_read_ms;
    runner->last_timing.subcode_output_read_step_ms.push_back(output_read_ms);
    runner->last_timing.subcode_host_copy_bytes += runner->last_timing.host_copy_bytes - host_copy_bytes_before;
    runner->last_timing.subcode_host_copy_fallback_count +=
        runner->last_timing.host_copy_fallback_count - host_copy_fallback_before;
    if (!codes.data || static_cast<int64_t>(codes.size) < num_code_groups) {
        std::ostringstream msg;
        msg << "paged split subcode graph returned too few codec groups (" << tensor_debug_string(codes_tensor)
            << ", expected_groups=" << num_code_groups << ")";
        throw std::runtime_error(msg.str());
    }
    if (read_host_next_embed && (!embed.data || static_cast<int64_t>(embed.size) < hidden_size)) {
        std::ostringstream msg;
        msg << "paged split subcode graph returned too small embed (" << tensor_debug_string(embed_tensor)
            << ", hidden_size=" << hidden_size << ")";
        throw std::runtime_error(msg.str());
    }
    double next_embed_ms = 0.0;
    measure_ms(next_embed_ms, [&]() {
        frame_codes.resize(static_cast<size_t>(num_code_groups));
        std::memcpy(frame_codes.data(), codes.data, static_cast<size_t>(num_code_groups) * sizeof(int64_t));
        if (read_host_next_embed) {
            next_embed.resize(static_cast<size_t>(hidden_size));
            if (runner->subcode_outputs_next_embed) {
                std::memcpy(next_embed.data(), embed.data, static_cast<size_t>(hidden_size) * sizeof(float));
            } else {
                for (int64_t i = 0; i < hidden_size; ++i) {
                    next_embed[static_cast<size_t>(i)] =
                        embed.data[static_cast<size_t>(i)] + tts_pad_embed[static_cast<size_t>(i)];
                }
            }
        }
    });
    runner->last_timing.subcode_next_embed_ms += next_embed_ms;
    runner->last_timing.subcode_next_embed_step_ms.push_back(next_embed_ms);
}

void run_paged_kv_impl(
    NativeCodegen* runner,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    const NativeSamplingConfig& sampling,
    int64_t* out_codes,
    int64_t* out_count,
    double* elapsed_ms,
    FrameCallback callback,
    void* user_data) {
    if (!runner || !sequence || !tts_pad_embed || !out_count || (!out_codes && !callback)) {
        throw std::runtime_error("invalid null pointer passed to native paged KV codegen");
    }
    if (prompt_len <= 0 || hidden_size <= 0 || max_new_tokens <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid native paged KV shape argument");
    }
    const auto started = std::chrono::steady_clock::now();
    *out_count = 0;
    runner->last_timing = NativeCodegen::RunTiming{};
    runner->last_timing.buffer_reuse = runner->paged_static_decode_enabled;
    runner->last_timing.no_repeat_fast_path = true;
    runner->codegen_trace.clear();
    const int64_t trace_frame_limit = std::max<int64_t>(0, env_int64("QWEN3_TTS_OV_NATIVE_TRACE_CODEGEN_FRAMES", 0));
    const bool use_repetition_penalty = std::abs(repetition_penalty - 1.0f) > 1e-6f;
    if (sampling.do_sample && !runner->paged_split_subcode) {
        throw std::runtime_error(
            "native paged-KV fused codegen does not support do_sample=true; "
            "use split-subcode paged-KV or the full-AR reference path");
    }
    if (use_repetition_penalty && !runner->paged_split_subcode) {
        throw std::runtime_error(
            "native paged-KV fused codegen does not support repetition_penalty != 1.0; "
            "use the full-AR reference path or export a paged graph with host-side logits");
    }

    const int64_t total_capacity_tokens = prompt_len + max_new_tokens + 1;
    const int64_t num_blocks = std::max<int64_t>(
        1,
        (total_capacity_tokens + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
    const bool static_decode_capacity_ok =
        runner->paged_static_decode_enabled && num_blocks <= runner->paged_static_decode_block_capacity;
    const int64_t cache_blocks = static_decode_capacity_ok ? runner->paged_static_decode_block_capacity : num_blocks;
    auto& prefill_request = runner->prefill_request;
    auto* decode_request = &runner->prefill_request;
    const auto& cache_tensors = get_paged_kv_cache_tensors(runner, runner->prefill_model, cache_blocks);
    bind_named_tensors(prefill_request, cache_tensors);
    bool use_static_decode = static_decode_capacity_ok;
    if (use_static_decode) {
        try {
            bind_named_tensors(runner->decode_request, cache_tensors);
            decode_request = &runner->decode_request;
        } catch (const std::exception& exc) {
            if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                std::cerr << "paged static decode cache bind failed; falling back to dynamic request: "
                          << exc.what() << std::endl;
            }
            use_static_decode = false;
            decode_request = &runner->prefill_request;
        }
    }

    std::vector<int64_t> position_ids;
    std::vector<int32_t> block_indices;
    std::vector<float> allow_eos_buffer;
    std::vector<int64_t> beam_idx_buffer;
    std::vector<float> next_embed(static_cast<size_t>(hidden_size));
    const int64_t graph_unroll = std::max<int64_t>(1, runner->unroll);
    PagedDecodeStepBindings decode_prebind;
    const bool decode_prebind_requested = env_enabled("QWEN3_TTS_OV_NATIVE_DECODE_STEP_PREBIND", false);
    if (use_static_decode && decode_prebind_requested) {
        std::string prebind_error;
        if (prepare_decode_step_prebind(
                decode_prebind,
                *decode_request,
                next_embed.data(),
                hidden_size,
                runner->paged_kv_block_size,
                cache_blocks,
                tts_pad_embed,
                static_cast<size_t>(std::max<int64_t>(1, graph_unroll)),
                &prebind_error)) {
            runner->last_timing.decode_step_prebind_enabled = true;
        } else {
            runner->last_timing.decode_step_prebind_fallback_count += 1;
            if (env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_DECODE_STEP_PREBIND", false)) {
                throw std::runtime_error("required decode step prebind failed: " + prebind_error);
            }
        }
    }
    std::vector<uint8_t> repeated_first_codes;
    if (use_repetition_penalty) {
        repeated_first_codes.assign(static_cast<size_t>(vocab_size), 0);
        runner->last_timing.no_repeat_fast_path = false;
    }
    std::mt19937_64 rng(sampling.seed);
    RemoteEmbedChain remote_embed = make_paged_remote_embed_chain(runner, hidden_size);
    if (runner->paged_split_subcode) {
        remote_embed.enabled = false;
    }
    if (use_static_decode && !env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_REMOTE_EMBED", false)) {
        remote_embed.enabled = false;
    }
    SplitSubcodeRemoteNextEmbedChain split_remote_next_embed =
        make_split_subcode_remote_next_embed_chain(runner, hidden_size);
    if (split_remote_next_embed.requested && !split_remote_next_embed.enabled) {
        runner->last_timing.split_subcode_remote_next_embed_fallback_count += 1;
        if (split_remote_next_embed.required) {
            throw std::runtime_error(
                "required split-subcode remote next_embed setup failed: " + split_remote_next_embed.failure);
        }
    }
    runner->last_remote_embed_used = remote_embed.enabled;
    std::vector<int64_t>& split_frame_codes = runner->scratch.subcode_frame_codes;

    int64_t generated = 0;
    bool stop = false;
    auto make_allow_eos_values = [&](int64_t frame_count) {
        std::vector<float> values(static_cast<size_t>(std::max<int64_t>(1, frame_count)), 0.0f);
        for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
            values[static_cast<size_t>(i)] = (generated + i) >= min_new_tokens ? 1.0f : 0.0f;
        }
        return values;
    };
    auto mark_codec_frame = [&]() {
        const double token_ms = elapsed_ms_since(started);
        if (runner->last_timing.generated_frames == 0) {
            runner->last_timing.ttft_ms = token_ms;
        }
        runner->last_timing.last_token_ms = token_ms;
        runner->last_timing.generated_frames += 1;
    };
    auto emit_many = [&](const int64_t* first_codes, const int64_t* codes, int64_t frame_count) {
        if (!first_codes || !codes || stop || generated >= max_new_tokens || frame_count <= 0) {
            return;
        }
        for (int64_t frame = 0; frame < frame_count && !stop && generated < max_new_tokens; ++frame) {
            const int64_t* first_code = first_codes + frame;
            const int64_t* frame_codes = codes + frame * num_code_groups;
            if (*first_code == eos_token_id && generated >= min_new_tokens) {
                stop = true;
                return;
            }
            if (out_codes) {
                std::memcpy(
                    out_codes + generated * num_code_groups,
                    frame_codes,
                    static_cast<size_t>(num_code_groups) * sizeof(int64_t));
            }
            mark_codec_frame();
            if (callback) {
                int callback_rc = 0;
                double callback_ms = 0.0;
                measure_ms(callback_ms, [&]() {
                    callback_rc = callback(frame_codes, 1, num_code_groups, user_data);
                });
                runner->last_timing.codegen_callback_ms += callback_ms;
                runner->last_timing.callback_ms += callback_ms;
                if (callback_rc != 0) {
                    throw std::runtime_error("native paged KV callback requested stop");
                }
            }
            generated += 1;
        }
    };

    auto append_trace = [&](
        const std::string& label,
        int64_t frame_index,
        int64_t first_code,
        const int64_t* codes,
        const ov::Tensor* hidden_tensor,
        const float* embed_data) {
        if (trace_frame_limit <= 0 || static_cast<int64_t>(runner->codegen_trace.size()) >= trace_frame_limit) {
            return;
        }
        NativeCodegen::CodegenTraceFrame item;
        item.frame = frame_index;
        item.label = label;
        item.first_code = first_code;
        if (codes && num_code_groups > 0) {
            item.codes.assign(codes, codes + num_code_groups);
        }
        if (hidden_tensor) {
            try {
                const auto hidden = tensor_to_host_f32_view(runner, *hidden_tensor, "trace hidden");
                const auto stats = f32_l2_and_max_abs(hidden.data, std::min<size_t>(hidden.size, static_cast<size_t>(hidden_size)));
                item.hidden_l2 = stats.first;
                item.hidden_max_abs = stats.second;
            } catch (...) {
                item.hidden_l2 = -1.0;
                item.hidden_max_abs = -1.0;
            }
        }
        if (embed_data) {
            const auto stats = f32_l2_and_max_abs(embed_data, static_cast<size_t>(hidden_size));
            item.embed_l2 = stats.first;
            item.embed_max_abs = stats.second;
        }
        runner->codegen_trace.push_back(std::move(item));
    };

    const ov::Tensor* next_remote_input = nullptr;
    bool split_remote_host_backup_valid = false;

    auto run_step = [&](
        ov::InferRequest& step_request,
        const std::string& step_request_label,
        const float* embeds,
        int64_t seq_len,
        int64_t position_start,
        int64_t past_len,
        const std::vector<float>& allow_eos_values,
        const ov::Tensor* remote_input,
        ov::Tensor* remote_output) {
        double bind_ms = 0.0;
        measure_ms(bind_ms, [&]() {
            const bool has_remote_input = remote_input && static_cast<bool>(*remote_input);
            bool used_prebind = false;
            auto bind_host_inputs = [&]() {
                bind_paged_step_inputs(
                    step_request,
                    embeds,
                    seq_len,
                    hidden_size,
                    position_start,
                    past_len,
                    runner->paged_kv_block_size,
                    tts_pad_embed,
                    allow_eos_values,
                    position_ids,
                    block_indices,
                    allow_eos_buffer,
                    beam_idx_buffer,
                    true);
            };
            auto mark_split_remote_decode_input_ok = [&]() {
                if (split_remote_next_embed.enabled && has_remote_input) {
                    split_remote_next_embed.verified_decode_input = true;
                }
            };
            if (
                decode_prebind.enabled &&
                use_static_decode &&
                &step_request == decode_request &&
                seq_len == 1) {
                try {
                    double prebind_update_ms = 0.0;
                    measure_ms(prebind_update_ms, [&]() {
                        update_prebound_decode_step_inputs(
                            decode_prebind,
                            step_request,
                            embeds,
                            position_start,
                            past_len,
                            allow_eos_values,
                            remote_input);
                    });
                    runner->last_timing.decode_step_prebind_update_ms += prebind_update_ms;
                    runner->last_timing.decode_step_prebind_update_step_ms.push_back(prebind_update_ms);
                    used_prebind = true;
                    mark_split_remote_decode_input_ok();
                } catch (const std::exception& exc) {
                    decode_prebind.enabled = false;
                    runner->last_timing.decode_step_prebind_fallback_count += 1;
                    if (env_enabled("QWEN3_TTS_OV_NATIVE_REQUIRE_DECODE_STEP_PREBIND", false)) {
                        throw std::runtime_error(std::string("required decode step prebind update failed: ") + exc.what());
                    }
                }
            }
            if (!used_prebind) {
                bind_paged_step_inputs(
                    step_request,
                    embeds,
                    seq_len,
                    hidden_size,
                    position_start,
                    past_len,
                    runner->paged_kv_block_size,
                    tts_pad_embed,
                    allow_eos_values,
                    position_ids,
                    block_indices,
                    allow_eos_buffer,
                    beam_idx_buffer,
                    !has_remote_input);
                if (has_remote_input) {
                    try {
                        step_request.set_tensor("inputs_embeds", *remote_input);
                        mark_split_remote_decode_input_ok();
                    } catch (const std::exception& exc) {
                        runner->last_timing.split_subcode_remote_next_embed_fallback_count += 1;
                        if (split_remote_next_embed.required || !split_remote_host_backup_valid) {
                            throw std::runtime_error(
                                std::string("split-subcode remote next_embed input bind failed: ") + exc.what());
                        }
                        split_remote_next_embed.enabled = false;
                        next_remote_input = nullptr;
                        bind_host_inputs();
                    } catch (...) {
                        runner->last_timing.split_subcode_remote_next_embed_fallback_count += 1;
                        if (split_remote_next_embed.required || !split_remote_host_backup_valid) {
                            throw std::runtime_error("split-subcode remote next_embed input bind failed");
                        }
                        split_remote_next_embed.enabled = false;
                        next_remote_input = nullptr;
                        bind_host_inputs();
                    }
                }
            }
            if (remote_output && static_cast<bool>(*remote_output)) {
                step_request.set_output_tensor(2, *remote_output);
            }
        });
        runner->last_timing.tensor_bind_ms += bind_ms;
        runner->last_timing.codegen_bind_step_ms.push_back(bind_ms);
        double step_infer_ms = 0.0;
        measure_ms(step_infer_ms, [&]() {
            step_request.infer();
        });
        runner->last_timing.codegen_infer_ms += step_infer_ms;
        if (step_request_label.find("prefill") != std::string::npos) {
            runner->last_timing.codegen_prefill_infer_ms += step_infer_ms;
            runner->last_timing.codegen_prefill_count += 1;
            runner->last_timing.codegen_prefill_step_ms.push_back(step_infer_ms);
        } else {
            runner->last_timing.codegen_decode_infer_ms += step_infer_ms;
            runner->last_timing.codegen_decode_count += 1;
            runner->last_timing.codegen_decode_step_ms.push_back(step_infer_ms);
        }
        record_request_profile(runner, step_request_label, step_request);
        if (runner->paged_split_subcode) {
            auto first_output_tensor = step_request.get_output_tensor(0);
            auto last_hidden_tensor = step_request.get_output_tensor(1);
            int64_t first_code = 0;
            double sampling_ms = 0.0;
            measure_ms(sampling_ms, [&]() {
                if (runner->paged_seed_outputs_first_code) {
                    if (sampling.do_sample || use_repetition_penalty) {
                        throw std::runtime_error(
                            "paged top1 seed split-subcode supports greedy repetition_penalty=1.0 only");
                    }
                    first_code = tensor_to_host_i64_scalar(runner, first_output_tensor, "paged top1 first_code");
                } else {
                    first_code = select_first_code_from_logits(
                        runner,
                        first_output_tensor,
                        generated,
                        min_new_tokens,
                        vocab_size,
                        eos_token_id,
                        use_repetition_penalty ? &repeated_first_codes : nullptr,
                        repetition_penalty,
                        &sampling,
                        &rng);
                }
            });
            runner->last_timing.sampling_ms += sampling_ms;
            runner->last_timing.codegen_sampling_step_ms.push_back(sampling_ms);
            if (first_code == eos_token_id && generated >= min_new_tokens) {
                stop = true;
                return;
            }
            ov::Tensor* split_remote_output = nullptr;
            if (split_remote_next_embed.enabled) {
                split_remote_output = &split_remote_next_embed.outputs[split_remote_next_embed.next_output];
            }
            bool split_remote_written = false;
            const bool need_host_next_embed =
                trace_frame_limit > 0 ||
                (split_remote_next_embed.enabled && !split_remote_next_embed.verified_decode_input);
            run_paged_split_subcode_step(
                runner,
                last_hidden_tensor,
                first_code,
                runner->paged_seed_outputs_first_code ? &first_output_tensor : nullptr,
                tts_pad_embed,
                hidden_size,
                num_code_groups,
                split_frame_codes,
                next_embed,
                split_remote_output,
                need_host_next_embed,
                &split_remote_written,
                &sampling,
                &rng);
            split_remote_host_backup_valid = need_host_next_embed || !split_remote_written;
            append_trace(
                "split_subcode",
                generated,
                first_code,
                split_frame_codes.data(),
                &last_hidden_tensor,
                need_host_next_embed && !next_embed.empty() ? next_embed.data() : nullptr);
            if (split_remote_written && split_remote_output) {
                next_remote_input = split_remote_output;
                split_remote_next_embed.next_output = 1 - split_remote_next_embed.next_output;
            } else {
                if (split_remote_output) {
                    split_remote_next_embed.enabled = false;
                }
                next_remote_input = nullptr;
            }
            emit_many(&first_code, split_frame_codes.data(), 1);
            if (
                use_repetition_penalty &&
                first_code >= 0 &&
                static_cast<size_t>(first_code) < repeated_first_codes.size()) {
                repeated_first_codes[static_cast<size_t>(first_code)] = 1;
            }
        } else {
            auto first_codes_tensor = step_request.get_output_tensor(0);
            auto codes_tensor = step_request.get_output_tensor(1);
            auto frame_embed_tensor = step_request.get_output_tensor(2);
            const int64_t output_frames = static_cast<int64_t>(codes_tensor.get_size()) / num_code_groups;
            const int64_t* first_codes = tensor_data<int64_t>(first_codes_tensor);
            const int64_t* codes = tensor_data<int64_t>(codes_tensor);
            const float* frame_embed = tensor_data<float>(frame_embed_tensor);
            for (int64_t frame = 0; frame < output_frames; ++frame) {
                append_trace(
                    "graph_fused",
                    generated + frame,
                    first_codes ? first_codes[frame] : 0,
                    codes ? codes + frame * num_code_groups : nullptr,
                    nullptr,
                    frame + 1 == output_frames ? frame_embed : nullptr);
            }
            emit_many(first_codes, codes, output_frames);
            if (!remote_output || !static_cast<bool>(*remote_output)) {
                std::memcpy(next_embed.data(), frame_embed, static_cast<size_t>(hidden_size) * sizeof(float));
            }
        }
    };

    ov::Tensor* current_remote_output = remote_embed.enabled ? &remote_embed.decode_outputs[0] : nullptr;
    run_step(
        prefill_request,
        "codegen_paged_kv_prefill",
        sequence,
        prompt_len,
        0,
        0,
        make_allow_eos_values(graph_unroll),
        nullptr,
        current_remote_output);
    if (remote_embed.enabled) {
        next_remote_input = current_remote_output;
        remote_embed.next_decode_output = 1;
    }
    while (!stop && generated < max_new_tokens) {
        const int64_t position = prompt_len + generated - 1;
        current_remote_output = remote_embed.enabled ? &remote_embed.decode_outputs[remote_embed.next_decode_output] : nullptr;
        run_step(
            *decode_request,
            use_static_decode ? "codegen_paged_kv_decode_static" : "codegen_paged_kv_decode_dynamic",
            next_embed.data(),
            1,
            position,
            position,
            make_allow_eos_values(graph_unroll),
            next_remote_input,
            current_remote_output);
        if (remote_embed.enabled) {
            next_remote_input = current_remote_output;
            remote_embed.next_decode_output = 1 - remote_embed.next_decode_output;
        }
    }

    *out_count = generated;
    if (generated == 0) {
        throw std::runtime_error("native paged KV codegen stopped before producing any codec token");
    }
    runner->last_timing.total_ms = elapsed_ms_since(started);
    if (elapsed_ms) {
        *elapsed_ms = runner->last_timing.total_ms;
    }
}

void run_paged_kv_sequence_batch_impl(
    NativeCodegen* runner,
    const float* sequence,
    int64_t total_prompt_tokens,
    const int64_t* prompt_lens,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t batch_size,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    const NativeSamplingConfig& sampling,
    int64_t* out_counts,
    double* out_ttft_ms,
    double* out_last_token_ms,
    double* elapsed_ms,
    int64_t* out_codes = nullptr) {
    if (!runner || !sequence || !prompt_lens || !tts_pad_embed || !out_counts || !out_ttft_ms || !out_last_token_ms) {
        throw std::runtime_error("invalid null pointer passed to native continuous batch codegen");
    }
    if (!runner->paged_kv_enabled || !runner->paged_split_subcode) {
        throw std::runtime_error("native continuous batch requires paged-KV split-subcode runner");
    }
    if (sampling.do_sample) {
        throw std::runtime_error("native continuous batch currently supports greedy do_sample=false only");
    }
    if (std::abs(repetition_penalty - 1.0f) > 1e-6f) {
        throw std::runtime_error("native continuous batch currently requires repetition_penalty=1.0");
    }
    if (total_prompt_tokens <= 0 || hidden_size <= 0 || batch_size <= 0 || max_new_tokens <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid native continuous batch shape argument");
    }
    const auto started = std::chrono::steady_clock::now();
    runner->last_timing = NativeCodegen::RunTiming{};
    runner->last_timing.no_repeat_fast_path = true;
    runner->codegen_trace.clear();
    std::fill(out_counts, out_counts + batch_size, 0);
    std::fill(out_ttft_ms, out_ttft_ms + batch_size, -1.0);
    std::fill(out_last_token_ms, out_last_token_ms + batch_size, 0.0);
    if (out_codes) {
        std::fill(
            out_codes,
            out_codes + static_cast<size_t>(batch_size * max_new_tokens * num_code_groups),
            -1);
    }
    const char* policy_env = std::getenv("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_POLICY");
    std::string batch_policy = normalize_continuous_batch_policy(
        policy_env && std::strlen(policy_env) > 0 ? policy_env : "raw_fused");
    if (!valid_continuous_batch_policy(batch_policy)) {
        throw std::runtime_error(continuous_batch_policy_error());
    }
    runner->continuous_batch_policy = batch_policy;
    runner->batch_decode_unroll = 1;
    runner->last_timing.batch_decode_unroll_requested = runner->batch_decode_unroll_requested;
    runner->last_timing.batch_decode_unroll = runner->batch_decode_unroll;
    if (runner->batch_decode_unroll_requested > 1) {
        runner->last_timing.batch_decode_unroll_fallback_count += 1;
    }

    std::vector<int64_t> prompt_lengths(static_cast<size_t>(batch_size), 0);
    std::vector<int64_t> blocks_per_sequence(static_cast<size_t>(batch_size), 0);
    int64_t checked_total_prompt_tokens = 0;
    int64_t total_cache_blocks = 0;
    for (int64_t seq = 0; seq < batch_size; ++seq) {
        const int64_t prompt_len = prompt_lens[seq];
        if (prompt_len <= 0) {
            throw std::runtime_error("native continuous batch prompt_lens must be positive");
        }
        prompt_lengths[static_cast<size_t>(seq)] = prompt_len;
        checked_total_prompt_tokens += prompt_len;
        const int64_t capacity_tokens = prompt_len + max_new_tokens + 1;
        const int64_t blocks = std::max<int64_t>(
            1,
            (capacity_tokens + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
        blocks_per_sequence[static_cast<size_t>(seq)] = blocks;
        total_cache_blocks += blocks;
    }
    if (checked_total_prompt_tokens != total_prompt_tokens) {
        throw std::runtime_error("native continuous batch total_prompt_tokens does not match prompt_lens sum");
    }
    auto& request = runner->prefill_request;
    const auto& cache_tensors = get_paged_kv_cache_tensors(runner, runner->prefill_model, total_cache_blocks);
    bind_named_tensors(request, cache_tensors);
    if (runner->paged_fused_batch_decode_enabled) {
        bind_named_tensors(runner->decode_request, cache_tensors);
    }

    std::vector<PagedBatchSequenceState> states(static_cast<size_t>(batch_size));
    int64_t next_block_offset = 0;
    for (int64_t seq = 0; seq < batch_size; ++seq) {
        auto& state = states[static_cast<size_t>(seq)];
        state.prompt_len = prompt_lengths[static_cast<size_t>(seq)];
        state.block_offset = next_block_offset;
        state.block_capacity = blocks_per_sequence[static_cast<size_t>(seq)];
        next_block_offset += state.block_capacity;
        state.next_embed.assign(static_cast<size_t>(hidden_size), 0.0f);
    }

    std::vector<int64_t> position_ids;
    std::vector<int32_t> past_lens_buffer;
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<float> allow_eos_buffer;
    std::vector<int64_t> beam_idx_buffer;
    std::vector<int32_t> score_aggregation_window_buffer;
    std::vector<int64_t> frame_codes;
    std::vector<float> next_embeds;
    std::vector<int64_t> first_codes;

    auto active_count = [&]() {
        int64_t count = 0;
        for (const auto& state : states) {
            if (!state.stop && state.generated < max_new_tokens) {
                ++count;
            }
        }
        return count;
    };
    auto mark_sequence_frame = [&](int64_t seq_index) {
        auto& state = states[static_cast<size_t>(seq_index)];
        const double token_ms = elapsed_ms_since(started);
        if (state.generated == 0) {
            state.ttft_ms = token_ms;
            out_ttft_ms[seq_index] = token_ms;
        }
        state.last_token_ms = token_ms;
        out_last_token_ms[seq_index] = token_ms;
        state.generated += 1;
        out_counts[seq_index] = state.generated;
        runner->last_timing.generated_frames += 1;
        runner->last_timing.last_token_ms = token_ms;
        if (runner->last_timing.ttft_ms <= 0.0) {
            runner->last_timing.ttft_ms = token_ms;
        }
    };
    auto record_frame_codes = [&](int64_t seq_index, size_t row) {
        if (!out_codes) {
            return;
        }
        auto& state = states[static_cast<size_t>(seq_index)];
        if (state.generated < 0 || state.generated >= max_new_tokens) {
            return;
        }
        const size_t src_offset = row * static_cast<size_t>(num_code_groups);
        if (frame_codes.size() < src_offset + static_cast<size_t>(num_code_groups)) {
            throw std::runtime_error("native continuous batch frame_codes output is too small");
        }
        const size_t dst_offset =
            (static_cast<size_t>(seq_index) * static_cast<size_t>(max_new_tokens) +
             static_cast<size_t>(state.generated)) *
            static_cast<size_t>(num_code_groups);
        std::memcpy(
            out_codes + dst_offset,
            frame_codes.data() + src_offset,
            static_cast<size_t>(num_code_groups) * sizeof(int64_t));
    };
    auto select_first_codes = [&](
        const ov::Tensor& first_output_tensor,
        const std::vector<int64_t>& seq_indices,
        std::vector<int64_t>& selected,
        const std::vector<int64_t>& row_offsets = {}) {
        selected.assign(seq_indices.size(), eos_token_id);
        int64_t max_row = static_cast<int64_t>(seq_indices.size()) - 1;
        if (!row_offsets.empty()) {
            if (row_offsets.size() != seq_indices.size()) {
                throw std::runtime_error("native continuous batch row_offsets size mismatch");
            }
            for (const int64_t row : row_offsets) {
                max_row = std::max<int64_t>(max_row, row);
            }
        }
        if (runner->paged_seed_outputs_first_code) {
            const auto first_code_view = tensor_to_host_view<int64_t>(runner, first_output_tensor, "batch first_code");
            if (!first_code_view.data || first_code_view.size <= static_cast<size_t>(max_row)) {
                throw std::runtime_error("native continuous batch first_code output is too small");
            }
            for (size_t i = 0; i < seq_indices.size(); ++i) {
                const int64_t row = row_offsets.empty() ? static_cast<int64_t>(i) : row_offsets[i];
                selected[i] = first_code_view.data[static_cast<size_t>(row)];
            }
            return;
        }
        const auto logits = tensor_to_host_f32_view(runner, first_output_tensor, "batch logits");
        int64_t effective_vocab_size = vocab_size;
        const auto shape = first_output_tensor.get_shape();
        if (!shape.empty() && shape.back() > 0) {
            effective_vocab_size = static_cast<int64_t>(shape.back());
        }
        if (
            !logits.data ||
            effective_vocab_size <= 0 ||
            logits.size < static_cast<size_t>(max_row + 1) * static_cast<size_t>(effective_vocab_size)) {
            std::ostringstream msg;
            msg << "native continuous batch logits output is invalid: "
                << tensor_debug_string(first_output_tensor)
                << ", logits_size=" << logits.size
                << ", selected_rows=" << seq_indices.size()
                << ", effective_vocab_size=" << effective_vocab_size;
            throw std::runtime_error(msg.str());
        }
        for (size_t i = 0; i < seq_indices.size(); ++i) {
            const int64_t seq_index = seq_indices[i];
            const int64_t row = row_offsets.empty() ? static_cast<int64_t>(i) : row_offsets[i];
            selected[i] = select_first_code_greedy_from_logits_row(
                logits.data + static_cast<size_t>(row) * static_cast<size_t>(effective_vocab_size),
                effective_vocab_size,
                states[static_cast<size_t>(seq_index)].generated,
                min_new_tokens,
                eos_token_id);
        }
    };

    std::vector<int64_t> seq_indices;
    std::vector<int64_t> seq_lens;
    std::vector<int64_t> position_starts;
    std::vector<int64_t> past_lens;
    std::vector<int64_t> block_offsets;
    std::vector<int64_t> block_counts;
    std::vector<float> allow_eos;
    std::vector<float> decode_embeds;

    const bool use_batched_prefill =
        continuous_policy_uses_batched_prefill(batch_policy) &&
        batch_size > 1 &&
        env_enabled("QWEN3_TTS_OV_NATIVE_BATCH_PREFILL", false);
    if (use_batched_prefill) {
        runner->last_timing.batch_prefill_enabled = true;
        seq_indices.resize(static_cast<size_t>(batch_size));
        seq_lens.resize(static_cast<size_t>(batch_size));
        position_starts.assign(static_cast<size_t>(batch_size), 0);
        past_lens.assign(static_cast<size_t>(batch_size), 0);
        block_offsets.resize(static_cast<size_t>(batch_size));
        block_counts.resize(static_cast<size_t>(batch_size));
        allow_eos.assign(static_cast<size_t>(batch_size), min_new_tokens <= 0 ? 1.0f : 0.0f);
        std::vector<int64_t> prefill_row_offsets(static_cast<size_t>(batch_size), 0);
        int64_t token_offset = 0;
        for (int64_t seq = 0; seq < batch_size; ++seq) {
            auto& state = states[static_cast<size_t>(seq)];
            seq_indices[static_cast<size_t>(seq)] = seq;
            seq_lens[static_cast<size_t>(seq)] = state.prompt_len;
            block_offsets[static_cast<size_t>(seq)] = state.block_offset;
            block_counts[static_cast<size_t>(seq)] = std::max<int64_t>(
                1,
                (state.prompt_len + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
            prefill_row_offsets[static_cast<size_t>(seq)] = token_offset + state.prompt_len - 1;
            token_offset += state.prompt_len;
        }
        double bind_ms = 0.0;
        measure_ms(bind_ms, [&]() {
            bind_paged_batch_step_inputs(
                request,
                sequence,
                seq_lens,
                hidden_size,
                position_starts,
                past_lens,
                runner->paged_kv_block_size,
                tts_pad_embed,
                allow_eos,
                block_offsets,
                block_counts,
                position_ids,
                past_lens_buffer,
                subsequence_begins,
                block_indices,
                block_indices_begins,
                allow_eos_buffer,
                beam_idx_buffer,
                score_aggregation_window_buffer);
        });
        runner->last_timing.tensor_bind_ms += bind_ms;
        runner->last_timing.codegen_bind_step_ms.push_back(bind_ms);
        double prefill_ms = 0.0;
        measure_ms(prefill_ms, [&]() {
            request.infer();
        });
        runner->last_timing.codegen_infer_ms += prefill_ms;
        runner->last_timing.codegen_prefill_infer_ms += prefill_ms;
        runner->last_timing.codegen_prefill_count += 1;
        runner->last_timing.codegen_prefill_step_ms.push_back(prefill_ms);
        record_request_profile(runner, "codegen_paged_kv_prefill_batch", request);

        auto first_output_tensor = request.get_output_tensor(0);
        auto last_hidden_tensor = request.get_output_tensor(1);
        select_first_codes(first_output_tensor, seq_indices, first_codes, prefill_row_offsets);
        std::vector<float> prefill_last_hidden(static_cast<size_t>(batch_size * hidden_size), 0.0f);
        const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, "continuous batch prefill last_hidden");
        if (!hidden_view.data || hidden_view.size < static_cast<size_t>(total_prompt_tokens * hidden_size)) {
            throw std::runtime_error("native continuous batch prefill hidden output is invalid");
        }
        for (int64_t seq = 0; seq < batch_size; ++seq) {
            const int64_t row = prefill_row_offsets[static_cast<size_t>(seq)];
            std::memcpy(
                prefill_last_hidden.data() + static_cast<size_t>(seq * hidden_size),
                hidden_view.data + static_cast<size_t>(row * hidden_size),
                static_cast<size_t>(hidden_size) * sizeof(float));
        }
        const bool use_batched_prefill_subcode =
            runner->batch_subcode_enabled &&
            env_enabled("QWEN3_TTS_OV_NATIVE_BATCH_PREFILL_SUBCODE", false);
        if (use_batched_prefill_subcode) {
            ov::Tensor subcode_hidden_tensor(
                ov::element::f32,
                ov::Shape{static_cast<size_t>(batch_size), 1, static_cast<size_t>(hidden_size)},
                prefill_last_hidden.data());
            run_paged_split_subcode_batch(
                runner,
                subcode_hidden_tensor,
                first_codes,
                tts_pad_embed,
                hidden_size,
                num_code_groups,
                frame_codes,
                next_embeds);
        } else {
            frame_codes.assign(static_cast<size_t>(batch_size * num_code_groups), 0);
            next_embeds.assign(static_cast<size_t>(batch_size * hidden_size), 0.0f);
            for (int64_t seq = 0; seq < batch_size; ++seq) {
                ov::Tensor row_hidden_tensor(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    prefill_last_hidden.data() + static_cast<size_t>(seq * hidden_size));
                std::vector<int64_t> row_first_code{first_codes[static_cast<size_t>(seq)]};
                std::vector<int64_t> row_codes;
                std::vector<float> row_next_embed;
                run_paged_split_subcode_batch(
                    runner,
                    row_hidden_tensor,
                    row_first_code,
                    tts_pad_embed,
                    hidden_size,
                    num_code_groups,
                    row_codes,
                    row_next_embed);
                if (row_codes.size() < static_cast<size_t>(num_code_groups)) {
                    throw std::runtime_error("native continuous batched prefill subcode returned too few codec groups");
                }
                if (row_next_embed.size() < static_cast<size_t>(hidden_size)) {
                    throw std::runtime_error("native continuous batched prefill subcode returned too small next_embed");
                }
                std::memcpy(
                    frame_codes.data() + static_cast<size_t>(seq * num_code_groups),
                    row_codes.data(),
                    static_cast<size_t>(num_code_groups) * sizeof(int64_t));
                std::memcpy(
                    next_embeds.data() + static_cast<size_t>(seq * hidden_size),
                    row_next_embed.data(),
                    static_cast<size_t>(hidden_size) * sizeof(float));
            }
        }
        for (int64_t seq = 0; seq < batch_size; ++seq) {
            auto& state = states[static_cast<size_t>(seq)];
            const int64_t first_code = first_codes[static_cast<size_t>(seq)];
            if (first_code == eos_token_id && state.generated >= min_new_tokens) {
                state.stop = true;
                continue;
            }
            record_frame_codes(seq, static_cast<size_t>(seq));
            std::memcpy(
                state.next_embed.data(),
                next_embeds.data() + static_cast<size_t>(seq * hidden_size),
                static_cast<size_t>(hidden_size) * sizeof(float));
            mark_sequence_frame(seq);
        }
    } else {
        int64_t prompt_token_offset = 0;
        for (int64_t seq = 0; seq < batch_size; ++seq) {
            auto& state = states[static_cast<size_t>(seq)];
            seq_indices.assign(1, seq);
            seq_lens.assign(1, state.prompt_len);
            position_starts.assign(1, 0);
            past_lens.assign(1, 0);
            block_offsets.assign(1, state.block_offset);
            block_counts.assign(
                1,
                std::max<int64_t>(
                    1,
                    (state.prompt_len + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size));
            allow_eos.assign(1, min_new_tokens <= 0 ? 1.0f : 0.0f);
            const float* sequence_ptr = sequence + static_cast<size_t>(prompt_token_offset * hidden_size);
            prompt_token_offset += state.prompt_len;

            double bind_ms = 0.0;
            measure_ms(bind_ms, [&]() {
                bind_paged_batch_step_inputs(
                    request,
                    sequence_ptr,
                    seq_lens,
                    hidden_size,
                    position_starts,
                    past_lens,
                    runner->paged_kv_block_size,
                    tts_pad_embed,
                    allow_eos,
                    block_offsets,
                    block_counts,
                    position_ids,
                    past_lens_buffer,
                    subsequence_begins,
                    block_indices,
                    block_indices_begins,
                    allow_eos_buffer,
                    beam_idx_buffer,
                    score_aggregation_window_buffer);
            });
            runner->last_timing.tensor_bind_ms += bind_ms;
            runner->last_timing.codegen_bind_step_ms.push_back(bind_ms);
            double prefill_ms = 0.0;
            measure_ms(prefill_ms, [&]() {
                request.infer();
            });
            runner->last_timing.codegen_infer_ms += prefill_ms;
            runner->last_timing.codegen_prefill_infer_ms += prefill_ms;
            runner->last_timing.codegen_prefill_count += 1;
            runner->last_timing.codegen_prefill_step_ms.push_back(prefill_ms);
            record_request_profile(runner, "codegen_paged_kv_prefill_sequence", request);

            auto first_output_tensor = request.get_output_tensor(0);
            auto last_hidden_tensor = request.get_output_tensor(1);
            std::vector<int64_t> prefill_row_offsets{state.prompt_len - 1};
            select_first_codes(first_output_tensor, seq_indices, first_codes, prefill_row_offsets);
            ov::Tensor subcode_hidden_tensor = last_hidden_tensor;
            std::vector<float> prefill_last_hidden;
            if (last_hidden_tensor.get_size() > static_cast<size_t>(hidden_size)) {
                const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, "continuous prefill last_hidden");
                if (!hidden_view.data || hidden_view.size < static_cast<size_t>(state.prompt_len * hidden_size)) {
                    throw std::runtime_error("native continuous batch prefill hidden output is invalid");
                }
                prefill_last_hidden.assign(
                    hidden_view.data + static_cast<size_t>((state.prompt_len - 1) * hidden_size),
                    hidden_view.data + static_cast<size_t>(state.prompt_len * hidden_size));
                subcode_hidden_tensor = ov::Tensor(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    prefill_last_hidden.data());
            }
            run_paged_split_subcode_batch(
                runner,
                subcode_hidden_tensor,
                first_codes,
                tts_pad_embed,
                hidden_size,
                num_code_groups,
                frame_codes,
                next_embeds);
            const int64_t first_code = first_codes[0];
            if (first_code == eos_token_id && state.generated >= min_new_tokens) {
                state.stop = true;
                continue;
            }
            record_frame_codes(seq, 0);
            std::memcpy(
                state.next_embed.data(),
                next_embeds.data(),
                static_cast<size_t>(hidden_size) * sizeof(float));
            mark_sequence_frame(seq);
        }
    }

    while (active_count() > 0) {
        seq_indices.clear();
        for (int64_t seq = 0; seq < batch_size; ++seq) {
            const auto& state = states[static_cast<size_t>(seq)];
            if (!state.stop && state.generated < max_new_tokens) {
                seq_indices.push_back(seq);
            }
        }
        if (batch_policy == "context_bucketed" && seq_indices.size() > 1) {
            int64_t selected_block_count = std::numeric_limits<int64_t>::max();
            for (const int64_t seq_index : seq_indices) {
                const auto& state = states[static_cast<size_t>(seq_index)];
                const int64_t position = state.prompt_len + state.generated - 1;
                const int64_t count = std::max<int64_t>(
                    1,
                    (position + 1 + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
                selected_block_count = std::min<int64_t>(selected_block_count, count);
            }
            std::vector<int64_t> bucketed;
            bucketed.reserve(seq_indices.size());
            for (const int64_t seq_index : seq_indices) {
                const auto& state = states[static_cast<size_t>(seq_index)];
                const int64_t position = state.prompt_len + state.generated - 1;
                const int64_t count = std::max<int64_t>(
                    1,
                    (position + 1 + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
                if (count == selected_block_count) {
                    bucketed.push_back(seq_index);
                }
            }
            if (!bucketed.empty()) {
                seq_indices.swap(bucketed);
            }
        }
        const size_t active = seq_indices.size();
        if (runner->last_timing.active_batch_histogram.size() <= active) {
            runner->last_timing.active_batch_histogram.resize(active + 1, 0);
        }
        runner->last_timing.active_batch_histogram[active] += 1;
        runner->last_timing.batch_buffer_reuse = true;
        decode_embeds.resize(active * static_cast<size_t>(hidden_size));
        seq_lens.assign(active, 1);
        position_starts.assign(active, 0);
        past_lens.assign(active, 0);
        block_offsets.assign(active, 0);
        block_counts.assign(active, 1);
        allow_eos.assign(active, 0.0f);
        for (size_t row = 0; row < active; ++row) {
            const int64_t seq_index = seq_indices[row];
            const auto& state = states[static_cast<size_t>(seq_index)];
            std::memcpy(
                decode_embeds.data() + row * static_cast<size_t>(hidden_size),
                state.next_embed.data(),
                static_cast<size_t>(hidden_size) * sizeof(float));
            const int64_t position = state.prompt_len + state.generated - 1;
            position_starts[row] = position;
            past_lens[row] = position;
            block_offsets[row] = state.block_offset;
            block_counts[row] = std::max<int64_t>(
                1,
                (position + 1 + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
            allow_eos[row] = state.generated >= min_new_tokens ? 1.0f : 0.0f;
        }
        const bool use_fused_batch_decode =
            runner->paged_fused_batch_decode_enabled &&
            !(continuous_policy_uses_single_row_fast_path(batch_policy) && active == 1);
        ov::InferRequest& decode_request = use_fused_batch_decode
            ? runner->decode_request
            : request;
        if (use_fused_batch_decode) {
            runner->last_timing.batch_fused_decode_step_count += 1;
            runner->last_timing.batch_fused_decode_token_count += static_cast<int64_t>(active);
        } else {
            runner->last_timing.batch_single_decode_step_count += 1;
            runner->last_timing.batch_single_decode_token_count += static_cast<int64_t>(active);
            if (runner->paged_fused_batch_decode_enabled &&
                continuous_policy_uses_single_row_fast_path(batch_policy) &&
                active == 1) {
                runner->last_timing.batch_fused_decode_active1_bypass_count += 1;
            }
        }
        double decode_bind_ms = 0.0;
        measure_ms(decode_bind_ms, [&]() {
            bind_paged_batch_step_inputs(
                decode_request,
                decode_embeds.data(),
                seq_lens,
                hidden_size,
                position_starts,
                past_lens,
                runner->paged_kv_block_size,
                tts_pad_embed,
                allow_eos,
                block_offsets,
                block_counts,
                position_ids,
                past_lens_buffer,
                subsequence_begins,
                block_indices,
                block_indices_begins,
                allow_eos_buffer,
                beam_idx_buffer,
                score_aggregation_window_buffer);
        });
        runner->last_timing.tensor_bind_ms += decode_bind_ms;
        runner->last_timing.codegen_bind_step_ms.push_back(decode_bind_ms);
        double decode_ms = 0.0;
        measure_ms(decode_ms, [&]() {
            decode_request.infer();
        });
        runner->last_timing.codegen_infer_ms += decode_ms;
        runner->last_timing.codegen_decode_infer_ms += decode_ms;
        runner->last_timing.codegen_decode_count += 1;
        runner->last_timing.codegen_decode_step_ms.push_back(decode_ms);
        record_request_profile(
            runner,
            use_fused_batch_decode
                ? "codegen_paged_kv_decode_batch_fused"
                : "codegen_paged_kv_decode_batch",
            decode_request);
        auto first_output_tensor = decode_request.get_output_tensor(0);
        if (use_fused_batch_decode) {
            auto codes_tensor = decode_request.get_output_tensor(1);
            auto embed_tensor = decode_request.get_output_tensor(2);
            double output_read_ms = 0.0;
            HostTensorView<int64_t> first_code_view;
            HostTensorView<int64_t> codes_view;
            FloatTensorView embed_view;
            measure_ms(output_read_ms, [&]() {
                first_code_view = tensor_to_host_view<int64_t>(
                    runner,
                    first_output_tensor,
                    "continuous fused batch first_code");
                codes_view = tensor_to_host_view<int64_t>(
                    runner,
                    codes_tensor,
                    "continuous fused batch codes");
                embed_view = tensor_to_host_f32_view(
                    runner,
                    embed_tensor,
                    "continuous fused batch frame_embed");
            });
            runner->last_timing.subcode_output_read_ms += output_read_ms;
            runner->last_timing.subcode_output_read_step_ms.push_back(output_read_ms);
            if (!first_code_view.data || first_code_view.size < active) {
                throw std::runtime_error("native continuous fused batch first_code output is too small");
            }
            if (!codes_view.data || codes_view.size < active * static_cast<size_t>(num_code_groups)) {
                throw std::runtime_error("native continuous fused batch codes output is too small");
            }
            if (!embed_view.data || embed_view.size < active * static_cast<size_t>(hidden_size)) {
                throw std::runtime_error("native continuous fused batch frame_embed output is too small");
            }
            first_codes.assign(first_code_view.data, first_code_view.data + active);
            frame_codes.assign(codes_view.data, codes_view.data + active * static_cast<size_t>(num_code_groups));
            next_embeds.assign(embed_view.data, embed_view.data + active * static_cast<size_t>(hidden_size));
        } else {
            auto last_hidden_tensor = decode_request.get_output_tensor(1);
            select_first_codes(first_output_tensor, seq_indices, first_codes);
            const bool batch_subcode = env_enabled("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_SUBCODE", false);
            if (batch_subcode || active == 1) {
                run_paged_split_subcode_batch(
                    runner,
                    last_hidden_tensor,
                    first_codes,
                    tts_pad_embed,
                    hidden_size,
                    num_code_groups,
                    frame_codes,
                    next_embeds);
            } else {
                const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, "continuous decode last_hidden");
                if (!hidden_view.data || hidden_view.size < active * static_cast<size_t>(hidden_size)) {
                    throw std::runtime_error("native continuous batch decode hidden output is invalid");
                }
                frame_codes.assign(active * static_cast<size_t>(num_code_groups), 0);
                next_embeds.assign(active * static_cast<size_t>(hidden_size), 0.0f);
                for (size_t row = 0; row < active; ++row) {
                    std::vector<float> row_hidden(
                        hidden_view.data + row * static_cast<size_t>(hidden_size),
                        hidden_view.data + (row + 1) * static_cast<size_t>(hidden_size));
                    ov::Tensor row_hidden_tensor(
                        ov::element::f32,
                        ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                        row_hidden.data());
                    std::vector<int64_t> row_first_code{first_codes[row]};
                    std::vector<int64_t> row_codes;
                    std::vector<float> row_next_embed;
                    run_paged_split_subcode_batch(
                        runner,
                        row_hidden_tensor,
                        row_first_code,
                        tts_pad_embed,
                        hidden_size,
                        num_code_groups,
                        row_codes,
                        row_next_embed);
                    if (row_codes.size() >= static_cast<size_t>(num_code_groups)) {
                        std::memcpy(
                            frame_codes.data() + row * static_cast<size_t>(num_code_groups),
                            row_codes.data(),
                            static_cast<size_t>(num_code_groups) * sizeof(int64_t));
                    }
                    if (row_next_embed.size() >= static_cast<size_t>(hidden_size)) {
                        std::memcpy(
                            next_embeds.data() + row * static_cast<size_t>(hidden_size),
                            row_next_embed.data(),
                            static_cast<size_t>(hidden_size) * sizeof(float));
                    }
                }
            }
        }
        for (size_t row = 0; row < active; ++row) {
            const int64_t seq_index = seq_indices[row];
            auto& state = states[static_cast<size_t>(seq_index)];
            const int64_t first_code = first_codes[row];
            if (first_code == eos_token_id && state.generated >= min_new_tokens) {
                state.stop = true;
                continue;
            }
            record_frame_codes(seq_index, row);
            std::memcpy(
                state.next_embed.data(),
                next_embeds.data() + row * static_cast<size_t>(hidden_size),
                static_cast<size_t>(hidden_size) * sizeof(float));
            mark_sequence_frame(seq_index);
        }
    }

    runner->last_timing.total_ms = elapsed_ms_since(started);
    if (elapsed_ms) {
        *elapsed_ms = runner->last_timing.total_ms;
    }
}

void online_batch_reset_impl(NativeCodegen* runner, int64_t max_cache_blocks) {
    if (!runner || !runner->paged_kv_enabled || !runner->paged_split_subcode) {
        throw std::runtime_error("online continuous batching requires paged-KV split-subcode runner");
    }
    if (max_cache_blocks <= 0) {
        throw std::runtime_error("online continuous batching max_cache_blocks must be positive");
    }
    runner->online_batch_ready = true;
    runner->online_batch_next_id = 1;
    runner->online_batch_hidden_size = 0;
    runner->online_batch_vocab_size = 0;
    runner->online_batch_num_code_groups = 0;
    runner->online_batch_eos_token_id = 0;
    runner->online_batch_cache_block_capacity = max_cache_blocks;
    runner->online_batch_next_block_offset = 0;
    runner->online_batch_total_requests = 0;
    runner->online_batch_finished_requests = 0;
    runner->online_batch_tts_pad_embed.clear();
    runner->online_batch_sequences.clear();
    runner->online_batch_pending_ids.clear();
    runner->online_batch_free_blocks.clear();
    runner->last_timing = NativeCodegen::RunTiming{};
    runner->codegen_trace.clear();
    runner->online_scheduler = env_string("QWEN3_TTS_OV_NATIVE_SCHEDULER", runner->online_scheduler);
    if (runner->online_scheduler != "legacy" && runner->online_scheduler != "layered") {
        throw std::runtime_error("QWEN3_TTS_OV_NATIVE_SCHEDULER must be legacy or layered");
    }
    runner->prefill_mode = env_string(
        "QWEN3_TTS_OV_NATIVE_PREFILL_MODE",
        env_enabled("QWEN3_TTS_OV_NATIVE_BATCH_PREFILL", false) ? "dynamic_ragged" : runner->prefill_mode);
    if (runner->prefill_mode == "dynamic") {
        runner->prefill_mode = "dynamic_ragged";
    }
    if (
        runner->prefill_mode != "serial" &&
        runner->prefill_mode != "dynamic_ragged" &&
        runner->prefill_mode != "bucketed_padded") {
        throw std::runtime_error("QWEN3_TTS_OV_NATIVE_PREFILL_MODE must be serial, dynamic_ragged, or bucketed_padded");
    }
    runner->prefill_seq_buckets = env_int64_list(
        "QWEN3_TTS_OV_NATIVE_PREFILL_SEQ_BUCKETS",
        std::vector<int64_t>{128, 256, 512, 1024});
    runner->prefill_batch_buckets = env_int64_list(
        "QWEN3_TTS_OV_NATIVE_PREFILL_BATCH_BUCKETS",
        std::vector<int64_t>{1, 2, 4, 8});
    runner->decode_batch_buckets = env_int64_list(
        "QWEN3_TTS_OV_NATIVE_DECODE_BATCH_BUCKETS",
        std::vector<int64_t>{1, 2, 4, 8, 16});
    runner->max_num_batched_tokens = std::max<int64_t>(
        1,
        env_int64("QWEN3_TTS_OV_NATIVE_MAX_NUM_BATCHED_TOKENS", 16));
    runner->paged_kv_cache_tensors = make_paged_kv_cache_tensors(runner->prefill_model, max_cache_blocks);
    runner->paged_kv_cache_tensor_blocks = max_cache_blocks;
    bind_named_tensors(runner->prefill_request, runner->paged_kv_cache_tensors);
    bind_named_tensors(runner->online_logits_decode_request, runner->paged_kv_cache_tensors);
    if (runner->paged_fused_batch_decode_enabled) {
        bind_named_tensors(runner->decode_request, runner->paged_kv_cache_tensors);
    }
}

std::pair<int64_t, int64_t> online_allocate_blocks(NativeCodegen* runner, int64_t blocks) {
    if (!runner || blocks <= 0) {
        throw std::runtime_error("invalid online continuous batch block allocation");
    }
    for (auto it = runner->online_batch_free_blocks.begin(); it != runner->online_batch_free_blocks.end(); ++it) {
        if (it->second < blocks) {
            continue;
        }
        const int64_t offset = it->first;
        if (it->second == blocks) {
            runner->online_batch_free_blocks.erase(it);
        } else {
            it->first += blocks;
            it->second -= blocks;
        }
        return {offset, blocks};
    }
    if (runner->online_batch_next_block_offset + blocks > runner->online_batch_cache_block_capacity) {
        throw std::runtime_error("online continuous batch KV cache capacity exceeded; increase max_cache_blocks");
    }
    const int64_t offset = runner->online_batch_next_block_offset;
    runner->online_batch_next_block_offset += blocks;
    return {offset, blocks};
}

void online_release_blocks(NativeCodegen* runner, int64_t offset, int64_t blocks) {
    if (!runner || blocks <= 0 || offset < 0) {
        return;
    }
    runner->online_batch_free_blocks.push_back({offset, blocks});
    std::sort(
        runner->online_batch_free_blocks.begin(),
        runner->online_batch_free_blocks.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<int64_t, int64_t>> merged;
    for (const auto& item : runner->online_batch_free_blocks) {
        if (merged.empty() || merged.back().first + merged.back().second < item.first) {
            merged.push_back(item);
        } else {
            const int64_t end = std::max(merged.back().first + merged.back().second, item.first + item.second);
            merged.back().second = end - merged.back().first;
        }
    }
    runner->online_batch_free_blocks.swap(merged);
    while (!runner->online_batch_free_blocks.empty()) {
        auto& last = runner->online_batch_free_blocks.back();
        if (last.first + last.second != runner->online_batch_next_block_offset) {
            break;
        }
        runner->online_batch_next_block_offset = last.first;
        runner->online_batch_free_blocks.pop_back();
    }
}

int64_t online_batch_add_sequence_impl(
    NativeCodegen* runner,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    const NativeSamplingConfig& sampling) {
    if (!runner || !runner->online_batch_ready) {
        throw std::runtime_error("online continuous batch session is not initialized");
    }
    if (!sequence || !tts_pad_embed || prompt_len <= 0 || hidden_size <= 0 || max_new_tokens <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid online continuous batch sequence arguments");
    }
    if (runner->online_batch_hidden_size == 0) {
        runner->online_batch_hidden_size = hidden_size;
        runner->online_batch_vocab_size = vocab_size;
        runner->online_batch_num_code_groups = num_code_groups;
        runner->online_batch_eos_token_id = eos_token_id;
        runner->online_batch_tts_pad_embed.assign(
            tts_pad_embed,
            tts_pad_embed + static_cast<size_t>(hidden_size));
    } else if (
        runner->online_batch_hidden_size != hidden_size ||
        runner->online_batch_vocab_size != vocab_size ||
        runner->online_batch_num_code_groups != num_code_groups ||
        runner->online_batch_eos_token_id != eos_token_id) {
        throw std::runtime_error("online continuous batch requests must share hidden/vocab/codegroup/eos shape");
    }
    const int64_t capacity_tokens = prompt_len + max_new_tokens + 1;
    const int64_t blocks = std::max<int64_t>(
        1,
        (capacity_tokens + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
    const auto allocated_blocks = online_allocate_blocks(runner, blocks);
    OnlineBatchSequence item;
    item.id = runner->online_batch_next_id++;
    item.prompt_len = prompt_len;
    item.hidden_size = hidden_size;
    item.max_new_tokens = max_new_tokens;
    item.min_new_tokens = min_new_tokens;
    item.vocab_size = vocab_size;
    item.num_code_groups = num_code_groups;
    item.eos_token_id = eos_token_id;
    item.repetition_penalty = repetition_penalty;
    item.do_sample = sampling.do_sample;
    item.top_k = sampling.top_k;
    item.top_p = sampling.top_p;
    item.temperature = sampling.temperature;
    item.seed = sampling.seed;
    item.rng.seed(sampling.seed == 0 ? static_cast<uint64_t>(item.id) : sampling.seed);
    if (std::abs(repetition_penalty - 1.0f) > 1e-6f) {
        item.repeated_first_codes.assign(static_cast<size_t>(std::max<int64_t>(1, vocab_size)), 0);
    }
    item.block_offset = allocated_blocks.first;
    item.block_capacity = allocated_blocks.second;
    item.next_embed.assign(static_cast<size_t>(hidden_size), 0.0f);
    item.prompt.assign(sequence, sequence + static_cast<size_t>(prompt_len * hidden_size));
    item.queued_at_ms = 0.0;
    const int64_t id = item.id;
    runner->online_batch_sequences.emplace(id, std::move(item));
    runner->online_batch_pending_ids.push_back(id);
    runner->online_batch_total_requests += 1;
    return id;
}

struct OnlineBatchEvent {
    int64_t id = 0;
    int64_t kind = 0;  // 1=code, 2=final, 3=code+final
    std::vector<int64_t> codes;
};

std::vector<int64_t> online_select_first_codes(
    NativeCodegen* runner,
    const ov::Tensor& first_output_tensor,
    const std::vector<int64_t>& ids,
    const std::vector<int64_t>& row_offsets) {
    std::vector<int64_t> selected(ids.size(), runner->online_batch_eos_token_id);
    int64_t max_row = static_cast<int64_t>(ids.size()) - 1;
    if (!row_offsets.empty()) {
        if (row_offsets.size() != ids.size()) {
            throw std::runtime_error("online continuous batch row_offsets size mismatch");
        }
        for (const int64_t row : row_offsets) {
            max_row = std::max<int64_t>(max_row, row);
        }
    }
    if (runner->paged_seed_outputs_first_code) {
        for (const int64_t id : ids) {
            const auto it = runner->online_batch_sequences.find(id);
            if (it == runner->online_batch_sequences.end()) {
                continue;
            }
            if (it->second.do_sample || std::abs(it->second.repetition_penalty - 1.0f) > 1e-6f) {
                throw std::runtime_error(
                    "online continuous batching sampled/repetition mode requires logits seed graph; "
                    "disable top1 seed or use greedy repetition_penalty=1.0");
            }
        }
        const auto first_code_view = tensor_to_host_view<int64_t>(runner, first_output_tensor, "online first_code");
        if (!first_code_view.data || first_code_view.size <= static_cast<size_t>(max_row)) {
            throw std::runtime_error("online continuous batch first_code output is too small");
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            const int64_t row = row_offsets.empty() ? static_cast<int64_t>(i) : row_offsets[i];
            selected[i] = first_code_view.data[static_cast<size_t>(row)];
        }
        return selected;
    }
    const auto logits = tensor_to_host_f32_view(runner, first_output_tensor, "online logits");
    int64_t effective_vocab_size = runner->online_batch_vocab_size;
    const auto shape = first_output_tensor.get_shape();
    if (!shape.empty() && shape.back() > 0) {
        effective_vocab_size = static_cast<int64_t>(shape.back());
    }
    if (
        !logits.data ||
        effective_vocab_size <= 0 ||
        logits.size < static_cast<size_t>(max_row + 1) * static_cast<size_t>(effective_vocab_size)) {
        throw std::runtime_error("online continuous batch logits output is invalid");
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto it = runner->online_batch_sequences.find(ids[i]);
        if (it == runner->online_batch_sequences.end()) {
            throw std::runtime_error("online continuous batch sequence disappeared during selection");
        }
        const int64_t row = row_offsets.empty() ? static_cast<int64_t>(i) : row_offsets[i];
        NativeSamplingConfig sampling;
        sampling.do_sample = it->second.do_sample;
        sampling.top_k = it->second.top_k;
        sampling.top_p = it->second.top_p;
        sampling.temperature = it->second.temperature;
        sampling.seed = it->second.seed;
        selected[i] = select_first_code_from_logits_row(
            logits.data + static_cast<size_t>(row) * static_cast<size_t>(effective_vocab_size),
            effective_vocab_size,
            it->second.generated,
            it->second.min_new_tokens,
            it->second.eos_token_id,
            it->second.repeated_first_codes.empty() ? nullptr : &it->second.repeated_first_codes,
            it->second.repetition_penalty,
            &sampling,
            &it->second.rng);
    }
    return selected;
}

void online_record_event(
    NativeCodegen* runner,
    OnlineBatchSequence& state,
    int64_t first_code,
    const int64_t* frame_codes,
    std::vector<OnlineBatchEvent>& events) {
    OnlineBatchEvent event;
    event.id = state.id;
    event.kind = 1;
    event.codes.assign(frame_codes, frame_codes + static_cast<size_t>(state.num_code_groups));
    state.generated += 1;
    if (
        first_code >= 0 &&
        first_code != state.eos_token_id &&
        !state.repeated_first_codes.empty() &&
        static_cast<size_t>(first_code) < state.repeated_first_codes.size()) {
        state.repeated_first_codes[static_cast<size_t>(first_code)] = 1;
    }
    state.last_token_ms = 0.0;
    runner->last_timing.generated_frames += 1;
    if (state.generated >= state.max_new_tokens) {
        state.stop = true;
        event.kind = 3;
        runner->online_batch_finished_requests += 1;
    }
    events.push_back(std::move(event));
}

void online_record_final(NativeCodegen* runner, OnlineBatchSequence& state, std::vector<OnlineBatchEvent>& events) {
    if (!state.stop) {
        state.stop = true;
        runner->online_batch_finished_requests += 1;
    }
    OnlineBatchEvent event;
    event.id = state.id;
    event.kind = 2;
    events.push_back(std::move(event));
}

void online_batch_step_impl(
    NativeCodegen* runner,
    int64_t max_decode_batch,
    int64_t max_events,
    int64_t* out_ids,
    int64_t* out_kinds,
    int64_t* out_codes,
    int64_t* out_event_count,
    double* elapsed_ms) {
    if (!runner || !runner->online_batch_ready) {
        throw std::runtime_error("online continuous batch session is not initialized");
    }
    if (!out_ids || !out_kinds || !out_codes || !out_event_count || max_events <= 0) {
        throw std::runtime_error("invalid online continuous batch output buffers");
    }
    if (max_decode_batch <= 0) {
        max_decode_batch = 1;
    }
    const auto started = std::chrono::steady_clock::now();
    const std::string configured_policy = runner->continuous_batch_policy;
    runner->last_timing = NativeCodegen::RunTiming{};
    runner->last_timing.no_repeat_fast_path = true;
    runner->continuous_batch_policy =
        configured_policy.rfind("online_", 0) == 0 ? configured_policy : "online_" + configured_policy;
    const bool layered_scheduler =
        runner->online_scheduler == "layered" ||
        configured_policy == "layered_vllm" ||
        configured_policy == "online_layered_vllm";
    if (layered_scheduler) {
        runner->online_scheduler = "layered";
    }
    runner->last_timing.online_scheduler = runner->online_scheduler;
    runner->last_timing.prefill_mode = runner->prefill_mode;
    runner->last_timing.max_num_batched_tokens = runner->max_num_batched_tokens;
    std::vector<OnlineBatchEvent> events;
    events.reserve(static_cast<size_t>(max_events));
    std::vector<int64_t> ids;
    std::vector<int64_t> seq_lens;
    std::vector<int64_t> position_starts;
    std::vector<int64_t> past_lens;
    std::vector<int64_t> block_offsets;
    std::vector<int64_t> block_counts;
    std::vector<float> allow_eos;
    std::vector<int64_t> position_ids;
    std::vector<int32_t> past_lens_buffer;
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<float> allow_eos_buffer;
    std::vector<int64_t> beam_idx_buffer;
    std::vector<int32_t> score_aggregation_window_buffer;
    std::vector<int64_t> frame_codes;
    std::vector<float> next_embeds;

    const int64_t hidden_size = runner->online_batch_hidden_size;
    const int64_t num_code_groups = runner->online_batch_num_code_groups;
    const float* tts_pad_embed = runner->online_batch_tts_pad_embed.data();
    auto sequence_requires_logits_sampling = [](const OnlineBatchSequence& state) {
        return state.do_sample || std::abs(state.repetition_penalty - 1.0f) > 1e-6f;
    };
    std::string prefill_mode = runner->prefill_mode;
    if (prefill_mode.empty()) {
        prefill_mode = env_enabled("QWEN3_TTS_OV_NATIVE_BATCH_PREFILL", false) ? "dynamic_ragged" : "serial";
    }
    if (prefill_mode == "dynamic") {
        prefill_mode = "dynamic_ragged";
    }
    if (prefill_mode == "bucketed_padded") {
        throw std::runtime_error(
            "online continuous batching prefill_mode=bucketed_padded requires exported fixed BxS prefill graphs; "
            "the current native online runner only supports serial and dynamic_ragged prefill");
    }
    const bool online_batch_prefill_enabled = prefill_mode == "dynamic_ragged";
    runner->last_timing.prefill_mode = prefill_mode;
    auto ready_decode_count = [&]() {
        int64_t count = 0;
        for (const auto& item : runner->online_batch_sequences) {
            const auto& state = item.second;
            if (!state.stop && !state.pending_prefill && state.generated > 0 && state.generated < state.max_new_tokens) {
                ++count;
            }
        }
        return count;
    };
    for (const auto& item : runner->online_batch_sequences) {
        if (sequence_requires_logits_sampling(item.second)) {
            runner->last_timing.no_repeat_fast_path = false;
            break;
        }
    }
    auto erase_finished = [&]() {
        std::vector<int64_t> erase_ids;
        for (const auto& item : runner->online_batch_sequences) {
            if (item.second.stop) {
                erase_ids.push_back(item.first);
            }
        }
        for (const int64_t id : erase_ids) {
            auto it = runner->online_batch_sequences.find(id);
            if (it != runner->online_batch_sequences.end()) {
                online_release_blocks(runner, it->second.block_offset, it->second.block_capacity);
                runner->online_batch_sequences.erase(it);
            }
        }
    };

    if (hidden_size > 0 && !runner->online_batch_pending_ids.empty() && events.size() < static_cast<size_t>(max_events)) {
        const int64_t decode_ready = ready_decode_count();
        const int64_t decode_token_budget =
            layered_scheduler ? std::min<int64_t>(max_decode_batch, runner->max_num_batched_tokens) : max_decode_batch;
        int64_t requested_prefill_limit = online_batch_prefill_enabled ? max_decode_batch : 1;
        if (layered_scheduler) {
            const int64_t remaining_token_budget =
                std::max<int64_t>(0, runner->max_num_batched_tokens - std::min<int64_t>(decode_ready, decode_token_budget));
            requested_prefill_limit = std::min<int64_t>(
                requested_prefill_limit,
                bucket_limit_le(runner->prefill_batch_buckets, remaining_token_budget));
            runner->last_timing.prefill_token_budget = remaining_token_budget;
            runner->last_timing.decode_token_budget = decode_token_budget;
        }
        const int64_t limit = std::min<int64_t>(
            requested_prefill_limit,
            std::min<int64_t>(runner->online_batch_pending_ids.size(), max_events - static_cast<int64_t>(events.size())));
        ids.clear();
        ids.reserve(static_cast<size_t>(limit));
        if (layered_scheduler && limit > 0) {
            int64_t anchor_bucket = 0;
            std::deque<int64_t> kept;
            while (!runner->online_batch_pending_ids.empty()) {
                const int64_t id = runner->online_batch_pending_ids.front();
                runner->online_batch_pending_ids.pop_front();
                auto it = runner->online_batch_sequences.find(id);
                if (it == runner->online_batch_sequences.end()) {
                    continue;
                }
                const int64_t seq_bucket = bucket_ge(runner->prefill_seq_buckets, it->second.prompt_len);
                if (anchor_bucket == 0) {
                    anchor_bucket = seq_bucket;
                }
                if (seq_bucket == anchor_bucket && static_cast<int64_t>(ids.size()) < limit) {
                    ids.push_back(id);
                } else {
                    kept.push_back(id);
                }
            }
            runner->online_batch_pending_ids.swap(kept);
            runner->last_timing.prefill_seq_bucket = anchor_bucket;
            runner->last_timing.prefill_batch_bucket = bucket_ge(
                runner->prefill_batch_buckets,
                static_cast<int64_t>(ids.size()));
            runner->last_timing.prefill_seq_bucket_histogram.push_back(anchor_bucket);
            runner->last_timing.prefill_batch_bucket_histogram.push_back(runner->last_timing.prefill_batch_bucket);
        } else {
            for (int64_t i = 0; i < limit; ++i) {
                const int64_t id = runner->online_batch_pending_ids.front();
                runner->online_batch_pending_ids.pop_front();
                if (runner->online_batch_sequences.count(id)) {
                    ids.push_back(id);
                }
            }
            runner->last_timing.prefill_batch_bucket = static_cast<int64_t>(ids.size());
        }
        if (!ids.empty()) {
            runner->last_timing.batch_prefill_enabled = ids.size() > 1;
            std::vector<float> flat_prompt;
            std::vector<int64_t> row_offsets(ids.size(), 0);
            int64_t token_offset = 0;
            seq_lens.resize(ids.size());
            position_starts.assign(ids.size(), 0);
            past_lens.assign(ids.size(), 0);
            block_offsets.resize(ids.size());
            block_counts.resize(ids.size());
            allow_eos.resize(ids.size());
            for (size_t row = 0; row < ids.size(); ++row) {
                auto& state = runner->online_batch_sequences.at(ids[row]);
                seq_lens[row] = state.prompt_len;
                block_offsets[row] = state.block_offset;
                block_counts[row] = std::max<int64_t>(
                    1,
                    (state.prompt_len + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
                allow_eos[row] = state.min_new_tokens <= 0 ? 1.0f : 0.0f;
                row_offsets[row] = token_offset + state.prompt_len - 1;
                token_offset += state.prompt_len;
                flat_prompt.insert(flat_prompt.end(), state.prompt.begin(), state.prompt.end());
            }
            double bind_ms = 0.0;
            measure_ms(bind_ms, [&]() {
                bind_paged_batch_step_inputs(
                    runner->prefill_request,
                    flat_prompt.data(),
                    seq_lens,
                    hidden_size,
                    position_starts,
                    past_lens,
                    runner->paged_kv_block_size,
                    tts_pad_embed,
                    allow_eos,
                    block_offsets,
                    block_counts,
                    position_ids,
                    past_lens_buffer,
                    subsequence_begins,
                    block_indices,
                    block_indices_begins,
                    allow_eos_buffer,
                    beam_idx_buffer,
                    score_aggregation_window_buffer);
            });
            runner->last_timing.tensor_bind_ms += bind_ms;
            runner->last_timing.codegen_bind_step_ms.push_back(bind_ms);
            double prefill_ms = 0.0;
            measure_ms(prefill_ms, [&]() {
                runner->prefill_request.infer();
            });
            runner->last_timing.codegen_infer_ms += prefill_ms;
            runner->last_timing.codegen_prefill_infer_ms += prefill_ms;
            runner->last_timing.codegen_prefill_count += 1;
            runner->last_timing.codegen_prefill_step_ms.push_back(prefill_ms);
            record_request_profile(runner, "online_paged_kv_prefill_batch", runner->prefill_request);
            auto first_output_tensor = runner->prefill_request.get_output_tensor(0);
            auto last_hidden_tensor = runner->prefill_request.get_output_tensor(1);
            double select_ms = 0.0;
            std::vector<int64_t> first_codes;
            measure_ms(select_ms, [&]() {
                first_codes = online_select_first_codes(runner, first_output_tensor, ids, row_offsets);
            });
            runner->last_timing.sampling_ms += select_ms;
            runner->last_timing.codegen_sampling_step_ms.push_back(select_ms);
            const auto hidden_view = tensor_to_host_f32_view(runner, last_hidden_tensor, "online prefill last_hidden");
            if (!hidden_view.data || hidden_view.size < static_cast<size_t>(token_offset * hidden_size)) {
                throw std::runtime_error("online continuous batch prefill hidden output is invalid");
            }
            for (size_t row = 0; row < ids.size(); ++row) {
                auto& state = runner->online_batch_sequences.at(ids[row]);
                const int64_t first_code = first_codes[row];
                if (first_code == state.eos_token_id && state.generated >= state.min_new_tokens) {
                    online_record_final(runner, state, events);
                    continue;
                }
                const int64_t hidden_row = row_offsets[row];
                ov::Tensor row_hidden_tensor(
                    ov::element::f32,
                    ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                    const_cast<float*>(hidden_view.data + static_cast<size_t>(hidden_row * hidden_size)));
                std::vector<int64_t> row_first_code{first_code};
                std::vector<int64_t> row_codes;
                std::vector<float> row_next_embed;
                run_paged_split_subcode_batch(
                    runner,
                    row_hidden_tensor,
                    row_first_code,
                    tts_pad_embed,
                    hidden_size,
                    num_code_groups,
                    row_codes,
                    row_next_embed);
                if (row_codes.size() < static_cast<size_t>(num_code_groups) ||
                    row_next_embed.size() < static_cast<size_t>(hidden_size)) {
                    throw std::runtime_error("online continuous batch prefill subcode output is too small");
                }
                std::memcpy(state.next_embed.data(), row_next_embed.data(), static_cast<size_t>(hidden_size) * sizeof(float));
                state.pending_prefill = false;
                online_record_event(runner, state, first_code, row_codes.data(), events);
            }
        }
    }

    if (hidden_size > 0 && events.size() < static_cast<size_t>(max_events)) {
        ids.clear();
        const int64_t raw_decode_limit = std::min<int64_t>(
            max_decode_batch,
            max_events - static_cast<int64_t>(events.size()));
        const int64_t decode_limit = layered_scheduler
            ? std::min<int64_t>(raw_decode_limit, bucket_limit_le(runner->decode_batch_buckets, runner->max_num_batched_tokens))
            : raw_decode_limit;
        for (const auto& item : runner->online_batch_sequences) {
            const auto& state = item.second;
            if (!state.stop && !state.pending_prefill && state.generated > 0 && state.generated < state.max_new_tokens) {
                ids.push_back(item.first);
                if (static_cast<int64_t>(ids.size()) >= decode_limit) {
                    break;
                }
            }
        }
        const size_t active = ids.size();
        if (active > 0) {
            if (runner->last_timing.active_batch_histogram.size() <= active) {
                runner->last_timing.active_batch_histogram.resize(active + 1, 0);
            }
            runner->last_timing.active_batch_histogram[active] += 1;
            runner->last_timing.decode_token_budget = decode_limit;
            runner->last_timing.decode_batch_bucket = layered_scheduler
                ? bucket_ge(runner->decode_batch_buckets, static_cast<int64_t>(active))
                : static_cast<int64_t>(active);
            runner->last_timing.decode_batch_bucket_histogram.push_back(runner->last_timing.decode_batch_bucket);
            runner->last_timing.batch_buffer_reuse = true;
            std::vector<float> decode_embeds(active * static_cast<size_t>(hidden_size));
            seq_lens.assign(active, 1);
            position_starts.assign(active, 0);
            past_lens.assign(active, 0);
            block_offsets.assign(active, 0);
            block_counts.assign(active, 1);
            allow_eos.assign(active, 0.0f);
            for (size_t row = 0; row < active; ++row) {
                const auto& state = runner->online_batch_sequences.at(ids[row]);
                std::memcpy(
                    decode_embeds.data() + row * static_cast<size_t>(hidden_size),
                    state.next_embed.data(),
                    static_cast<size_t>(hidden_size) * sizeof(float));
                const int64_t position = state.prompt_len + state.generated - 1;
                position_starts[row] = position;
                past_lens[row] = position;
                block_offsets[row] = state.block_offset;
                block_counts[row] = std::max<int64_t>(
                    1,
                    (position + 1 + runner->paged_kv_block_size - 1) / runner->paged_kv_block_size);
                allow_eos[row] = state.generated >= state.min_new_tokens ? 1.0f : 0.0f;
            }
            bool active_batch_requires_logits = false;
            for (const int64_t id : ids) {
                const auto& state = runner->online_batch_sequences.at(id);
                if (sequence_requires_logits_sampling(state)) {
                    active_batch_requires_logits = true;
                    break;
                }
            }
            const bool use_fused_batch_decode =
                runner->paged_fused_batch_decode_enabled && active > 1 && !active_batch_requires_logits;
            if (use_fused_batch_decode) {
                runner->last_timing.batch_fused_decode_step_count += 1;
                runner->last_timing.batch_fused_decode_token_count += static_cast<int64_t>(active);
            } else {
                runner->last_timing.batch_single_decode_step_count += 1;
                runner->last_timing.batch_single_decode_token_count += static_cast<int64_t>(active);
                if (runner->paged_fused_batch_decode_enabled && active == 1) {
                    runner->last_timing.batch_fused_decode_active1_bypass_count += 1;
                }
                if (runner->paged_fused_batch_decode_enabled && active > 1 && active_batch_requires_logits) {
                    runner->last_timing.batch_fused_decode_logits_bypass_count += 1;
                }
            }
            ov::InferRequest& decode_request =
                use_fused_batch_decode ? runner->decode_request : runner->online_logits_decode_request;
            double bind_ms = 0.0;
            measure_ms(bind_ms, [&]() {
                bind_paged_batch_step_inputs(
                    decode_request,
                    decode_embeds.data(),
                    seq_lens,
                    hidden_size,
                    position_starts,
                    past_lens,
                    runner->paged_kv_block_size,
                    tts_pad_embed,
                    allow_eos,
                    block_offsets,
                    block_counts,
                    position_ids,
                    past_lens_buffer,
                    subsequence_begins,
                    block_indices,
                    block_indices_begins,
                    allow_eos_buffer,
                    beam_idx_buffer,
                    score_aggregation_window_buffer);
            });
            runner->last_timing.tensor_bind_ms += bind_ms;
            runner->last_timing.codegen_bind_step_ms.push_back(bind_ms);
            double decode_ms = 0.0;
            measure_ms(decode_ms, [&]() {
                decode_request.infer();
            });
            runner->last_timing.codegen_infer_ms += decode_ms;
            runner->last_timing.codegen_decode_infer_ms += decode_ms;
            runner->last_timing.codegen_decode_count += 1;
            runner->last_timing.codegen_decode_step_ms.push_back(decode_ms);
            record_request_profile(
                runner,
                use_fused_batch_decode ? "online_paged_kv_decode_batch_fused" : "online_paged_kv_decode_single",
                decode_request);
            if (use_fused_batch_decode) {
                auto first_tensor = decode_request.get_output_tensor(0);
                auto codes_tensor = decode_request.get_output_tensor(1);
                auto embed_tensor = decode_request.get_output_tensor(2);
                const auto first_view = tensor_to_host_view<int64_t>(runner, first_tensor, "online fused first_code");
                const auto codes_view = tensor_to_host_view<int64_t>(runner, codes_tensor, "online fused codes");
                const auto embed_view = tensor_to_host_f32_view(runner, embed_tensor, "online fused next_embed");
                if (!first_view.data || first_view.size < active ||
                    !codes_view.data || codes_view.size < active * static_cast<size_t>(num_code_groups) ||
                    !embed_view.data || embed_view.size < active * static_cast<size_t>(hidden_size)) {
                    throw std::runtime_error("online fused batch output is too small");
                }
                for (size_t row = 0; row < active; ++row) {
                    auto& state = runner->online_batch_sequences.at(ids[row]);
                    if (first_view.data[row] == state.eos_token_id && state.generated >= state.min_new_tokens) {
                        online_record_final(runner, state, events);
                        continue;
                    }
                    std::memcpy(
                        state.next_embed.data(),
                        embed_view.data + row * static_cast<size_t>(hidden_size),
                        static_cast<size_t>(hidden_size) * sizeof(float));
                    online_record_event(
                        runner,
                        state,
                        first_view.data[row],
                        codes_view.data + row * static_cast<size_t>(num_code_groups),
                        events);
                }
            } else {
                auto first_tensor = decode_request.get_output_tensor(0);
                auto hidden_tensor = decode_request.get_output_tensor(1);
                double select_ms = 0.0;
                std::vector<int64_t> first_codes;
                measure_ms(select_ms, [&]() {
                    first_codes = online_select_first_codes(runner, first_tensor, ids, {});
                });
                runner->last_timing.sampling_ms += select_ms;
                runner->last_timing.codegen_sampling_step_ms.push_back(select_ms);
                if (active_batch_requires_logits && active > 1) {
                    run_online_sampled_subcode(
                        runner,
                        hidden_tensor,
                        first_codes,
                        tts_pad_embed,
                        hidden_size,
                        num_code_groups,
                        frame_codes,
                        next_embeds);
                } else {
                    run_paged_split_subcode_batch(
                        runner,
                        hidden_tensor,
                        first_codes,
                        tts_pad_embed,
                        hidden_size,
                        num_code_groups,
                        frame_codes,
                        next_embeds);
                }
                for (size_t row = 0; row < active; ++row) {
                    auto& state = runner->online_batch_sequences.at(ids[row]);
                    if (first_codes[row] == state.eos_token_id && state.generated >= state.min_new_tokens) {
                        online_record_final(runner, state, events);
                        continue;
                    }
                    std::memcpy(
                        state.next_embed.data(),
                        next_embeds.data() + row * static_cast<size_t>(hidden_size),
                        static_cast<size_t>(hidden_size) * sizeof(float));
                    online_record_event(
                        runner,
                        state,
                        first_codes[row],
                        frame_codes.data() + row * static_cast<size_t>(num_code_groups),
                        events);
                }
            }
        }
    }

    erase_finished();
    const int64_t emit_count = std::min<int64_t>(events.size(), max_events);
    for (int64_t i = 0; i < emit_count; ++i) {
        out_ids[i] = events[static_cast<size_t>(i)].id;
        out_kinds[i] = events[static_cast<size_t>(i)].kind;
        int64_t* dst = out_codes + static_cast<size_t>(i * num_code_groups);
        std::fill(dst, dst + num_code_groups, -1);
        if (!events[static_cast<size_t>(i)].codes.empty()) {
            std::memcpy(dst, events[static_cast<size_t>(i)].codes.data(), static_cast<size_t>(num_code_groups) * sizeof(int64_t));
        }
    }
    *out_event_count = emit_count;
    runner->last_timing.total_ms = elapsed_ms_since(started);
    if (elapsed_ms) {
        *elapsed_ms = runner->last_timing.total_ms;
    }
}

void online_batch_cancel_impl(NativeCodegen* runner, int64_t request_id) {
    if (!runner || !runner->online_batch_ready) {
        return;
    }
    auto it = runner->online_batch_sequences.find(request_id);
    if (it != runner->online_batch_sequences.end()) {
        online_release_blocks(runner, it->second.block_offset, it->second.block_capacity);
        runner->online_batch_sequences.erase(it);
    }
    std::deque<int64_t> kept;
    while (!runner->online_batch_pending_ids.empty()) {
        const int64_t id = runner->online_batch_pending_ids.front();
        runner->online_batch_pending_ids.pop_front();
        if (id != request_id) {
            kept.push_back(id);
        }
    }
    runner->online_batch_pending_ids.swap(kept);
}

std::string online_batch_stats_json(const NativeCodegen& runner) {
    int64_t active = 0;
    int64_t pending = 0;
    int64_t active_blocks = 0;
    int64_t free_blocks = 0;
    for (const auto& item : runner.online_batch_sequences) {
        if (item.second.pending_prefill) {
            ++pending;
        } else if (!item.second.stop) {
            ++active;
        }
        if (!item.second.stop) {
            active_blocks += item.second.block_capacity;
        }
    }
    for (const auto& item : runner.online_batch_free_blocks) {
        free_blocks += item.second;
    }
    std::string json = "{";
    json += "\"ready\": ";
    json += runner.online_batch_ready ? "true" : "false";
    json += ", \"active\": " + std::to_string(active);
    json += ", \"pending\": " + std::to_string(pending);
    json += ", \"total_requests\": " + std::to_string(runner.online_batch_total_requests);
    json += ", \"finished_requests\": " + std::to_string(runner.online_batch_finished_requests);
    json += ", \"cache_block_capacity\": " + std::to_string(runner.online_batch_cache_block_capacity);
    json += ", \"cache_blocks_used\": " + std::to_string(runner.online_batch_next_block_offset);
    json += ", \"cache_blocks_active\": " + std::to_string(active_blocks);
    json += ", \"cache_blocks_free\": " + std::to_string(free_blocks);
    json += ", \"batch_subcode_compiled\": ";
    json += runner.batch_subcode_enabled ? "true" : "false";
    json += ", \"paged_fused_batch_decode_enabled\": ";
    json += runner.paged_fused_batch_decode_enabled ? "true" : "false";
    json += ", \"continuous_batch_policy\": \"" + json_escape_native(runner.continuous_batch_policy) + "\"";
    json += ", \"online_scheduler\": \"" + json_escape_native(runner.online_scheduler) + "\"";
    json += ", \"prefill_mode\": \"" + json_escape_native(runner.prefill_mode) + "\"";
    json += ", \"max_num_batched_tokens\": " + std::to_string(runner.max_num_batched_tokens);
    json += ", \"last_prefill_seq_bucket\": " + std::to_string(runner.last_timing.prefill_seq_bucket);
    json += ", \"last_prefill_batch_bucket\": " + std::to_string(runner.last_timing.prefill_batch_bucket);
    json += ", \"last_decode_batch_bucket\": " + std::to_string(runner.last_timing.decode_batch_bucket);
    json += ", \"last_prefill_token_budget\": " + std::to_string(runner.last_timing.prefill_token_budget);
    json += ", \"last_decode_token_budget\": " + std::to_string(runner.last_timing.decode_token_budget);
    json += ", \"last_batch_fused_decode_step_count\": " +
            std::to_string(runner.last_timing.batch_fused_decode_step_count);
    json += ", \"last_batch_fused_decode_token_count\": " +
            std::to_string(runner.last_timing.batch_fused_decode_token_count);
    json += ", \"last_batch_single_decode_step_count\": " +
            std::to_string(runner.last_timing.batch_single_decode_step_count);
    json += ", \"last_batch_single_decode_token_count\": " +
            std::to_string(runner.last_timing.batch_single_decode_token_count);
    json += ", \"last_batch_fused_decode_active1_bypass_count\": " +
            std::to_string(runner.last_timing.batch_fused_decode_active1_bypass_count);
    json += ", \"last_batch_fused_decode_logits_bypass_count\": " +
            std::to_string(runner.last_timing.batch_fused_decode_logits_bypass_count);
    json += ", \"last_batch_subcode_used\": ";
    json += runner.last_timing.batch_subcode_enabled ? "true" : "false";
    json += ", \"sampled_batch_subcode_policy\": \"" +
            json_escape_native(runner.last_timing.sampled_batch_subcode_policy) + "\"";
    json += ", \"sampled_batch_subcode_used\": ";
    json += runner.last_timing.sampled_batch_subcode_used ? "true" : "false";
    json += ", \"sampled_batch_subcode_verified\": ";
    json += runner.last_timing.sampled_batch_subcode_verified ? "true" : "false";
    json += ", \"sampled_batch_subcode_fallback_count\": " +
            std::to_string(runner.last_timing.sampled_batch_subcode_fallback_count);
    json += ", \"sampled_batch_subcode_mismatch_count\": " +
            std::to_string(runner.last_timing.sampled_batch_subcode_mismatch_count);
    json += ", \"sampled_batch_subcode_code_mismatch_count\": " +
            std::to_string(runner.last_timing.sampled_batch_subcode_code_mismatch_count);
    json += ", \"sampled_batch_subcode_embed_mismatch_count\": " +
            std::to_string(runner.last_timing.sampled_batch_subcode_embed_mismatch_count);
    json += ", \"sampled_batch_subcode_max_abs_diff\": " +
            std::to_string(runner.last_timing.sampled_batch_subcode_max_abs_diff);
    json += ", \"sampled_batch_subcode_fallback_reason\": \"" +
            json_escape_native(runner.last_timing.sampled_batch_subcode_fallback_reason) + "\"";
    json += ", \"sampled_subcode_parallel_rows\": ";
    json += runner.last_timing.sampled_subcode_parallel_rows ? "true" : "false";
    json += ", \"sampled_subcode_parallel_row_count\": " +
            std::to_string(runner.last_timing.sampled_subcode_parallel_row_count);
    json += ", \"last_no_repeat_fast_path\": ";
    json += runner.last_timing.no_repeat_fast_path ? "true" : "false";
    json += ", \"last_codegen_decode_count\": " + std::to_string(runner.last_timing.codegen_decode_count);
    json += ", \"last_codegen_subcode_count\": " + std::to_string(runner.last_timing.codegen_subcode_count);
    append_int_vector_json(json, "last_active_batch_histogram", runner.last_timing.active_batch_histogram);
    append_int_vector_json(json, "last_prefill_seq_bucket_histogram", runner.last_timing.prefill_seq_bucket_histogram);
    append_int_vector_json(json, "last_prefill_batch_bucket_histogram", runner.last_timing.prefill_batch_bucket_histogram);
    append_int_vector_json(json, "last_decode_batch_bucket_histogram", runner.last_timing.decode_batch_bucket_histogram);
    json += "}";
    return json;
}

void run_paged_kv_repeat_batch_impl(
    NativeCodegen* runner,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t batch_size,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    const NativeSamplingConfig& sampling,
    int64_t* out_counts,
    double* out_ttft_ms,
    double* out_last_token_ms,
    double* elapsed_ms) {
    if (!sequence || prompt_len <= 0 || hidden_size <= 0 || batch_size <= 0) {
        throw std::runtime_error("invalid native continuous repeat-batch shape argument");
    }
    std::vector<int64_t> prompt_lens(static_cast<size_t>(batch_size), prompt_len);
    std::vector<float> repeated(static_cast<size_t>(batch_size * prompt_len * hidden_size));
    for (int64_t seq = 0; seq < batch_size; ++seq) {
        std::memcpy(
            repeated.data() + static_cast<size_t>(seq * prompt_len * hidden_size),
            sequence,
            static_cast<size_t>(prompt_len * hidden_size) * sizeof(float));
    }
    run_paged_kv_sequence_batch_impl(
        runner,
        repeated.data(),
        batch_size * prompt_len,
        prompt_lens.data(),
        hidden_size,
        tts_pad_embed,
        batch_size,
        max_new_tokens,
        min_new_tokens,
        repetition_penalty,
        vocab_size,
        num_code_groups,
        eos_token_id,
        sampling,
        out_counts,
        out_ttft_ms,
        out_last_token_ms,
        elapsed_ms);
}

void run_unroll4_statefulmask_impl(
    NativeCodegen* runner,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t* out_codes,
    int64_t* out_count,
    double* elapsed_ms,
    FrameCallback callback,
    void* user_data) {
    if (!runner || !sequence || !tts_pad_embed || !out_count || (!out_codes && !callback)) {
        throw std::runtime_error("invalid null pointer passed to native codegen");
    }
    if (prompt_len <= 0 || hidden_size <= 0 || max_new_tokens <= 0 || vocab_size <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid native codegen shape argument");
    }
    const int64_t unroll = runner->unroll;
    const auto started = std::chrono::steady_clock::now();
    *out_count = 0;
    runner->last_timing = NativeCodegen::RunTiming{};
    runner->last_timing.buffer_reuse = env_enabled("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE", true);

    runner->prefill_request.reset_state();
    runner->decode_request.reset_state();
    const bool prefill_uses_repetition = compiled_model_has_input(runner->prefill_model, "repeated_mask");
    const bool decode_uses_repetition = compiled_model_has_input(runner->decode_model, "repetition_penalty");
    const bool decode_has_repeated_mask_input = compiled_model_has_input(runner->decode_model, "repeated_mask");
    const bool decode_has_repeated_mask_state = request_has_repeated_mask_state(runner->decode_request);
    const bool uses_repetition =
        prefill_uses_repetition || decode_uses_repetition || decode_has_repeated_mask_input || decode_has_repeated_mask_state;
    runner->last_timing.no_repeat_fast_path = !uses_repetition;

    std::vector<int64_t> local_positions;
    std::vector<int64_t> local_decode_cache_position;
    std::vector<float> local_attention_mask;
    std::vector<float> local_repeated_mask;
    std::vector<float> local_allow_eos;
    std::vector<float> local_penalty;
    std::vector<float> local_next_embed;
    auto& positions = runner->last_timing.buffer_reuse ? runner->scratch.positions : local_positions;
    auto& attention_mask = runner->last_timing.buffer_reuse ? runner->scratch.attention_mask : local_attention_mask;
    auto& repeated_mask = runner->last_timing.buffer_reuse ? runner->scratch.repeated_mask : local_repeated_mask;
    auto& allow_eos = runner->last_timing.buffer_reuse ? runner->scratch.allow_eos : local_allow_eos;
    auto& penalty = runner->last_timing.buffer_reuse ? runner->scratch.penalty : local_penalty;
    auto& next_embed = runner->last_timing.buffer_reuse ? runner->scratch.next_embed : local_next_embed;
    auto& decode_cache_position =
        runner->last_timing.buffer_reuse ? runner->scratch.decode_cache_position : local_decode_cache_position;

    measure_ms(runner->last_timing.host_prepare_ms, [&]() {
        fill_positions(positions, 0, prompt_len);
        fill_attention_mask(attention_mask, prompt_len, runner->bucket);
        if (uses_repetition) {
            repeated_mask.assign(static_cast<size_t>(vocab_size), 0.0f);
        } else {
            repeated_mask.clear();
        }
        fill_allow_eos(allow_eos, 0, min_new_tokens, unroll);
        penalty.assign(1, repetition_penalty);
        decode_cache_position.assign(1, 0);
    });
    RemoteEmbedChain remote_embed = make_remote_embed_chain(runner, hidden_size);
    runner->last_remote_embed_used = remote_embed.enabled;

    measure_ms(runner->last_timing.tensor_bind_ms, [&]() {
        runner->prefill_request.set_tensor(
            "inputs_embeds",
            ov::Tensor(ov::element::f32, ov::Shape{1, static_cast<size_t>(prompt_len), static_cast<size_t>(hidden_size)}, sequence));
        runner->prefill_request.set_tensor(
            "cache_position",
            ov::Tensor(ov::element::i64, ov::Shape{static_cast<size_t>(prompt_len)}, positions.data()));
        runner->prefill_request.set_tensor(
            "attention_mask",
            ov::Tensor(
                ov::element::f32,
                ov::Shape{1, 1, static_cast<size_t>(prompt_len), static_cast<size_t>(runner->bucket)},
                attention_mask.data()));
        runner->prefill_request.set_tensor(
            "tts_pad_embed",
            ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, tts_pad_embed));
        if (prefill_uses_repetition) {
            runner->prefill_request.set_tensor(
                "repeated_mask",
                ov::Tensor(ov::element::f32, ov::Shape{1, static_cast<size_t>(vocab_size)}, repeated_mask.data()));
        }
        runner->prefill_request.set_tensor(
            "allow_eos_steps",
            ov::Tensor(ov::element::f32, ov::Shape{static_cast<size_t>(unroll)}, allow_eos.data()));
        if (prefill_uses_repetition) {
            runner->prefill_request.set_tensor(
                "repetition_penalty",
                ov::Tensor(ov::element::f32, ov::Shape{1}, penalty.data()));
        }
        if (remote_embed.enabled) {
            runner->prefill_request.set_output_tensor(2, remote_embed.prefill_output);
        }
    });
    measure_ms(runner->last_timing.codegen_infer_ms, [&]() {
        runner->prefill_request.infer();
    });
    record_request_profile(runner, "codegen_prefill", runner->prefill_request);

    auto first_codes_tensor = runner->prefill_request.get_output_tensor(0);
    auto codes_tensor = runner->prefill_request.get_output_tensor(1);
    auto frame_embed_tensor = runner->prefill_request.get_output_tensor(2);

    const int64_t* first_codes = tensor_data<int64_t>(first_codes_tensor);
    const int64_t* codes = tensor_data<int64_t>(codes_tensor);
    if (prefill_uses_repetition) {
        auto repeated_mask_tensor = runner->prefill_request.get_output_tensor(3);
        const float* repeated_out = tensor_data<float>(repeated_mask_tensor);
        if (repeated_mask.size() != static_cast<size_t>(vocab_size)) {
            repeated_mask.resize(static_cast<size_t>(vocab_size));
        }
        std::memcpy(repeated_mask.data(), repeated_out, static_cast<size_t>(vocab_size) * sizeof(float));
    }

    int64_t generated = 0;
    bool stop = false;
    auto mark_codec_frames = [&](int64_t count) {
        if (count <= 0) {
            return;
        }
        const double token_ms = elapsed_ms_since(started);
        if (runner->last_timing.generated_frames == 0) {
            runner->last_timing.ttft_ms = token_ms;
        }
        runner->last_timing.last_token_ms = token_ms;
        runner->last_timing.generated_frames += count;
    };
    auto emit_codes = [&](const int64_t* first, const int64_t* all_codes, int64_t limit) {
        int64_t emit_count = 0;
        for (int64_t offset = 0; offset < limit; ++offset) {
            if (first[offset] == eos_token_id) {
                stop = true;
                break;
            }
            ++emit_count;
        }
        if (emit_count == 0) {
            return;
        }
        if (out_codes) {
            std::memcpy(
                out_codes + generated * num_code_groups,
                all_codes,
                static_cast<size_t>(emit_count * num_code_groups) * sizeof(int64_t));
        }
        mark_codec_frames(emit_count);
        if (callback) {
            int callback_rc = 0;
            double callback_ms = 0.0;
            measure_ms(callback_ms, [&]() {
                callback_rc = callback(all_codes, emit_count, num_code_groups, user_data);
            });
            runner->last_timing.codegen_callback_ms += callback_ms;
            runner->last_timing.callback_ms += callback_ms;
            if (callback_rc != 0) {
                throw std::runtime_error("native codegen callback requested stop");
            }
        }
        generated += emit_count;
    };

    emit_codes(first_codes, codes, std::min<int64_t>(unroll, max_new_tokens));

    if (remote_embed.enabled) {
        remote_embed.next_input = frame_embed_tensor;
    } else {
        const float* frame_embed = tensor_data<float>(frame_embed_tensor);
        next_embed.resize(static_cast<size_t>(hidden_size));
        std::memcpy(next_embed.data(), frame_embed, static_cast<size_t>(hidden_size) * sizeof(float));
    }
    bool decode_ready = false;
    bool decode_inputs_bound = false;

    while (!stop && generated < max_new_tokens) {
        if (!decode_ready) {
            copy_matching_states(runner->prefill_request, runner->decode_request);
            if (decode_has_repeated_mask_state && !set_repeated_mask_state(runner->decode_request, repeated_mask)) {
                throw std::runtime_error("decode graph does not expose repeated_mask state");
            }
            decode_ready = true;
        }

        const int64_t step = generated;
        decode_cache_position[0] = prompt_len + generated - 1;
        measure_ms(runner->last_timing.host_prepare_ms, [&]() {
            fill_allow_eos(allow_eos, step, min_new_tokens, unroll);
        });

        measure_ms(runner->last_timing.tensor_bind_ms, [&]() {
            const bool bind_static_inputs = !runner->last_timing.buffer_reuse || !decode_inputs_bound;
            if (remote_embed.enabled) {
                runner->decode_request.set_tensor("inputs_embeds", remote_embed.next_input);
                runner->decode_request.set_output_tensor(2, remote_embed.decode_outputs[remote_embed.next_decode_output]);
            } else if (bind_static_inputs) {
                runner->decode_request.set_tensor(
                    "inputs_embeds",
                    ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, next_embed.data()));
            }
            if (bind_static_inputs) {
                runner->decode_request.set_tensor(
                    "cache_position",
                    ov::Tensor(ov::element::i64, ov::Shape{1}, decode_cache_position.data()));
                runner->decode_request.set_tensor(
                    "tts_pad_embed",
                    ov::Tensor(ov::element::f32, ov::Shape{1, 1, static_cast<size_t>(hidden_size)}, tts_pad_embed));
                if (decode_has_repeated_mask_input) {
                    runner->decode_request.set_tensor(
                        "repeated_mask",
                        ov::Tensor(ov::element::f32, ov::Shape{1, static_cast<size_t>(vocab_size)}, repeated_mask.data()));
                }
                runner->decode_request.set_tensor(
                    "allow_eos_steps",
                    ov::Tensor(ov::element::f32, ov::Shape{static_cast<size_t>(unroll)}, allow_eos.data()));
                if (decode_uses_repetition) {
                    runner->decode_request.set_tensor(
                        "repetition_penalty",
                        ov::Tensor(ov::element::f32, ov::Shape{1}, penalty.data()));
                }
                decode_inputs_bound = runner->last_timing.buffer_reuse;
            }
        });
        measure_ms(runner->last_timing.codegen_infer_ms, [&]() {
            runner->decode_request.infer();
        });
        record_request_profile(runner, "codegen_decode", runner->decode_request);

        first_codes_tensor = runner->decode_request.get_output_tensor(0);
        codes_tensor = runner->decode_request.get_output_tensor(1);
        frame_embed_tensor = runner->decode_request.get_output_tensor(2);
        first_codes = tensor_data<int64_t>(first_codes_tensor);
        codes = tensor_data<int64_t>(codes_tensor);

        const int64_t remaining = max_new_tokens - generated;
        emit_codes(first_codes, codes, std::min<int64_t>(unroll, remaining));
        if (decode_has_repeated_mask_input) {
            auto repeated_mask_tensor = runner->decode_request.get_output_tensor(3);
            const float* repeated_out = tensor_data<float>(repeated_mask_tensor);
            if (repeated_mask.size() != static_cast<size_t>(vocab_size)) {
                repeated_mask.resize(static_cast<size_t>(vocab_size));
            }
            std::memcpy(repeated_mask.data(), repeated_out, static_cast<size_t>(vocab_size) * sizeof(float));
        }
        if (remote_embed.enabled) {
            remote_embed.next_input = frame_embed_tensor;
            remote_embed.next_decode_output = 1 - remote_embed.next_decode_output;
        } else {
            const float* frame_embed = tensor_data<float>(frame_embed_tensor);
            if (next_embed.size() != static_cast<size_t>(hidden_size)) {
                next_embed.resize(static_cast<size_t>(hidden_size));
            }
            std::memcpy(next_embed.data(), frame_embed, static_cast<size_t>(hidden_size) * sizeof(float));
        }
    }

    *out_count = generated;
    if (generated == 0) {
        throw std::runtime_error("native codegen stopped before producing any codec token");
    }
    const auto stopped = std::chrono::steady_clock::now();
    runner->last_timing.total_ms = ov::genai::PerfMetrics::get_microsec(stopped - started) / 1000.0;
    if (elapsed_ms) {
        *elapsed_ms = runner->last_timing.total_ms;
    }
}

class Qwen3TTSGenAIPipeline {
public:
    Qwen3TTSGenAIPipeline(
        const char* prefill_xml,
        const char* decode_xml,
        const char* device,
        const char* cache_dir,
        const char* cache_mode) {
        if (!prefill_xml || !decode_xml || !device) {
            throw std::runtime_error("prefill_xml, decode_xml, and device are required");
        }
        const auto config = compile_config_for_device(cache_dir, cache_mode, device);
        m_runner.profile_enabled = env_enabled("QWEN3_TTS_OV_NATIVE_PERF_COUNT", false);
        m_runner.bucket = parse_cache_bucket(prefill_xml);
        if (m_runner.bucket <= 0) {
            throw std::runtime_error("failed to parse cache bucket from prefill graph path");
        }
        m_runner.unroll = parse_unroll_steps(prefill_xml);
        if (m_runner.unroll <= 1) {
            throw std::runtime_error("native codegen requires a fused cache unroll graph");
        }
        m_runner.prefill_model = m_runner.core.compile_model(prefill_xml, device, config);
        m_runner.decode_model = m_runner.core.compile_model(decode_xml, device, config);
        m_runner.prefill_request = m_runner.prefill_model.create_infer_request();
        m_runner.decode_request = m_runner.decode_model.create_infer_request();
    }

    Qwen3TTSGenAIPipeline(
        const char* paged_seed_xml,
        const char* device,
        const char* cache_dir,
        const char* cache_mode,
        const char* kv_cache_precision,
        int64_t kv_heads,
        int64_t kv_block_size,
        int64_t kv_head_dim,
        const char* subcode_xml = nullptr) {
        if (!paged_seed_xml || !device) {
            throw std::runtime_error("paged_seed_xml and device are required");
        }
        const bool split_subcode = subcode_xml && std::strlen(subcode_xml) > 0;
        if (kv_heads <= 0 || kv_block_size <= 0 || kv_head_dim <= 0) {
            throw std::runtime_error("invalid paged KV cache shape");
        }
        auto config = compile_config_for_device(cache_dir, cache_mode, device);
        const std::string precision = kv_cache_precision && std::strlen(kv_cache_precision) > 0 ? kv_cache_precision : "f16";
        const char* cache_input_precision_env = std::getenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION");
        const std::string cache_input_precision =
            cache_input_precision_env && std::strlen(cache_input_precision_env) > 0
                ? cache_input_precision_env
                : "f32";
        config[ov::hint::kv_cache_precision.name()] = parse_element_type(precision);
        m_runner.profile_enabled = env_enabled("QWEN3_TTS_OV_NATIVE_PERF_COUNT", false);
        m_runner.paged_kv_enabled = true;
        m_runner.paged_kv_block_size = kv_block_size;
        m_runner.paged_kv_heads = kv_heads;
        m_runner.paged_kv_head_dim = kv_head_dim;
        m_runner.paged_kv_precision = precision;
        m_runner.paged_kv_cache_input_precision = cache_input_precision;
        m_runner.paged_split_subcode = split_subcode;
        m_runner.subcode_outputs_next_embed =
            split_subcode && lower_text(std::filesystem::path(subcode_xml).filename().string()).find("next_embed") != std::string::npos;
        const char* continuous_policy_env = std::getenv("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_POLICY");
        std::string compile_continuous_policy = normalize_continuous_batch_policy(
            continuous_policy_env && std::strlen(continuous_policy_env) > 0 ? continuous_policy_env : "raw_fused");
        if (!valid_continuous_batch_policy(compile_continuous_policy)) {
            throw std::runtime_error(continuous_batch_policy_error());
        }
        m_runner.continuous_batch_policy = compile_continuous_policy;
        const char* batch_unroll_env = std::getenv("QWEN3_TTS_OV_NATIVE_BATCH_DECODE_UNROLL");
        if (batch_unroll_env && std::strlen(batch_unroll_env) > 0) {
            std::string batch_unroll_text = lower_text(batch_unroll_env);
            if (batch_unroll_text != "auto") {
                m_runner.batch_decode_unroll_requested = std::max<int64_t>(1, std::stoll(batch_unroll_text));
            }
        }
        m_runner.bucket = 0;
        m_runner.unroll = parse_unroll_steps(paged_seed_xml);
        auto model = convert_paged_kv_seed_model(
            m_runner.core,
            paged_seed_xml,
            kv_heads,
            kv_block_size,
            kv_head_dim,
            parse_element_type(cache_input_precision));
        auto decode_model = model->clone();
        const char* fused_batch_decode_env = std::getenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_FUSED_BATCH_DECODE_XML");
        if (fused_batch_decode_env && std::strlen(fused_batch_decode_env) > 0) {
            decode_model = convert_paged_kv_seed_model(
                m_runner.core,
                fused_batch_decode_env,
                kv_heads,
                kv_block_size,
                kv_head_dim,
                parse_element_type(cache_input_precision));
            m_runner.paged_fused_batch_decode_enabled = true;
            m_runner.paged_fused_batch_decode_graph = fused_batch_decode_env;
        }
        const bool want_static_decode =
            !m_runner.paged_fused_batch_decode_enabled &&
            env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE", false);
        const char* static_blocks_env = std::getenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_BLOCKS");
        int64_t static_blocks = 128;
        if (static_blocks_env && std::strlen(static_blocks_env) > 0) {
            static_blocks = std::max<int64_t>(1, std::stoll(static_blocks_env));
        }
        const char* static_mode_env = std::getenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE");
        std::string static_mode = lower_text(static_mode_env && std::strlen(static_mode_env) > 0 ? static_mode_env : "minimal");
        if (static_mode != "minimal" && static_mode != "full") {
            throw std::runtime_error("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE must be minimal or full");
        }
        m_runner.paged_static_decode_requested = want_static_decode;
        m_runner.paged_static_decode_failure.clear();
        m_runner.paged_static_decode_block_capacity = static_blocks;
        m_runner.paged_static_decode_mode = want_static_decode ? static_mode : "dynamic";
        bool static_decode_ready = false;
        if (want_static_decode) {
            try {
                static_decode_ready = reshape_paged_decode_model(
                    decode_model,
                    kv_heads,
                    kv_block_size,
                    kv_head_dim,
                    static_blocks,
                    static_mode);
            } catch (const std::exception& exc) {
                static_decode_ready = false;
                m_runner.paged_static_decode_failure = std::string("reshape_failed: ") + exc.what();
                if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                    std::cerr << "paged static decode reshape failed: " << exc.what() << std::endl;
                }
            }
            if (!static_decode_ready && m_runner.paged_static_decode_failure.empty()) {
                m_runner.paged_static_decode_failure = "reshape_not_applicable";
            }
        }
        m_runner.prefill_model = m_runner.core.compile_model(model, device, config);
        m_runner.paged_seed_outputs_first_code =
            split_subcode && compiled_model_first_output_is_first_code(m_runner.prefill_model);
        m_runner.prefill_request = m_runner.prefill_model.create_infer_request();
        m_runner.online_logits_decode_request = m_runner.prefill_model.create_infer_request();
        if (m_runner.paged_fused_batch_decode_enabled || static_decode_ready) {
            try {
                m_runner.decode_model = m_runner.core.compile_model(decode_model, device, config);
                m_runner.decode_request = m_runner.decode_model.create_infer_request();
                m_runner.paged_static_decode_enabled = static_decode_ready;
                m_runner.paged_static_decode_failure.clear();
            } catch (const std::exception& exc) {
                m_runner.paged_fused_batch_decode_enabled = false;
                m_runner.paged_static_decode_enabled = false;
                m_runner.paged_static_decode_mode = "dynamic";
                m_runner.paged_static_decode_failure = std::string("compile_failed: ") + exc.what();
                if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                    std::cerr << "paged static decode compile failed: " << exc.what() << std::endl;
                }
            }
        } else {
            m_runner.paged_static_decode_mode = "dynamic";
            if (want_static_decode && m_runner.paged_static_decode_failure.empty()) {
                m_runner.paged_static_decode_failure = "static_decode_disabled";
            }
        }
        if (split_subcode) {
            const char* subcode_device_env = std::getenv("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE");
            const std::string subcode_device =
                subcode_device_env && std::strlen(subcode_device_env) > 0 ? subcode_device_env : device;
            auto subcode_config = compile_config_for_device(cache_dir, cache_mode, subcode_device.c_str());
            auto single_subcode_model = m_runner.core.read_model(subcode_xml);
            m_runner.subcode_model = m_runner.core.compile_model(single_subcode_model, subcode_device, subcode_config);
            m_runner.subcode_request = m_runner.subcode_model.create_infer_request();

            const bool want_batch_subcode =
                env_enabled("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH", false) &&
                (
                    env_enabled("QWEN3_TTS_OV_NATIVE_BATCH_PREFILL_SUBCODE", false) ||
                    env_enabled("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_SUBCODE", false) ||
                    sampled_batch_subcode_policy_from_env() != "off");
            if (want_batch_subcode) {
                const char* batch_subcode_xml_env = std::getenv("QWEN3_TTS_OV_NATIVE_BATCH_SUBCODE_XML");
                const bool has_batch_subcode_xml =
                    batch_subcode_xml_env && std::strlen(batch_subcode_xml_env) > 0;
                if (has_batch_subcode_xml) {
                    try {
                        auto batch_subcode_model = m_runner.core.read_model(batch_subcode_xml_env);
                        m_runner.batch_subcode_model =
                            m_runner.core.compile_model(batch_subcode_model, subcode_device, subcode_config);
                        m_runner.batch_subcode_request = m_runner.batch_subcode_model.create_infer_request();
                        m_runner.batch_subcode_enabled = true;
                    } catch (const std::exception& exc) {
                        m_runner.batch_subcode_enabled = false;
                        if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                            std::cerr << "true batch subcode compile failed: " << exc.what() << std::endl;
                        }
                    }
                } else if (env_enabled("QWEN3_TTS_OV_NATIVE_LEGACY_RESHAPE_BATCH_SUBCODE", false)) {
                    try {
                        auto subcode_model = m_runner.core.read_model(subcode_xml);
                        int64_t subcode_hidden = 2048;
                        for (const auto& input : subcode_model->inputs()) {
                            std::string name;
                            try {
                                name = input.get_any_name();
                            } catch (...) {
                                continue;
                            }
                            if (name != "past_hidden") {
                                continue;
                            }
                            const auto shape = input.get_partial_shape();
                            if (!shape.rank().is_dynamic() && shape.rank().get_length() > 0) {
                                const auto last = shape[shape.rank().get_length() - 1];
                                if (last.is_static()) {
                                    subcode_hidden = last.get_length();
                                }
                            }
                        }
                        std::map<std::string, ov::PartialShape> subcode_shapes;
                        subcode_shapes["past_hidden"] = ov::PartialShape{ov::Dimension::dynamic(), 1, subcode_hidden};
                        subcode_shapes["first_code"] = ov::PartialShape{ov::Dimension::dynamic(), 1};
                        subcode_model->reshape(subcode_shapes);
                        m_runner.batch_subcode_model =
                            m_runner.core.compile_model(subcode_model, subcode_device, subcode_config);
                        m_runner.batch_subcode_request = m_runner.batch_subcode_model.create_infer_request();
                        m_runner.batch_subcode_enabled = true;
                    } catch (const std::exception& exc) {
                        m_runner.batch_subcode_enabled = false;
                        if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                            std::cerr << "legacy reshaped subcode dynamic batch compile failed: " << exc.what()
                                      << std::endl;
                        }
                    }
                } else {
                    m_runner.batch_subcode_enabled = false;
                    if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                        std::cerr
                            << "batch subcode disabled: QWEN3_TTS_OV_NATIVE_BATCH_SUBCODE_XML is not set"
                            << std::endl;
                    }
                }
            }
        }
    }

    NativeCodegen& runner() {
        return m_runner;
    }

    ov::genai::SpeechGenerationConfig make_generation_config(
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t eos_token_id) const {
        if (max_new_tokens <= 0 || min_new_tokens < 0) {
            throw std::runtime_error("invalid generation token limits");
        }
        ov::genai::SpeechGenerationConfig config;
        config.max_new_tokens = static_cast<size_t>(max_new_tokens);
        config.min_new_tokens = static_cast<size_t>(min_new_tokens);
        config.repetition_penalty = repetition_penalty;
        config.eos_token_id = eos_token_id;
        config.do_sample = false;
        config.validate();
        return config;
    }

    void configure_voice_design_prompt(
        const char* tokenizer_dir,
        const char* text_embedding_xml,
        const char* codec_embedding_xml,
        const char* device,
        const char* cache_dir,
        const char* cache_mode,
        int64_t tts_bos_token_id,
        int64_t tts_eos_token_id,
        int64_t tts_pad_token_id,
        int64_t codec_pad_id,
        int64_t codec_bos_id) {
        if (!tokenizer_dir || !text_embedding_xml || !codec_embedding_xml || !device) {
            throw std::runtime_error("tokenizer_dir, text_embedding_xml, codec_embedding_xml, and device are required");
        }
        const auto config = compile_config_for_device(cache_dir, cache_mode, device);
        m_runner.tokenizer = std::make_unique<ov::genai::Tokenizer>(std::filesystem::path(tokenizer_dir));
        m_runner.text_embedding_model = m_runner.core.compile_model(text_embedding_xml, device, config);
        m_runner.codec_embedding_model = m_runner.core.compile_model(codec_embedding_xml, device, config);
        m_runner.text_embedding_request = m_runner.text_embedding_model.create_infer_request();
        m_runner.codec_embedding_request = m_runner.codec_embedding_model.create_infer_request();
        m_runner.prompt_embedding_cache.clear();
        m_runner.prompt_embedding_cache_hits = 0;
        m_runner.prompt_embedding_cache_misses = 0;
        m_runner.tts_bos_token_id = tts_bos_token_id;
        m_runner.tts_eos_token_id = tts_eos_token_id;
        m_runner.tts_pad_token_id = tts_pad_token_id;
        m_runner.codec_pad_id = codec_pad_id;
        m_runner.codec_bos_id = codec_bos_id;
        m_runner.voice_design_prompt_ready = true;
    }

    PromptEmbeds build_voice_design_prompt(
        const std::string& text,
        const std::string& instruct,
        const int64_t* codec_prefill,
        int64_t codec_prefill_len,
        int64_t max_prompt_tokens) {
        if (!m_runner.voice_design_prompt_ready || !m_runner.tokenizer) {
            throw std::runtime_error("native voice design prompt pipeline is not configured");
        }
        if (!codec_prefill || codec_prefill_len <= 0) {
            throw std::runtime_error("codec_prefill is required");
        }
        if (max_prompt_tokens <= 0) {
            throw std::runtime_error("max_prompt_tokens must be positive");
        }

        std::vector<int64_t> input_ids = tokenize_to_ids(*m_runner.tokenizer, build_assistant_text(text));
        std::vector<int64_t> instruct_ids;
        if (!instruct.empty()) {
            instruct_ids = tokenize_to_ids(*m_runner.tokenizer, build_instruct_text(instruct));
        }
        if (static_cast<int64_t>(input_ids.size()) > max_prompt_tokens) {
            throw std::runtime_error("text prompt exceeds max_prompt_tokens");
        }
        if (!instruct_ids.empty() && static_cast<int64_t>(instruct_ids.size()) > max_prompt_tokens) {
            throw std::runtime_error("instruct prompt exceeds max_prompt_tokens");
        }
        if (input_ids.size() < 8) {
            throw std::runtime_error("assistant prompt is too short for Qwen3-TTS layout");
        }

        int64_t hidden_size = 0;
        std::vector<float> output;
        if (!instruct_ids.empty()) {
            auto instruct_embed = embed_ids_cached(
                &m_runner,
                m_runner.text_embedding_request,
                "text_embedding",
                "text_embedding:instruct",
                instruct_ids,
                &hidden_size);
            output.insert(output.end(), instruct_embed.begin(), instruct_embed.end());
        }

        auto tts_special = embed_ids_cached(
            &m_runner,
            m_runner.text_embedding_request,
            "text_embedding",
            "text_embedding:tts_special",
            {m_runner.tts_bos_token_id, m_runner.tts_eos_token_id, m_runner.tts_pad_token_id},
            &hidden_size);
        const float* tts_bos = tts_special.data();
        const float* tts_eos = tts_special.data() + static_cast<size_t>(hidden_size);
        const float* tts_pad = tts_special.data() + static_cast<size_t>(2 * hidden_size);

        std::vector<int64_t> codec_ids(codec_prefill, codec_prefill + codec_prefill_len);
        codec_ids.push_back(m_runner.codec_pad_id);
        codec_ids.push_back(m_runner.codec_bos_id);
        auto codec_embed = embed_ids_cached(
            &m_runner,
            m_runner.codec_embedding_request,
            "codec_embedding",
            "codec_embedding:prefill",
            codec_ids,
            &hidden_size);
        auto input_embed = embed_ids(&m_runner, m_runner.text_embedding_request, "text_embedding", input_ids, &hidden_size);

        append_embedding_slice(output, input_embed, 0, 3, hidden_size);
        const int64_t prefill_count = static_cast<int64_t>(codec_ids.size()) - 1;
        for (int64_t i = 0; i < prefill_count; ++i) {
            const float* text_side = (i == prefill_count - 1) ? tts_bos : tts_pad;
            const float* codec_side = codec_embed.data() + static_cast<size_t>(i * hidden_size);
            append_sum_token(output, text_side, codec_side, hidden_size);
        }

        const int64_t input_len = static_cast<int64_t>(input_ids.size());
        const int64_t text_body_start = 3;
        const int64_t text_body_count = input_len - 8;
        for (int64_t i = 0; i < text_body_count; ++i) {
            const float* text_token = input_embed.data() + static_cast<size_t>((text_body_start + i) * hidden_size);
            const float* codec_pad = codec_embed.data() + static_cast<size_t>((codec_ids.size() - 2) * hidden_size);
            append_sum_token(output, text_token, codec_pad, hidden_size);
        }
        const float* codec_pad = codec_embed.data() + static_cast<size_t>((codec_ids.size() - 2) * hidden_size);
        append_sum_token(output, tts_eos, codec_pad, hidden_size);
        const float* codec_bos = codec_embed.data() + static_cast<size_t>((codec_ids.size() - 1) * hidden_size);
        append_sum_token(output, tts_pad, codec_bos, hidden_size);

        PromptEmbeds result;
        result.sequence = std::move(output);
        result.tts_pad_embed.assign(tts_pad, tts_pad + hidden_size);
        result.hidden_size = hidden_size;
        result.prompt_len = static_cast<int64_t>(result.sequence.size()) / hidden_size;
        return result;
    }

    void generate_codes(
        const float* sequence,
        int64_t prompt_len,
        int64_t hidden_size,
        const float* tts_pad_embed,
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t vocab_size,
        int64_t num_code_groups,
        int64_t eos_token_id,
        int64_t do_sample,
        int64_t top_k,
        float top_p,
        float temperature,
        uint64_t seed,
        int64_t* out_codes,
        int64_t* out_count,
        double* elapsed_ms,
        FrameCallback callback,
        void* user_data) {
        const auto config = make_generation_config(max_new_tokens, min_new_tokens, repetition_penalty, eos_token_id);
        NativeSamplingConfig sampling;
        sampling.do_sample = do_sample != 0;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.temperature = temperature;
        sampling.seed = seed;
        if (m_runner.paged_kv_enabled) {
            run_paged_kv_impl(
                &m_runner,
                sequence,
                prompt_len,
                hidden_size,
                tts_pad_embed,
                static_cast<int64_t>(config.max_new_tokens),
                static_cast<int64_t>(config.min_new_tokens),
                config.repetition_penalty,
                vocab_size,
                num_code_groups,
                config.eos_token_id,
                sampling,
                out_codes,
                out_count,
                elapsed_ms,
                callback,
                user_data);
            return;
        }
        if (sampling.do_sample) {
            throw std::runtime_error("native stateful bucket codegen does not support do_sample=true");
        }
        run_unroll4_statefulmask_impl(
            &m_runner,
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            static_cast<int64_t>(config.max_new_tokens),
            static_cast<int64_t>(config.min_new_tokens),
            config.repetition_penalty,
            vocab_size,
            num_code_groups,
            config.eos_token_id,
            out_codes,
            out_count,
            elapsed_ms,
            callback,
            user_data);
    }

    void generate_repeat_batch(
        const float* sequence,
        int64_t prompt_len,
        int64_t hidden_size,
        const float* tts_pad_embed,
        int64_t batch_size,
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t vocab_size,
        int64_t num_code_groups,
        int64_t eos_token_id,
        int64_t do_sample,
        int64_t top_k,
        float top_p,
        float temperature,
        uint64_t seed,
        int64_t* out_counts,
        double* out_ttft_ms,
        double* out_last_token_ms,
        double* elapsed_ms) {
        const auto config = make_generation_config(max_new_tokens, min_new_tokens, repetition_penalty, eos_token_id);
        NativeSamplingConfig sampling;
        sampling.do_sample = do_sample != 0;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.temperature = temperature;
        sampling.seed = seed;
        run_paged_kv_repeat_batch_impl(
            &m_runner,
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            batch_size,
            static_cast<int64_t>(config.max_new_tokens),
            static_cast<int64_t>(config.min_new_tokens),
            config.repetition_penalty,
            vocab_size,
            num_code_groups,
            config.eos_token_id,
            sampling,
            out_counts,
            out_ttft_ms,
            out_last_token_ms,
            elapsed_ms);
    }

    void generate_sequence_batch(
        const float* sequences,
        int64_t total_prompt_tokens,
        const int64_t* prompt_lens,
        int64_t hidden_size,
        const float* tts_pad_embed,
        int64_t batch_size,
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t vocab_size,
        int64_t num_code_groups,
        int64_t eos_token_id,
        int64_t do_sample,
        int64_t top_k,
        float top_p,
        float temperature,
        uint64_t seed,
        int64_t* out_counts,
        double* out_ttft_ms,
        double* out_last_token_ms,
        double* elapsed_ms,
        int64_t* out_codes = nullptr) {
        const auto config = make_generation_config(max_new_tokens, min_new_tokens, repetition_penalty, eos_token_id);
        NativeSamplingConfig sampling;
        sampling.do_sample = do_sample != 0;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.temperature = temperature;
        sampling.seed = seed;
        run_paged_kv_sequence_batch_impl(
            &m_runner,
            sequences,
            total_prompt_tokens,
            prompt_lens,
            hidden_size,
            tts_pad_embed,
            batch_size,
            static_cast<int64_t>(config.max_new_tokens),
            static_cast<int64_t>(config.min_new_tokens),
            config.repetition_penalty,
            vocab_size,
            num_code_groups,
            config.eos_token_id,
            sampling,
            out_counts,
            out_ttft_ms,
            out_last_token_ms,
            elapsed_ms,
            out_codes);
    }

    void set_stream_decoders(
        const char* first_decoder_xml,
        const char* steady_decoder_xml,
        const char* device,
        const char* cache_dir,
        const char* cache_mode,
        int64_t first_context_frames,
        int64_t first_chunk_frames,
        int64_t steady_context_frames,
        int64_t steady_chunk_frames,
        int64_t num_code_groups,
        int64_t decode_upsample_rate) {
        if (!first_decoder_xml || !steady_decoder_xml || !device) {
            throw std::runtime_error("first_decoder_xml, steady_decoder_xml, and device are required");
        }
        if (first_chunk_frames <= 0 || steady_chunk_frames <= 0 || num_code_groups <= 0 || decode_upsample_rate <= 0) {
            throw std::runtime_error("invalid stream decoder configuration");
        }
        const auto config = compile_config_for_device(cache_dir, cache_mode, device);
        m_runner.first_stream_decoder_model = m_runner.core.compile_model(first_decoder_xml, device, config);
        m_runner.steady_stream_decoder_model = m_runner.core.compile_model(steady_decoder_xml, device, config);
        m_runner.first_stream_decoder_request = m_runner.first_stream_decoder_model.create_infer_request();
        m_runner.steady_stream_decoder_request = m_runner.steady_stream_decoder_model.create_infer_request();
        const auto first_input_spec =
            stream_decoder_input_spec(m_runner.first_stream_decoder_model, first_context_frames + first_chunk_frames);
        const auto steady_input_spec =
            stream_decoder_input_spec(m_runner.steady_stream_decoder_model, steady_context_frames + steady_chunk_frames);
        m_runner.first_context_frames = first_context_frames;
        m_runner.first_chunk_frames = first_chunk_frames;
        m_runner.first_stream_decoder_static_input = first_input_spec.first;
        m_runner.first_stream_decoder_input_frames = first_input_spec.second;
        m_runner.steady_context_frames = steady_context_frames;
        m_runner.steady_chunk_frames = steady_chunk_frames;
        m_runner.steady_stream_decoder_static_input = steady_input_spec.first;
        m_runner.steady_stream_decoder_input_frames = steady_input_spec.second;
        m_runner.stream_num_code_groups = num_code_groups;
        m_runner.decode_upsample_rate = decode_upsample_rate;
        m_runner.stream_decoders_ready = true;
    }

    void stream_audio(
        const float* sequence,
        int64_t prompt_len,
        int64_t hidden_size,
        const float* tts_pad_embed,
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t vocab_size,
        int64_t num_code_groups,
        int64_t eos_token_id,
        int64_t do_sample,
        int64_t top_k,
        float top_p,
        float temperature,
        uint64_t seed,
        const int64_t* prefix_codes,
        int64_t prefix_frames,
        AudioCallback callback,
        void* user_data,
        int64_t* out_count,
        double* elapsed_ms) {
        if (!m_runner.stream_decoders_ready) {
            throw std::runtime_error("native stream decoders are not configured");
        }
        if (!callback) {
            throw std::runtime_error("native audio stream callback is required");
        }
        if (prefix_frames < 0) {
            throw std::runtime_error("prefix_frames must not be negative");
        }
        if (prefix_frames > 0 && !prefix_codes) {
            throw std::runtime_error("prefix_codes is required when prefix_frames > 0");
        }
        if (num_code_groups != m_runner.stream_num_code_groups) {
            throw std::runtime_error("audio stream code group count does not match configured stream decoder");
        }

        const auto stream_started = std::chrono::steady_clock::now();
        NativeAudioStreamState state;
        state.runner = &m_runner;
        state.callback = callback;
        state.user_data = user_data;
        state.async_decode = env_enabled("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", true);
        m_runner.last_async_decode = state.async_decode;
        state.codegen_started = std::chrono::steady_clock::now();
        if (prefix_frames > 0) {
            const size_t prefix_values = static_cast<size_t>(prefix_frames * num_code_groups);
            state.all_codes.assign(prefix_codes, prefix_codes + prefix_values);
            state.emitted_frames = prefix_frames;
        }
        state.all_codes.reserve(static_cast<size_t>((prefix_frames + max_new_tokens) * num_code_groups));

        start_decode_worker(state);
        try {
            generate_codes(
                sequence,
                prompt_len,
                hidden_size,
                tts_pad_embed,
                max_new_tokens,
                min_new_tokens,
                repetition_penalty,
                vocab_size,
                num_code_groups,
                eos_token_id,
                do_sample,
                top_k,
                top_p,
                temperature,
                seed,
                nullptr,
                out_count,
                elapsed_ms,
                native_audio_frame_callback,
                &state);

            decode_and_emit_audio(state, true);
            finish_decode_worker(state);
            m_runner.last_timing.total_ms = elapsed_ms_since(stream_started);
            if (elapsed_ms) {
                *elapsed_ms = m_runner.last_timing.total_ms;
            }
        } catch (...) {
            cancel_decode_worker(state);
            throw;
        }
        m_perf_metrics.num_generated_samples += static_cast<size_t>(std::max<int64_t>(0, *out_count) * m_runner.decode_upsample_rate);
    }

    void stream_voice_design_audio(
        const char* text,
        const char* instruct,
        const int64_t* codec_prefill,
        int64_t codec_prefill_len,
        int64_t max_prompt_tokens,
        int64_t max_new_tokens,
        int64_t min_new_tokens,
        float repetition_penalty,
        int64_t vocab_size,
        int64_t num_code_groups,
        int64_t eos_token_id,
        int64_t do_sample,
        int64_t top_k,
        float top_p,
        float temperature,
        uint64_t seed,
        AudioCallback callback,
        void* user_data,
        int64_t* out_count,
        double* elapsed_ms) {
        if (!text) {
            throw std::runtime_error("text is required");
        }
        PromptEmbeds prompt = build_voice_design_prompt(
            text,
            instruct ? instruct : "",
            codec_prefill,
            codec_prefill_len,
            max_prompt_tokens);
        stream_audio(
            prompt.sequence.data(),
            prompt.prompt_len,
            prompt.hidden_size,
            prompt.tts_pad_embed.data(),
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            nullptr,
            0,
            callback,
            user_data,
            out_count,
            elapsed_ms);
    }

    ov::genai::SpeechGenerationPerfMetrics get_performance_metrics() const {
        return m_perf_metrics;
    }

    std::string get_profile_json() const {
        return native_profile_json(m_runner);
    }

    std::string get_timing_json() const {
        return native_timing_json(m_runner);
    }

    void reset_profile() {
        m_runner.profile_ops.clear();
    }

    void release_run_buffers() {
        m_runner.paged_kv_cache_tensors.clear();
        m_runner.paged_kv_cache_tensors.shrink_to_fit();
        m_runner.paged_kv_cache_tensor_blocks = 0;
        m_runner.last_remote_embed_used = false;
        m_runner.scratch.positions.clear();
        m_runner.scratch.decode_cache_position.clear();
        m_runner.scratch.attention_mask.clear();
        m_runner.scratch.repeated_mask.clear();
        m_runner.scratch.allow_eos.clear();
        m_runner.scratch.penalty.clear();
        m_runner.scratch.next_embed.clear();
        if (static_cast<bool>(m_runner.prefill_model)) {
            m_runner.prefill_request = m_runner.prefill_model.create_infer_request();
            m_runner.online_logits_decode_request = m_runner.prefill_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.decode_model)) {
            m_runner.decode_request = m_runner.decode_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.text_embedding_model)) {
            m_runner.text_embedding_request = m_runner.text_embedding_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.codec_embedding_model)) {
            m_runner.codec_embedding_request = m_runner.codec_embedding_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.subcode_model)) {
            m_runner.subcode_request = m_runner.subcode_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.first_stream_decoder_model)) {
            m_runner.first_stream_decoder_request = m_runner.first_stream_decoder_model.create_infer_request();
        }
        if (static_cast<bool>(m_runner.steady_stream_decoder_model)) {
            m_runner.steady_stream_decoder_request = m_runner.steady_stream_decoder_model.create_infer_request();
        }
    }

private:
    NativeCodegen m_runner;
    ov::genai::SpeechGenerationPerfMetrics m_perf_metrics;
};

}  // namespace

extern "C" {

int qwen3_tts_codegen_create(
    const char* prefill_xml,
    const char* decode_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    void** out_handle,
    char** error) {
    return guarded(error, [&]() {
        if (!prefill_xml || !decode_xml || !device || !out_handle) {
            throw std::runtime_error("prefill_xml, decode_xml, device, and out_handle are required");
        }
        auto pipeline = std::make_unique<Qwen3TTSGenAIPipeline>(prefill_xml, decode_xml, device, cache_dir, cache_mode);
        *out_handle = pipeline.release();
    });
}

int qwen3_tts_codegen_create_paged_kv(
    const char* paged_seed_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    const char* kv_cache_precision,
    int64_t kv_heads,
    int64_t kv_block_size,
    int64_t kv_head_dim,
    void** out_handle,
    char** error) {
    return guarded(error, [&]() {
        if (!paged_seed_xml || !device || !out_handle) {
            throw std::runtime_error("paged_seed_xml, device, and out_handle are required");
        }
        auto pipeline = std::make_unique<Qwen3TTSGenAIPipeline>(
            paged_seed_xml,
            device,
            cache_dir,
            cache_mode,
            kv_cache_precision,
            kv_heads,
            kv_block_size,
            kv_head_dim);
        *out_handle = pipeline.release();
    });
}

int qwen3_tts_codegen_create_paged_kv_split(
    const char* paged_talker_seed_xml,
    const char* subcode_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    const char* kv_cache_precision,
    int64_t kv_heads,
    int64_t kv_block_size,
    int64_t kv_head_dim,
    void** out_handle,
    char** error) {
    return guarded(error, [&]() {
        if (!paged_talker_seed_xml || !subcode_xml || !device || !out_handle) {
            throw std::runtime_error("paged_talker_seed_xml, subcode_xml, device, and out_handle are required");
        }
        auto pipeline = std::make_unique<Qwen3TTSGenAIPipeline>(
            paged_talker_seed_xml,
            device,
            cache_dir,
            cache_mode,
            kv_cache_precision,
            kv_heads,
            kv_block_size,
            kv_head_dim,
            subcode_xml);
        *out_handle = pipeline.release();
    });
}

int qwen3_tts_codegen_destroy(void* handle, char** error) {
    return guarded(error, [&]() {
        delete static_cast<Qwen3TTSGenAIPipeline*>(handle);
    });
}

int qwen3_tts_codegen_run_unroll4_statefulmask(
    void* handle,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int64_t* out_codes,
    int64_t* out_count,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->generate_codes(
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            out_codes,
            out_count,
            elapsed_ms,
            nullptr,
            nullptr);
    });
}

int qwen3_tts_codegen_run_paged_kv_repeat_batch(
    void* handle,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t batch_size,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int64_t* out_counts,
    double* out_ttft_ms,
    double* out_last_token_ms,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->generate_repeat_batch(
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            batch_size,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            out_counts,
            out_ttft_ms,
            out_last_token_ms,
            elapsed_ms);
    });
}

int qwen3_tts_codegen_run_paged_kv_sequence_batch(
    void* handle,
    const float* sequences,
    int64_t total_prompt_tokens,
    const int64_t* prompt_lens,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t batch_size,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int64_t* out_counts,
    double* out_ttft_ms,
    double* out_last_token_ms,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->generate_sequence_batch(
            sequences,
            total_prompt_tokens,
            prompt_lens,
            hidden_size,
            tts_pad_embed,
            batch_size,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            out_counts,
            out_ttft_ms,
            out_last_token_ms,
            elapsed_ms);
    });
}

int qwen3_tts_codegen_run_paged_kv_sequence_batch_codes(
    void* handle,
    const float* sequences,
    int64_t total_prompt_tokens,
    const int64_t* prompt_lens,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t batch_size,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int64_t* out_counts,
    int64_t* out_codes,
    double* out_ttft_ms,
    double* out_last_token_ms,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        if (!out_codes) {
            throw std::runtime_error("out_codes is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->generate_sequence_batch(
            sequences,
            total_prompt_tokens,
            prompt_lens,
            hidden_size,
            tts_pad_embed,
            batch_size,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            out_counts,
            out_ttft_ms,
            out_last_token_ms,
            elapsed_ms,
            out_codes);
    });
}

int qwen3_tts_codegen_online_batch_reset(
    void* handle,
    int64_t max_cache_blocks,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        online_batch_reset_impl(&static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner(), max_cache_blocks);
    });
}

int qwen3_tts_codegen_online_batch_add_sequence(
    void* handle,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int64_t* out_request_id,
    char** error) {
    return guarded(error, [&]() {
        if (!handle || !out_request_id) {
            throw std::runtime_error("native pipeline handle and out_request_id are required");
        }
        NativeSamplingConfig sampling;
        sampling.do_sample = do_sample != 0;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.temperature = temperature;
        sampling.seed = seed;
        *out_request_id = online_batch_add_sequence_impl(
            &static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner(),
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            sampling);
    });
}

int qwen3_tts_codegen_online_batch_step(
    void* handle,
    int64_t max_decode_batch,
    int64_t max_events,
    int64_t* out_request_ids,
    int64_t* out_event_kinds,
    int64_t* out_codes,
    int64_t* out_event_count,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        online_batch_step_impl(
            &static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner(),
            max_decode_batch,
            max_events,
            out_request_ids,
            out_event_kinds,
            out_codes,
            out_event_count,
            elapsed_ms);
    });
}

int qwen3_tts_codegen_online_batch_cancel(
    void* handle,
    int64_t request_id,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        online_batch_cancel_impl(&static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner(), request_id);
    });
}

int qwen3_tts_codegen_online_batch_get_stats_json(
    void* handle,
    char** out_json,
    char** error) {
    return guarded(error, [&]() {
        if (!handle || !out_json) {
            throw std::runtime_error("native pipeline handle and out_json are required");
        }
        auto json = online_batch_stats_json(static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner());
        char* buffer = static_cast<char*>(std::malloc(json.size() + 1));
        if (!buffer) {
            throw std::runtime_error("failed to allocate stats json buffer");
        }
        std::memcpy(buffer, json.c_str(), json.size() + 1);
        *out_json = buffer;
    });
}

int qwen3_tts_codegen_run_unroll4_statefulmask_stream(
    void* handle,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    FrameCallback callback,
    void* user_data,
    int64_t* out_count,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->generate_codes(
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            nullptr,
            out_count,
            elapsed_ms,
            callback,
            user_data);
    });
}

int qwen3_tts_codegen_set_stream_decoders(
    void* handle,
    const char* first_decoder_xml,
    const char* steady_decoder_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    int64_t first_context_frames,
    int64_t first_chunk_frames,
    int64_t steady_context_frames,
    int64_t steady_chunk_frames,
    int64_t num_code_groups,
    int64_t decode_upsample_rate,
    char** error) {
    return guarded(error, [&]() {
        if (!handle || !first_decoder_xml || !steady_decoder_xml || !device) {
            throw std::runtime_error("handle, first_decoder_xml, steady_decoder_xml, and device are required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->set_stream_decoders(
            first_decoder_xml,
            steady_decoder_xml,
            device,
            cache_dir,
            cache_mode,
            first_context_frames,
            first_chunk_frames,
            steady_context_frames,
            steady_chunk_frames,
            num_code_groups,
            decode_upsample_rate);
    });
}

int qwen3_tts_codegen_configure_voice_design_prompt(
    void* handle,
    const char* tokenizer_dir,
    const char* text_embedding_xml,
    const char* codec_embedding_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    int64_t tts_bos_token_id,
    int64_t tts_eos_token_id,
    int64_t tts_pad_token_id,
    int64_t codec_pad_id,
    int64_t codec_bos_id,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->configure_voice_design_prompt(
            tokenizer_dir,
            text_embedding_xml,
            codec_embedding_xml,
            device,
            cache_dir,
            cache_mode,
            tts_bos_token_id,
            tts_eos_token_id,
            tts_pad_token_id,
            codec_pad_id,
            codec_bos_id);
    });
}

int qwen3_tts_codegen_run_unroll4_statefulmask_audio_stream(
    void* handle,
    const float* sequence,
    int64_t prompt_len,
    int64_t hidden_size,
    const float* tts_pad_embed,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    const int64_t* prefix_codes,
    int64_t prefix_frames,
    AudioCallback callback,
    void* user_data,
    int64_t* out_count,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->stream_audio(
            sequence,
            prompt_len,
            hidden_size,
            tts_pad_embed,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            prefix_codes,
            prefix_frames,
            callback,
            user_data,
            out_count,
            elapsed_ms);
    });
}

int qwen3_tts_codegen_run_voice_design_audio_stream(
    void* handle,
    const char* text,
    const char* instruct,
    const int64_t* codec_prefill,
    int64_t codec_prefill_len,
    int64_t max_prompt_tokens,
    int64_t max_new_tokens,
    int64_t min_new_tokens,
    float repetition_penalty,
    int64_t vocab_size,
    int64_t num_code_groups,
    int64_t eos_token_id,
    int64_t do_sample,
    int64_t top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    AudioCallback callback,
    void* user_data,
    int64_t* out_count,
    double* elapsed_ms,
    char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("native pipeline handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->stream_voice_design_audio(
            text,
            instruct,
            codec_prefill,
            codec_prefill_len,
            max_prompt_tokens,
            max_new_tokens,
            min_new_tokens,
            repetition_penalty,
            vocab_size,
            num_code_groups,
            eos_token_id,
            do_sample,
            top_k,
            top_p,
            temperature,
            seed,
            callback,
            user_data,
            out_count,
            elapsed_ms);
    });
}

int qwen3_tts_codegen_get_last_remote_embed_used(void* handle, int64_t* out_used, char** error) {
    return guarded(error, [&]() {
        if (!handle || !out_used) {
            throw std::runtime_error("handle and out_used are required");
        }
        *out_used = static_cast<Qwen3TTSGenAIPipeline*>(handle)->runner().last_remote_embed_used ? 1 : 0;
    });
}

int qwen3_tts_codegen_reset_profile(void* handle, char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->reset_profile();
    });
}

int qwen3_tts_codegen_get_profile_json(void* handle, char** out_json, char** error) {
    return guarded(error, [&]() {
        if (!handle || !out_json) {
            throw std::runtime_error("handle and out_json are required");
        }
        *out_json = dup_cstr(static_cast<Qwen3TTSGenAIPipeline*>(handle)->get_profile_json());
        if (!*out_json) {
            throw std::runtime_error("failed to allocate native profile JSON");
        }
    });
}

int qwen3_tts_codegen_get_last_timing_json(void* handle, char** out_json, char** error) {
    return guarded(error, [&]() {
        if (!handle || !out_json) {
            throw std::runtime_error("handle and out_json are required");
        }
        *out_json = dup_cstr(static_cast<Qwen3TTSGenAIPipeline*>(handle)->get_timing_json());
        if (!*out_json) {
            throw std::runtime_error("failed to allocate native timing JSON");
        }
    });
}

int qwen3_tts_codegen_release_run_buffers(void* handle, char** error) {
    return guarded(error, [&]() {
        if (!handle) {
            throw std::runtime_error("handle is required");
        }
        static_cast<Qwen3TTSGenAIPipeline*>(handle)->release_run_buffers();
    });
}

void qwen3_tts_codegen_free_error(char* error) {
    std::free(error);
}

}  // extern "C"
