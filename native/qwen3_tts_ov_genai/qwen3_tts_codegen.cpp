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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
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
#include <vector>

namespace {

constexpr float NEG_INF = -3.4028234663852886e38f;

struct NamedTensor {
    std::string name;
    ov::Tensor tensor;
};

struct NativeCodegen {
    ov::Core core;
    ov::CompiledModel prefill_model;
    ov::CompiledModel decode_model;
    ov::CompiledModel text_embedding_model;
    ov::CompiledModel codec_embedding_model;
    ov::CompiledModel subcode_model;
    ov::CompiledModel first_stream_decoder_model;
    ov::CompiledModel steady_stream_decoder_model;
    ov::InferRequest prefill_request;
    ov::InferRequest decode_request;
    ov::InferRequest text_embedding_request;
    ov::InferRequest codec_embedding_request;
    ov::InferRequest subcode_request;
    ov::InferRequest first_stream_decoder_request;
    ov::InferRequest steady_stream_decoder_request;
    std::unique_ptr<ov::genai::Tokenizer> tokenizer;
    int64_t bucket = 0;
    int64_t unroll = 4;
    bool stream_decoders_ready = false;
    bool voice_design_prompt_ready = false;
    bool profile_enabled = false;
    bool paged_kv_enabled = false;
    bool paged_split_subcode = false;
    bool paged_static_decode_enabled = false;
    int64_t paged_kv_block_size = 8;
    int64_t paged_kv_heads = 16;
    int64_t paged_kv_head_dim = 128;
    int64_t paged_static_decode_block_capacity = 0;
    std::string paged_static_decode_mode = "dynamic";
    std::string paged_kv_precision = "f16";
    std::string paged_kv_cache_input_precision = "f32";
    int64_t paged_kv_cache_tensor_blocks = 0;
    std::vector<NamedTensor> paged_kv_cache_tensors;
    int64_t tts_bos_token_id = 0;
    int64_t tts_eos_token_id = 0;
    int64_t tts_pad_token_id = 0;
    int64_t codec_pad_id = 0;
    int64_t codec_bos_id = 0;
    int64_t first_context_frames = 0;
    int64_t first_chunk_frames = 8;
    int64_t steady_context_frames = 25;
    int64_t steady_chunk_frames = 12;
    int64_t stream_num_code_groups = 16;
    int64_t decode_upsample_rate = 2000;
    bool last_remote_embed_used = false;
    struct ProfileEntry {
        std::string label;
        std::string node_name;
        std::string node_type;
        std::string exec_type;
        double real_time_ms = 0.0;
        double cpu_time_ms = 0.0;
        int64_t count = 0;
    };
    struct RunTiming {
        bool buffer_reuse = true;
        bool no_repeat_fast_path = false;
        bool kv_cache_tensor_reuse = false;
        double host_prepare_ms = 0.0;
        double tensor_bind_ms = 0.0;
        double codegen_infer_ms = 0.0;
        double sampling_ms = 0.0;
        double decode_infer_ms = 0.0;
        double callback_ms = 0.0;
        double codegen_callback_ms = 0.0;
        double decode_callback_ms = 0.0;
        double total_ms = 0.0;
    };
    struct ScratchBuffers {
        std::vector<int64_t> positions;
        std::vector<int64_t> decode_cache_position;
        std::vector<float> attention_mask;
        std::vector<float> repeated_mask;
        std::vector<float> allow_eos;
        std::vector<float> penalty;
        std::vector<float> next_embed;
    };
    std::unordered_map<std::string, ProfileEntry> profile_ops;
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
    for (const auto& input : model->inputs()) {
        std::string name;
        try {
            name = input.get_any_name();
        } catch (...) {
            continue;
        }
        if (name != "tts_pad_embed") {
            continue;
        }
        const auto pshape = input.get_partial_shape();
        if (pshape.rank().is_dynamic() || pshape.rank().get_length() == 0) {
            continue;
        }
        const auto dim = pshape[pshape.rank().get_length() - 1];
        if (dim.is_static()) {
            return dim.get_length();
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

bool env_enabled(const char* name, bool default_value) {
    return enabled_env(name, default_value);
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
    json += "  ]\n";
    json += "}";
    return json;
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
    json += ", \"paged_static_decode_enabled\": ";
    json += runner.paged_static_decode_enabled ? "true" : "false";
    json += ", \"paged_static_decode_mode\": \"" + json_escape_native(runner.paged_static_decode_mode) + "\"";
    json += ", \"host_prepare_ms\": " + std::to_string(item.host_prepare_ms);
    json += ", \"tensor_bind_ms\": " + std::to_string(item.tensor_bind_ms);
    json += ", \"codegen_infer_ms\": " + std::to_string(item.codegen_infer_ms);
    json += ", \"sampling_ms\": " + std::to_string(item.sampling_ms);
    json += ", \"decode_infer_ms\": " + std::to_string(item.decode_infer_ms);
    json += ", \"callback_ms\": " + std::to_string(item.callback_ms);
    json += ", \"codegen_callback_ms\": " + std::to_string(item.codegen_callback_ms);
    json += ", \"decode_callback_ms\": " + std::to_string(item.decode_callback_ms);
    json += ", \"total_ms\": " + std::to_string(item.total_ms);
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

struct NativeDecodeTask {
    std::vector<int64_t> window;
    std::vector<int64_t> new_codes;
    int64_t new_frames = 0;
    int64_t num_code_groups = 0;
    int64_t chunk_index = 0;
    bool is_final = false;
    double codegen_ms = 0.0;
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
        execute_decode_task(state, task);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.worker_error) {
            std::rethrow_exception(state.worker_error);
        }
        state.decode_queue.push_back(std::move(task));
    }
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
    std::vector<int64_t>& beam_idx_buffer) {
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

int64_t select_first_code_from_logits(
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
    const float* logits_base = tensor_data<float>(logits_tensor);
    if (!logits_base || size == 0 || vocab_size <= 0 || static_cast<size_t>(vocab_size) > size) {
        throw std::runtime_error("invalid paged split logits tensor");
    }
    const size_t rows = std::max<size_t>(1, size / static_cast<size_t>(vocab_size));
    const float* logits = logits_base + (rows - 1) * static_cast<size_t>(vocab_size);
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

const float* last_hidden_vector_ptr(const ov::Tensor& last_hidden_tensor, int64_t hidden_size) {
    const float* base = tensor_data<float>(last_hidden_tensor);
    const size_t size = last_hidden_tensor.get_size();
    if (!base || hidden_size <= 0 || size < static_cast<size_t>(hidden_size)) {
        throw std::runtime_error("invalid paged split last_hidden tensor");
    }
    const size_t rows = std::max<size_t>(1, size / static_cast<size_t>(hidden_size));
    return base + (rows - 1) * static_cast<size_t>(hidden_size);
}

void run_paged_split_subcode_step(
    NativeCodegen* runner,
    const ov::Tensor& last_hidden_tensor,
    int64_t first_code,
    const float* tts_pad_embed,
    int64_t hidden_size,
    int64_t num_code_groups,
    std::vector<int64_t>& frame_codes,
    std::vector<float>& next_embed) {
    if (!runner || !tts_pad_embed || hidden_size <= 0 || num_code_groups <= 0) {
        throw std::runtime_error("invalid paged split subcode arguments");
    }
    if (!runner->paged_split_subcode) {
        throw std::runtime_error("paged split subcode graph is not configured");
    }
    auto& request = runner->subcode_request;
    const float* last_hidden = last_hidden_vector_ptr(last_hidden_tensor, hidden_size);
    int64_t first_code_buffer[1] = {first_code};
    measure_ms(runner->last_timing.tensor_bind_ms, [&]() {
        request.set_tensor(
            "past_hidden",
            ov::Tensor(
                ov::element::f32,
                ov::Shape{1, 1, static_cast<size_t>(hidden_size)},
                const_cast<float*>(last_hidden)));
        request.set_tensor(
            "first_code",
            ov::Tensor(ov::element::i64, ov::Shape{1, 1}, first_code_buffer));
    });
    measure_ms(runner->last_timing.codegen_infer_ms, [&]() {
        request.infer();
    });
    record_request_profile(runner, "codegen_paged_kv_subcode", request);
    auto codes_tensor = request.get_output_tensor(0);
    auto sum_embed_tensor = request.get_output_tensor(1);
    const int64_t* codes = tensor_data<int64_t>(codes_tensor);
    const float* sum_embed = tensor_data<float>(sum_embed_tensor);
    if (static_cast<int64_t>(codes_tensor.get_size()) < num_code_groups) {
        throw std::runtime_error("paged split subcode graph returned too few codec groups");
    }
    frame_codes.assign(codes, codes + num_code_groups);
    next_embed.resize(static_cast<size_t>(hidden_size));
    for (int64_t i = 0; i < hidden_size; ++i) {
        next_embed[static_cast<size_t>(i)] =
            sum_embed[static_cast<size_t>(i)] + tts_pad_embed[static_cast<size_t>(i)];
    }
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
    std::vector<uint8_t> repeated_first_codes;
    if (use_repetition_penalty) {
        repeated_first_codes.assign(static_cast<size_t>(vocab_size), 0);
        runner->last_timing.no_repeat_fast_path = false;
    }
    std::mt19937_64 rng(
        sampling.seed
            ? sampling.seed
            : static_cast<uint64_t>(
                  std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    RemoteEmbedChain remote_embed = make_paged_remote_embed_chain(runner, hidden_size);
    if (runner->paged_split_subcode) {
        remote_embed.enabled = false;
    }
    if (use_static_decode && !env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_REMOTE_EMBED", false)) {
        remote_embed.enabled = false;
    }
    runner->last_remote_embed_used = remote_embed.enabled;
    const int64_t graph_unroll = std::max<int64_t>(1, runner->unroll);

    int64_t generated = 0;
    bool stop = false;
    auto make_allow_eos_values = [&](int64_t frame_count) {
        std::vector<float> values(static_cast<size_t>(std::max<int64_t>(1, frame_count)), 0.0f);
        for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
            values[static_cast<size_t>(i)] = (generated + i) >= min_new_tokens ? 1.0f : 0.0f;
        }
        return values;
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
        measure_ms(runner->last_timing.tensor_bind_ms, [&]() {
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
                beam_idx_buffer);
            if (remote_input && static_cast<bool>(*remote_input)) {
                step_request.set_tensor("inputs_embeds", *remote_input);
            }
            if (remote_output && static_cast<bool>(*remote_output)) {
                step_request.set_output_tensor(2, *remote_output);
            }
        });
        measure_ms(runner->last_timing.codegen_infer_ms, [&]() {
            step_request.infer();
        });
        record_request_profile(runner, step_request_label, step_request);
        if (runner->paged_split_subcode) {
            auto logits_tensor = step_request.get_output_tensor(0);
            auto last_hidden_tensor = step_request.get_output_tensor(1);
            int64_t first_code = 0;
            measure_ms(runner->last_timing.sampling_ms, [&]() {
                first_code = select_first_code_from_logits(
                    logits_tensor,
                    generated,
                    min_new_tokens,
                    vocab_size,
                    eos_token_id,
                    use_repetition_penalty ? &repeated_first_codes : nullptr,
                    repetition_penalty,
                    &sampling,
                    &rng);
            });
            if (first_code == eos_token_id && generated >= min_new_tokens) {
                stop = true;
                return;
            }
            std::vector<int64_t> frame_codes;
            run_paged_split_subcode_step(
                runner,
                last_hidden_tensor,
                first_code,
                tts_pad_embed,
                hidden_size,
                num_code_groups,
                frame_codes,
                next_embed);
            emit_many(&first_code, frame_codes.data(), 1);
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
            emit_many(tensor_data<int64_t>(first_codes_tensor), tensor_data<int64_t>(codes_tensor), output_frames);
            if (!remote_output || !static_cast<bool>(*remote_output)) {
                const float* frame_embed = tensor_data<float>(frame_embed_tensor);
                std::memcpy(next_embed.data(), frame_embed, static_cast<size_t>(hidden_size) * sizeof(float));
            }
        }
    };

    ov::Tensor* current_remote_output = remote_embed.enabled ? &remote_embed.decode_outputs[0] : nullptr;
    const ov::Tensor* next_remote_input = nullptr;
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
        const auto config = compile_config(cache_dir, cache_mode);
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
        auto config = compile_config(cache_dir, cache_mode);
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
        const bool want_static_decode = env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE", false);
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
                if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                    std::cerr << "paged static decode reshape failed: " << exc.what() << std::endl;
                }
            }
        }
        m_runner.prefill_model = m_runner.core.compile_model(model, device, config);
        m_runner.prefill_request = m_runner.prefill_model.create_infer_request();
        if (static_decode_ready) {
            try {
                m_runner.decode_model = m_runner.core.compile_model(decode_model, device, config);
                m_runner.decode_request = m_runner.decode_model.create_infer_request();
                m_runner.paged_static_decode_enabled = true;
            } catch (const std::exception& exc) {
                m_runner.paged_static_decode_enabled = false;
                m_runner.paged_static_decode_mode = "dynamic";
                if (enabled_env("QWEN3_TTS_OV_NATIVE_DEBUG_GRAPH", false)) {
                    std::cerr << "paged static decode compile failed: " << exc.what() << std::endl;
                }
            }
        } else {
            m_runner.paged_static_decode_mode = "dynamic";
        }
        if (split_subcode) {
            const char* subcode_device_env = std::getenv("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE");
            const std::string subcode_device =
                subcode_device_env && std::strlen(subcode_device_env) > 0 ? subcode_device_env : device;
            auto subcode_config = compile_config(cache_dir, cache_mode);
            m_runner.subcode_model = m_runner.core.compile_model(subcode_xml, subcode_device, subcode_config);
            m_runner.subcode_request = m_runner.subcode_model.create_infer_request();
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
        const auto config = compile_config(cache_dir, cache_mode);
        m_runner.tokenizer = std::make_unique<ov::genai::Tokenizer>(std::filesystem::path(tokenizer_dir));
        m_runner.text_embedding_model = m_runner.core.compile_model(text_embedding_xml, device, config);
        m_runner.codec_embedding_model = m_runner.core.compile_model(codec_embedding_xml, device, config);
        m_runner.text_embedding_request = m_runner.text_embedding_model.create_infer_request();
        m_runner.codec_embedding_request = m_runner.codec_embedding_model.create_infer_request();
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
            auto instruct_embed = embed_ids(&m_runner, m_runner.text_embedding_request, "text_embedding", instruct_ids, &hidden_size);
            output.insert(output.end(), instruct_embed.begin(), instruct_embed.end());
        }

        auto tts_special = embed_ids(
            &m_runner,
            m_runner.text_embedding_request,
            "text_embedding",
            {m_runner.tts_bos_token_id, m_runner.tts_eos_token_id, m_runner.tts_pad_token_id},
            &hidden_size);
        const float* tts_bos = tts_special.data();
        const float* tts_eos = tts_special.data() + static_cast<size_t>(hidden_size);
        const float* tts_pad = tts_special.data() + static_cast<size_t>(2 * hidden_size);

        std::vector<int64_t> codec_ids(codec_prefill, codec_prefill + codec_prefill_len);
        codec_ids.push_back(m_runner.codec_pad_id);
        codec_ids.push_back(m_runner.codec_bos_id);
        auto codec_embed = embed_ids(&m_runner, m_runner.codec_embedding_request, "codec_embedding", codec_ids, &hidden_size);
        auto input_embed = embed_ids(&m_runner, m_runner.text_embedding_request, "text_embedding", input_ids, &hidden_size);

        append_embedding_slice(output, input_embed, 0, 3, hidden_size);
        const int64_t prefill_count = static_cast<int64_t>(codec_ids.size()) - 1;
        for (int64_t i = 0; i < prefill_count; ++i) {
            const float* text_side = (i == prefill_count - 1) ? tts_bos : tts_pad;
            const float* codec_side = codec_embed.data() + static_cast<size_t>(i * hidden_size);
            append_sum_token(output, text_side, codec_side, hidden_size);
        }
        output.resize(output.size() - static_cast<size_t>(hidden_size));

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
        const auto config = compile_config(cache_dir, cache_mode);
        m_runner.first_stream_decoder_model = m_runner.core.compile_model(first_decoder_xml, device, config);
        m_runner.steady_stream_decoder_model = m_runner.core.compile_model(steady_decoder_xml, device, config);
        m_runner.first_stream_decoder_request = m_runner.first_stream_decoder_model.create_infer_request();
        m_runner.steady_stream_decoder_request = m_runner.steady_stream_decoder_model.create_infer_request();
        m_runner.first_context_frames = first_context_frames;
        m_runner.first_chunk_frames = first_chunk_frames;
        m_runner.steady_context_frames = steady_context_frames;
        m_runner.steady_chunk_frames = steady_chunk_frames;
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
        state.async_decode = env_enabled("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", false);
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
