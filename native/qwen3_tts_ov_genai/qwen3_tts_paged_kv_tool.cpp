#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace {

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::string compile_device;
    std::string kv_cache_precision = "f16";
    std::string kv_cache_input_precision = "f32";
    bool dummy_infer = false;
    int64_t dummy_blocks = 1;
    int64_t dummy_tokens = 1;
    bool per_layer_block_indices = false;
    bool score_outputs = false;
    bool allow_score_aggregation = true;
    bool allow_cache_rotation = false;
    bool allow_xattention = false;
    bool allow_adaptive_rkv = false;
    int64_t kv_cache_heads = 16;
    int64_t kv_cache_block_size = 8;
    int64_t kv_cache_head_dim = 128;
};

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << ch;
            break;
        }
    }
    return out.str();
}

void usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " --input MODEL.xml --output PAGED_MODEL.xml [options]\n"
        << "\nOptions:\n"
        << "  --per-layer-block-indices\n"
        << "  --score-outputs\n"
        << "  --no-score-aggregation\n"
        << "  --allow-cache-rotation\n"
        << "  --allow-xattention\n"
        << "  --allow-adaptive-rkv\n"
        << "  --kv-cache-heads N\n"
        << "  --kv-cache-block-size N\n"
        << "  --kv-cache-head-dim N\n"
        << "  --compile-device DEVICE       Compile converted in-memory graph for diagnostics\n"
        << "  --kv-cache-precision f16|f32|u8|i8|u4|i4\n"
        << "  --kv-cache-input-precision f16|f32|u8|i8|u4|i4\n"
        << "  --dummy-infer                 Run one zero-filled diagnostic inference after compile\n"
        << "  --dummy-blocks N              Number of KV cache blocks for --dummy-infer\n"
        << "  --dummy-tokens N              Number of scheduled tokens for --dummy-infer\n";
}

ov::element::Type parse_element_type(const std::string& value) {
    if (value == "f16" || value == "float16") {
        return ov::element::f16;
    }
    if (value == "bf16" || value == "bfloat16") {
        return ov::element::bf16;
    }
    if (value == "f32" || value == "float32") {
        return ov::element::f32;
    }
    if (value == "u8" || value == "uint8") {
        return ov::element::u8;
    }
    if (value == "i8" || value == "int8") {
        return ov::element::i8;
    }
    if (value == "u4" || value == "uint4") {
        return ov::element::u4;
    }
    if (value == "i4" || value == "int4") {
        return ov::element::i4;
    }
    throw std::runtime_error("unsupported --kv-cache-precision: " + value);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string item = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (item == "--input") {
            args.input = require_value("--input");
        } else if (item == "--output") {
            args.output = require_value("--output");
        } else if (item == "--per-layer-block-indices") {
            args.per_layer_block_indices = true;
        } else if (item == "--score-outputs") {
            args.score_outputs = true;
        } else if (item == "--no-score-aggregation") {
            args.allow_score_aggregation = false;
        } else if (item == "--allow-cache-rotation") {
            args.allow_cache_rotation = true;
        } else if (item == "--allow-xattention") {
            args.allow_xattention = true;
        } else if (item == "--allow-adaptive-rkv") {
            args.allow_adaptive_rkv = true;
        } else if (item == "--kv-cache-heads") {
            args.kv_cache_heads = std::stoll(require_value("--kv-cache-heads"));
        } else if (item == "--kv-cache-block-size") {
            args.kv_cache_block_size = std::stoll(require_value("--kv-cache-block-size"));
        } else if (item == "--kv-cache-head-dim") {
            args.kv_cache_head_dim = std::stoll(require_value("--kv-cache-head-dim"));
        } else if (item == "--compile-device") {
            args.compile_device = require_value("--compile-device");
        } else if (item == "--kv-cache-precision") {
            args.kv_cache_precision = require_value("--kv-cache-precision");
        } else if (item == "--kv-cache-input-precision") {
            args.kv_cache_input_precision = require_value("--kv-cache-input-precision");
        } else if (item == "--dummy-infer") {
            args.dummy_infer = true;
        } else if (item == "--dummy-blocks") {
            args.dummy_blocks = std::stoll(require_value("--dummy-blocks"));
        } else if (item == "--dummy-tokens") {
            args.dummy_tokens = std::stoll(require_value("--dummy-tokens"));
        } else if (item == "--help" || item == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + item);
        }
    }
    if (args.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (args.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    return args;
}

std::map<std::string, size_t> op_counts(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, size_t> counts;
    for (const auto& op : model->get_ops()) {
        counts[op->get_type_name()] += 1;
    }
    return counts;
}

std::string lower_text(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string op_scope(const std::shared_ptr<ov::Node>& op) {
    const std::string name = lower_text(op->get_friendly_name());
    const std::string type = op->get_type_name();
    if (name.find("code_predictor") != std::string::npos ||
        name.find("subcode") != std::string::npos ||
        name.find("mtp") != std::string::npos) {
        return "subcode_predictor";
    }
    if (type == "PagedAttentionExtension") {
        return "talker_attention";
    }
    if (type == "ScaledDotProductAttention" &&
        (name.find("__module.talker") != std::string::npos ||
         name.find("self.talker") != std::string::npos)) {
        return "talker_attention";
    }
    if (name.find("__module.talker.model.layers") != std::string::npos ||
        name.find("self.talker.model.layers") != std::string::npos) {
        if (name.find(".mlp.") != std::string::npos) {
            return "talker_mlp";
        }
        if (name.find("self_attn") != std::string::npos || type == "ScaledDotProductAttention") {
            return "talker_attention";
        }
        if (name.find("layernorm") != std::string::npos || name.find("_norm") != std::string::npos) {
            return "talker_norm";
        }
        return "talker_layer_other";
    }
    if (name.find("__module.talker.codec_head") != std::string::npos ||
        name.find("self.talker.codec_head") != std::string::npos) {
        return "talker_codec_head";
    }
    if (name.find("__module.talker") != std::string::npos ||
        name.find("self.talker") != std::string::npos) {
        return "talker_other";
    }
    return "other";
}

std::map<std::string, std::map<std::string, size_t>> scoped_op_counts(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, std::map<std::string, size_t>> counts;
    for (const auto& op : model->get_ops()) {
        counts[op_scope(op)][op->get_type_name()] += 1;
    }
    return counts;
}

bool input_name_starts_with(const std::shared_ptr<ov::Model>& model, const std::string& prefix) {
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            if (name.rfind(prefix, 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

bool input_has_name(const std::shared_ptr<ov::Model>& model, const std::string& target) {
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            if (name == target) {
                return true;
            }
        }
    }
    return false;
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
    if (changed > 0) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

size_t specialize_kv_cache_parameters(const std::shared_ptr<ov::Model>& model,
                                      int64_t heads,
                                      int64_t block_size,
                                      int64_t head_dim,
                                      ov::element::Type cache_element_type) {
    if (heads <= 0 || block_size <= 0 || head_dim <= 0) {
        return 0;
    }
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
        parameter->set_partial_shape(ov::PartialShape({ov::Dimension::dynamic(), heads, head_dim, block_size}));
        parameter->validate_and_infer_types();
        ++changed;
    }
    if (changed > 0) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

size_t restore_unregistered_parameters(const std::shared_ptr<ov::Model>& model) {
    std::set<const ov::Node*> registered;
    for (const auto& param : model->get_parameters()) {
        registered.insert(param.get());
    }
    ov::ParameterVector missing;
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

void print_counts_json(const std::string& field, const std::map<std::string, size_t>& counts) {
    std::cout << "\"" << field << "\":{";
    bool first = true;
    for (const auto& [name, count] : counts) {
        if (!first) {
            std::cout << ",";
        }
        first = false;
        std::cout << "\"" << json_escape(name) << "\":" << count;
    }
    std::cout << "}";
}

void print_scoped_counts_json(
    const std::string& field,
    const std::map<std::string, std::map<std::string, size_t>>& counts) {
    std::cout << "\"" << field << "\":{";
    bool first_scope = true;
    for (const auto& [scope, scoped_counts] : counts) {
        if (!first_scope) {
            std::cout << ",";
        }
        first_scope = false;
        std::cout << "\"" << json_escape(scope) << "\":{";
        bool first_type = true;
        for (const auto& [type, count] : scoped_counts) {
            if (!first_type) {
                std::cout << ",";
            }
            first_type = false;
            std::cout << "\"" << json_escape(type) << "\":" << count;
        }
        std::cout << "}";
    }
    std::cout << "}";
}

size_t scoped_type_count(
    const std::map<std::string, std::map<std::string, size_t>>& counts,
    const std::string& scope,
    const std::string& type) {
    const auto scope_it = counts.find(scope);
    if (scope_it == counts.end()) {
        return 0;
    }
    const auto type_it = scope_it->second.find(type);
    return type_it == scope_it->second.end() ? 0 : type_it->second;
}

void print_inputs_json(const std::shared_ptr<ov::Model>& model) {
    std::cout << "\"inputs\":[";
    bool first_input = true;
    for (const auto& input : model->inputs()) {
        if (!first_input) {
            std::cout << ",";
        }
        first_input = false;
        std::string any_name;
        try {
            any_name = input.get_any_name();
        } catch (...) {
            any_name = "";
        }
        std::cout << "{\"name\":\"" << json_escape(any_name) << "\",";
        std::cout << "\"shape\":\"" << json_escape(input.get_partial_shape().to_string()) << "\",";
        std::cout << "\"type\":\"" << json_escape(input.get_element_type().to_string()) << "\"}";
    }
    std::cout << "]";
}

void print_compiled_cache_inputs_json(const ov::CompiledModel& compiled_model) {
    std::cout << "\"compiled_cache_inputs\":[";
    bool first = true;
    for (const auto& input : compiled_model.inputs()) {
        std::string any_name;
        try {
            any_name = input.get_any_name();
        } catch (...) {
            any_name = "";
        }
        if (any_name.rfind("key_cache.", 0) != 0 && any_name.rfind("value_cache.", 0) != 0) {
            continue;
        }
        if (!first) {
            std::cout << ",";
        }
        first = false;
        std::cout << "{\"name\":\"" << json_escape(any_name) << "\",";
        std::cout << "\"shape\":\"" << json_escape(input.get_partial_shape().to_string()) << "\",";
        std::cout << "\"type\":\"" << json_escape(input.get_element_type().to_string()) << "\"}";
    }
    std::cout << "]";
}

ov::Shape concrete_shape(const ov::PartialShape& partial_shape, int64_t dynamic_value) {
    if (partial_shape.rank().is_dynamic()) {
        throw std::runtime_error("dynamic rank is not supported by diagnostic tensor allocator");
    }
    ov::Shape shape;
    shape.reserve(partial_shape.rank().get_length());
    for (const auto& dim : partial_shape) {
        shape.push_back(static_cast<size_t>(dim.is_static() ? dim.get_length() : dynamic_value));
    }
    return shape;
}

void zero_tensor(ov::Tensor& tensor) {
    std::memset(tensor.data(), 0, tensor.get_byte_size());
}

void set_i32_tensor(ov::InferRequest& request, const std::string& name, const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    request.set_tensor(name, tensor);
}

void run_dummy_infer(ov::CompiledModel& compiled_model, int64_t num_blocks, int64_t num_tokens) {
    if (num_blocks <= 0 || num_tokens <= 0) {
        throw std::runtime_error("--dummy-blocks and --dummy-tokens must be positive");
    }
    auto request = compiled_model.create_infer_request();
    for (const auto& input : compiled_model.inputs()) {
        const auto name = input.get_any_name();
        const auto type = input.get_element_type();
        ov::Shape shape;

        if (name == "inputs_embeds") {
            shape = {static_cast<size_t>(num_tokens), 2048};
        } else if (name == "position_ids") {
            shape = {3, static_cast<size_t>(num_tokens)};
        } else if (name == "tts_pad_embed") {
            shape = {1, 1, 2048};
        } else if (name == "allow_eos") {
            shape = {1};
        } else if (name == "score_aggregation_window") {
            set_i32_tensor(request, name, {1});
            continue;
        } else if (name == "past_lens") {
            set_i32_tensor(request, name, {0});
            continue;
        } else if (name == "subsequence_begins") {
            set_i32_tensor(request, name, {0, static_cast<int32_t>(num_tokens)});
            continue;
        } else if (name == "block_indices") {
            set_i32_tensor(request, name, {0});
            continue;
        } else if (name == "block_indices_begins") {
            set_i32_tensor(request, name, {0, 1});
            continue;
        } else if (name == "max_context_len") {
            ov::Tensor tensor(ov::element::i32, ov::Shape{});
            tensor.data<int32_t>()[0] = static_cast<int32_t>(num_tokens);
            request.set_tensor(name, tensor);
            continue;
        } else if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
            shape = concrete_shape(input.get_partial_shape(), num_blocks);
        } else {
            shape = concrete_shape(input.get_partial_shape(), num_tokens);
        }

        ov::Tensor tensor(type, shape);
        zero_tensor(tensor);
        request.set_tensor(name, tensor);
    }
    request.infer();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        ov::Core core;
        auto model = core.read_model(args.input);
        const size_t readvalue_initializers_added = add_readvalue_initializers(model);
        const auto before = op_counts(model);
        const auto scoped_before = scoped_op_counts(model);

        std::string validation_warning;
        std::string pass_error;
        bool pass_ok = true;
        try {
            ov::pass::SDPAToPagedAttention(
                args.per_layer_block_indices,
                args.score_outputs,
                args.allow_score_aggregation,
                args.allow_cache_rotation,
                args.allow_xattention,
                args.allow_adaptive_rkv)
                .run_on_model(model);
        } catch (const std::exception& exc) {
            pass_ok = false;
            pass_error = exc.what();
        }
        const size_t restored_parameters = restore_unregistered_parameters(model);
        const size_t specialized_kv_cache_parameters = specialize_kv_cache_parameters(
            model,
            args.kv_cache_heads,
            args.kv_cache_block_size,
            args.kv_cache_head_dim,
            parse_element_type(args.kv_cache_input_precision));
        try {
            model->validate_nodes_and_infer_types();
        } catch (const std::exception& exc) {
            validation_warning = exc.what();
        }

        const auto after = op_counts(model);
        const auto scoped_after = scoped_op_counts(model);

        bool compile_ok = true;
        std::string compile_error;
        bool dummy_infer_ok = true;
        std::string dummy_infer_error;
        std::shared_ptr<ov::CompiledModel> compiled_model;
        if (!args.compile_device.empty()) {
            try {
                ov::AnyMap compile_config{
                    ov::hint::kv_cache_precision(parse_element_type(args.kv_cache_precision)),
                };
                auto compiled = core.compile_model(model, args.compile_device, compile_config);
                compiled_model = std::make_shared<ov::CompiledModel>(std::move(compiled));
                if (args.dummy_infer) {
                    try {
                        run_dummy_infer(*compiled_model, args.dummy_blocks, args.dummy_tokens);
                    } catch (const std::exception& exc) {
                        dummy_infer_ok = false;
                        dummy_infer_error = exc.what();
                    }
                }
            } catch (const std::exception& exc) {
                compile_ok = false;
                compile_error = exc.what();
            }
        }

        std::filesystem::create_directories(args.output.parent_path());
        ov::save_model(model, args.output);

        const bool has_key_cache = input_name_starts_with(model, "key_cache.");
        const bool has_value_cache = input_name_starts_with(model, "value_cache.");
        const bool has_block_indices =
            input_has_name(model, "block_indices") || input_name_starts_with(model, "block_indices.");
        const bool has_block_indices_begins = input_has_name(model, "block_indices_begins");
        const bool paged_kv_ready = has_key_cache && has_value_cache && has_block_indices && has_block_indices_begins;

        std::cout << "{";
        std::cout << "\"ok\":" << (pass_ok ? "true" : "false") << ",";
        if (!pass_error.empty()) {
            std::cout << "\"error\":\"" << json_escape(pass_error) << "\",";
        }
        if (!validation_warning.empty()) {
            std::cout << "\"validation_warning\":\"" << json_escape(validation_warning) << "\",";
        }
        std::cout << "\"input\":\"" << json_escape(args.input.string()) << "\",";
        std::cout << "\"output\":\"" << json_escape(args.output.string()) << "\",";
        std::cout << "\"has_key_cache\":" << (has_key_cache ? "true" : "false") << ",";
        std::cout << "\"has_value_cache\":" << (has_value_cache ? "true" : "false") << ",";
        std::cout << "\"has_block_indices\":" << (has_block_indices ? "true" : "false") << ",";
        std::cout << "\"has_block_indices_begins\":" << (has_block_indices_begins ? "true" : "false") << ",";
        std::cout << "\"paged_kv_ready\":" << (paged_kv_ready ? "true" : "false") << ",";
        std::cout << "\"readvalue_initializers_added\":" << readvalue_initializers_added << ",";
        std::cout << "\"specialized_kv_cache_parameters\":" << specialized_kv_cache_parameters << ",";
        std::cout << "\"restored_unregistered_parameters\":" << restored_parameters << ",";
        std::cout << "\"kv_cache_precision\":\"" << json_escape(args.kv_cache_precision) << "\",";
        std::cout << "\"kv_cache_input_precision\":\"" << json_escape(args.kv_cache_input_precision) << "\",";
        const size_t talker_sdpa_before = scoped_type_count(scoped_before, "talker_attention", "ScaledDotProductAttention");
        const size_t talker_sdpa_after = scoped_type_count(scoped_after, "talker_attention", "ScaledDotProductAttention");
        const size_t talker_paged_after = scoped_type_count(scoped_after, "talker_attention", "PagedAttentionExtension");
        std::cout << "\"attention_conversion\":{";
        std::cout << "\"talker_sdpa_before\":" << talker_sdpa_before << ",";
        std::cout << "\"talker_sdpa_after\":" << talker_sdpa_after << ",";
        std::cout << "\"talker_paged_after\":" << talker_paged_after << ",";
        std::cout << "\"talker_conversion_complete\":"
                  << (talker_sdpa_before > 0 && talker_sdpa_after == 0 && talker_paged_after > 0 ? "true" : "false");
        std::cout << "},";
        if (!args.compile_device.empty()) {
            std::cout << "\"compile_device\":\"" << json_escape(args.compile_device) << "\",";
            std::cout << "\"compile_ok\":" << (compile_ok ? "true" : "false") << ",";
            if (!compile_error.empty()) {
                std::cout << "\"compile_error\":\"" << json_escape(compile_error) << "\",";
            }
            if (args.dummy_infer) {
                std::cout << "\"dummy_infer_ok\":" << (dummy_infer_ok ? "true" : "false") << ",";
                if (!dummy_infer_error.empty()) {
                    std::cout << "\"dummy_infer_error\":\"" << json_escape(dummy_infer_error) << "\",";
                }
            }
        }
        print_counts_json("op_counts_before", before);
        std::cout << ",";
        print_counts_json("op_counts_after", after);
        std::cout << ",";
        print_scoped_counts_json("scoped_op_counts_before", scoped_before);
        std::cout << ",";
        print_scoped_counts_json("scoped_op_counts_after", scoped_after);
        std::cout << ",";
        print_inputs_json(model);
        if (compiled_model) {
            std::cout << ",";
            print_compiled_cache_inputs_json(*compiled_model);
        }
        std::cout << "}" << std::endl;
        if (!pass_ok || !compile_ok || !dummy_infer_ok) {
            return 1;
        }
        return paged_kv_ready ? 0 : 2;
    } catch (const std::exception& exc) {
        std::cout << "{\"ok\":false,\"error\":\"" << json_escape(exc.what()) << "\"}" << std::endl;
        return 1;
    }
}
