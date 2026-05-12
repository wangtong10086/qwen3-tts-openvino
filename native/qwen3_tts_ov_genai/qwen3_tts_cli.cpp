// Copyright (C) 2026 Qwen3-TTS OpenVINO contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_tts_codegen.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path ir_dir = "openvino_full";
    std::string device = "GPU";
    std::string decoder_device = "GPU";
    std::string prompt_device = "CPU";
    std::filesystem::path cache_dir = ".cache/qwen3-tts-ov/native-openvino-cache";
    std::string cache_mode = "OPTIMIZE_SPEED";
    std::string text = "你好，这是 native C++ VoiceDesign 测试。";
    std::string instruct = "用自然、清晰的中文女声朗读。";
    std::filesystem::path output = "outputs/native_cpp.wav";
    std::filesystem::path profile_json;
    int64_t warmup_generations = 0;
    bool ov_profile = false;
    std::vector<int64_t> codec_prefill = {2154, 2156, 2055, 2157};
    int64_t max_prompt_tokens = 512;
    int64_t max_new_tokens = 16;
    int64_t min_new_tokens = 2;
    float repetition_penalty = 1.05f;
    int64_t vocab_size = 3072;
    int64_t num_code_groups = 16;
    int64_t eos_token_id = 2150;
    int64_t tts_bos_token_id = 151672;
    int64_t tts_eos_token_id = 151673;
    int64_t tts_pad_token_id = 151671;
    int64_t codec_pad_id = 2148;
    int64_t codec_bos_id = 2149;
    int64_t sample_rate = 24000;
    int64_t decode_upsample_rate = 2000;
    int64_t first_context_frames = 0;
    int64_t first_chunk_frames = 8;
    int64_t steady_context_frames = 25;
    int64_t steady_chunk_frames = 12;
    std::string prefill_graph = "fused_cache_step_unroll4_exact_cache96_int8_sym_fused.xml";
    std::string decode_graph = "fused_cache_decode_unroll4_exact_statefulmask_cache96_int8_sym_fused.xml";
    std::string first_decoder_graph = "speech_decoder_stream_c0_t8.xml";
    std::string steady_decoder_graph = "speech_decoder_stream_c25_t12.xml";
    std::string text_embedding_graph = "text_embedding.xml";
    std::string codec_embedding_graph = "codec_embedding.xml";
};

struct ChunkTiming {
    int64_t index = 0;
    int64_t frames = 0;
    int64_t samples = 0;
    int64_t is_final = 0;
    double arrival_ms = 0.0;
    double codegen_ms = 0.0;
    double decode_ms = 0.0;
};

struct AudioState {
    std::vector<float> audio;
    std::vector<ChunkTiming> chunks_detail;
    Clock::time_point started;
    bool started_set = false;
    double first_audio_ms = 0.0;
    int64_t chunks = 0;
    int64_t frames = 0;
};

double since_ms(const Clock::time_point& start, const Clock::time_point& stop = Clock::now()) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::vector<int64_t> parse_i64_csv(const std::string& value) {
    std::vector<int64_t> out;
    size_t start = 0;
    while (start < value.size()) {
        const size_t comma = value.find(',', start);
        const std::string item = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!item.empty()) {
            out.push_back(std::stoll(item));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Runs Qwen3-TTS VoiceDesign through the native OpenVINO GenAI-style C++ pipeline.\n\n"
        << "Options:\n"
        << "  --ir-dir DIR                    OpenVINO IR directory [openvino_full]\n"
        << "  --device DEVICE                 Codegen/prompt device [GPU]\n"
        << "  --decoder-device DEVICE         Streaming decoder device [GPU]\n"
        << "  --prompt-device DEVICE          Token/text/codec embedding device [CPU]\n"
        << "  --cache-dir DIR                 OpenVINO model cache directory [.cache/qwen3-tts-ov/native-openvino-cache]\n"
        << "  --text TEXT                     Text to synthesize\n"
        << "  --instruct TEXT                 VoiceDesign instruction\n"
        << "  --output WAV                    Output WAV path [outputs/native_cpp.wav]\n"
        << "  --profile-json PATH             Write native timing JSON; use '-' for stdout\n"
        << "  --ov-profile                    Enable OpenVINO perf count and include op profile in JSON\n"
        << "  --warmup-generations N          Run N unmeasured generations before profiling [0]\n"
        << "  --max-new-tokens N              Codec frames to generate [16]\n"
        << "  --min-new-tokens N              Minimum codec frames before EOS [2]\n"
        << "  --codec-prefill IDS             Comma-separated codec prefill ids [Chinese]\n"
        << "  --prefill-graph FILE            Prefill graph filename\n"
        << "  --decode-graph FILE             Stateful-mask decode-unroll graph filename\n"
        << "  --first-decoder-graph FILE      First streaming decoder graph filename\n"
        << "  --steady-decoder-graph FILE     Steady streaming decoder graph filename\n"
        << "  --help                          Show this help\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--ir-dir") {
            options.ir_dir = require_value("--ir-dir");
        } else if (arg == "--device") {
            options.device = require_value("--device");
        } else if (arg == "--decoder-device") {
            options.decoder_device = require_value("--decoder-device");
        } else if (arg == "--prompt-device") {
            options.prompt_device = require_value("--prompt-device");
        } else if (arg == "--cache-dir") {
            options.cache_dir = require_value("--cache-dir");
        } else if (arg == "--text") {
            options.text = require_value("--text");
        } else if (arg == "--instruct") {
            options.instruct = require_value("--instruct");
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else if (arg == "--profile-json") {
            options.profile_json = require_value("--profile-json");
        } else if (arg == "--ov-profile") {
            options.ov_profile = true;
        } else if (arg == "--warmup-generations") {
            options.warmup_generations = std::stoll(require_value("--warmup-generations"));
        } else if (arg == "--max-new-tokens") {
            options.max_new_tokens = std::stoll(require_value("--max-new-tokens"));
        } else if (arg == "--min-new-tokens") {
            options.min_new_tokens = std::stoll(require_value("--min-new-tokens"));
        } else if (arg == "--codec-prefill") {
            options.codec_prefill = parse_i64_csv(require_value("--codec-prefill"));
        } else if (arg == "--prefill-graph") {
            options.prefill_graph = require_value("--prefill-graph");
        } else if (arg == "--decode-graph") {
            options.decode_graph = require_value("--decode-graph");
        } else if (arg == "--first-decoder-graph") {
            options.first_decoder_graph = require_value("--first-decoder-graph");
        } else if (arg == "--steady-decoder-graph") {
            options.steady_decoder_graph = require_value("--steady-decoder-graph");
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    return options;
}

std::string json_escape(const std::string& value) {
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

void ensure_tokenizer_extension_path() {
    if (std::getenv("OPENVINO_TOKENIZERS_PATH_GENAI")) {
        return;
    }
#ifdef DEFAULT_OPENVINO_TOKENIZERS_PATH
    setenv("OPENVINO_TOKENIZERS_PATH_GENAI", DEFAULT_OPENVINO_TOKENIZERS_PATH, 0);
#endif
}

void throw_on_error(int rc, char* error) {
    if (rc == 0) {
        return;
    }
    std::string message = error ? error : "native pipeline failed";
    if (error) {
        qwen3_tts_codegen_free_error(error);
    }
    throw std::runtime_error(message);
}

int audio_callback(
    const float* audio,
    int64_t num_samples,
    const int64_t*,
    int64_t num_frames,
    int64_t,
    int64_t is_final,
    double codegen_ms,
    double decode_ms,
    void* user_data) {
    auto* state = static_cast<AudioState*>(user_data);
    const double arrival_ms = state->started_set ? since_ms(state->started) : 0.0;
    if (num_samples > 0 && state->first_audio_ms <= 0.0) {
        state->first_audio_ms = arrival_ms;
    }
    if (audio && num_samples > 0) {
        state->audio.insert(state->audio.end(), audio, audio + num_samples);
    }
    state->frames += num_frames;
    state->chunks += 1;
    state->chunks_detail.push_back(
        ChunkTiming{state->chunks, num_frames, num_samples, is_final, arrival_ms, codegen_ms, decode_ms});
    std::cerr << "chunk=" << state->chunks
              << " frames=" << num_frames
              << " samples=" << num_samples
              << " codegen_ms=" << codegen_ms
              << " decode_ms=" << decode_ms
              << "\n";
    return 0;
}

int null_audio_callback(
    const float*,
    int64_t,
    const int64_t*,
    int64_t,
    int64_t,
    int64_t,
    double,
    double,
    void*) {
    return 0;
}

template <typename T>
void write_le(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_wav(const std::filesystem::path& path, const std::vector<float>& audio, int64_t sample_rate) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output WAV: " + path.string());
    }
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;

    out.write("RIFF", 4);
    write_le<uint32_t>(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_le<uint32_t>(out, 16);
    write_le<uint16_t>(out, 1);
    write_le<uint16_t>(out, channels);
    write_le<uint32_t>(out, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(out, byte_rate);
    write_le<uint16_t>(out, block_align);
    write_le<uint16_t>(out, bits_per_sample);
    out.write("data", 4);
    write_le<uint32_t>(out, data_bytes);
    for (float sample : audio) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const auto pcm = static_cast<int16_t>(clipped * 32767.0f);
        write_le<int16_t>(out, pcm);
    }
}

void write_profile_json(
    const Options& options,
    const AudioState& state,
    int64_t generated_frames,
    double generation_elapsed_ms,
    int64_t remote_embed,
    double total_ms,
    double create_ms,
    double stream_decoder_ms,
    double prompt_config_ms,
    double generate_ms,
    double wav_ms,
    double warmup_ms,
    const std::string& native_profile_json) {
    std::string json;
    json += "{\n";
    json += "  \"status\": \"ok\",\n";
    json += "  \"ir_dir\": \"" + json_escape(options.ir_dir.string()) + "\",\n";
    json += "  \"device\": \"" + json_escape(options.device) + "\",\n";
    json += "  \"decoder_device\": \"" + json_escape(options.decoder_device) + "\",\n";
    json += "  \"prompt_device\": \"" + json_escape(options.prompt_device) + "\",\n";
    json += "  \"output\": \"" + json_escape(options.output.string()) + "\",\n";
    json += "  \"generated_frames\": " + std::to_string(generated_frames) + ",\n";
    json += "  \"samples\": " + std::to_string(state.audio.size()) + ",\n";
    json += "  \"sample_rate\": " + std::to_string(options.sample_rate) + ",\n";
    json += "  \"audio_ms\": " + std::to_string(state.audio.empty() ? 0.0 : (1000.0 * state.audio.size() / options.sample_rate)) + ",\n";
    json += "  \"first_audio_ms\": " + std::to_string(state.first_audio_ms) + ",\n";
    json += "  \"native_remote_embed\": " + std::to_string(remote_embed) + ",\n";
    json += "  \"timings\": {\n";
    json += "    \"total_ms\": " + std::to_string(total_ms) + ",\n";
    json += "    \"create_codegen_ms\": " + std::to_string(create_ms) + ",\n";
    json += "    \"set_stream_decoders_ms\": " + std::to_string(stream_decoder_ms) + ",\n";
    json += "    \"configure_prompt_ms\": " + std::to_string(prompt_config_ms) + ",\n";
    json += "    \"warmup_ms\": " + std::to_string(warmup_ms) + ",\n";
    json += "    \"generate_ms\": " + std::to_string(generate_ms) + ",\n";
    json += "    \"generation_reported_ms\": " + std::to_string(generation_elapsed_ms) + ",\n";
    json += "    \"write_wav_ms\": " + std::to_string(wav_ms) + "\n";
    json += "  },\n";
    const double audio_ms = state.audio.empty() ? 0.0 : (1000.0 * state.audio.size() / options.sample_rate);
    const double callback_compute_ms = [&]() {
        double total = 0.0;
        for (const auto& chunk : state.chunks_detail) {
            total += chunk.codegen_ms + chunk.decode_ms;
        }
        return total;
    }();
    const double stream_compute_ms = [&]() {
        double total = 0.0;
        for (const auto& chunk : state.chunks_detail) {
            if (chunk.samples > 0) {
                total += chunk.codegen_ms + chunk.decode_ms;
            }
        }
        return total;
    }();
    json += "  \"stream_compute_ms\": " + std::to_string(stream_compute_ms) + ",\n";
    json += "  \"callback_compute_ms\": " + std::to_string(callback_compute_ms) + ",\n";
    json += "  \"generate_rtf\": " + std::to_string(audio_ms > 0.0 ? generate_ms / audio_ms : 0.0) + ",\n";
    json += "  \"stream_compute_rtf\": " + std::to_string(audio_ms > 0.0 ? stream_compute_ms / audio_ms : 0.0) + ",\n";
    json += "  \"chunks\": [\n";
    for (size_t i = 0; i < state.chunks_detail.size(); ++i) {
        const auto& chunk = state.chunks_detail[i];
        json += "    {\"index\": " + std::to_string(chunk.index) +
                ", \"frames\": " + std::to_string(chunk.frames) +
                ", \"samples\": " + std::to_string(chunk.samples) +
                ", \"is_final\": " + std::to_string(chunk.is_final) +
                ", \"arrival_ms\": " + std::to_string(chunk.arrival_ms) +
                ", \"codegen_ms\": " + std::to_string(chunk.codegen_ms) +
                ", \"decode_ms\": " + std::to_string(chunk.decode_ms) + "}";
        json += (i + 1 == state.chunks_detail.size()) ? "\n" : ",\n";
    }
    json += "  ],\n";
    json += "  \"native_ov_profile\": " + (native_profile_json.empty() ? std::string("null") : native_profile_json) + "\n";
    json += "}\n";

    if (options.profile_json.empty()) {
        return;
    }
    if (options.profile_json == "-") {
        std::cout << json;
        return;
    }
    if (!options.profile_json.parent_path().empty()) {
        std::filesystem::create_directories(options.profile_json.parent_path());
    }
    std::ofstream out(options.profile_json);
    if (!out) {
        throw std::runtime_error("failed to write profile JSON: " + options.profile_json.string());
    }
    out << json;
}

std::filesystem::path join(const std::filesystem::path& dir, const std::string& file) {
    return dir / file;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ensure_tokenizer_extension_path();
        Options options = parse_args(argc, argv);
        if (options.ov_profile) {
            setenv("QWEN3_TTS_OV_NATIVE_PERF_COUNT", "1", 1);
        }
        const auto total_started = Clock::now();
        std::filesystem::create_directories(options.cache_dir);
        void* handle = nullptr;
        char* error = nullptr;
        const auto prefill = join(options.ir_dir, options.prefill_graph);
        const auto decode = join(options.ir_dir, options.decode_graph);
        const auto create_started = Clock::now();
        throw_on_error(
            qwen3_tts_codegen_create(
                prefill.string().c_str(),
                decode.string().c_str(),
                options.device.c_str(),
                options.cache_dir.string().c_str(),
                options.cache_mode.c_str(),
                &handle,
                &error),
            error);
        const double create_ms = since_ms(create_started);

        const auto stream_decoder_started = Clock::now();
        throw_on_error(
            qwen3_tts_codegen_set_stream_decoders(
                handle,
                join(options.ir_dir, options.first_decoder_graph).string().c_str(),
                join(options.ir_dir, options.steady_decoder_graph).string().c_str(),
                options.decoder_device.c_str(),
                options.cache_dir.string().c_str(),
                options.cache_mode.c_str(),
                options.first_context_frames,
                options.first_chunk_frames,
                options.steady_context_frames,
                options.steady_chunk_frames,
                options.num_code_groups,
                options.decode_upsample_rate,
                &error),
            error);
        const double stream_decoder_ms = since_ms(stream_decoder_started);

        const auto prompt_config_started = Clock::now();
        throw_on_error(
            qwen3_tts_codegen_configure_voice_design_prompt(
                handle,
                options.ir_dir.string().c_str(),
                join(options.ir_dir, options.text_embedding_graph).string().c_str(),
                join(options.ir_dir, options.codec_embedding_graph).string().c_str(),
                options.prompt_device.c_str(),
                options.cache_dir.string().c_str(),
                options.cache_mode.c_str(),
                options.tts_bos_token_id,
                options.tts_eos_token_id,
                options.tts_pad_token_id,
                options.codec_pad_id,
                options.codec_bos_id,
                &error),
            error);
        const double prompt_config_ms = since_ms(prompt_config_started);

        if (options.warmup_generations < 0) {
            throw std::runtime_error("--warmup-generations must be non-negative");
        }
        double warmup_ms = 0.0;
        if (options.warmup_generations > 0) {
            const auto warmup_started = Clock::now();
            for (int64_t i = 0; i < options.warmup_generations; ++i) {
                int64_t warmup_count = 0;
                double warmup_elapsed = 0.0;
                error = nullptr;
                throw_on_error(
                    qwen3_tts_codegen_run_voice_design_audio_stream(
                        handle,
                        options.text.c_str(),
                        options.instruct.c_str(),
                        options.codec_prefill.data(),
                        static_cast<int64_t>(options.codec_prefill.size()),
                        options.max_prompt_tokens,
                        options.max_new_tokens,
                        options.min_new_tokens,
                        options.repetition_penalty,
                        options.vocab_size,
                        options.num_code_groups,
                        options.eos_token_id,
                        null_audio_callback,
                        nullptr,
                        &warmup_count,
                        &warmup_elapsed,
                        &error),
                    error);
                std::cerr << "warmup=" << (i + 1)
                          << " generated_frames=" << warmup_count
                          << " elapsed_ms=" << warmup_elapsed
                          << "\n";
            }
            warmup_ms = since_ms(warmup_started);
        }

        AudioState state;
        int64_t out_count = 0;
        double elapsed_ms = 0.0;
        const auto generate_started = Clock::now();
        state.started = generate_started;
        state.started_set = true;
        error = nullptr;
        throw_on_error(
            qwen3_tts_codegen_run_voice_design_audio_stream(
                handle,
                options.text.c_str(),
                options.instruct.c_str(),
                options.codec_prefill.data(),
                static_cast<int64_t>(options.codec_prefill.size()),
                options.max_prompt_tokens,
                options.max_new_tokens,
                options.min_new_tokens,
                options.repetition_penalty,
                options.vocab_size,
                options.num_code_groups,
                options.eos_token_id,
                audio_callback,
                &state,
                &out_count,
                &elapsed_ms,
                &error),
            error);
        const double generate_ms = since_ms(generate_started);

        const auto wav_started = Clock::now();
        write_wav(options.output, state.audio, options.sample_rate);
        const double wav_ms = since_ms(wav_started);
        int64_t remote_embed = 0;
        throw_on_error(qwen3_tts_codegen_get_last_remote_embed_used(handle, &remote_embed, &error), error);
        char* profile_json = nullptr;
        error = nullptr;
        throw_on_error(qwen3_tts_codegen_get_profile_json(handle, &profile_json, &error), error);
        std::string native_profile_json = profile_json ? profile_json : "";
        if (profile_json) {
            qwen3_tts_codegen_free_error(profile_json);
        }
        throw_on_error(qwen3_tts_codegen_destroy(handle, &error), error);
        const double total_ms = since_ms(total_started);
        write_profile_json(
            options,
            state,
            out_count,
            elapsed_ms,
            remote_embed,
            total_ms,
            create_ms,
            stream_decoder_ms,
            prompt_config_ms,
            generate_ms,
            wav_ms,
            warmup_ms,
            native_profile_json);
        std::cerr << "generated_frames=" << out_count
                  << " samples=" << state.audio.size()
                  << " elapsed_ms=" << elapsed_ms
                  << " native_remote_embed=" << remote_embed
                  << " output=" << options.output.string()
                  << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
