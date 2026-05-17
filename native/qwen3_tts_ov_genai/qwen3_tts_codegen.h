// Copyright (C) 2026 Qwen3-TTS OpenVINO contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

extern "C" {

using Qwen3TTSFrameCallback = int (*)(const int64_t* codes, int64_t num_frames, int64_t num_code_groups, void* user_data);
using Qwen3TTSAudioCallback = int (*)(
    const float* audio,
    int64_t num_samples,
    const int64_t* codes,
    int64_t num_frames,
    int64_t num_code_groups,
    int64_t is_final,
    double codegen_ms,
    double decode_ms,
    void* user_data);

int qwen3_tts_codegen_create(
    const char* prefill_xml,
    const char* decode_xml,
    const char* device,
    const char* cache_dir,
    const char* cache_mode,
    void** out_handle,
    char** error);

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
    char** error);

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
    char** error);

int qwen3_tts_codegen_destroy(void* handle, char** error);

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
    char** error);

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
    char** error);

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
    Qwen3TTSAudioCallback callback,
    void* user_data,
    int64_t* out_count,
    double* elapsed_ms,
    char** error);

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
    char** error);

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
    char** error);

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
    char** error);

int qwen3_tts_codegen_online_batch_reset(
    void* handle,
    int64_t max_cache_blocks,
    char** error);

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
    char** error);

int qwen3_tts_codegen_online_batch_step(
    void* handle,
    int64_t max_decode_batch,
    int64_t max_events,
    int64_t* out_request_ids,
    int64_t* out_event_kinds,
    int64_t* out_codes,
    int64_t* out_event_count,
    double* elapsed_ms,
    char** error);

int qwen3_tts_codegen_online_batch_cancel(
    void* handle,
    int64_t request_id,
    char** error);

int qwen3_tts_codegen_online_batch_get_stats_json(
    void* handle,
    char** out_json,
    char** error);

int qwen3_tts_codegen_get_last_remote_embed_used(void* handle, int64_t* out_used, char** error);

int qwen3_tts_codegen_reset_profile(void* handle, char** error);

int qwen3_tts_codegen_get_profile_json(void* handle, char** out_json, char** error);

int qwen3_tts_codegen_get_last_timing_json(void* handle, char** out_json, char** error);

int qwen3_tts_codegen_release_run_buffers(void* handle, char** error);

void qwen3_tts_codegen_free_error(char* error);

}  // extern "C"
