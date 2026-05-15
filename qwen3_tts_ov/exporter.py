import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("OV_TELEMETRY_DISABLE", "1")
if "ZE_ENABLE_ALT_DRIVERS" not in os.environ:
    default_level_zero_driver = "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020"
    if os.path.exists(default_level_zero_driver):
        os.environ["ZE_ENABLE_ALT_DRIVERS"] = default_level_zero_driver

import openvino as ov
import openvino._pyopenvino as ov_private
import torch
import transformers.models.mimi.modeling_mimi as mimi_modeling
from transformers import AutoConfig, AutoModel, AutoTokenizer

import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as tokenizer_v2
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb,
    repeat_kv,
)
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


NEG_INF = -3.4028234663852886e38
RMS_EXPORT_MODES = ("default", "canonical", "inline")
SUBCODE_EXPORT_MODES = ("recompute", "cached")
ATTENTION_KERNELS = ("exact", "sdpa")
TOKENIZER_PORTABLE_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def normalize_rms_export_mode(value: str | None) -> str:
    mode = str(value or "default").strip().lower().replace("-", "_")
    if mode not in RMS_EXPORT_MODES:
        raise ValueError(f"rms_export_mode must be one of {', '.join(RMS_EXPORT_MODES)}")
    return mode


def normalize_subcode_export_mode(value: str | None) -> str:
    mode = str(value or "recompute").strip().lower().replace("-", "_")
    if mode not in SUBCODE_EXPORT_MODES:
        raise ValueError(f"subcode_export_mode must be one of {', '.join(SUBCODE_EXPORT_MODES)}")
    return mode


def normalize_attention_kernel(value: str | None) -> str:
    kernel = str(value or "sdpa").strip().lower().replace("-", "_")
    if kernel not in ATTENTION_KERNELS:
        raise ValueError(f"attention kernel must be one of {', '.join(ATTENTION_KERNELS)}")
    return kernel


def canonical_rms_norm(hidden_states, norm, rms_export_mode: str = "default"):
    if normalize_rms_export_mode(rms_export_mode) == "default":
        return norm(hidden_states)
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = torch.mean(torch.pow(x, 2.0), dim=-1, keepdim=True)
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    x = x / torch.sqrt(variance + eps)
    return norm.weight * x.to(input_dtype)


def rms_graph_suffix(rms_export_mode: str) -> str:
    mode = normalize_rms_export_mode(rms_export_mode)
    if mode == "default":
        return ""
    return f"_rms_{mode}"


def fused_codegen_graph_suffix(rms_export_mode: str, subcode_export_mode: str) -> str:
    suffix = "_cachedsub" if normalize_subcode_export_mode(subcode_export_mode) == "cached" else ""
    return suffix + rms_graph_suffix(rms_export_mode)


def causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, length, length), 0.0, device=device, dtype=dtype)
    return torch.triu(mask.fill_(NEG_INF), diagonal=1)


class TextEmbeddingWrapper(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embedding = talker.get_text_embeddings().eval()
        self.projection = talker.text_projection.eval()

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))


class CodecEmbeddingWrapper(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embedding = talker.get_input_embeddings().eval()

    def forward(self, input_ids):
        return self.embedding(input_ids)


class CodeFrameEmbeddingWrapper(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.first_embedding = talker.get_input_embeddings().eval()
        self.sub_embeddings = talker.code_predictor.get_input_embeddings().eval()

    def forward(self, audio_codes):
        embeds = [self.first_embedding(audio_codes[:, :, 0])]
        for index, embedding in enumerate(self.sub_embeddings):
            embeds.append(embedding(audio_codes[:, :, index + 1]))
        return torch.stack(embeds, dim=0).sum(0)


class SpeechEncoderWrapper(torch.nn.Module):
    def __init__(self, speech_tokenizer):
        super().__init__()
        self.model = speech_tokenizer.model.eval()
        self.valid_num_quantizers = int(self.model.encoder_valid_num_quantizers)

    def forward(self, input_values, padding_mask):
        encoded = self.model.encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
        audio_codes = encoded.audio_codes[:, : self.valid_num_quantizers].transpose(1, 2)
        return audio_codes


class SpeakerEncoderWrapper(torch.nn.Module):
    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder.eval()

    def forward(self, mels):
        return self.speaker_encoder(mels)


class TalkerNoCacheWrapper(torch.nn.Module):
    def __init__(self, talker, rms_export_mode: str = "default"):
        super().__init__()
        self.model = talker.model.eval()
        self.codec_head = talker.codec_head.eval()
        self.rms_export_mode = normalize_rms_export_mode(rms_export_mode)

    def forward(self, inputs_embeds):
        batch, seq_len = inputs_embeds.shape[:2]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch, -1)
        text_position_ids = position_ids[0]
        additive_mask = causal_mask(seq_len, inputs_embeds.device, inputs_embeds.dtype)

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=additive_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]
        hidden_states = self.model.norm(hidden_states)
        last_hidden = hidden_states[:, -1:, :]
        logits = self.codec_head(last_hidden)[:, -1, :]
        return logits, last_hidden


class StatefulTalkerWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        max_cache_len: int,
        attention_kernel: str = "exact",
        rms_export_mode: str = "default",
    ):
        super().__init__()
        self.model = talker.model.eval()
        self.codec_head = talker.codec_head.eval()
        self.max_cache_len = int(max_cache_len)
        if attention_kernel not in {"exact", "sdpa"}:
            raise ValueError(f"unsupported attention_kernel={attention_kernel!r}")
        self.attention_kernel = attention_kernel
        self.rms_export_mode = normalize_rms_export_mode(rms_export_mode)

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value, cache_position):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            attn.rope_scaling["mrope_section"],
            attn.rope_scaling["interleaved"],
        )

        scatter_index = cache_position.view(1, 1, -1, 1).expand(
            key_states.shape[0],
            key_states.shape[1],
            key_states.shape[2],
            key_states.shape[3],
        )
        full_key = past_key.scatter(2, scatter_index, key_states)
        full_value = past_value.scatter(2, scatter_index, value_states)

        active_len = cache_position[-1] + 1
        active_key = full_key[:, :, :active_len, :]
        active_value = full_value[:, :, :active_len, :]
        key_for_attention = repeat_kv(active_key, attn.num_key_value_groups)
        value_for_attention = repeat_kv(active_value, attn.num_key_value_groups)
        if self.attention_kernel == "sdpa":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_for_attention,
                value_for_attention,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attn.scaling,
            )
        else:
            attn_weights = torch.matmul(query_states, key_for_attention.transpose(2, 3)) * attn.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attention)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), full_key, full_value

    def _forward_position_ids(self, inputs_embeds, cache_position, position_ids, attention_mask, *past_key_values):
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        present_key_values = []

        for layer_index, decoder_layer in enumerate(self.model.layers):
            past_key = past_key_values[layer_index * 2]
            past_value = past_key_values[layer_index * 2 + 1]

            residual = hidden_states
            normed_states = canonical_rms_norm(hidden_states, decoder_layer.input_layernorm, self.rms_export_mode)
            attn_output, present_key, present_value = self._attention(
                decoder_layer,
                normed_states,
                attention_mask,
                position_embeddings,
                past_key,
                past_value,
                cache_position,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = canonical_rms_norm(hidden_states, decoder_layer.post_attention_layernorm, self.rms_export_mode)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            present_key_values.extend([present_key, present_value])

        hidden_states = canonical_rms_norm(hidden_states, self.model.norm, self.rms_export_mode)
        last_hidden = hidden_states[:, -1:, :]
        logits = self.codec_head(last_hidden)[:, -1, :]
        return (logits, last_hidden, *present_key_values)

    def forward(self, inputs_embeds, cache_position, attention_mask, *past_key_values):
        batch = inputs_embeds.shape[0]
        position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)
        return self._forward_position_ids(inputs_embeds, cache_position, position_ids, attention_mask, *past_key_values)


class SubcodeGreedyWrapper(torch.nn.Module):
    def __init__(self, talker, attention_kernel: str = "sdpa", rms_export_mode: str = "default"):
        super().__init__()
        self.first_embedding = talker.get_input_embeddings().eval()
        self.predictor = talker.code_predictor.eval()
        self.predictor_model = talker.code_predictor.model.eval()
        self.small_to_mtp_projection = talker.code_predictor.small_to_mtp_projection.eval()
        self.lm_head = talker.code_predictor.lm_head.eval()
        self.sub_embeddings = talker.code_predictor.get_input_embeddings().eval()
        self.attention_kernel = normalize_attention_kernel(attention_kernel)
        self.rms_export_mode = normalize_rms_export_mode(rms_export_mode)

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_for_attention = repeat_kv(key_states, attn.num_key_value_groups)
        value_for_attention = repeat_kv(value_states, attn.num_key_value_groups)
        if self.attention_kernel == "sdpa":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_for_attention,
                value_for_attention,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attn.scaling,
            )
        else:
            attn_weights = torch.matmul(query_states, key_for_attention.transpose(2, 3)) * attn.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attention)
        return attn.o_proj(attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous())

    def _predict_next(self, embeds, head_index: int):
        inputs_embeds = torch.cat(embeds, dim=1)
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        batch, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).view(1, -1)
        additive_mask = causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        position_embeddings = self.predictor_model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.predictor_model.layers:
            residual = hidden_states
            normed_states = canonical_rms_norm(hidden_states, decoder_layer.input_layernorm, self.rms_export_mode)
            hidden_states = residual + self._attention(
                decoder_layer,
                normed_states,
                additive_mask,
                position_embeddings,
            )

            residual = hidden_states
            hidden_states = canonical_rms_norm(hidden_states, decoder_layer.post_attention_layernorm, self.rms_export_mode)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = canonical_rms_norm(hidden_states, self.predictor_model.norm, self.rms_export_mode)
        logits = self.lm_head[head_index](hidden_states[:, -1:, :])[:, -1, :]
        return logits

    def forward(self, past_hidden, first_code):
        first_embed = self.first_embedding(first_code)
        embeds = [past_hidden, first_embed]
        sub_codes = []
        sub_embeds = []

        for index in range(len(self.lm_head)):
            logits = self._predict_next(embeds, index)
            next_code = torch.argmax(logits.to(torch.float32), dim=-1, keepdim=True)
            sub_codes.append(next_code)
            next_embed = self.sub_embeddings[index](next_code)
            sub_embeds.append(next_embed)
            embeds.append(next_embed)

        codes = torch.cat([first_code] + sub_codes, dim=1)
        sum_embed = torch.cat([first_embed] + sub_embeds, dim=1).sum(1, keepdim=True)
        return codes, sum_embed


class SubcodeGreedyCachedWrapper(torch.nn.Module):
    def __init__(self, talker, attention_kernel: str = "sdpa", rms_export_mode: str = "default"):
        super().__init__()
        self.first_embedding = talker.get_input_embeddings().eval()
        self.predictor_model = talker.code_predictor.model.eval()
        self.small_to_mtp_projection = talker.code_predictor.small_to_mtp_projection.eval()
        self.lm_head = talker.code_predictor.lm_head.eval()
        self.sub_embeddings = talker.code_predictor.get_input_embeddings().eval()
        self.max_sub_len = talker.code_predictor.config.num_code_groups + 1
        self.attention_kernel = normalize_attention_kernel(attention_kernel)
        self.rms_export_mode = normalize_rms_export_mode(rms_export_mode)

    def _mask(self, cache_position, query_len, dtype):
        columns = torch.arange(self.max_sub_len, device=cache_position.device, dtype=cache_position.dtype)
        allowed = columns.view(1, -1) <= cache_position.view(-1, 1)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=cache_position.device),
            torch.full((), NEG_INF, dtype=dtype, device=cache_position.device),
        )
        return mask.view(1, 1, query_len, self.max_sub_len)

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value, cache_position):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        scatter_index = cache_position.view(1, 1, -1, 1).expand(
            key_states.shape[0],
            key_states.shape[1],
            key_states.shape[2],
            key_states.shape[3],
        )
        full_key = past_key.scatter(2, scatter_index, key_states)
        full_value = past_value.scatter(2, scatter_index, value_states)

        key_for_attention = repeat_kv(full_key, attn.num_key_value_groups)
        value_for_attention = repeat_kv(full_value, attn.num_key_value_groups)
        if self.attention_kernel == "sdpa":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_for_attention,
                value_for_attention,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attn.scaling,
            )
        else:
            attn_weights = torch.matmul(query_states, key_for_attention.transpose(2, 3)) * attn.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attention)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), full_key, full_value

    def _run_predictor(self, inputs_embeds, cache_position, past_key_values):
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        position_ids = cache_position.view(1, -1)
        position_embeddings = self.predictor_model.rotary_emb(hidden_states, position_ids)
        attention_mask = self._mask(cache_position, hidden_states.shape[1], hidden_states.dtype)
        present_key_values = []

        for layer_index, decoder_layer in enumerate(self.predictor_model.layers):
            past_key = past_key_values[layer_index * 2]
            past_value = past_key_values[layer_index * 2 + 1]

            residual = hidden_states
            normed_states = canonical_rms_norm(hidden_states, decoder_layer.input_layernorm, self.rms_export_mode)
            attn_output, present_key, present_value = self._attention(
                decoder_layer,
                normed_states,
                attention_mask,
                position_embeddings,
                past_key,
                past_value,
                cache_position,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = canonical_rms_norm(hidden_states, decoder_layer.post_attention_layernorm, self.rms_export_mode)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            present_key_values.extend([present_key, present_value])

        return canonical_rms_norm(hidden_states, self.predictor_model.norm, self.rms_export_mode), present_key_values

    def forward(self, past_hidden, first_code):
        first_embed = self.first_embedding(first_code)
        cache_shape = (
            past_hidden.shape[0],
            self.predictor_model.config.num_key_value_heads,
            self.max_sub_len,
            self.predictor_model.config.head_dim,
        )
        past_key_values = [
            torch.zeros(cache_shape, dtype=past_hidden.dtype, device=past_hidden.device)
            for _ in range(self.predictor_model.config.num_hidden_layers * 2)
        ]

        hidden_states, past_key_values = self._run_predictor(
            torch.cat([past_hidden, first_embed], dim=1),
            torch.arange(2, dtype=torch.long, device=past_hidden.device),
            past_key_values,
        )

        sub_codes = []
        sub_embeds = []
        logits = self.lm_head[0](hidden_states[:, -1:, :])[:, -1, :]

        for index in range(len(self.lm_head)):
            next_code = torch.argmax(logits.to(torch.float32), dim=-1, keepdim=True)
            sub_codes.append(next_code)
            next_embed = self.sub_embeddings[index](next_code)
            sub_embeds.append(next_embed)
            if index + 1 < len(self.lm_head):
                hidden_states, past_key_values = self._run_predictor(
                    next_embed,
                    torch.tensor([index + 2], dtype=torch.long, device=past_hidden.device),
                    past_key_values,
                )
                logits = self.lm_head[index + 1](hidden_states[:, -1:, :])[:, -1, :]

        codes = torch.cat([first_code] + sub_codes, dim=1)
        sum_embed = torch.cat([first_embed] + sub_embeds, dim=1).sum(1, keepdim=True)
        return codes, sum_embed


class SubcodeGreedyCachedNextEmbedWrapper(SubcodeGreedyCachedWrapper):
    def forward(self, past_hidden, first_code, tts_pad_embed):
        codes, sum_embed = super().forward(past_hidden, first_code)
        return codes, sum_embed + tts_pad_embed


def make_subcode_wrapper(
    talker,
    subcode_export_mode: str,
    rms_export_mode: str = "default",
    attention_kernel: str = "sdpa",
):
    cls = SubcodeGreedyCachedWrapper if normalize_subcode_export_mode(subcode_export_mode) == "cached" else SubcodeGreedyWrapper
    return cls(
        talker,
        attention_kernel=normalize_attention_kernel(attention_kernel),
        rms_export_mode=rms_export_mode,
    )


class FusedNoCacheCodecStepWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "recompute",
        subcode_attention_kernel: str = "sdpa",
    ):
        super().__init__()
        self.talker = TalkerNoCacheWrapper(talker, rms_export_mode=rms_export_mode)
        self.subcode = make_subcode_wrapper(
            talker,
            subcode_export_mode,
            rms_export_mode=rms_export_mode,
            attention_kernel=subcode_attention_kernel,
        )
        vocab_size = talker.config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        suppress_from = vocab_size - 1024
        suppress_add = torch.zeros(vocab_size, dtype=torch.float32)
        suppress_add[suppress_from:] = NEG_INF
        suppress_add[eos_id] = 0.0
        self.eos_id = int(eos_id)
        self.register_buffer("suppress_add", suppress_add.view(1, -1), persistent=False)

    def forward(self, inputs_embeds, tts_pad_embed, repeated_mask, allow_eos, repetition_penalty):
        logits, last_hidden = self.talker(inputs_embeds)
        scores = logits.to(torch.float32) + self.suppress_add

        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )

        repeated = repeated_mask > 0.5
        penalized_scores = torch.where(
            scores < 0,
            scores * repetition_penalty.view(1, 1),
            scores / repetition_penalty.view(1, 1),
        )
        scores = torch.where(repeated, penalized_scores, scores)

        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return first_code, codes, frame_embed


class FusedCacheCodecStepWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        max_cache_len: int,
        attention_kernel: str = "exact",
        rms_export_mode: str = "default",
        subcode_export_mode: str = "recompute",
        subcode_attention_kernel: str = "sdpa",
    ):
        super().__init__()
        self.talker = StatefulTalkerWrapper(
            talker,
            max_cache_len,
            attention_kernel=attention_kernel,
            rms_export_mode=rms_export_mode,
        )
        self.subcode = make_subcode_wrapper(
            talker,
            subcode_export_mode,
            rms_export_mode=rms_export_mode,
            attention_kernel=subcode_attention_kernel,
        )
        vocab_size = talker.config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        suppress_from = vocab_size - 1024
        suppress_add = torch.zeros(vocab_size, dtype=torch.float32)
        suppress_add[suppress_from:] = NEG_INF
        suppress_add[eos_id] = 0.0
        self.eos_id = int(eos_id)
        self.register_buffer("suppress_add", suppress_add.view(1, -1), persistent=False)

    def forward(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        repeated_mask,
        allow_eos,
        repetition_penalty,
        *past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            cache_position,
            attention_mask,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add

        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )

        repeated = repeated_mask > 0.5
        penalized_scores = torch.where(
            scores < 0,
            scores * repetition_penalty.view(1, 1),
            scores / repetition_penalty.view(1, 1),
        )
        scores = torch.where(repeated, penalized_scores, scores)

        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return (first_code, codes, frame_embed, *present_key_values)


class FullCacheStatefulTalkerWrapper(StatefulTalkerWrapper):
    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value, cache_position):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            attn.rope_scaling["mrope_section"],
            attn.rope_scaling["interleaved"],
        )

        scatter_index = cache_position.view(1, 1, -1, 1).expand(
            key_states.shape[0],
            key_states.shape[1],
            key_states.shape[2],
            key_states.shape[3],
        )
        full_key = past_key.scatter(2, scatter_index, key_states)
        full_value = past_value.scatter(2, scatter_index, value_states)

        key_for_attention = repeat_kv(full_key, attn.num_key_value_groups)
        value_for_attention = repeat_kv(full_value, attn.num_key_value_groups)
        if self.attention_kernel == "sdpa":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_for_attention,
                value_for_attention,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=attn.scaling,
            )
        else:
            attn_weights = torch.matmul(query_states, key_for_attention.transpose(2, 3)) * attn.scaling
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_for_attention)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), full_key, full_value


class DynamicStatefulTalkerWrapper(torch.nn.Module):
    def __init__(self, talker, rms_export_mode: str = "default"):
        super().__init__()
        self.model = talker.model.eval()
        self.codec_head = talker.codec_head.eval()
        self.rms_export_mode = normalize_rms_export_mode(rms_export_mode)

    def _last_hidden(self, hidden_states):
        return hidden_states[:, -1:, :]

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            attn.rope_scaling["mrope_section"],
            attn.rope_scaling["interleaved"],
        )

        full_key = torch.cat([past_key, key_states], dim=2)
        full_value = torch.cat([past_value, value_states], dim=2)
        key_for_attention = repeat_kv(full_key, attn.num_key_value_groups)
        value_for_attention = repeat_kv(full_value, attn.num_key_value_groups)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_for_attention,
            value_for_attention,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), full_key, full_value

    def _forward_position_ids(self, inputs_embeds, position_ids, attention_mask, *past_key_values):
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        present_key_values = []

        for layer_index, decoder_layer in enumerate(self.model.layers):
            past_key = past_key_values[layer_index * 2]
            past_value = past_key_values[layer_index * 2 + 1]

            residual = hidden_states
            normed_states = canonical_rms_norm(hidden_states, decoder_layer.input_layernorm, self.rms_export_mode)
            attn_output, present_key, present_value = self._attention(
                decoder_layer,
                normed_states,
                attention_mask,
                position_embeddings,
                past_key,
                past_value,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = canonical_rms_norm(hidden_states, decoder_layer.post_attention_layernorm, self.rms_export_mode)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            present_key_values.extend([present_key, present_value])

        hidden_states = canonical_rms_norm(hidden_states, self.model.norm, self.rms_export_mode)
        last_hidden = self._last_hidden(hidden_states)
        logits = self.codec_head(last_hidden)[:, -1, :]
        return (logits, last_hidden, *present_key_values)

    def _forward_impl(self, inputs_embeds, cache_position, attention_mask, *past_key_values):
        batch = inputs_embeds.shape[0]
        position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)
        return self._forward_position_ids(inputs_embeds, position_ids, attention_mask, *past_key_values)

    def forward(self, inputs_embeds, cache_position, attention_mask, *past_key_values):
        return self._forward_impl(inputs_embeds, cache_position, attention_mask, *past_key_values)


class DynamicStatefulTalkerNoMaskWrapper(DynamicStatefulTalkerWrapper):
    def forward(self, inputs_embeds, cache_position, *past_key_values):
        return self._forward_impl(inputs_embeds, cache_position, None, *past_key_values)


class DynamicStatefulTalkerPositionIdsNoMaskWrapper(DynamicStatefulTalkerWrapper):
    def forward(self, inputs_embeds, position_ids, *past_key_values):
        return self._forward_position_ids(inputs_embeds, position_ids, None, *past_key_values)


class DynamicStatefulTalkerPositionIdsBeamWrapper(DynamicStatefulTalkerWrapper):
    def forward(self, inputs_embeds, position_ids, attention_mask, beam_idx, *past_key_values):
        gathered = [torch.index_select(past_value, 0, beam_idx) for past_value in past_key_values]
        return self._forward_position_ids(inputs_embeds, position_ids, attention_mask, *gathered)


class DynamicStatefulTalkerPagedSeedWrapper(DynamicStatefulTalkerPositionIdsBeamWrapper):
    def _last_hidden(self, hidden_states):
        # PagedAttention seed graphs use token-major inputs [total_tokens, 1, hidden].
        # Gather the scheduled token before codec_head so prompt prefill does not
        # compute logits for every prompt token.
        return hidden_states[-1:, :, :]

    def _forward_position_ids(self, inputs_embeds, position_ids, attention_mask, *past_key_values):
        logits, last_hidden, *present_key_values = super()._forward_position_ids(
            inputs_embeds,
            position_ids,
            attention_mask,
            *past_key_values,
        )
        # SDPAToPagedAttention exposes token-major inputs_embeds [total_tokens, hidden].
        # Keep only the last scheduled token for codec sampling; otherwise prompt
        # prefill tries to sample every prompt token and hits fixed one-frame
        # output reshapes in the fused codec wrapper.
        return (logits[-1:, :], last_hidden[-1:, :, :], *present_key_values)

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = canonical_rms_norm(
            attn.q_proj(hidden_states).view(hidden_shape),
            attn.q_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        key_states = canonical_rms_norm(
            attn.k_proj(hidden_states).view(hidden_shape),
            attn.k_norm,
            self.rms_export_mode,
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            attn.rope_scaling["mrope_section"],
            attn.rope_scaling["interleaved"],
        )

        # OpenVINO's SDPAToPagedAttention pass matches cache Concat directly.
        # Store already-expanded GQA K/V in the paged seed graph so the SDPA
        # K/V inputs are Concat outputs. Use explicit Concat instead of
        # expand/broadcast repeat_kv; the latter can be misclassified as the
        # pass' GQA UBR pattern and produce an incorrect KV-head count.
        if attn.num_key_value_groups > 1:
            key_states = torch.cat([key_states for _ in range(attn.num_key_value_groups)], dim=1)
            value_states = torch.cat([value_states for _ in range(attn.num_key_value_groups)], dim=1)
        full_key = torch.cat([past_key, key_states], dim=2)
        full_value = torch.cat([past_value, value_states], dim=2)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            full_key,
            full_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), full_key, full_value


class DynamicStatefulTalkerPagedGQASeedWrapper(DynamicStatefulTalkerPositionIdsBeamWrapper):
    def _last_hidden(self, hidden_states):
        return hidden_states[-1:, :, :]

    def _forward_position_ids(self, inputs_embeds, position_ids, attention_mask, *past_key_values):
        logits, last_hidden, *present_key_values = super()._forward_position_ids(
            inputs_embeds,
            position_ids,
            attention_mask,
            *past_key_values,
        )
        return (logits[-1:, :], last_hidden[-1:, :, :], *present_key_values)


class DynamicStatefulTalkerPagedTop1SeedWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        gqa_cache: bool = False,
    ):
        super().__init__()
        wrapper_cls = DynamicStatefulTalkerPagedGQASeedWrapper if gqa_cache else DynamicStatefulTalkerPagedSeedWrapper
        self.talker = wrapper_cls(talker, rms_export_mode=rms_export_mode)
        vocab_size = talker.config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        suppress_from = vocab_size - 1024
        suppress_add = torch.zeros(vocab_size, dtype=torch.float32)
        suppress_add[suppress_from:] = NEG_INF
        suppress_add[eos_id] = 0.0
        self.eos_id = int(eos_id)
        self.register_buffer("suppress_add", suppress_add.view(1, -1), persistent=False)

    def forward(self, inputs_embeds, position_ids, attention_mask, beam_idx, allow_eos, *past_key_values):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            position_ids,
            attention_mask,
            beam_idx,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        return (first_code, last_hidden, *present_key_values)


class DynamicFusedCacheCodecStepWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__()
        self.talker = DynamicStatefulTalkerWrapper(talker, rms_export_mode=rms_export_mode)
        self.subcode = make_subcode_wrapper(
            talker,
            subcode_export_mode,
            rms_export_mode=rms_export_mode,
            attention_kernel=subcode_attention_kernel,
        )
        vocab_size = talker.config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        suppress_from = vocab_size - 1024
        suppress_add = torch.zeros(vocab_size, dtype=torch.float32)
        suppress_add[suppress_from:] = NEG_INF
        suppress_add[eos_id] = 0.0
        self.eos_id = int(eos_id)
        self.no_repeat = bool(no_repeat)
        self.register_buffer("suppress_add", suppress_add.view(1, -1), persistent=False)

    def forward(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        allow_eos,
        *past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            cache_position,
            attention_mask,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return (first_code, codes, frame_embed, *present_key_values)


class DynamicFusedCacheCodecStepNoMaskWrapper(DynamicFusedCacheCodecStepWrapper):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerNoMaskWrapper(talker, rms_export_mode=rms_export_mode)

    def forward(
        self,
        inputs_embeds,
        cache_position,
        tts_pad_embed,
        allow_eos,
        *past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            cache_position,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return (first_code, codes, frame_embed, *present_key_values)


class DynamicFusedCacheCodecStepPositionIdsNoMaskWrapper(DynamicFusedCacheCodecStepNoMaskWrapper):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerPositionIdsNoMaskWrapper(talker, rms_export_mode=rms_export_mode)


class DynamicFusedCacheCodecStepPositionIdsBeamWrapper(DynamicFusedCacheCodecStepWrapper):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerPositionIdsBeamWrapper(talker, rms_export_mode=rms_export_mode)

    def forward(
        self,
        inputs_embeds,
        position_ids,
        attention_mask,
        beam_idx,
        tts_pad_embed,
        allow_eos,
        *past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            position_ids,
            attention_mask,
            beam_idx,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return (first_code, codes, frame_embed, *present_key_values)


class DynamicFusedCacheCodecStepPagedSeedWrapper(DynamicFusedCacheCodecStepPositionIdsBeamWrapper):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerPagedSeedWrapper(talker, rms_export_mode=rms_export_mode)


class DynamicFusedCacheCodecStepPagedGQASeedWrapper(DynamicFusedCacheCodecStepPositionIdsBeamWrapper):
    def __init__(
        self,
        talker,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerPagedGQASeedWrapper(talker, rms_export_mode=rms_export_mode)


class DynamicFusedCacheCodecUnrollPagedSeedWrapper(DynamicFusedCacheCodecStepPagedSeedWrapper):
    def __init__(
        self,
        talker,
        unroll_steps: int,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.unroll_steps = int(unroll_steps)

    def _step(
        self,
        inputs_embeds,
        position_ids,
        attention_mask,
        beam_idx,
        tts_pad_embed,
        allow_eos,
        state,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            position_ids,
            attention_mask,
            beam_idx,
            *state,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return first_code, codes, frame_embed, present_key_values

    def forward(
        self,
        inputs_embeds,
        position_ids,
        attention_mask,
        beam_idx,
        tts_pad_embed,
        allow_eos_steps,
        *past_key_values,
    ):
        next_inputs = inputs_embeds
        next_position_ids = position_ids
        next_attention_mask = attention_mask
        next_beam_idx = beam_idx
        state = list(past_key_values)
        first_items = []
        code_items = []
        frame_embed = tts_pad_embed
        start_position = position_ids[:, -1:, :]
        for index in range(self.unroll_steps):
            first_code, codes, frame_embed, state = self._step(
                next_inputs,
                next_position_ids,
                next_attention_mask,
                next_beam_idx,
                tts_pad_embed,
                allow_eos_steps[index],
                state,
            )
            first_items.append(first_code)
            code_items.append(codes.unsqueeze(1))
            next_inputs = frame_embed
            next_position_ids = start_position + (index + 1)
            next_beam_idx = beam_idx[:1]
            next_attention_mask = torch.zeros(
                (1, 1, 1, state[0].shape[2] + 1),
                dtype=next_inputs.dtype,
                device=next_inputs.device,
            )
        first_codes = torch.cat(first_items, dim=1)
        codes = torch.cat(code_items, dim=1)
        return (first_codes, codes, frame_embed, *state)


class DynamicFusedCacheCodecUnrollPagedGQASeedWrapper(DynamicFusedCacheCodecUnrollPagedSeedWrapper):
    def __init__(
        self,
        talker,
        unroll_steps: int,
        rms_export_mode: str = "default",
        subcode_export_mode: str = "cached",
        subcode_attention_kernel: str = "sdpa",
        no_repeat: bool = True,
    ):
        super().__init__(
            talker,
            unroll_steps=unroll_steps,
            rms_export_mode=rms_export_mode,
            subcode_export_mode=subcode_export_mode,
            subcode_attention_kernel=subcode_attention_kernel,
            no_repeat=no_repeat,
        )
        self.talker = DynamicStatefulTalkerPagedGQASeedWrapper(talker, rms_export_mode=rms_export_mode)


class FusedCacheCodecUnrollWrapper(torch.nn.Module):
    def __init__(
        self,
        talker,
        max_cache_len: int,
        unroll_steps: int,
        attention_kernel: str = "exact",
        rms_export_mode: str = "default",
        subcode_export_mode: str = "recompute",
        subcode_attention_kernel: str = "sdpa",
    ):
        super().__init__()
        self.talker = FullCacheStatefulTalkerWrapper(
            talker,
            max_cache_len,
            attention_kernel=attention_kernel,
            rms_export_mode=rms_export_mode,
        )
        self.subcode = make_subcode_wrapper(
            talker,
            subcode_export_mode,
            rms_export_mode=rms_export_mode,
            attention_kernel=subcode_attention_kernel,
        )
        self.max_cache_len = int(max_cache_len)
        self.unroll_steps = int(unroll_steps)
        vocab_size = talker.config.vocab_size
        eos_id = talker.config.codec_eos_token_id
        suppress_from = vocab_size - 1024
        suppress_add = torch.zeros(vocab_size, dtype=torch.float32)
        suppress_add[suppress_from:] = NEG_INF
        suppress_add[eos_id] = 0.0
        self.eos_id = int(eos_id)
        self.register_buffer("suppress_add", suppress_add.view(1, -1), persistent=False)

    def _mask(self, cache_position, query_len, dtype):
        columns = torch.arange(self.max_cache_len, device=cache_position.device, dtype=cache_position.dtype)
        allowed = columns.view(1, -1) <= cache_position.view(-1, 1)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=cache_position.device),
            torch.full((), NEG_INF, dtype=dtype, device=cache_position.device),
        )
        return mask.view(1, 1, query_len, self.max_cache_len)

    def _step(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        repeated_mask,
        allow_eos,
        repetition_penalty,
        past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            cache_position,
            attention_mask,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        repeated = repeated_mask > 0.5
        penalized_scores = torch.where(
            scores < 0,
            scores * repetition_penalty.view(1, 1),
            scores / repetition_penalty.view(1, 1),
        )
        scores = torch.where(repeated, penalized_scores, scores)
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        repeated_mask = repeated_mask.scatter(1, first_code, torch.ones_like(first_code, dtype=repeated_mask.dtype))
        return first_code, codes, frame_embed, repeated_mask, present_key_values

    def forward(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        repeated_mask,
        allow_eos_steps,
        repetition_penalty,
        *past_key_values,
    ):
        next_inputs = inputs_embeds
        next_position = cache_position
        next_mask = attention_mask
        state = list(past_key_values)
        first_items = []
        code_items = []
        frame_embed = tts_pad_embed
        start_position = cache_position[-1:]
        for index in range(self.unroll_steps):
            first_code, codes, frame_embed, repeated_mask, state = self._step(
                next_inputs,
                next_position,
                next_mask,
                tts_pad_embed,
                repeated_mask,
                allow_eos_steps[index],
                repetition_penalty,
                state,
            )
            first_items.append(first_code)
            code_items.append(codes.unsqueeze(1))
            next_inputs = frame_embed
            next_position = start_position + (index + 1)
            next_mask = self._mask(next_position, 1, next_inputs.dtype)
        first_codes = torch.cat(first_items, dim=1)
        codes = torch.cat(code_items, dim=1)
        return (first_codes, codes, frame_embed, repeated_mask, *state)


class FusedCacheCodecDecodeUnrollWrapper(FusedCacheCodecUnrollWrapper):
    def forward(
        self,
        inputs_embeds,
        cache_position,
        tts_pad_embed,
        repeated_mask,
        allow_eos_steps,
        repetition_penalty,
        *past_key_values,
    ):
        next_inputs = inputs_embeds
        next_position = cache_position
        next_mask = self._mask(next_position, 1, next_inputs.dtype)
        state = list(past_key_values)
        first_items = []
        code_items = []
        frame_embed = tts_pad_embed
        start_position = cache_position[-1:]
        for index in range(self.unroll_steps):
            first_code, codes, frame_embed, repeated_mask, state = self._step(
                next_inputs,
                next_position,
                next_mask,
                tts_pad_embed,
                repeated_mask,
                allow_eos_steps[index],
                repetition_penalty,
                state,
            )
            first_items.append(first_code)
            code_items.append(codes.unsqueeze(1))
            next_inputs = frame_embed
            next_position = start_position + (index + 1)
            next_mask = self._mask(next_position, 1, next_inputs.dtype)
        first_codes = torch.cat(first_items, dim=1)
        codes = torch.cat(code_items, dim=1)
        return (first_codes, codes, frame_embed, repeated_mask, *state)


class FusedCacheCodecUnrollNoRepeatWrapper(FusedCacheCodecUnrollWrapper):
    def _step_no_repeat(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        allow_eos,
        past_key_values,
    ):
        logits, last_hidden, *present_key_values = self.talker(
            inputs_embeds,
            cache_position,
            attention_mask,
            *past_key_values,
        )
        scores = logits.to(torch.float32) + self.suppress_add
        eos_score = torch.where(
            allow_eos.view(1) > 0,
            logits[:, self.eos_id].to(torch.float32),
            torch.full_like(logits[:, self.eos_id].to(torch.float32), NEG_INF),
        )
        scores = torch.cat(
            [scores[:, : self.eos_id], eos_score.view(1, 1), scores[:, self.eos_id + 1 :]],
            dim=-1,
        )
        first_code = torch.argmax(scores, dim=-1, keepdim=True).to(torch.long)
        codes, sum_embed = self.subcode(last_hidden, first_code)
        frame_embed = sum_embed + tts_pad_embed
        return first_code, codes, frame_embed, present_key_values

    def forward(
        self,
        inputs_embeds,
        cache_position,
        attention_mask,
        tts_pad_embed,
        allow_eos_steps,
        *past_key_values,
    ):
        next_inputs = inputs_embeds
        next_position = cache_position
        next_mask = attention_mask
        state = list(past_key_values)
        first_items = []
        code_items = []
        frame_embed = tts_pad_embed
        start_position = cache_position[-1:]
        for index in range(self.unroll_steps):
            first_code, codes, frame_embed, state = self._step_no_repeat(
                next_inputs,
                next_position,
                next_mask,
                tts_pad_embed,
                allow_eos_steps[index],
                state,
            )
            first_items.append(first_code)
            code_items.append(codes.unsqueeze(1))
            next_inputs = frame_embed
            next_position = start_position + (index + 1)
            next_mask = self._mask(next_position, 1, next_inputs.dtype)
        first_codes = torch.cat(first_items, dim=1)
        codes = torch.cat(code_items, dim=1)
        return (first_codes, codes, frame_embed, *state)


class FusedCacheCodecDecodeUnrollNoRepeatWrapper(FusedCacheCodecUnrollNoRepeatWrapper):
    def forward(
        self,
        inputs_embeds,
        cache_position,
        tts_pad_embed,
        allow_eos_steps,
        *past_key_values,
    ):
        next_inputs = inputs_embeds
        next_position = cache_position
        next_mask = self._mask(next_position, 1, next_inputs.dtype)
        state = list(past_key_values)
        first_items = []
        code_items = []
        frame_embed = tts_pad_embed
        start_position = cache_position[-1:]
        for index in range(self.unroll_steps):
            first_code, codes, frame_embed, state = self._step_no_repeat(
                next_inputs,
                next_position,
                next_mask,
                tts_pad_embed,
                allow_eos_steps[index],
                state,
            )
            first_items.append(first_code)
            code_items.append(codes.unsqueeze(1))
            next_inputs = frame_embed
            next_position = start_position + (index + 1)
            next_mask = self._mask(next_position, 1, next_inputs.dtype)
        first_codes = torch.cat(first_items, dim=1)
        codes = torch.cat(code_items, dim=1)
        return (first_codes, codes, frame_embed, *state)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder.eval()

    def forward(self, audio_codes):
        codes = torch.clamp(audio_codes, min=0).transpose(1, 2)
        return self.decoder(codes).squeeze(1)


class DecodeStreamWrapper(torch.nn.Module):
    def __init__(self, decoder, left_context_frames: int):
        super().__init__()
        self.decoder = decoder.eval()
        self.left_context_samples = int(left_context_frames) * int(getattr(decoder, "total_upsample", 1920))

    def forward(self, audio_codes):
        codes = torch.clamp(audio_codes, min=0).transpose(1, 2)
        audio = self.decoder(codes).squeeze(1)
        return audio[:, self.left_context_samples :]


def save_openvino_model(module: torch.nn.Module, example_input, path: Path, input_shapes=None, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return
    started = time.time()
    kwargs = {"example_input": example_input}
    if input_shapes is not None:
        kwargs["input"] = input_shapes
    ov_model = ov.convert_model(module.eval(), **kwargs)
    ov.save_model(ov_model, path, compress_to_fp16=True)
    print(f"saved {path} in {time.time() - started:.1f}s", flush=True)


def make_stateful_names(num_hidden_layers: int):
    input_names = ["inputs_embeds", "cache_position", "attention_mask"]
    output_names = ["logits", "last_hidden"]
    state_pairs = {}
    for layer_index in range(num_hidden_layers):
        input_names.extend([f"past_key_{layer_index}", f"past_value_{layer_index}"])
        output_names.extend([f"present_key_{layer_index}", f"present_value_{layer_index}"])
        state_pairs[f"past_key_{layer_index}"] = f"present_key_{layer_index}"
        state_pairs[f"past_value_{layer_index}"] = f"present_value_{layer_index}"
    return input_names, output_names, state_pairs


def apply_stateful_names(ov_model, input_names, output_names, state_pairs):
    for ov_input, name in zip(ov_model.inputs, input_names):
        ov_input.get_tensor().set_names({name})
    for ov_output, name in zip(ov_model.outputs, output_names):
        ov_output.get_tensor().set_names({name})
    ov_private.passes.MakeStateful(state_pairs).run_on_model(ov_model)


def apply_io_names(ov_model, input_names, output_names):
    for ov_input, name in zip(ov_model.inputs, input_names):
        ov_input.get_tensor().set_names({name})
    for ov_output, name in zip(ov_model.outputs, output_names):
        ov_output.get_tensor().set_names({name})


def save_subcode_cached_next_embed_model(
    talker,
    path: Path,
    attention_kernel: str = "sdpa",
    rms_export_mode: str = "default",
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    wrapper = SubcodeGreedyCachedNextEmbedWrapper(
        talker,
        attention_kernel=attention_kernel,
        rms_export_mode=rms_export_mode,
    )
    example_inputs = (
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
    )
    input_shapes = [
        ov.PartialShape([1, 1, config.hidden_size]),
        ov.PartialShape([1, 1]),
        ov.PartialShape([1, 1, config.hidden_size]),
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)
    apply_io_names(ov_model, ["past_hidden", "first_code", "tts_pad_embed"], ["codes", "next_embed"])
    ov.save_model(ov_model, path, compress_to_fp16=True)
    print(f"saved cached subcode next-embed {attention_kernel} {path} in {time.time() - started:.1f}s", flush=True)


def save_stateful_talker_model(
    talker,
    path: Path,
    example_seq_len: int,
    max_cache_len: int,
    attention_kernel: str,
    rms_export_mode: str = "default",
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    cache_shape = (1, kv_heads, max_cache_len, head_dim)
    wrapper = StatefulTalkerWrapper(
        talker,
        max_cache_len,
        attention_kernel=attention_kernel,
        rms_export_mode=rms_export_mode,
    )

    example_inputs = (
        torch.zeros((1, example_seq_len, config.hidden_size), dtype=torch.float32),
        torch.arange(example_seq_len, dtype=torch.long),
        torch.zeros((1, 1, example_seq_len, example_seq_len), dtype=torch.float32),
        *[
            torch.zeros(cache_shape, dtype=torch.float32)
            for _ in range(config.num_hidden_layers * 2)
        ],
    )
    input_shapes = [
        ov.PartialShape([1, -1, config.hidden_size]),
        ov.PartialShape([-1]),
        ov.PartialShape([1, 1, -1, -1]),
        *[ov.PartialShape(cache_shape) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)

    input_names, output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)

    ov.save_model(ov_model, path, compress_to_fp16=True)
    print(f"saved stateful {attention_kernel} {path} in {time.time() - started:.1f}s", flush=True)


def save_fused_cache_step_model(
    talker,
    path: Path,
    example_seq_len: int,
    max_cache_len: int,
    attention_kernel: str,
    rms_export_mode: str = "default",
    subcode_export_mode: str = "recompute",
    subcode_attention_kernel: str = "sdpa",
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    cache_shape = (1, kv_heads, max_cache_len, head_dim)
    wrapper = FusedCacheCodecStepWrapper(
        talker,
        max_cache_len,
        attention_kernel=attention_kernel,
        rms_export_mode=rms_export_mode,
        subcode_export_mode=subcode_export_mode,
        subcode_attention_kernel=subcode_attention_kernel,
    )

    example_inputs = (
        torch.zeros((1, example_seq_len, config.hidden_size), dtype=torch.float32),
        torch.arange(example_seq_len, dtype=torch.long),
        torch.zeros((1, 1, example_seq_len, example_seq_len), dtype=torch.float32),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.zeros((1, config.vocab_size), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        torch.full((1,), 1.05, dtype=torch.float32),
        *[
            torch.zeros(cache_shape, dtype=torch.float32)
            for _ in range(config.num_hidden_layers * 2)
        ],
    )
    input_shapes = [
        ov.PartialShape([1, -1, config.hidden_size]),
        ov.PartialShape([-1]),
        ov.PartialShape([1, 1, -1, -1]),
        ov.PartialShape([1, 1, config.hidden_size]),
        ov.PartialShape([1, config.vocab_size]),
        ov.PartialShape([1]),
        ov.PartialShape([1]),
        *[ov.PartialShape(cache_shape) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)

    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    input_names = [
        "inputs_embeds",
        "cache_position",
        "attention_mask",
        "tts_pad_embed",
        "repeated_mask",
        "allow_eos",
        "repetition_penalty",
        *input_names[3:],
    ]
    output_names = ["first_code", "codes", "frame_embed", *stateful_output_names[2:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)

    ov.save_model(ov_model, path, compress_to_fp16=True)
    print(f"saved fused cache {attention_kernel} {path} in {time.time() - started:.1f}s", flush=True)


def save_paged_kv_seed_talker_model(
    talker,
    path: Path,
    example_seq_len: int,
    rms_export_mode: str = "default",
    gqa_cache: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads if gqa_cache else config.num_attention_heads
    head_dim = config.head_dim
    past_shape = (1, kv_heads, 0, head_dim)
    wrapper_cls = DynamicStatefulTalkerPagedGQASeedWrapper if gqa_cache else DynamicStatefulTalkerPagedSeedWrapper
    wrapper = wrapper_cls(talker, rms_export_mode=rms_export_mode)
    position_ids = torch.arange(example_seq_len, dtype=torch.long).view(1, -1, 1).expand(3, -1, 1)
    example_inputs = (
        torch.zeros((example_seq_len, 1, config.hidden_size), dtype=torch.float32),
        position_ids,
        torch.zeros((example_seq_len, 1, 1, 1), dtype=torch.float32),
        torch.zeros((example_seq_len,), dtype=torch.long),
        *[torch.zeros(past_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([-1, 1, config.hidden_size]),
        ov.PartialShape([3, -1, 1]),
        ov.PartialShape([-1, 1, 1, -1]),
        ov.PartialShape([-1]),
        *[ov.PartialShape([1, kv_heads, -1, head_dim]) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)
    input_names, output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    input_names = ["inputs_embeds", "position_ids", "attention_mask", "beam_idx", *input_names[3:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)
    ov.save_model(ov_model, path, compress_to_fp16=True)
    suffix = " gqa" if gqa_cache else ""
    print(f"saved paged-kv{suffix} seed talker {path} in {time.time() - started:.1f}s", flush=True)


def save_paged_kv_seed_talker_top1_model(
    talker,
    path: Path,
    example_seq_len: int,
    rms_export_mode: str = "default",
    gqa_cache: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads if gqa_cache else config.num_attention_heads
    head_dim = config.head_dim
    past_shape = (1, kv_heads, 0, head_dim)
    wrapper = DynamicStatefulTalkerPagedTop1SeedWrapper(
        talker,
        rms_export_mode=rms_export_mode,
        gqa_cache=gqa_cache,
    )
    position_ids = torch.arange(example_seq_len, dtype=torch.long).view(1, -1, 1).expand(3, -1, 1)
    example_inputs = (
        torch.zeros((example_seq_len, 1, config.hidden_size), dtype=torch.float32),
        position_ids,
        torch.zeros((example_seq_len, 1, 1, 1), dtype=torch.float32),
        torch.zeros((example_seq_len,), dtype=torch.long),
        torch.ones((1,), dtype=torch.float32),
        *[torch.zeros(past_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([-1, 1, config.hidden_size]),
        ov.PartialShape([3, -1, 1]),
        ov.PartialShape([-1, 1, 1, -1]),
        ov.PartialShape([-1]),
        ov.PartialShape([1]),
        *[ov.PartialShape([1, kv_heads, -1, head_dim]) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)
    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    input_names = ["inputs_embeds", "position_ids", "attention_mask", "beam_idx", "allow_eos", *input_names[3:]]
    output_names = ["first_code", "last_hidden", *stateful_output_names[2:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)
    ov.save_model(ov_model, path, compress_to_fp16=True)
    suffix = " gqa" if gqa_cache else ""
    print(f"saved paged-kv{suffix} top1 seed talker {path} in {time.time() - started:.1f}s", flush=True)


def save_paged_kv_seed_fused_model(
    talker,
    path: Path,
    example_seq_len: int,
    rms_export_mode: str = "default",
    subcode_export_mode: str = "cached",
    subcode_attention_kernel: str = "sdpa",
    gqa_cache: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads if gqa_cache else config.num_attention_heads
    head_dim = config.head_dim
    past_shape = (1, kv_heads, 0, head_dim)
    wrapper_cls = DynamicFusedCacheCodecStepPagedGQASeedWrapper if gqa_cache else DynamicFusedCacheCodecStepPagedSeedWrapper
    wrapper = wrapper_cls(
        talker,
        rms_export_mode=rms_export_mode,
        subcode_export_mode=subcode_export_mode,
        subcode_attention_kernel=subcode_attention_kernel,
        no_repeat=True,
    )
    seed_len = 1
    position_ids = torch.arange(seed_len, dtype=torch.long).view(1, -1, 1).expand(3, -1, 1)
    example_inputs = (
        torch.zeros((seed_len, 1, config.hidden_size), dtype=torch.float32),
        position_ids,
        torch.zeros((seed_len, 1, 1, 1), dtype=torch.float32),
        torch.zeros((seed_len,), dtype=torch.long),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        *[torch.zeros(past_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([-1, 1, config.hidden_size]),
        ov.PartialShape([3, -1, 1]),
        ov.PartialShape([-1, 1, 1, -1]),
        ov.PartialShape([-1]),
        ov.PartialShape([1, 1, config.hidden_size]),
        ov.PartialShape([1]),
        *[ov.PartialShape([1, kv_heads, -1, head_dim]) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)
    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    input_names = [
        "inputs_embeds",
        "position_ids",
        "attention_mask",
        "beam_idx",
        "tts_pad_embed",
        "allow_eos",
        *input_names[3:],
    ]
    output_names = ["first_code", "codes", "frame_embed", *stateful_output_names[2:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)
    ov.save_model(ov_model, path, compress_to_fp16=True)
    suffix = " gqa" if gqa_cache else ""
    subcode_suffix = f" subcode-{normalize_attention_kernel(subcode_attention_kernel)}"
    print(f"saved paged-kv{suffix}{subcode_suffix} seed fused cache {path} in {time.time() - started:.1f}s", flush=True)


def save_paged_kv_seed_fused_unroll_model(
    talker,
    path: Path,
    example_seq_len: int,
    unroll_steps: int,
    rms_export_mode: str = "default",
    subcode_export_mode: str = "cached",
    subcode_attention_kernel: str = "sdpa",
    gqa_cache: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads if gqa_cache else config.num_attention_heads
    head_dim = config.head_dim
    past_shape = (1, kv_heads, 0, head_dim)
    wrapper_cls = DynamicFusedCacheCodecUnrollPagedGQASeedWrapper if gqa_cache else DynamicFusedCacheCodecUnrollPagedSeedWrapper
    wrapper = wrapper_cls(
        talker,
        unroll_steps=int(unroll_steps),
        rms_export_mode=rms_export_mode,
        subcode_export_mode=subcode_export_mode,
        subcode_attention_kernel=subcode_attention_kernel,
        no_repeat=True,
    )
    seed_len = 1
    position_ids = torch.arange(seed_len, dtype=torch.long).view(1, -1, 1).expand(3, -1, 1)
    example_inputs = (
        torch.zeros((seed_len, 1, config.hidden_size), dtype=torch.float32),
        position_ids,
        torch.zeros((seed_len, 1, 1, 1), dtype=torch.float32),
        torch.zeros((seed_len,), dtype=torch.long),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.ones((int(unroll_steps),), dtype=torch.float32),
        *[torch.zeros(past_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([-1, 1, config.hidden_size]),
        ov.PartialShape([3, -1, 1]),
        ov.PartialShape([-1, 1, 1, -1]),
        ov.PartialShape([-1]),
        ov.PartialShape([1, 1, config.hidden_size]),
        ov.PartialShape([int(unroll_steps)]),
        *[ov.PartialShape([1, kv_heads, -1, head_dim]) for _ in range(config.num_hidden_layers * 2)],
    ]
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)
    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    input_names = [
        "inputs_embeds",
        "position_ids",
        "attention_mask",
        "beam_idx",
        "tts_pad_embed",
        "allow_eos_steps",
        *input_names[3:],
    ]
    output_names = ["first_codes", "codes", "frame_embed", *stateful_output_names[2:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)
    ov.save_model(ov_model, path, compress_to_fp16=True)
    suffix = " gqa" if gqa_cache else ""
    subcode_suffix = f" subcode-{normalize_attention_kernel(subcode_attention_kernel)}"
    print(
        f"saved paged-kv{suffix}{subcode_suffix} seed fused cache unroll{unroll_steps} {path} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


def save_fused_cache_unroll_model(
    talker,
    path: Path,
    example_seq_len: int,
    max_cache_len: int,
    unroll_steps: int,
    attention_kernel: str,
    rms_export_mode: str = "default",
    subcode_export_mode: str = "recompute",
    subcode_attention_kernel: str = "sdpa",
    no_repeat: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    example_seq_len = min(int(example_seq_len), int(max_cache_len) - int(unroll_steps) + 1)
    if example_seq_len < 1:
        raise ValueError(
            f"cache length {max_cache_len} is too small for fused cache unroll{unroll_steps}; "
            "increase --cache-buckets or reduce --fused-cache-unroll-steps"
        )
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    cache_shape = (1, kv_heads, max_cache_len, head_dim)
    wrapper_cls = FusedCacheCodecUnrollNoRepeatWrapper if no_repeat else FusedCacheCodecUnrollWrapper
    wrapper = wrapper_cls(
        talker,
        max_cache_len=max_cache_len,
        unroll_steps=unroll_steps,
        attention_kernel=attention_kernel,
        rms_export_mode=rms_export_mode,
        subcode_export_mode=subcode_export_mode,
        subcode_attention_kernel=subcode_attention_kernel,
    )

    base_inputs = (
        torch.zeros((1, example_seq_len, config.hidden_size), dtype=torch.float32),
        torch.arange(example_seq_len, dtype=torch.long),
        torch.zeros((1, 1, example_seq_len, max_cache_len), dtype=torch.float32),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
    )
    penalty_inputs = () if no_repeat else (
        torch.zeros((1, config.vocab_size), dtype=torch.float32),
        torch.ones((unroll_steps,), dtype=torch.float32),
        torch.full((1,), 1.05, dtype=torch.float32),
    )
    no_repeat_inputs = (torch.ones((unroll_steps,), dtype=torch.float32),) if no_repeat else ()
    example_inputs = (
        *base_inputs,
        *penalty_inputs,
        *no_repeat_inputs,
        *[torch.zeros(cache_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([1, -1, config.hidden_size]),
        ov.PartialShape([-1]),
        ov.PartialShape([1, 1, -1, max_cache_len]),
        ov.PartialShape([1, 1, config.hidden_size]),
    ]
    if no_repeat:
        input_shapes.extend([ov.PartialShape([unroll_steps])])
    else:
        input_shapes.extend([
        ov.PartialShape([1, config.vocab_size]),
        ov.PartialShape([unroll_steps]),
        ov.PartialShape([1]),
        ])
    input_shapes.extend([ov.PartialShape(cache_shape) for _ in range(config.num_hidden_layers * 2)])
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)

    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    if no_repeat:
        input_names = [
            "inputs_embeds",
            "cache_position",
            "attention_mask",
            "tts_pad_embed",
            "allow_eos_steps",
            *input_names[3:],
        ]
        output_names = ["first_codes", "codes", "frame_embed", *stateful_output_names[2:]]
    else:
        input_names = [
            "inputs_embeds",
            "cache_position",
            "attention_mask",
            "tts_pad_embed",
            "repeated_mask",
            "allow_eos_steps",
            "repetition_penalty",
            *input_names[3:],
        ]
        output_names = ["first_codes", "codes", "frame_embed", "repeated_mask", *stateful_output_names[2:]]
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)

    ov.save_model(ov_model, path, compress_to_fp16=True)
    norepeat_suffix = " norepeat" if no_repeat else ""
    print(
        f"saved fused cache unroll{unroll_steps}{norepeat_suffix} {attention_kernel} {path} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


def save_fused_cache_decode_unroll_model(
    talker,
    path: Path,
    max_cache_len: int,
    unroll_steps: int,
    attention_kernel: str,
    rms_export_mode: str = "default",
    subcode_export_mode: str = "recompute",
    stateful_repeated_mask: bool = False,
    no_repeat: bool = False,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.exists() and path.with_suffix(".bin").exists():
        return

    started = time.time()
    config = talker.config
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    cache_shape = (1, kv_heads, max_cache_len, head_dim)
    wrapper_cls = FusedCacheCodecDecodeUnrollNoRepeatWrapper if no_repeat else FusedCacheCodecDecodeUnrollWrapper
    wrapper = wrapper_cls(
        talker,
        max_cache_len=max_cache_len,
        unroll_steps=unroll_steps,
        attention_kernel=attention_kernel,
        rms_export_mode=rms_export_mode,
        subcode_export_mode=subcode_export_mode,
    )

    base_inputs = (
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.long),
        torch.zeros((1, 1, config.hidden_size), dtype=torch.float32),
    )
    penalty_inputs = () if no_repeat else (
        torch.zeros((1, config.vocab_size), dtype=torch.float32),
        torch.ones((unroll_steps,), dtype=torch.float32),
        torch.full((1,), 1.05, dtype=torch.float32),
    )
    no_repeat_inputs = (torch.ones((unroll_steps,), dtype=torch.float32),) if no_repeat else ()
    example_inputs = (
        *base_inputs,
        *penalty_inputs,
        *no_repeat_inputs,
        *[torch.zeros(cache_shape, dtype=torch.float32) for _ in range(config.num_hidden_layers * 2)],
    )
    input_shapes = [
        ov.PartialShape([1, 1, config.hidden_size]),
        ov.PartialShape([1]),
        ov.PartialShape([1, 1, config.hidden_size]),
    ]
    if no_repeat:
        input_shapes.extend([ov.PartialShape([unroll_steps])])
    else:
        input_shapes.extend([
        ov.PartialShape([1, config.vocab_size]),
        ov.PartialShape([unroll_steps]),
        ov.PartialShape([1]),
        ])
    input_shapes.extend([ov.PartialShape(cache_shape) for _ in range(config.num_hidden_layers * 2)])
    ov_model = ov.convert_model(wrapper.eval(), example_input=example_inputs, input=input_shapes)

    input_names, stateful_output_names, state_pairs = make_stateful_names(config.num_hidden_layers)
    if no_repeat:
        input_names = [
            "inputs_embeds",
            "cache_position",
            "tts_pad_embed",
            "allow_eos_steps",
            *input_names[3:],
        ]
        output_names = ["first_codes", "codes", "frame_embed", *stateful_output_names[2:]]
    else:
        input_names = [
            "inputs_embeds",
            "cache_position",
            "tts_pad_embed",
            "repeated_mask",
            "allow_eos_steps",
            "repetition_penalty",
            *input_names[3:],
        ]
        output_names = ["first_codes", "codes", "frame_embed", "repeated_mask_out", *stateful_output_names[2:]]
    if stateful_repeated_mask and not no_repeat:
        state_pairs["repeated_mask"] = "repeated_mask_out"
    apply_stateful_names(ov_model, input_names, output_names, state_pairs)

    ov.save_model(ov_model, path, compress_to_fp16=True)
    mask_suffix = " statefulmask" if stateful_repeated_mask else ""
    mask_suffix = f"{mask_suffix} norepeat" if no_repeat else mask_suffix
    print(
        f"saved fused cache decode unroll{unroll_steps}{mask_suffix} {attention_kernel} {path} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


def load_model(model_dir: str):
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    model = AutoModel.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    return model


def export_decoder(model_dir: str, out_dir: Path, decoder_tokens: int) -> Path:
    path = out_dir / f"speech_decoder_t{decoder_tokens}.xml"
    if path.exists() and path.with_suffix(".bin").exists():
        return path

    tokenizer_v2.create_causal_mask = lambda **kwargs: None
    tokenizer_v2.create_sliding_window_causal_mask = lambda **kwargs: None

    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(model_dir, "speech_tokenizer"),
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    wrapper = DecodeWrapper(speech_tokenizer.model.decoder)
    num_quantizers = speech_tokenizer.model.config.decoder_config.num_quantizers
    example = torch.zeros((1, decoder_tokens, num_quantizers), dtype=torch.long)
    save_openvino_model(wrapper, (example,), path)
    return path


def export_stream_decoder(
    model_dir: str,
    out_dir: Path,
    chunk_frames: int,
    left_context_frames: int,
    input_shape: str = "static",
    force: bool = False,
) -> Path:
    path = out_dir / f"speech_decoder_stream_c{left_context_frames}_t{chunk_frames}.xml"
    if path.exists() and path.with_suffix(".bin").exists() and not force:
        return path

    speech_tokenizer = load_speech_tokenizer(model_dir)
    wrapper = DecodeStreamWrapper(speech_tokenizer.model.decoder, left_context_frames)
    num_quantizers = speech_tokenizer.model.config.decoder_config.num_quantizers
    example_len = int(left_context_frames) + int(chunk_frames)
    example = torch.zeros((1, example_len, num_quantizers), dtype=torch.long)
    if input_shape == "static":
        ov_input_shapes = [ov.PartialShape([1, example_len, num_quantizers])]
    elif input_shape == "dynamic":
        ov_input_shapes = [ov.PartialShape([1, -1, num_quantizers])]
    else:
        raise ValueError(f"unsupported stream decoder input shape: {input_shape!r}")
    save_openvino_model(
        wrapper,
        (example,),
        path,
        ov_input_shapes,
        force=force,
    )
    return path


def load_speech_tokenizer(model_dir: str, attn_implementation: str = "sdpa"):
    tokenizer_v2.create_causal_mask = lambda **kwargs: None
    tokenizer_v2.create_sliding_window_causal_mask = lambda **kwargs: None
    return Qwen3TTSTokenizer.from_pretrained(
        os.path.join(model_dir, "speech_tokenizer"),
        dtype=torch.float32,
        attn_implementation=attn_implementation,
    )


def force_attention_implementation(module: torch.nn.Module, attn_implementation: str) -> None:
    for submodule in module.modules():
        config = getattr(submodule, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = attn_implementation


def traceable_mimi_causal_mask(
    config,
    input_embeds,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
):
    batch, query_len = input_embeds.shape[:2]
    device = input_embeds.device
    dtype = input_embeds.dtype
    if cache_position is None:
        cache_position = torch.arange(query_len, device=device, dtype=torch.long)
    key_len = query_len
    if past_key_values is not None:
        key_len = int(past_key_values.get_seq_length()) + query_len
    key_position = torch.arange(key_len, device=device, dtype=torch.long)
    blocked = key_position.view(1, -1) > cache_position.view(-1, 1)
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window:
        blocked = blocked | (key_position.view(1, -1) <= (cache_position.view(-1, 1) - int(sliding_window)))
    mask = torch.zeros((1, 1, query_len, key_len), device=device, dtype=dtype)
    mask = mask.masked_fill(blocked.view(1, 1, query_len, key_len), NEG_INF)
    if attention_mask is not None:
        padding = attention_mask[:, None, None, :key_len] == 0
        mask = mask.expand(batch, 1, query_len, key_len).clone()
        mask = mask.masked_fill(padding, NEG_INF)
    return mask


def export_speech_encoder(model_dir: str, out_dir: Path, force: bool = False) -> Path:
    path = out_dir / "speech_encoder.xml"
    if path.exists() and path.with_suffix(".bin").exists() and not force:
        return path
    # The Mimi encoder's SDPA mask path uses nested torch.vmap/custom_function
    # tracing in recent Transformers/PyTorch builds, which OpenVINO conversion
    # can fail to trace. Eager attention avoids that path and keeps the exported
    # reference-audio encoder deterministic.
    mimi_modeling.create_causal_mask = traceable_mimi_causal_mask
    speech_tokenizer = load_speech_tokenizer(model_dir, attn_implementation="eager")
    force_attention_implementation(speech_tokenizer.model.encoder, "eager")
    wrapper = SpeechEncoderWrapper(speech_tokenizer)
    example_len = int(getattr(speech_tokenizer.model, "input_sample_rate", 24000) * 3)
    save_openvino_model(
        wrapper,
        (torch.zeros((1, example_len), dtype=torch.float32), torch.ones((1, example_len), dtype=torch.long)),
        path,
        [ov.PartialShape([1, -1]), ov.PartialShape([1, -1])],
        force=force,
    )
    return path


def export_speaker_encoder(model, out_dir: Path, force: bool = False) -> Path | None:
    if getattr(model, "speaker_encoder", None) is None:
        return None
    path = out_dir / "speaker_encoder.xml"
    if path.exists() and path.with_suffix(".bin").exists() and not force:
        return path
    wrapper = SpeakerEncoderWrapper(model.speaker_encoder)
    save_openvino_model(
        wrapper,
        (torch.zeros((1, 256, 128), dtype=torch.float32),),
        path,
        [ov.PartialShape([1, -1, 128])],
        force=force,
    )
    return path


def parse_int_list(value: str) -> list[int]:
    items = []
    for item in value.split(","):
        item = item.strip()
        if item:
            items.append(int(item))
    return sorted(set(items))


def parse_str_list(value: str) -> list[str]:
    items = []
    for item in value.split(","):
        item = item.strip()
        if item:
            items.append(item)
    return sorted(set(items))


def export_openvino_tokenizer(model_dir: str, out_dir: Path, force: bool = False) -> dict[str, str]:
    tokenizer_path = out_dir / "openvino_tokenizer.xml"
    detokenizer_path = out_dir / "openvino_detokenizer.xml"
    if tokenizer_path.exists() and detokenizer_path.exists() and not force:
        return {
            "tokenizer": tokenizer_path.name,
            "detokenizer": detokenizer_path.name,
        }
    try:
        from openvino_tokenizers import convert_tokenizer
    except Exception as exc:
        raise RuntimeError(
            "exporting OpenVINO tokenizer IR requires openvino-tokenizers; install the export/native extras"
        ) from exc

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    converted = convert_tokenizer(tokenizer, with_detokenizer=True)
    if not isinstance(converted, tuple) or len(converted) != 2:
        raise RuntimeError("openvino_tokenizers.convert_tokenizer did not return tokenizer/detokenizer models")
    ov_tokenizer, ov_detokenizer = converted
    ov.save_model(ov_tokenizer, tokenizer_path)
    ov.save_model(ov_detokenizer, detokenizer_path)
    print(f"exported OpenVINO tokenizer IR in {time.time() - started:.1f}s", flush=True)
    return {
        "tokenizer": tokenizer_path.name,
        "detokenizer": detokenizer_path.name,
    }


def update_manifest_tokenizer_ir(out_dir: Path, tokenizer_ir: dict[str, str]) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["tokenizer_ir"] = tokenizer_ir
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def update_manifest_subcode_graphs(out_dir: Path) -> None:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    graphs = manifest.setdefault("graphs", {})
    for key, graph in {
        "subcode_greedy": "subcode_greedy.xml",
        "subcode_greedy_cached": "subcode_greedy_cached.xml",
        "subcode_greedy_cached_next_embed": "subcode_greedy_cached_next_embed.xml",
        "subcode_greedy_exact": "subcode_greedy_exact.xml",
        "subcode_greedy_cached_exact": "subcode_greedy_cached_exact.xml",
    }.items():
        if (out_dir / graph).exists() and (out_dir / graph).with_suffix(".bin").exists():
            graphs[key] = graph
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def cache_graphs(prefix: str, cache_buckets: list[int], cache_kernels: list[str], out_dir: Path):
    def maybe_graph(name: str) -> str | None:
        return name if (out_dir / name).exists() else None

    return {
        kernel: {
            str(length): graph
            for length in cache_buckets
            if (graph := maybe_graph(f"{prefix}_{kernel}_cache{length}.xml"))
        }
        for kernel in cache_kernels
    }


def add_graph_suffix(path: str, suffix: str) -> str:
    if not suffix:
        return path
    item = Path(path)
    return f"{item.stem}{suffix}{item.suffix}"


def paged_kv_fused_seed_key(
    *,
    gqa_cache: bool = False,
    subcode_attention_kernel: str = "sdpa",
    unroll_steps: int | None = None,
) -> str:
    key = "fused_cache_step"
    if unroll_steps is not None:
        key += f"_unroll{int(unroll_steps)}"
    if gqa_cache:
        key += "_gqa"
    if normalize_attention_kernel(subcode_attention_kernel) != "sdpa":
        key += f"_subcode_{normalize_attention_kernel(subcode_attention_kernel)}"
    return key


def paged_kv_fused_seed_filename(
    *,
    gqa_cache: bool = False,
    subcode_attention_kernel: str = "sdpa",
    unroll_steps: int | None = None,
) -> str:
    stem = "fused_cache_step"
    if unroll_steps is not None:
        stem += f"_unroll{int(unroll_steps)}"
    stem += "_sdpa_paged"
    if gqa_cache:
        stem += "_gqa"
    if normalize_attention_kernel(subcode_attention_kernel) != "sdpa":
        stem += f"_subcode_{normalize_attention_kernel(subcode_attention_kernel)}"
    return f"{stem}_seed.xml"


def codegen_variant_feature_suffix(rms_export_mode: str, subcode_export_mode: str) -> str:
    features = []
    subcode_mode = normalize_subcode_export_mode(subcode_export_mode)
    rms_mode = normalize_rms_export_mode(rms_export_mode)
    if subcode_mode == "cached":
        features.append("cachedsub")
    if rms_mode != "default":
        features.append("rms" if rms_mode == "canonical" else f"rms_{rms_mode}")
    return "_".join(features)


def codegen_variant_names(rms_export_mode: str, subcode_export_mode: str, fused_cache_kernels: list[str]) -> list[str]:
    suffix = codegen_variant_feature_suffix(rms_export_mode, subcode_export_mode)
    if not suffix or not fused_cache_kernels:
        return []
    names = [f"fp16_fused_{suffix}"]
    if "sdpa" in fused_cache_kernels:
        names.append(f"fp16_sdpa_fused_{suffix}")
    return names


def build_codegen_variant_graphs(
    out_dir: Path,
    cache_buckets: list[int],
    cache_kernels: list[str],
    fused_cache_kernels: list[str],
    fused_cache_unroll_steps: list[int],
    fused_cache_decode_unroll_steps: list[int],
    fused_cache_stateful_mask_steps: list[int],
    fused_cache_norepeat_steps: list[int],
    rms_suffix: str,
    fused_suffix: str,
) -> dict:
    variant_graphs = {}
    def maybe_graph(name: str) -> str | None:
        return name if (out_dir / name).exists() else None

    if rms_suffix:
        variant_graphs.update(
            {
                "talker": maybe_graph(add_graph_suffix("talker_no_cache.xml", rms_suffix)),
                "talker_stateful": add_graph_suffix(
                    f"talker_stateful_exact_cache{max(cache_buckets)}.xml",
                    rms_suffix,
                ) if cache_buckets and (out_dir / add_graph_suffix(f"talker_stateful_exact_cache{max(cache_buckets)}.xml", rms_suffix)).exists() else None,
                "talker_stateful_buckets": {
                    kernel: {
                        str(length): graph
                        for length in cache_buckets
                        if (graph := maybe_graph(add_graph_suffix(f"talker_stateful_{kernel}_cache{length}.xml", rms_suffix)))
                    }
                    for kernel in cache_kernels
                },
                "subcode_greedy": maybe_graph(add_graph_suffix("subcode_greedy.xml", rms_suffix)),
                "subcode_greedy_cached": maybe_graph(add_graph_suffix("subcode_greedy_cached.xml", rms_suffix)),
                "subcode_greedy_cached_next_embed": maybe_graph(
                    add_graph_suffix("subcode_greedy_cached_next_embed.xml", rms_suffix)
                ),
            }
        )
    if fused_suffix:
        variant_graphs.update(
            {
        "fused_cache_step_buckets": {
            kernel: {
                str(length): graph
                for length in cache_buckets
                if (graph := maybe_graph(add_graph_suffix(f"fused_cache_step_{kernel}_cache{length}.xml", fused_suffix)))
            }
            for kernel in fused_cache_kernels
        },
        "fused_cache_step_unroll_buckets": {
            kernel: {
                str(unroll): {
                    str(length): graph
                    for length in cache_buckets
                    if (graph := maybe_graph(add_graph_suffix(
                        f"fused_cache_step_unroll{unroll}_{kernel}_cache{length}.xml",
                        fused_suffix,
                    )))
                }
                for unroll in fused_cache_unroll_steps
            }
            for kernel in fused_cache_kernels
        },
        "fused_cache_decode_unroll_buckets": {
            kernel: {
                str(unroll): {
                    str(length): graph
                    for length in cache_buckets
                    if (graph := maybe_graph(add_graph_suffix(
                        f"fused_cache_decode_unroll{unroll}_{kernel}_cache{length}.xml",
                        fused_suffix,
                    )))
                }
                for unroll in fused_cache_decode_unroll_steps
            }
            for kernel in fused_cache_kernels
        },
        "fused_cache_decode_unroll_stateful_mask_buckets": {
            kernel: {
                str(unroll): {
                    str(length): graph
                    for length in cache_buckets
                    if (graph := maybe_graph(add_graph_suffix(
                        f"fused_cache_decode_unroll{unroll}_{kernel}_statefulmask_cache{length}.xml",
                        fused_suffix,
                    )))
                }
                for unroll in fused_cache_stateful_mask_steps
            }
            for kernel in fused_cache_kernels
        },
        "fused_cache_step_unroll_norepeat_buckets": {
            kernel: {
                str(unroll): {
                    str(length): graph
                    for length in cache_buckets
                    if (graph := maybe_graph(add_graph_suffix(
                        f"fused_cache_step_unroll{unroll}_{kernel}_norepeat_cache{length}.xml",
                        fused_suffix,
                    )))
                }
                for unroll in fused_cache_norepeat_steps
            }
            for kernel in fused_cache_kernels
        },
        "fused_cache_decode_unroll_norepeat_buckets": {
            kernel: {
                str(unroll): {
                    str(length): graph
                    for length in cache_buckets
                    if (graph := maybe_graph(add_graph_suffix(
                        f"fused_cache_decode_unroll{unroll}_{kernel}_norepeat_cache{length}.xml",
                        fused_suffix,
                    )))
                }
                for unroll in fused_cache_norepeat_steps
            }
            for kernel in fused_cache_kernels
        },
        "fused_no_cache_step": maybe_graph(add_graph_suffix("fused_no_cache_step.xml", fused_suffix)),
            }
        )
    return variant_graphs


def update_codegen_variant_manifest(
    out_dir: Path,
    rms_export_mode: str,
    subcode_export_mode: str,
    cache_buckets: list[int],
    cache_kernels: list[str],
    fused_cache_kernels: list[str],
    fused_cache_unroll_steps: list[int],
    fused_cache_decode_unroll_steps: list[int],
    fused_cache_stateful_mask_steps: list[int],
    fused_cache_norepeat_steps: list[int],
) -> None:
    rms_suffix = rms_graph_suffix(rms_export_mode)
    fused_suffix = fused_codegen_graph_suffix(rms_export_mode, subcode_export_mode)
    if not rms_suffix and not fused_suffix:
        return
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    variant_graphs = build_codegen_variant_graphs(
        out_dir,
        cache_buckets,
        cache_kernels,
        fused_cache_kernels,
        fused_cache_unroll_steps,
        fused_cache_decode_unroll_steps,
        fused_cache_stateful_mask_steps,
        fused_cache_norepeat_steps,
        rms_suffix,
        fused_suffix,
    )
    variants = manifest.setdefault("graph_variants", {})
    for name in codegen_variant_names(rms_export_mode, subcode_export_mode, fused_cache_kernels):
        variants[name] = {
            "precision": "fp16_weights",
            "rms_export_mode": normalize_rms_export_mode(rms_export_mode),
            "subcode_export_mode": normalize_subcode_export_mode(subcode_export_mode),
            "graphs": variant_graphs,
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def copy_portable_tokenizer_files(model_dir: str, out_dir: Path) -> None:
    source_root = Path(model_dir)
    missing = []
    for name in TOKENIZER_PORTABLE_FILES:
        source = source_root / name
        dest = out_dir / name
        if not source.exists():
            missing.append(name)
            continue
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    if missing:
        raise FileNotFoundError(
            "missing tokenizer files required for portable OpenVINO runtime: " + ", ".join(missing)
        )


def write_manifest(
    model_dir: str,
    out_dir: Path,
    decoder_tokens: list[int],
    cache_buckets: list[int],
    cache_kernels: list[str],
    fused_cache_kernels: list[str],
    fused_cache_unroll_steps: list[int],
    fused_cache_decode_unroll_steps: list[int],
    fused_cache_stateful_mask_steps: list[int],
    fused_cache_norepeat_steps: list[int],
    paged_kv_unroll_steps: list[int],
    stream_decoder_chunks: list[int],
    stream_decoder_left_context: int,
    stream_decoder_first_chunks: list[int],
    stream_decoder_input_shape: str,
) -> None:
    manifest_path = out_dir / "manifest.json"
    copy_portable_tokenizer_files(model_dir, out_dir)
    previous_manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            previous_manifest = json.load(f)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "speech_tokenizer", "config.json"), "r", encoding="utf-8") as f:
        speech_config = json.load(f)

    max_cache_len = max(cache_buckets) if cache_buckets else 0
    default_cache_kernel = "exact" if "exact" in cache_kernels else (cache_kernels[0] if cache_kernels else None)
    if default_cache_kernel is None and fused_cache_kernels:
        default_cache_kernel = "exact" if "exact" in fused_cache_kernels else fused_cache_kernels[0]
    stateful_buckets = cache_graphs("talker_stateful", cache_buckets, cache_kernels, out_dir)
    fused_cache_buckets = cache_graphs("fused_cache_step", cache_buckets, fused_cache_kernels, out_dir)
    fused_cache_unroll_buckets = {
        kernel: {
            str(unroll): {
                str(length): graph
                for length in cache_buckets
                if (graph := f"fused_cache_step_unroll{unroll}_{kernel}_cache{length}.xml")
                and (out_dir / graph).exists()
            }
            for unroll in fused_cache_unroll_steps
        }
        for kernel in fused_cache_kernels
    }
    fused_cache_decode_unroll_buckets = {
        kernel: {
            str(unroll): {
                str(length): graph
                for length in cache_buckets
                if (graph := f"fused_cache_decode_unroll{unroll}_{kernel}_cache{length}.xml")
                and (out_dir / graph).exists()
            }
            for unroll in fused_cache_decode_unroll_steps
        }
        for kernel in fused_cache_kernels
    }
    fused_cache_decode_unroll_stateful_mask_buckets = {
        kernel: {
            str(unroll): {
                str(length): graph
                for length in cache_buckets
                if (graph := f"fused_cache_decode_unroll{unroll}_{kernel}_statefulmask_cache{length}.xml")
                and (out_dir / graph).exists()
            }
            for unroll in fused_cache_stateful_mask_steps
        }
        for kernel in fused_cache_kernels
    }
    fused_cache_unroll_norepeat_buckets = {
        kernel: {
            str(unroll): {
                str(length): graph
                for length in cache_buckets
                if (graph := f"fused_cache_step_unroll{unroll}_{kernel}_norepeat_cache{length}.xml")
                and (out_dir / graph).exists()
            }
            for unroll in fused_cache_norepeat_steps
        }
        for kernel in fused_cache_kernels
    }
    fused_cache_decode_unroll_norepeat_buckets = {
        kernel: {
            str(unroll): {
                str(length): graph
                for length in cache_buckets
                if (graph := f"fused_cache_decode_unroll{unroll}_{kernel}_norepeat_cache{length}.xml")
                and (out_dir / graph).exists()
            }
            for unroll in fused_cache_norepeat_steps
        }
        for kernel in fused_cache_kernels
    }
    stream_decoder_contexts = {}
    first_context_graphs = {}
    for chunk_frames in stream_decoder_first_chunks:
        graph = f"speech_decoder_stream_c0_t{chunk_frames}.xml"
        if (out_dir / graph).exists():
            first_context_graphs[str(chunk_frames)] = graph
    if not first_context_graphs:
        for path in sorted(out_dir.glob("speech_decoder_stream_c0_t*.xml")):
            chunk = path.stem.rsplit("_t", 1)[-1]
            if chunk.isdigit():
                first_context_graphs[chunk] = path.name
    if first_context_graphs:
        stream_decoder_contexts["0"] = first_context_graphs

    stream_decoder_graphs = {}
    for chunk_frames in stream_decoder_chunks:
        graph = f"speech_decoder_stream_c{stream_decoder_left_context}_t{chunk_frames}.xml"
        if (out_dir / graph).exists():
            stream_decoder_graphs[str(chunk_frames)] = graph
    if not stream_decoder_graphs:
        for path in sorted(out_dir.glob(f"speech_decoder_stream_c{stream_decoder_left_context}_t*.xml")):
            chunk = path.stem.rsplit("_t", 1)[-1]
            if chunk.isdigit():
                stream_decoder_graphs[chunk] = path.name
    if stream_decoder_graphs:
        stream_decoder_contexts[str(stream_decoder_left_context)] = stream_decoder_graphs
    manifest = {
        "format": "qwen3_tts_openvino_v3",
        "model_dir": ".",
        "tts_model_type": config.get("tts_model_type", "unknown"),
        "tokenizer_type": config.get("tokenizer_type"),
        "tts_model_size": config.get("tts_model_size"),
        "precision": "fp16_weights",
        "runtime_requires_torch": False,
        "talker_config": config["talker_config"],
        "max_cache_len": int(max_cache_len),
        "cache_buckets": cache_buckets,
        "cache_kernels": cache_kernels,
        "default_cache_kernel": default_cache_kernel,
        "default_cache_step": "split",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "talker": "talker_no_cache.xml",
            "talker_stateful": stateful_buckets.get("exact", {}).get(str(max_cache_len)),
            "talker_stateful_buckets": stateful_buckets,
            "fused_cache_step_buckets": fused_cache_buckets,
            "fused_cache_step_unroll_buckets": fused_cache_unroll_buckets,
            "fused_cache_decode_unroll_buckets": fused_cache_decode_unroll_buckets,
            "fused_cache_decode_unroll_stateful_mask_buckets": fused_cache_decode_unroll_stateful_mask_buckets,
            "fused_cache_step_unroll_norepeat_buckets": fused_cache_unroll_norepeat_buckets,
            "fused_cache_decode_unroll_norepeat_buckets": fused_cache_decode_unroll_norepeat_buckets,
            "paged_kv_seed": {
                key: graph
                for key, graph in {
                    "talker_stateful": "talker_stateful_sdpa_paged_seed.xml",
                    "fused_cache_step": "fused_cache_step_sdpa_paged_seed.xml",
                    "talker_subcode_greedy": "fused_cache_step_sdpa_paged_seed.xml",
                    "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml",
                    "talker_top1": "talker_top1_sdpa_paged_seed.xml",
                    "talker_top1_gqa": "talker_top1_sdpa_paged_gqa_seed.xml",
                    "fused_cache_step_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml",
                    "talker_subcode_greedy_gqa": "fused_cache_step_sdpa_paged_gqa_seed.xml",
                    "fused_cache_step_subcode_exact": paged_kv_fused_seed_filename(subcode_attention_kernel="exact"),
                    "talker_subcode_greedy_subcode_exact": paged_kv_fused_seed_filename(
                        subcode_attention_kernel="exact"
                    ),
                    "fused_cache_step_gqa_subcode_exact": paged_kv_fused_seed_filename(
                        gqa_cache=True,
                        subcode_attention_kernel="exact",
                    ),
                    "talker_subcode_greedy_gqa_subcode_exact": paged_kv_fused_seed_filename(
                        gqa_cache=True,
                        subcode_attention_kernel="exact",
                    ),
                    **{
                        f"fused_cache_step_unroll{unroll}": f"fused_cache_step_unroll{unroll}_sdpa_paged_seed.xml"
                        for unroll in paged_kv_unroll_steps
                    },
                    **{
                        f"fused_cache_step_unroll{unroll}_gqa": f"fused_cache_step_unroll{unroll}_sdpa_paged_gqa_seed.xml"
                        for unroll in paged_kv_unroll_steps
                    },
                    **{
                        paged_kv_fused_seed_key(subcode_attention_kernel="exact", unroll_steps=unroll): (
                            paged_kv_fused_seed_filename(subcode_attention_kernel="exact", unroll_steps=unroll)
                        )
                        for unroll in paged_kv_unroll_steps
                    },
                    **{
                        paged_kv_fused_seed_key(
                            gqa_cache=True,
                            subcode_attention_kernel="exact",
                            unroll_steps=unroll,
                        ): paged_kv_fused_seed_filename(
                            gqa_cache=True,
                            subcode_attention_kernel="exact",
                            unroll_steps=unroll,
                        )
                        for unroll in paged_kv_unroll_steps
                    },
                }.items()
                if (out_dir / graph).exists()
            },
            "subcode_greedy": "subcode_greedy.xml",
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            **(
                {"subcode_greedy_cached_next_embed": "subcode_greedy_cached_next_embed.xml"}
                if (out_dir / "subcode_greedy_cached_next_embed.xml").exists()
                else {}
            ),
            **{
                f"subcode_greedy_{kernel}": graph
                for kernel in ["exact"]
                if (graph := f"subcode_greedy_{kernel}.xml") and (out_dir / graph).exists()
            },
            **{
                f"subcode_greedy_cached_{kernel}": graph
                for kernel in ["exact"]
                if (graph := f"subcode_greedy_cached_{kernel}.xml") and (out_dir / graph).exists()
            },
            "code_frame_embedding": "code_frame_embedding.xml",
            "fused_no_cache_step": "fused_no_cache_step.xml",
            "speech_decoder": {
                str(t): graph
                for t in decoder_tokens
                if (graph := f"speech_decoder_t{t}.xml") and (out_dir / graph).exists()
            },
        },
        "ids": {
            "tts_bos_token_id": config["tts_bos_token_id"],
            "tts_eos_token_id": config["tts_eos_token_id"],
            "tts_pad_token_id": config["tts_pad_token_id"],
            "codec_bos_id": config["talker_config"]["codec_bos_id"],
            "codec_eos_token_id": config["talker_config"]["codec_eos_token_id"],
            "codec_think_id": config["talker_config"]["codec_think_id"],
            "codec_nothink_id": config["talker_config"]["codec_nothink_id"],
            "codec_pad_id": config["talker_config"]["codec_pad_id"],
            "codec_think_bos_id": config["talker_config"]["codec_think_bos_id"],
            "codec_think_eos_id": config["talker_config"]["codec_think_eos_id"],
            "codec_language_id": config["talker_config"]["codec_language_id"],
            "spk_id": config["talker_config"].get("spk_id", {}),
            "spk_is_dialect": config["talker_config"].get("spk_is_dialect", {}),
            "vocab_size": config["talker_config"]["vocab_size"],
            "suppress_from": config["talker_config"]["vocab_size"] - 1024,
        },
        "num_code_groups": config["talker_config"]["num_code_groups"],
        "sample_rate": speech_config["output_sample_rate"],
        "input_sample_rate": speech_config.get("input_sample_rate", speech_config["output_sample_rate"]),
        "encode_downsample_rate": speech_config.get("encode_downsample_rate", speech_config["decode_upsample_rate"]),
        "decode_upsample_rate": speech_config["decode_upsample_rate"],
    }
    if stream_decoder_graphs:
        manifest["graphs"]["streaming_decoder"] = stream_decoder_graphs
    fused_no_cache_graph = "fused_no_cache_step.xml"
    if not (out_dir / fused_no_cache_graph).exists():
        cached_graph = "fused_no_cache_step_cachedsub.xml"
        fused_no_cache_graph = cached_graph if (out_dir / cached_graph).exists() else None
    if fused_no_cache_graph:
        manifest["graphs"]["fused_no_cache_step"] = fused_no_cache_graph
    else:
        manifest["graphs"].pop("fused_no_cache_step", None)
    paged_seed_graphs = manifest["graphs"].get("paged_kv_seed") or {}
    if paged_seed_graphs:
        talker_config = config["talker_config"]
        kv_heads = int(talker_config.get("num_key_value_heads") or talker_config["num_attention_heads"])
        attention_heads = int(talker_config["num_attention_heads"])
        hidden_size = int(talker_config["hidden_size"])
        head_dim = int(talker_config.get("head_dim") or (hidden_size // attention_heads))
        default_seed = (
            "fused_cache_step_gqa"
            if paged_seed_graphs.get("fused_cache_step_gqa")
            else "fused_cache_step"
            if paged_seed_graphs.get("fused_cache_step")
            else "talker_stateful_gqa"
            if paged_seed_graphs.get("talker_stateful_gqa")
            else "talker_stateful"
        )
        manifest["paged_kv"] = {
            "backend": "openvino_sdpa_to_paged_attention",
            "default_seed": default_seed,
            "default_unroll": 1,
            "unroll_steps": [
                int(step)
                for step in paged_kv_unroll_steps
                if paged_seed_graphs.get(f"fused_cache_step_unroll{step}")
                or paged_seed_graphs.get(f"fused_cache_step_unroll{step}_gqa")
            ],
            "default_block_size": 8,
            "kv_cache_precision": "f16",
            "kv_cache_heads": attention_heads,
            "kv_cache_gqa_heads": kv_heads,
            "kv_cache_head_dim": head_dim,
            "kv_cache_layers": int(talker_config["num_hidden_layers"]),
            "max_position_embeddings": int(talker_config.get("max_position_embeddings", 32768)),
            "uses_remote_tensors_on_gpu": True,
            "seed_graphs": sorted(paged_seed_graphs),
        }
    if stream_decoder_contexts:
        manifest["streaming_decoder"] = {
            "left_context_frames": int(stream_decoder_left_context),
            "chunk_frames": stream_decoder_chunks or [int(item) for item in sorted(stream_decoder_graphs, key=int)],
            "first_chunk_frames": stream_decoder_first_chunks
            or [int(item) for item in sorted(first_context_graphs, key=int)],
            "default_strategy": "low_latency",
            "strategies": {
                "realtime": {
                    "initial_chunk_frames": 8,
                    "chunk_frames": 12,
                    "left_context_frames": int(stream_decoder_left_context),
                },
                "low_latency": {
                    "initial_chunk_frames": 8,
                    "chunk_frames": 12,
                    "left_context_frames": int(stream_decoder_left_context),
                },
                "smooth": {
                    "initial_chunk_frames": 12,
                    "chunk_frames": 24,
                    "left_context_frames": int(stream_decoder_left_context),
                },
                "balanced": {
                    "initial_chunk_frames": 12,
                    "chunk_frames": 12,
                    "left_context_frames": int(stream_decoder_left_context),
                },
                "stable": {
                    "initial_chunk_frames": 12,
                    "chunk_frames": 24,
                    "left_context_frames": int(stream_decoder_left_context),
                },
            },
            "graphs": stream_decoder_graphs,
            "contexts": stream_decoder_contexts,
            "input_shape": stream_decoder_input_shape,
            "output_format": "pcm_f32",
        }
    tokenizer_ir = {}
    if (out_dir / "openvino_tokenizer.xml").exists():
        tokenizer_ir["tokenizer"] = "openvino_tokenizer.xml"
    if (out_dir / "openvino_detokenizer.xml").exists():
        tokenizer_ir["detokenizer"] = "openvino_detokenizer.xml"
    if tokenizer_ir:
        manifest["tokenizer_ir"] = tokenizer_ir
    if (out_dir / "speech_encoder.xml").exists():
        manifest["graphs"]["speech_encoder"] = "speech_encoder.xml"
    if (out_dir / "speaker_encoder.xml").exists():
        manifest["graphs"]["speaker_encoder"] = "speaker_encoder.xml"
        manifest["speaker_encoder_sample_rate"] = config.get("speaker_encoder_config", {}).get("sample_rate", 24000)
    if previous_manifest.get("graph_variants"):
        manifest["graph_variants"] = previous_manifest["graph_variants"]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
    parser.add_argument("--model-type", default="auto", choices=["auto", "voice_design", "custom_voice", "base"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--example-seq-len", type=int, default=96)
    parser.add_argument("--max-cache-len", type=int, default=384, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-buckets",
        default="80,96,112,128,192,256,320,384",
        help="Comma-separated stateful talker cache lengths.",
    )
    parser.add_argument("--cache-kernels", default="exact,sdpa", help="Comma-separated stateful attention kernels.")
    parser.add_argument("--fused-cache-kernels", default="exact", help="Comma-separated fused cache kernels.")
    parser.add_argument(
        "--skip-fixed-cache-graphs",
        action="store_true",
        help=(
            "Skip legacy fixed-bucket stateful/fused cache graphs. "
            "Use this for the production fastest paged-KV split-subcode path to reduce export memory."
        ),
    )
    parser.add_argument("--fused-cache-unroll-steps", default="4,6,8,12", help="Comma-separated fused cache greedy unroll sizes.")
    parser.add_argument(
        "--fused-cache-decode-unroll-steps",
        default="4,8,12",
        help="Comma-separated steady-state fused cache greedy unroll sizes with in-graph masks.",
    )
    parser.add_argument(
        "--fused-cache-stateful-mask-steps",
        default="4,8,12",
        help="Comma-separated steady-state fused cache unroll sizes with repeated_mask stored as OpenVINO state.",
    )
    parser.add_argument(
        "--fused-cache-norepeat-steps",
        default="4",
        help="Comma-separated fused cache unroll sizes for greedy repetition_penalty=1.0 no-repeat fast graphs.",
    )
    parser.add_argument("--decoder-tokens", default="64,128,256")
    parser.add_argument("--stream-decoder-chunks", default="8,12,24")
    parser.add_argument("--stream-decoder-first-chunks", default="6,8,12")
    parser.add_argument("--stream-decoder-left-context", type=int, default=25)
    parser.add_argument(
        "--stream-decoder-input-shape",
        default="static",
        choices=["static", "dynamic"],
        help="Export streaming decoder graphs with fixed input frames for NPU or dynamic frames for legacy runtimes.",
    )
    parser.add_argument(
        "--rms-export-mode",
        default="default",
        choices=RMS_EXPORT_MODES,
        help=(
            "Export alternate codegen graphs with a canonical Python RMSNorm decomposition. "
            "Use 'canonical' to create fp16_fused_rms graph variants for RMS fusion experiments."
        ),
    )
    parser.add_argument(
        "--fused-subcode-mode",
        default="recompute",
        choices=SUBCODE_EXPORT_MODES,
        help="Use cached subcode predictor inside fused codegen graphs to avoid recomputing the subcode prefix each group.",
    )
    parser.add_argument(
        "--subcode-attention-kernels",
        default="sdpa",
        help="Comma-separated standalone subcode attention kernels to export: sdpa,exact.",
    )
    parser.add_argument("--force-stateful", action="store_true", help="Re-export stateful talker cache graphs.")
    parser.add_argument("--force-cache-graphs", action="store_true", help="Re-export stateful and fused cache graphs.")
    parser.add_argument(
        "--export-paged-kv-seed",
        action="store_true",
        help="Export dynamic SDPA past/present seed graphs that can be converted with the OpenVINO SDPAToPagedAttention pass.",
    )
    parser.add_argument(
        "--paged-kv-unroll-steps",
        default="4",
        help="Comma-separated paged-KV fused cache seed unroll sizes. Use an empty string to disable.",
    )
    parser.add_argument(
        "--paged-kv-subcode-attention-kernels",
        default="sdpa",
        help="Comma-separated cached subcode attention kernels for paged-KV fused seed graphs: sdpa,exact.",
    )
    parser.add_argument(
        "--force-paged-kv-seed",
        action="store_true",
        help="Re-export paged-KV dynamic SDPA seed graphs.",
    )
    parser.add_argument("--export-clone-graphs", action="store_true", help="Export speech/speaker encoder graphs when available.")
    parser.add_argument("--skip-tokenizer-ir", action="store_true", help="Do not export OpenVINO tokenizer/detokenizer IR.")
    parser.add_argument("--force-tokenizer-ir", action="store_true", help="Re-export OpenVINO tokenizer/detokenizer IR.")
    parser.add_argument("--tokenizer-only", action="store_true", help="Only export OpenVINO tokenizer IR and update manifest.")
    parser.add_argument("--subcode-only", action="store_true", help="Only export standalone subcode graphs and update manifest.")
    args = parser.parse_args()

    with open(os.path.join(args.model, "config.json"), "r", encoding="utf-8") as f:
        model_config = json.load(f)
    detected_model_type = model_config.get("tts_model_type", "unknown")
    if args.model_type != "auto" and args.model_type != detected_model_type:
        raise ValueError(f"--model-type={args.model_type!r} does not match model config tts_model_type={detected_model_type!r}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("openvino") / detected_model_type
    decoder_tokens = parse_int_list(args.decoder_tokens)
    stream_decoder_chunks = parse_int_list(args.stream_decoder_chunks)
    stream_decoder_first_chunks = parse_int_list(args.stream_decoder_first_chunks)
    cache_buckets = parse_int_list(args.cache_buckets) if args.cache_buckets else [int(args.max_cache_len)]
    cache_kernels = parse_str_list(args.cache_kernels)
    fused_cache_kernels = parse_str_list(args.fused_cache_kernels)
    fused_cache_unroll_steps = parse_int_list(args.fused_cache_unroll_steps)
    fused_cache_decode_unroll_steps = parse_int_list(args.fused_cache_decode_unroll_steps)
    fused_cache_stateful_mask_steps = parse_int_list(args.fused_cache_stateful_mask_steps)
    fused_cache_norepeat_steps = parse_int_list(args.fused_cache_norepeat_steps)
    paged_kv_unroll_steps = parse_int_list(args.paged_kv_unroll_steps)
    paged_kv_subcode_attention_kernels = [
        normalize_attention_kernel(item)
        for item in (parse_str_list(args.paged_kv_subcode_attention_kernels) or ["sdpa"])
    ]
    standalone_subcode_attention_kernels = [
        normalize_attention_kernel(item)
        for item in (parse_str_list(args.subcode_attention_kernels) or ["sdpa"])
    ]
    rms_export_mode = normalize_rms_export_mode(args.rms_export_mode)
    subcode_export_mode = normalize_subcode_export_mode(args.fused_subcode_mode)
    rms_suffix = rms_graph_suffix(rms_export_mode)
    fused_suffix = fused_codegen_graph_suffix(rms_export_mode, subcode_export_mode)

    def graph(name: str) -> str:
        return add_graph_suffix(name, rms_suffix)

    def fused_graph(name: str) -> str:
        return add_graph_suffix(name, fused_suffix)

    skip_fixed_cache_graphs = bool(args.skip_fixed_cache_graphs)
    if skip_fixed_cache_graphs:
        cache_kernels = []
        fused_cache_kernels = []
        fused_cache_unroll_steps = []
        fused_cache_decode_unroll_steps = []
        fused_cache_stateful_mask_steps = []
        fused_cache_norepeat_steps = []
    if not cache_buckets:
        raise ValueError("at least one cache bucket is required")
    for kernel in cache_kernels + fused_cache_kernels:
        if kernel not in {"exact", "sdpa"}:
            raise ValueError(f"unsupported cache kernel {kernel!r}")
    if args.tokenizer_only:
        tokenizer_ir = export_openvino_tokenizer(args.model, out_dir, force=args.force_tokenizer_ir)
        update_manifest_tokenizer_ir(out_dir, tokenizer_ir)
        print(f"tokenizer export complete: {out_dir / 'manifest.json'}", flush=True)
        return
    if args.subcode_only:
        started = time.time()
        model = load_model(args.model)
        talker = model.talker
        print(f"loaded PyTorch model for subcode export in {time.time() - started:.1f}s", flush=True)
        for subcode_kernel in standalone_subcode_attention_kernels:
            suffix = "" if subcode_kernel == "sdpa" else f"_{subcode_kernel}"
            save_openvino_model(
                SubcodeGreedyWrapper(
                    talker,
                    attention_kernel=subcode_kernel,
                    rms_export_mode=rms_export_mode,
                ),
                (
                    torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.long),
                ),
                out_dir / graph(f"subcode_greedy{suffix}.xml"),
                [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
                force=args.force_cache_graphs,
            )
            save_openvino_model(
                SubcodeGreedyCachedWrapper(
                    talker,
                    attention_kernel=subcode_kernel,
                    rms_export_mode=rms_export_mode,
                ),
                (
                    torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.long),
                ),
                out_dir / graph(f"subcode_greedy_cached{suffix}.xml"),
                [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
                force=args.force_cache_graphs,
            )
            if subcode_kernel == "sdpa":
                save_subcode_cached_next_embed_model(
                    talker,
                    out_dir / graph("subcode_greedy_cached_next_embed.xml"),
                    attention_kernel=subcode_kernel,
                    rms_export_mode=rms_export_mode,
                    force=args.force_cache_graphs,
                )
        update_manifest_subcode_graphs(out_dir)
        print(f"subcode export complete: {out_dir / 'manifest.json'}", flush=True)
        return

    core_paths = [
        out_dir / "text_embedding.xml",
        out_dir / "codec_embedding.xml",
        out_dir / graph("talker_no_cache.xml"),
        out_dir / graph("subcode_greedy.xml"),
        out_dir / graph("subcode_greedy_cached.xml"),
        out_dir / graph("subcode_greedy_cached_next_embed.xml"),
        out_dir / "code_frame_embedding.xml",
        out_dir / fused_graph("fused_no_cache_step.xml"),
    ]
    for kernel in standalone_subcode_attention_kernels:
        if kernel == "sdpa":
            continue
        core_paths.append(out_dir / graph(f"subcode_greedy_{kernel}.xml"))
        core_paths.append(out_dir / graph(f"subcode_greedy_cached_{kernel}.xml"))
    if not skip_fixed_cache_graphs:
        core_paths.extend(
            out_dir / graph(f"talker_stateful_{kernel}_cache{length}.xml")
            for kernel in cache_kernels
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_step_{kernel}_cache{length}.xml")
            for kernel in fused_cache_kernels
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_step_unroll{unroll}_{kernel}_cache{length}.xml")
            for kernel in fused_cache_kernels
            for unroll in fused_cache_unroll_steps
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_decode_unroll{unroll}_{kernel}_cache{length}.xml")
            for kernel in fused_cache_kernels
            for unroll in fused_cache_decode_unroll_steps
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_decode_unroll{unroll}_{kernel}_statefulmask_cache{length}.xml")
            for kernel in fused_cache_kernels
            for unroll in fused_cache_stateful_mask_steps
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_step_unroll{unroll}_{kernel}_norepeat_cache{length}.xml")
            for kernel in fused_cache_kernels
            for unroll in fused_cache_norepeat_steps
            for length in cache_buckets
        )
        core_paths.extend(
            out_dir / fused_graph(f"fused_cache_decode_unroll{unroll}_{kernel}_norepeat_cache{length}.xml")
            for kernel in fused_cache_kernels
            for unroll in fused_cache_norepeat_steps
            for length in cache_buckets
        )
    if args.export_paged_kv_seed:
        core_paths.extend(
            [
                out_dir / "talker_stateful_sdpa_paged_seed.xml",
                out_dir / "talker_stateful_sdpa_paged_gqa_seed.xml",
                out_dir / "talker_top1_sdpa_paged_seed.xml",
                out_dir / "talker_top1_sdpa_paged_gqa_seed.xml",
            ]
        )
        if not skip_fixed_cache_graphs:
            for subcode_kernel in paged_kv_subcode_attention_kernels:
                core_paths.append(out_dir / paged_kv_fused_seed_filename(subcode_attention_kernel=subcode_kernel))
                core_paths.append(out_dir / paged_kv_fused_seed_filename(gqa_cache=True, subcode_attention_kernel=subcode_kernel))
                for unroll in paged_kv_unroll_steps:
                    core_paths.append(
                        out_dir / paged_kv_fused_seed_filename(
                            subcode_attention_kernel=subcode_kernel,
                            unroll_steps=unroll,
                        )
                    )
                    core_paths.append(
                        out_dir / paged_kv_fused_seed_filename(
                            gqa_cache=True,
                            subcode_attention_kernel=subcode_kernel,
                            unroll_steps=unroll,
                        )
                    )
    force_cache_graphs = args.force_stateful or args.force_cache_graphs
    force_paged_kv_seed = args.force_paged_kv_seed or args.force_cache_graphs
    needs_core_export = force_cache_graphs or force_paged_kv_seed or any(
        not path.exists() or not path.with_suffix(".bin").exists() for path in core_paths
    )

    model = None
    if needs_core_export:
        started = time.time()
        model = load_model(args.model)
        talker = model.talker
        print(f"loaded PyTorch model for export in {time.time() - started:.1f}s", flush=True)

        save_openvino_model(
            TextEmbeddingWrapper(talker),
            (torch.zeros((1, 8), dtype=torch.long),),
            out_dir / "text_embedding.xml",
            [ov.PartialShape([1, -1])],
        )
        save_openvino_model(
            CodecEmbeddingWrapper(talker),
            (torch.zeros((1, 8), dtype=torch.long),),
            out_dir / "codec_embedding.xml",
            [ov.PartialShape([1, -1])],
        )
        save_openvino_model(
            CodeFrameEmbeddingWrapper(talker),
            (torch.zeros((1, 8, talker.config.num_code_groups), dtype=torch.long),),
            out_dir / "code_frame_embedding.xml",
            [ov.PartialShape([1, -1, talker.config.num_code_groups])],
        )
        save_openvino_model(
            TalkerNoCacheWrapper(talker, rms_export_mode=rms_export_mode),
            (torch.zeros((1, args.example_seq_len, talker.config.hidden_size), dtype=torch.float32),),
            out_dir / graph("talker_no_cache.xml"),
            [ov.PartialShape([1, -1, talker.config.hidden_size])],
        )
        for kernel in cache_kernels:
            for cache_len in cache_buckets:
                save_stateful_talker_model(
                    talker,
                    out_dir / graph(f"talker_stateful_{kernel}_cache{cache_len}.xml"),
                    min(args.example_seq_len, cache_len),
                    cache_len,
                    kernel,
                    rms_export_mode=rms_export_mode,
                    force=force_cache_graphs,
                )
        if args.export_paged_kv_seed:
            save_paged_kv_seed_talker_model(
                talker,
                out_dir / "talker_stateful_sdpa_paged_seed.xml",
                args.example_seq_len,
                rms_export_mode=rms_export_mode,
                force=force_paged_kv_seed,
            )
            save_paged_kv_seed_talker_model(
                talker,
                out_dir / "talker_stateful_sdpa_paged_gqa_seed.xml",
                args.example_seq_len,
                rms_export_mode=rms_export_mode,
                gqa_cache=True,
                force=force_paged_kv_seed,
            )
            save_paged_kv_seed_talker_top1_model(
                talker,
                out_dir / "talker_top1_sdpa_paged_seed.xml",
                args.example_seq_len,
                rms_export_mode=rms_export_mode,
                force=force_paged_kv_seed,
            )
            save_paged_kv_seed_talker_top1_model(
                talker,
                out_dir / "talker_top1_sdpa_paged_gqa_seed.xml",
                args.example_seq_len,
                rms_export_mode=rms_export_mode,
                gqa_cache=True,
                force=force_paged_kv_seed,
            )
        save_openvino_model(
            SubcodeGreedyWrapper(talker, rms_export_mode=rms_export_mode),
            (
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.long),
            ),
            out_dir / graph("subcode_greedy.xml"),
            [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
        )
        save_openvino_model(
            SubcodeGreedyCachedWrapper(talker, rms_export_mode=rms_export_mode),
            (
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.long),
            ),
            out_dir / graph("subcode_greedy_cached.xml"),
            [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
        )
        save_subcode_cached_next_embed_model(
            talker,
            out_dir / graph("subcode_greedy_cached_next_embed.xml"),
            rms_export_mode=rms_export_mode,
        )
        for subcode_kernel in standalone_subcode_attention_kernels:
            if subcode_kernel == "sdpa":
                continue
            save_openvino_model(
                SubcodeGreedyWrapper(
                    talker,
                    attention_kernel=subcode_kernel,
                    rms_export_mode=rms_export_mode,
                ),
                (
                    torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.long),
                ),
                out_dir / graph(f"subcode_greedy_{subcode_kernel}.xml"),
                [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
            )
            save_openvino_model(
                SubcodeGreedyCachedWrapper(
                    talker,
                    attention_kernel=subcode_kernel,
                    rms_export_mode=rms_export_mode,
                ),
                (
                    torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.long),
                ),
                out_dir / graph(f"subcode_greedy_cached_{subcode_kernel}.xml"),
                [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
            )
        save_openvino_model(
            FusedNoCacheCodecStepWrapper(
                talker,
                rms_export_mode=rms_export_mode,
                subcode_export_mode=subcode_export_mode,
            ),
            (
                torch.zeros((1, args.example_seq_len, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, talker.config.vocab_size), dtype=torch.float32),
                torch.ones((1,), dtype=torch.float32),
                torch.full((1,), 1.05, dtype=torch.float32),
            ),
            out_dir / fused_graph("fused_no_cache_step.xml"),
            [
                ov.PartialShape([1, -1, talker.config.hidden_size]),
                ov.PartialShape([1, 1, talker.config.hidden_size]),
                ov.PartialShape([1, talker.config.vocab_size]),
                ov.PartialShape([1]),
                ov.PartialShape([1]),
            ],
        )
        for kernel in fused_cache_kernels:
            for cache_len in cache_buckets:
                save_fused_cache_step_model(
                    talker,
                    out_dir / fused_graph(f"fused_cache_step_{kernel}_cache{cache_len}.xml"),
                    min(args.example_seq_len, cache_len),
                    cache_len,
                    kernel,
                    rms_export_mode=rms_export_mode,
                    subcode_export_mode=subcode_export_mode,
                    force=force_cache_graphs,
                )
                if args.export_paged_kv_seed and kernel == fused_cache_kernels[0] and cache_len == cache_buckets[0]:
                    for subcode_kernel in paged_kv_subcode_attention_kernels:
                        save_paged_kv_seed_fused_model(
                            talker,
                            out_dir / paged_kv_fused_seed_filename(subcode_attention_kernel=subcode_kernel),
                            args.example_seq_len,
                            rms_export_mode=rms_export_mode,
                            subcode_export_mode="cached",
                            subcode_attention_kernel=subcode_kernel,
                            force=force_paged_kv_seed,
                        )
                        save_paged_kv_seed_fused_model(
                            talker,
                            out_dir / paged_kv_fused_seed_filename(
                                gqa_cache=True,
                                subcode_attention_kernel=subcode_kernel,
                            ),
                            args.example_seq_len,
                            rms_export_mode=rms_export_mode,
                            subcode_export_mode="cached",
                            subcode_attention_kernel=subcode_kernel,
                            gqa_cache=True,
                            force=force_paged_kv_seed,
                        )
                        for paged_unroll_steps in paged_kv_unroll_steps:
                            save_paged_kv_seed_fused_unroll_model(
                                talker,
                                out_dir / paged_kv_fused_seed_filename(
                                    subcode_attention_kernel=subcode_kernel,
                                    unroll_steps=paged_unroll_steps,
                                ),
                                args.example_seq_len,
                                paged_unroll_steps,
                                rms_export_mode=rms_export_mode,
                                subcode_export_mode="cached",
                                subcode_attention_kernel=subcode_kernel,
                                force=force_paged_kv_seed,
                            )
                            save_paged_kv_seed_fused_unroll_model(
                                talker,
                                out_dir / paged_kv_fused_seed_filename(
                                    gqa_cache=True,
                                    subcode_attention_kernel=subcode_kernel,
                                    unroll_steps=paged_unroll_steps,
                                ),
                                args.example_seq_len,
                                paged_unroll_steps,
                                rms_export_mode=rms_export_mode,
                                subcode_export_mode="cached",
                                subcode_attention_kernel=subcode_kernel,
                                gqa_cache=True,
                                force=force_paged_kv_seed,
                            )
                for unroll_steps in fused_cache_unroll_steps:
                    save_fused_cache_unroll_model(
                        talker,
                        out_dir / fused_graph(f"fused_cache_step_unroll{unroll_steps}_{kernel}_cache{cache_len}.xml"),
                        min(args.example_seq_len, cache_len),
                        cache_len,
                        unroll_steps,
                        kernel,
                        rms_export_mode=rms_export_mode,
                        subcode_export_mode=subcode_export_mode,
                        force=force_cache_graphs,
                    )
                for unroll_steps in fused_cache_decode_unroll_steps:
                    save_fused_cache_decode_unroll_model(
                        talker,
                        out_dir / fused_graph(f"fused_cache_decode_unroll{unroll_steps}_{kernel}_cache{cache_len}.xml"),
                        cache_len,
                        unroll_steps,
                        kernel,
                        rms_export_mode=rms_export_mode,
                        subcode_export_mode=subcode_export_mode,
                        stateful_repeated_mask=False,
                        force=force_cache_graphs,
                    )
                for unroll_steps in fused_cache_stateful_mask_steps:
                    save_fused_cache_decode_unroll_model(
                        talker,
                        out_dir / fused_graph(f"fused_cache_decode_unroll{unroll_steps}_{kernel}_statefulmask_cache{cache_len}.xml"),
                        cache_len,
                        unroll_steps,
                        kernel,
                        rms_export_mode=rms_export_mode,
                        subcode_export_mode=subcode_export_mode,
                        stateful_repeated_mask=True,
                        force=force_cache_graphs,
                    )
                for unroll_steps in fused_cache_norepeat_steps:
                    save_fused_cache_unroll_model(
                        talker,
                        out_dir / fused_graph(f"fused_cache_step_unroll{unroll_steps}_{kernel}_norepeat_cache{cache_len}.xml"),
                        min(args.example_seq_len, cache_len),
                        cache_len,
                        unroll_steps,
                        kernel,
                        rms_export_mode=rms_export_mode,
                        subcode_export_mode=subcode_export_mode,
                        no_repeat=True,
                        force=force_cache_graphs,
                    )
                    save_fused_cache_decode_unroll_model(
                        talker,
                        out_dir / fused_graph(f"fused_cache_decode_unroll{unroll_steps}_{kernel}_norepeat_cache{cache_len}.xml"),
                        cache_len,
                        unroll_steps,
                        kernel,
                        rms_export_mode=rms_export_mode,
                        subcode_export_mode=subcode_export_mode,
                        no_repeat=True,
                        force=force_cache_graphs,
                    )
    else:
        print("core OpenVINO graphs already exist; skipping main model load", flush=True)

    if args.export_clone_graphs or detected_model_type == "base":
        export_speech_encoder(args.model, out_dir, force=force_cache_graphs)
        speaker_encoder_path = out_dir / "speaker_encoder.xml"
        if force_cache_graphs or not (speaker_encoder_path.exists() and speaker_encoder_path.with_suffix(".bin").exists()):
            if model is None:
                started = time.time()
                model = load_model(args.model)
                print(f"loaded PyTorch model for speaker encoder export in {time.time() - started:.1f}s", flush=True)
            export_speaker_encoder(model, out_dir, force=force_cache_graphs)

    for tokens in decoder_tokens:
        export_decoder(args.model, out_dir, tokens)
    for chunk_frames in stream_decoder_first_chunks:
        export_stream_decoder(
            args.model,
            out_dir,
            chunk_frames=chunk_frames,
            left_context_frames=0,
            input_shape=args.stream_decoder_input_shape,
            force=force_cache_graphs,
        )
    for chunk_frames in stream_decoder_chunks:
        export_stream_decoder(
            args.model,
            out_dir,
            chunk_frames=chunk_frames,
            left_context_frames=args.stream_decoder_left_context,
            input_shape=args.stream_decoder_input_shape,
            force=force_cache_graphs,
        )
    if not args.skip_tokenizer_ir:
        export_openvino_tokenizer(args.model, out_dir, force=args.force_tokenizer_ir)

    write_manifest(
        args.model,
        out_dir,
        decoder_tokens,
        cache_buckets,
        cache_kernels,
        fused_cache_kernels,
        fused_cache_unroll_steps,
        fused_cache_decode_unroll_steps,
        fused_cache_stateful_mask_steps,
        fused_cache_norepeat_steps,
        paged_kv_unroll_steps,
        stream_decoder_chunks,
        args.stream_decoder_left_context,
        stream_decoder_first_chunks,
        args.stream_decoder_input_shape,
    )
    update_codegen_variant_manifest(
        out_dir,
        rms_export_mode,
        subcode_export_mode,
        cache_buckets,
        cache_kernels,
        fused_cache_kernels,
        fused_cache_unroll_steps,
        fused_cache_decode_unroll_steps,
        fused_cache_stateful_mask_steps,
        fused_cache_norepeat_steps,
    )
    print(f"export complete: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
