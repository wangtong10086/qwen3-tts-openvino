import argparse
import json
import os
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
from transformers import AutoConfig, AutoModel

import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as tokenizer_v2
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb,
    repeat_kv,
)
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


NEG_INF = -3.4028234663852886e38


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
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model.eval()
        self.codec_head = talker.codec_head.eval()

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
    def __init__(self, talker, max_cache_len: int, attention_kernel: str = "exact"):
        super().__init__()
        self.model = talker.model.eval()
        self.codec_head = talker.codec_head.eval()
        self.max_cache_len = int(max_cache_len)
        if attention_kernel not in {"exact", "sdpa"}:
            raise ValueError(f"unsupported attention_kernel={attention_kernel!r}")
        self.attention_kernel = attention_kernel

    def _attention(self, layer, hidden_states, attention_mask, position_embeddings, past_key, past_value, cache_position):
        attn = layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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

    def forward(self, inputs_embeds, cache_position, attention_mask, *past_key_values):
        batch = inputs_embeds.shape[0]
        position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)
        text_position_ids = position_ids[0]

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        present_key_values = []

        for layer_index, decoder_layer in enumerate(self.model.layers):
            past_key = past_key_values[layer_index * 2]
            past_value = past_key_values[layer_index * 2 + 1]

            residual = hidden_states
            normed_states = decoder_layer.input_layernorm(hidden_states)
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
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            present_key_values.extend([present_key, present_value])

        hidden_states = self.model.norm(hidden_states)
        last_hidden = hidden_states[:, -1:, :]
        logits = self.codec_head(last_hidden)[:, -1, :]
        return (logits, last_hidden, *present_key_values)


class SubcodeGreedyWrapper(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.first_embedding = talker.get_input_embeddings().eval()
        self.predictor = talker.code_predictor.eval()
        self.predictor_model = talker.code_predictor.model.eval()
        self.small_to_mtp_projection = talker.code_predictor.small_to_mtp_projection.eval()
        self.lm_head = talker.code_predictor.lm_head.eval()
        self.sub_embeddings = talker.code_predictor.get_input_embeddings().eval()

    def _predict_next(self, embeds, head_index: int):
        inputs_embeds = torch.cat(embeds, dim=1)
        hidden_states = self.small_to_mtp_projection(inputs_embeds)
        batch, seq_len = hidden_states.shape[:2]
        position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).view(1, -1)
        additive_mask = causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        position_embeddings = self.predictor_model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.predictor_model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=additive_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.predictor_model.norm(hidden_states)
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
    def __init__(self, talker):
        super().__init__()
        self.first_embedding = talker.get_input_embeddings().eval()
        self.predictor_model = talker.code_predictor.model.eval()
        self.small_to_mtp_projection = talker.code_predictor.small_to_mtp_projection.eval()
        self.lm_head = talker.code_predictor.lm_head.eval()
        self.sub_embeddings = talker.code_predictor.get_input_embeddings().eval()
        self.max_sub_len = talker.code_predictor.config.num_code_groups + 1

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

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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
            normed_states = decoder_layer.input_layernorm(hidden_states)
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
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            present_key_values.extend([present_key, present_value])

        return self.predictor_model.norm(hidden_states), present_key_values

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


class FusedNoCacheCodecStepWrapper(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.talker = TalkerNoCacheWrapper(talker)
        self.subcode = SubcodeGreedyWrapper(talker)
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
    def __init__(self, talker, max_cache_len: int, attention_kernel: str = "exact"):
        super().__init__()
        self.talker = StatefulTalkerWrapper(talker, max_cache_len, attention_kernel=attention_kernel)
        self.subcode = SubcodeGreedyWrapper(talker)
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


class DecodeWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder.eval()

    def forward(self, audio_codes):
        codes = torch.clamp(audio_codes, min=0).transpose(1, 2)
        return self.decoder(codes).squeeze(1)


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


def save_stateful_talker_model(
    talker,
    path: Path,
    example_seq_len: int,
    max_cache_len: int,
    attention_kernel: str,
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
    wrapper = StatefulTalkerWrapper(talker, max_cache_len, attention_kernel=attention_kernel)

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
    wrapper = FusedCacheCodecStepWrapper(talker, max_cache_len, attention_kernel=attention_kernel)

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


def load_speech_tokenizer(model_dir: str):
    tokenizer_v2.create_causal_mask = lambda **kwargs: None
    tokenizer_v2.create_sliding_window_causal_mask = lambda **kwargs: None
    return Qwen3TTSTokenizer.from_pretrained(
        os.path.join(model_dir, "speech_tokenizer"),
        dtype=torch.float32,
        attn_implementation="sdpa",
    )


def export_speech_encoder(model_dir: str, out_dir: Path, force: bool = False) -> Path:
    path = out_dir / "speech_encoder.xml"
    if path.exists() and path.with_suffix(".bin").exists() and not force:
        return path
    speech_tokenizer = load_speech_tokenizer(model_dir)
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


def cache_graphs(prefix: str, cache_buckets: list[int], cache_kernels: list[str]):
    return {
        kernel: {str(length): f"{prefix}_{kernel}_cache{length}.xml" for length in cache_buckets}
        for kernel in cache_kernels
    }


def write_manifest(
    model_dir: str,
    out_dir: Path,
    decoder_tokens: list[int],
    cache_buckets: list[int],
    cache_kernels: list[str],
    fused_cache_kernels: list[str],
) -> None:
    manifest_path = out_dir / "manifest.json"
    previous_manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            previous_manifest = json.load(f)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "speech_tokenizer", "config.json"), "r", encoding="utf-8") as f:
        speech_config = json.load(f)

    max_cache_len = max(cache_buckets) if cache_buckets else 0
    stateful_buckets = cache_graphs("talker_stateful", cache_buckets, cache_kernels)
    fused_cache_buckets = cache_graphs("fused_cache_step", cache_buckets, fused_cache_kernels)
    manifest = {
        "format": "qwen3_tts_openvino_v3",
        "model_dir": str(Path(model_dir).resolve()),
        "tts_model_type": config.get("tts_model_type", "unknown"),
        "tokenizer_type": config.get("tokenizer_type"),
        "tts_model_size": config.get("tts_model_size"),
        "precision": "fp16_weights",
        "runtime_requires_torch": False,
        "max_cache_len": int(max_cache_len),
        "cache_buckets": cache_buckets,
        "cache_kernels": cache_kernels,
        "default_cache_kernel": "exact" if "exact" in cache_kernels else cache_kernels[0],
        "default_cache_step": "split",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "talker": "talker_no_cache.xml",
            "talker_stateful": stateful_buckets.get("exact", {}).get(str(max_cache_len)),
            "talker_stateful_buckets": stateful_buckets,
            "fused_cache_step_buckets": fused_cache_buckets,
            "subcode_greedy": "subcode_greedy.xml",
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "code_frame_embedding": "code_frame_embedding.xml",
            "fused_no_cache_step": "fused_no_cache_step.xml",
            "speech_decoder": {str(t): f"speech_decoder_t{t}.xml" for t in decoder_tokens},
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
        default="128,192,256,320,384",
        help="Comma-separated stateful talker cache lengths.",
    )
    parser.add_argument("--cache-kernels", default="exact,sdpa", help="Comma-separated stateful attention kernels.")
    parser.add_argument("--fused-cache-kernels", default="exact", help="Comma-separated fused cache kernels.")
    parser.add_argument("--decoder-tokens", default="64,128,256")
    parser.add_argument("--force-stateful", action="store_true", help="Re-export stateful talker cache graphs.")
    parser.add_argument("--force-cache-graphs", action="store_true", help="Re-export stateful and fused cache graphs.")
    parser.add_argument("--export-clone-graphs", action="store_true", help="Export speech/speaker encoder graphs when available.")
    args = parser.parse_args()

    with open(os.path.join(args.model, "config.json"), "r", encoding="utf-8") as f:
        model_config = json.load(f)
    detected_model_type = model_config.get("tts_model_type", "unknown")
    if args.model_type != "auto" and args.model_type != detected_model_type:
        raise ValueError(f"--model-type={args.model_type!r} does not match model config tts_model_type={detected_model_type!r}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("openvino") / detected_model_type
    decoder_tokens = parse_int_list(args.decoder_tokens)
    cache_buckets = parse_int_list(args.cache_buckets) if args.cache_buckets else [int(args.max_cache_len)]
    cache_kernels = parse_str_list(args.cache_kernels)
    fused_cache_kernels = parse_str_list(args.fused_cache_kernels)
    if not cache_buckets:
        raise ValueError("at least one cache bucket is required")
    for kernel in cache_kernels + fused_cache_kernels:
        if kernel not in {"exact", "sdpa"}:
            raise ValueError(f"unsupported cache kernel {kernel!r}")

    core_paths = [
        out_dir / "text_embedding.xml",
        out_dir / "codec_embedding.xml",
        out_dir / "talker_no_cache.xml",
        out_dir / "subcode_greedy.xml",
        out_dir / "subcode_greedy_cached.xml",
        out_dir / "code_frame_embedding.xml",
        out_dir / "fused_no_cache_step.xml",
    ]
    core_paths.extend(
        out_dir / f"talker_stateful_{kernel}_cache{length}.xml"
        for kernel in cache_kernels
        for length in cache_buckets
    )
    core_paths.extend(
        out_dir / f"fused_cache_step_{kernel}_cache{length}.xml"
        for kernel in fused_cache_kernels
        for length in cache_buckets
    )
    force_cache_graphs = args.force_stateful or args.force_cache_graphs
    needs_core_export = force_cache_graphs or any(
        not path.exists() or not path.with_suffix(".bin").exists() for path in core_paths
    )

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
            TalkerNoCacheWrapper(talker),
            (torch.zeros((1, args.example_seq_len, talker.config.hidden_size), dtype=torch.float32),),
            out_dir / "talker_no_cache.xml",
            [ov.PartialShape([1, -1, talker.config.hidden_size])],
        )
        for kernel in cache_kernels:
            for cache_len in cache_buckets:
                save_stateful_talker_model(
                    talker,
                    out_dir / f"talker_stateful_{kernel}_cache{cache_len}.xml",
                    min(args.example_seq_len, cache_len),
                    cache_len,
                    kernel,
                    force=force_cache_graphs,
                )
        save_openvino_model(
            SubcodeGreedyWrapper(talker),
            (
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.long),
            ),
            out_dir / "subcode_greedy.xml",
            [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
        )
        save_openvino_model(
            SubcodeGreedyCachedWrapper(talker),
            (
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.long),
            ),
            out_dir / "subcode_greedy_cached.xml",
            [ov.PartialShape([1, 1, talker.config.hidden_size]), ov.PartialShape([1, 1])],
        )
        save_openvino_model(
            FusedNoCacheCodecStepWrapper(talker),
            (
                torch.zeros((1, args.example_seq_len, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, 1, talker.config.hidden_size), dtype=torch.float32),
                torch.zeros((1, talker.config.vocab_size), dtype=torch.float32),
                torch.ones((1,), dtype=torch.float32),
                torch.full((1,), 1.05, dtype=torch.float32),
            ),
            out_dir / "fused_no_cache_step.xml",
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
                    out_dir / f"fused_cache_step_{kernel}_cache{cache_len}.xml",
                    min(args.example_seq_len, cache_len),
                    cache_len,
                    kernel,
                    force=force_cache_graphs,
                )
        with open(os.path.join(args.model, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        if args.export_clone_graphs or config.get("tts_model_type") == "base":
            export_speech_encoder(args.model, out_dir, force=force_cache_graphs)
            export_speaker_encoder(model, out_dir, force=force_cache_graphs)
    else:
        print("core OpenVINO graphs already exist; skipping main model load", flush=True)

    for tokens in decoder_tokens:
        export_decoder(args.model, out_dir, tokens)

    write_manifest(args.model, out_dir, decoder_tokens, cache_buckets, cache_kernels, fused_cache_kernels)
    print(f"export complete: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
