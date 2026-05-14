# 导出最快路径 IR

生产 profile `fastest` 需要导出 paged-KV talker seed graph、cached standalone subcode graph 和 streaming decoder graph，然后生成 `int8_sym_paged_talker_split` variant。默认生产路径不依赖 fixed-bucket cache、unroll 或 decode-unroll graph。

首次使用建议先按 [Quick Start](quick_start_zh.md) 走一键构建。本页用于需要手工控制导出、压缩和验证参数的场景。

## 推荐一键构建

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

该命令会构建 native、导出缺失 IR、执行 `--preset fastest` 压缩，并预热 OpenVINO cache。使用 `--dry-run` 可先查看具体命令。

默认 `--graph-set production` 是低内存构建路径。需要额外导出旧的 fixed-bucket/unroll 诊断图时显式添加：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --graph-set compat
```

## 分步导出

下面的命令用于调试或需要手工控制导出参数时使用。

## VoiceDesign

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --skip-fixed-cache-graphs \
  --cache-buckets 96 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps "" \
  --fused-cache-decode-unroll-steps "" \
  --fused-cache-stateful-mask-steps "" \
  --fused-cache-norepeat-steps "" \
  --export-paged-kv-seed \
  --paged-kv-unroll-steps "" \
  --paged-kv-subcode-attention-kernels sdpa \
  --decoder-tokens 256 \
  --stream-decoder-chunks 12,24 \
  --stream-decoder-first-chunks 8,12 \
  --stream-decoder-left-context 25 \
  --stream-decoder-input-shape static
```

`--stream-decoder-input-shape static` 是生产默认值，用于导出 NPU 可编译的固定 shape streaming decoder，例如 `c0_t8=[1,8,16]`、`c25_t24=[1,49,16]`。旧调试路径需要动态 shape 时可显式使用 `--stream-decoder-input-shape dynamic`。

## CustomVoice

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --model-type custom_voice \
  --out-dir openvino/custom_voice \
  --skip-fixed-cache-graphs \
  --cache-buckets 96 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps "" \
  --fused-cache-decode-unroll-steps "" \
  --fused-cache-stateful-mask-steps "" \
  --fused-cache-norepeat-steps "" \
  --export-paged-kv-seed \
  --paged-kv-unroll-steps "" \
  --paged-kv-subcode-attention-kernels sdpa \
  --decoder-tokens 256 \
  --stream-decoder-chunks 12,24 \
  --stream-decoder-first-chunks 8,12 \
  --stream-decoder-left-context 25 \
  --stream-decoder-input-shape static
```

## Base / VoiceClone

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --model-type base \
  --out-dir openvino/base \
  --skip-fixed-cache-graphs \
  --cache-buckets 96 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps "" \
  --fused-cache-decode-unroll-steps "" \
  --fused-cache-stateful-mask-steps "" \
  --fused-cache-norepeat-steps "" \
  --export-paged-kv-seed \
  --paged-kv-unroll-steps "" \
  --paged-kv-subcode-attention-kernels sdpa \
  --decoder-tokens 256 \
  --stream-decoder-chunks 12,24 \
  --stream-decoder-first-chunks 8,12 \
  --stream-decoder-left-context 25 \
  --stream-decoder-input-shape static \
  --export-clone-graphs
```

## INT8_SYM 压缩

对每个 IR 目录运行：

```bash
uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --preset fastest
```

压缩不会覆盖 FP16 原图，会在 manifest 中写入 `graph_variants.int8_sym_paged_talker_split`。

## 验证

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --dry-run
```

如果缺少必需 graph，`fastest` 会报错。不要通过切换到旧 profile 绕过错误；应重新导出或重新压缩对应 IR。
