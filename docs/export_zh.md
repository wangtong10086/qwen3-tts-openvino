# 导出最快路径 IR

生产 profile `fastest` 需要导出 cached-subcode fused graph、no-repeat unroll4 graph、decode-unroll graph 和 streaming decoder graph，然后生成 `int8_sym_fused_cachedsub` variant。

## VoiceDesign

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25
```

## CustomVoice

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --model-type custom_voice \
  --out-dir openvino/custom_voice \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25
```

## Base / VoiceClone

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --model-type base \
  --out-dir openvino/base \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25 \
  --export-clone-graphs
```

## INT8_SYM 压缩

对每个 IR 目录运行：

```bash
uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --source-variant fp16_fused_cachedsub \
  --variant int8_sym_fused_cachedsub \
  --mode int8_sym \
  --fused-cache-unroll-steps 4
```

压缩不会覆盖 FP16 原图，会在 manifest 中写入 `graph_variants.int8_sym_fused_cachedsub`。

## 验证

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --dry-run
```

如果缺少 required graph，`fastest` 会报错。不要通过切换到旧 profile 绕过错误；应重新导出或重新压缩对应 IR。
