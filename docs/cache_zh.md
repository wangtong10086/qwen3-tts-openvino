# OpenVINO 编译缓存

生产路径使用 `fastest` profile。它会编译 native C++ pipeline 所需的 unroll4、decode-unroll、streaming decoder 和基础 embedding 图。

## 预热

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --graphs core,stream,buckets \
  --preload-buckets warmup
```

默认缓存写入用户缓存目录，不写入 IR 目录。`--preload-buckets warmup` 只预热当前最快路径需要的 bucket，避免 iGPU USM 压力过大。

## 查看任务

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --dry-run
```

期望任务包含：

- core embedding graph
- `fused_cache_step_unroll4_*_norepeat_cache96_*`
- `fused_cache_decode_unroll4_*_norepeat_cache96_*`
- `speech_decoder_stream_c0_t8.xml`
- `speech_decoder_stream_c25_t24.xml`

## Sidecar

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --preload-modes voice_design \
  --preload-buckets warmup
```

`serve --realtime-profile auto` 会读取 `outputs/realtime_bench/streaming_profiles.json` 中已通过的 p90 summary；没有 benchmark 结果时回到 `fastest`。

## 注意事项

- 不要把 `openvino/**/ov_cache/`、`outputs/` 或用户缓存目录提交到 git。
- `--preload-buckets all` 只适合离线预编译，不建议在 sidecar 常驻启动时使用。
- 如果缺少 no-repeat 或 decode-unroll graph，`fastest` 会直接报错；这是预期行为，避免静默降级到慢路径。
