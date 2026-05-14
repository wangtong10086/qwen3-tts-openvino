# OpenVINO 编译缓存

`fastest` profile 依赖 native C++ pipeline、paged-KV seed graph、cached subcode graph 和 streaming decoder。预热的目标是提前触发 OpenVINO compile，避免用户首个请求承担编译开销。

首次部署请先完成 [Quick Start](quick_start_zh.md) 中的一键构建；已有 IR 时再单独运行本页的 cache warmup。

如果还没有导出/压缩 IR，优先使用一键构建：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

该命令默认使用低内存 production 图集合；旧 fixed-bucket/unroll 诊断图不会被导出或预热。

## 预热

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --graphs core,stream,buckets \
  --preload-buckets warmup
```

`fastest` 当前是 paged-KV no-cache profile，因此 `buckets` 不会强制编译旧 fixed-bucket 图；保留该参数是为了和其它诊断 profile 兼容。

Windows GPU+NPU 异构部署时，让预热和服务端使用同一套设备决策：

```powershell
uv run python -m qwen3_tts_ov cache-warmup `
  --ir-dir openvino/voice_design `
  --device GPU `
  --npu-offload decoder `
  --realtime-profile fastest `
  --graphs core,stream,buckets `
  --preload-buckets warmup
```

`--npu-offload auto` 会在检测到 OpenVINO `NPU` 时把 streaming decoder 缓存预热到 NPU；`decoder/require` 会在缺少 NPU 时直接失败。

## 查看将编译哪些图

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --graphs core,stream,buckets \
  --preload-buckets warmup \
  --dry-run
```

典型任务包括：

- `core:text_embedding`
- `core:codec_embedding`
- `core:code_frame_embedding`
- `core:paged_kv_seed:talker_stateful_gqa`
- `core:subcode_greedy_cached`
- `stream:c0_t8` 或旧 IR 中可用的 `stream:c0_t12`
- `stream:c25_t24`

如果缺少 paged-KV seed、cached subcode 或 streaming decoder 图，`fastest` 应直接报错。不要通过切到旧 profile 绕过错误；应重新导出或重新压缩 IR。

## 缓存位置

默认缓存写入用户缓存目录，不写入 IR 目录。可以显式指定：

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --ov-cache-dir ~/.cache/qwen3_tts_ov
```

如果磁盘空间不足，runtime 会关闭本次 compile 的 `CACHE_DIR`，避免写出不完整缓存。不要把 OpenVINO cache、`outputs/` 或用户缓存目录提交到 git。

## Sidecar 预热

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --preload-modes voice_design \
  --preload-buckets warmup
```

`--preload-buckets all` 适合离线预编译，不建议在 iGPU sidecar 常驻启动时使用。长文本会优先使用 manifest 中可用的 sampled paged-KV full-AR 加速图；质量评测生成的 `outputs/long_text_quality/quality_summary.json` 可用于覆盖内置选择。
