# OpenVINO 编译缓存

OpenVINO 的 `compile_model()` 会把 GPU kernel/blob 写入 `CACHE_DIR`。本项目默认使用用户级缓存目录，避免模型 IR 目录必须可写，也避免每次启动 sidecar 都从零编译。

默认目录：

- Linux: `$XDG_CACHE_HOME/qwen3-tts-ov/openvino-cache`，未设置时为 `~/.cache/qwen3-tts-ov/openvino-cache`
- Windows: `%LOCALAPPDATA%\\qwen3-tts-ov\\openvino-cache`
- 可用 `QWEN3_TTS_OV_CACHE_DIR` 或 `--ov-cache-dir` 覆盖

缓存会按 OpenVINO 版本、设备、模型 manifest、graph variant、cache kernel/step 和 compile config 分目录。更换驱动、OpenVINO 版本、设备或 IR 后建议重新 warmup。

## 离线填充缓存

首次部署或更新 IR 后运行。`--ir-dir` 必须指向包含 `manifest.json` 的导出目录；如果你当前只有旧目录 `openvino_full/manifest.json`，可以把下面的 `VOICE_DESIGN_IR` 改成 `openvino_full`。

```bash
export VOICE_DESIGN_IR=openvino/voice_design
test -f "$VOICE_DESIGN_IR/manifest.json"
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir "$VOICE_DESIGN_IR" \
  --device GPU \
  --decoder-device GPU \
  --mode cache \
  --cache-step fused \
  --graphs core,stream,buckets \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

`cache-warmup` 默认使用子进程逐图编译。每个图编译完成后子进程退出，GPU/USM 内存会释放；磁盘上的 OpenVINO cache 保留下来。

如需提前填充所有 cache bucket：

```bash
export VOICE_DESIGN_IR=openvino/voice_design
test -f "$VOICE_DESIGN_IR/manifest.json"
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir "$VOICE_DESIGN_IR" \
  --device GPU \
  --mode cache \
  --cache-step fused \
  --graphs core,stream,buckets \
  --preload-buckets all \
  --stream-decoders all
```

这只填充磁盘缓存，不表示服务启动后会常驻所有 bucket。

查看计划但不编译：

```bash
export VOICE_DESIGN_IR=openvino/voice_design
test -f "$VOICE_DESIGN_IR/manifest.json"
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir "$VOICE_DESIGN_IR" \
  --dry-run
```

## 服务启动

sidecar 启动时仍建议使用最小 resident warmup：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

`--model-root openvino` 需要如下布局：

```text
openvino/
  voice_design/manifest.json
  custom_voice/manifest.json
  base/manifest.json
```

只有单个 VoiceDesign IR 时，可以临时使用：

```bash
uv run python -m qwen3_tts_ov serve \
  --ir-dir openvino_full \
  --device GPU \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

`--preload-buckets all` 会把所有 bucket 编译并常驻在服务进程中，iGPU 上容易触发 USM OOM。需要全量预编译时使用 `cache-warmup --preload-buckets all`。

## 常用参数

- `--ov-cache-dir`: 指定 OpenVINO cache 目录。
- `--ov-cache-mode optimize_speed|optimize_size`: 默认 `optimize_speed`。
- `--disable-ov-cache`: 禁用 OpenVINO 编译缓存，用于排查缓存污染。
- `--graphs core,stream,buckets,decoder`: 控制 warmup 范围。
- `--no-subprocess`: 在当前进程顺序编译，调试用；生产不建议。
