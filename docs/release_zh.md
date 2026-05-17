# Release 使用说明

Release 面向最终调用方。调用方不需要 PyTorch、导出脚本或源码开发环境，只需要下载 GitHub Release 中的 runtime App 包。

首次启动时，如果本地没有 OpenVINO IR，release server 会自动从 Hugging Face 下载默认公开 IR 到用户缓存目录，然后继续启动。

新机器先看 [前置条件](prerequisites_zh.md)，常见启动/下载/设备问题见 [Troubleshooting](troubleshooting_zh.md)。

当前正式版本：`v0.1.4`

## 下载内容

Runtime App 包在 GitHub Release：

- Linux: `qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal.tar.zst`
- Windows: `qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal.zip`
- Release 页面：<https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.4>

默认自动下载的已编译 OpenVINO IR 在 Hugging Face：

- Model repo: <https://huggingface.co/waston10086/qwen3-tts-openvino-voice-design>
- 使用目录：`openvino_realtime/`
- 包含模式：
  - `openvino_realtime/voice_design`
  - `openvino_realtime/custom_voice`
  - `openvino_realtime/base`，用于 VoiceClone

离线部署或需要预下载时，可以从网页下载，也可以用 Hugging Face CLI：

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

手动下载完成后，模型根目录应为：

```text
qwen3-tts-openvino-ir/openvino_realtime
```

## Linux 启动

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal.tar.zst

cd qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal
./qwen3-tts-ov-server \
  --device GPU
```

打开：

```text
http://127.0.0.1:17860/
```

## Windows 启动

GPU-only 路径：

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal.zip -DestinationPath .

cd qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal
.\qwen3-tts-ov-server.exe `
  --device GPU
```

Intel GPU+NPU 机器可以显式把 streaming decoder 放到 NPU：

```powershell
.\qwen3-tts-ov-server.exe `
  --device GPU `
  --npu-offload decoder
```

该模式要求 Windows 原生 OpenVINO 能看到 `NPU`，并且 IR 中的 streaming decoder 是固定 shape。缺少 NPU 或 NPU decoder 编译失败时，`--npu-offload decoder` 会报错；需要自动回退时使用 `--npu-offload auto`。完整验证流程见 [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md)。

打开：

```text
http://127.0.0.1:17860/
```

## 调用接口

Web Demo：

```text
http://127.0.0.1:17860/
```

Web Demo 可以直接测试 VoiceDesign、CustomVoice、VoiceClone、长文本 full-AR、参考音频上传、模型一键下载、请求 JSON/curl 复制、自定义最终请求 JSON 和同时多请求 smoke。多请求 smoke 只用于现场观察，正式性能报告仍以源码仓库中的 benchmark 脚本为准。

查看模型是否已下载：

```bash
curl http://127.0.0.1:17860/v1/models
```

手动触发某个模式下载：

```bash
curl -X POST http://127.0.0.1:17860/v1/models/download \
  -H "content-type: application/json" \
  -d '{"mode":"voice_clone","sync":false}'
```

HTTP NDJSON：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d '{"mode":"voice_design","text":"你好，这是 release 包测试。","language":"Chinese","instruct":"A calm young female voice."}'
```

OpenAI-compatible Speech API：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容接口测试。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm"}' \
  --output speech.pcm
```

更多 CLI、HTTP、WebSocket 和 Python API 说明见 [运行接口](runtime_zh.md)。
完整请求字段、默认值和返回格式见 [API Reference](api_reference_zh.md)。

仓库内还有最小 Python 客户端，适合桌面应用或其他本地程序参考：

```bash
python examples/python/http_tts_wav.py --output outputs/example_http.wav
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

## 发布物边界

- GitHub Release 只包含 runtime App 包，不包含模型权重或 OpenVINO IR。
- Hugging Face model repo 存放已编译 OpenVINO IR，当前公开包含 VoiceDesign、CustomVoice 和 Base/VoiceClone realtime IR。release server 默认在缺少本地 IR 时自动下载 `openvino_realtime/`。
- 公开 HF IR 使用 `runtime-minimal` profile，只保留当前 `fastest` 生产路径需要的图；旧实验图和诊断图不发布，避免下载体积和用户理解成本膨胀。
- CustomVoice 需要 `custom_voice/manifest.json`，VoiceClone 需要 Base/VoiceClone IR 的 `base/manifest.json`。服务端 `/health` 会返回 `available_modes`，Web Demo 会显示缺失/已就绪，并可一键下载对应模式。
- VoiceClone 默认走 `ref_audio + ref_text` ICL 克隆路径，`x_vector_only` 默认关闭。最终用户在 Web Demo 中上传参考音频时，应同时填写对应参考文本；只有做 speaker embedding-only 对照实验时才开启 `x_vector_only`。
- OpenVINO compile cache 会在用户机器首次运行时生成，不随 release 分发。
- 需要私有分发 IR 时，可以使用 `scripts/package_ir.py` 自行打包；当前公开分发推荐直接使用 Hugging Face。

## 自动下载控制

默认启动即可自动下载：

```bash
./qwen3-tts-ov-server --device GPU
```

常用控制参数：

- `--no-auto-download-model`: 禁止启动时联网下载。
- `--model-root <path>`: 使用本地已下载或私有分发的 IR。
- `--model-cache-dir <path>`: 指定自动下载缓存目录。
- `--model-repo <repo>`、`--model-revision <rev>`、`--model-subdir <dir>`: 指定下载来源。
- `--max-continuous-prompt-tokens auto|0|N`: 长文本 full-AR prompt token 预算。默认 `auto`；GPU 路径使用 KV-cache planner 按显存和 KV/cache-input 精度计算，CPU-only 路径使用保守固定预算；`0` 表示关闭该预算保护。
- `--kv-cache-profile auto|fp16|bf16|u8|u8-input|u8-all`: paged-KV cache 显存档位。默认 `auto` 跟随最快路径，即 U8 KV cache；需要保守对照时使用 `fp16`。
- `--max-vram-ratio auto|N`: Web Demo 和服务端 prompt 预算使用的最大显存占比；例如 `70` 表示按 70% 可用显存计算长文本 token 上限。
- `--kv-cache-reserve-mb auto|N`: KV planner 预留给模型权重、中间 buffer 和驱动的显存。
- `--kv-cache-max-blocks auto|N`: 手动限制可计划的 KV block 数。
- `--kv-cache-preallocation auto|off|static`: `auto` 只计算预算；`static` 同时启用 native static decode block 容量。

缓存目录也可以用环境变量 `QWEN3_TTS_OV_MODEL_CACHE_DIR` 指定。

Web Demo 的模型下载按钮调用服务端 `/v1/models/download`，下载到当前 `model_root` 下的对应目录：

```text
VoiceDesign  -> openvino/voice_design
CustomVoice  -> openvino/custom_voice
VoiceClone   -> openvino/base
```

如果不同模式使用不同 Hugging Face repo/subdir，可以用环境变量覆盖：

```bash
export QWEN3_TTS_OV_MODEL_REPO_CUSTOM_VOICE=owner/custom-voice-ir
export QWEN3_TTS_OV_MODEL_SUBDIR_CUSTOM_VOICE=openvino_realtime
export QWEN3_TTS_OV_MODEL_REPO_VOICE_CLONE=owner/base-ir
export QWEN3_TTS_OV_MODEL_SUBDIR_VOICE_CLONE=openvino_realtime
```

## 长文本预算

长文本默认不切分输入文本，而是使用同一条 full-AR 自回归链路。为了避免过长 prompt 在部分 GPU/驱动上触发 OpenVINO USM 分配失败，server 会在推理前做 prompt token 预算检查。

默认 `--max-continuous-prompt-tokens auto` 会在 GPU 路径上使用 KV-cache planner：先读取模型层数、head 数、head dim、上下文长度、KV storage 精度和 cache input 精度，再按 `--max-vram-ratio` 与保留显存计算可用 KV blocks，最后得到 `effective_max_total_tokens` 和 `effective_max_continuous_prompt_tokens`。默认 `u8` 档位的 KV storage 是 U8，但 cache input 仍为 FP32，因此 planner 会按实际 cache input 做保守预算；需要进一步降低 block 预算占用时再评测 `u8-all`。

如果请求仍然超出预算，可以按机器显存/内存情况提高或关闭限制：

```bash
./qwen3-tts-ov-server --device GPU --max-continuous-prompt-tokens 4096

# 或关闭 prompt 预算保护，仅保留 OpenVINO/USM 运行时错误与重试
./qwen3-tts-ov-server --device GPU --max-continuous-prompt-tokens 0
```

服务端 `/health` 和流式 metadata 会返回 `max_continuous_prompt_tokens_config`、`effective_max_continuous_prompt_tokens`、`effective_max_total_tokens`、`preallocated_kv_blocks`、`kv_cache_budget_bytes` 和 `long_text_budget_policy`，用于确认实际生效值。

## KV Cache 显存档位

默认 release 路径使用 U8 paged-KV cache，降低长文本和长输出时的显存压力。需要显式指定时可以写：

```bash
./qwen3-tts-ov-server --device GPU --kv-cache-profile u8 --max-vram-ratio 70
```

可选档位：

- `fp16`: KV cache 为 FP16，保守质量对照路径。
- `bf16`: KV cache 为 BF16，用于对比硬件行为。
- `u8`: KV cache 存储为 U8，cache input 仍保持 FP32，默认生产路径。
- `u8-input`: KV cache 保持 FP16，但 cache input 改为 U8，主要用于实验。
- `u8-all`: KV cache 和 cache input 都使用 U8，显存更低，但需要额外做音质验证。

`/health` 的 `warmup`、`memory` 和 runtime metadata 会返回 `kv_cache_profile`、`native_paged_kv_precision`、`native_paged_kv_cache_input_precision`、`kv_cache_relative_to_fp16`。其中 `kv_cache_relative_to_fp16=0.5` 表示 KV cache 元素理论占用约为 FP16 的一半。这里压缩的是 paged-KV cache 存储，不等价于全部 attention 算子都以 INT8 计算。

## 系统要求

- Linux x86_64 或 Windows x64。
- Intel GPU 使用时需要目标机器已安装对应 GPU 驱动、OpenCL/Level Zero runtime，并且 OpenVINO `available_devices` 能看到 `GPU`。
- `--device CPU` 可作为 smoke/fallback，但实时性能不保证。
- `--device AUTO` 可用于实验，实际设备选择由 OpenVINO AUTO 插件决定。
- Windows GPU+NPU 使用 `--device GPU --npu-offload auto|decoder|audio|all|require`；严格验证用 `decoder`，需要回退用 `auto`。
- Windows runtime 必须使用 Windows 构建产物；不要使用 Linux 交叉编译出的 DLL。

更完整的 Python、uv、OpenVINO、Intel GPU/NPU 和缓存说明见 [前置条件](prerequisites_zh.md)。

## 维护者发布流程

正式 runtime 发布由 GitHub Actions 的 `release-runtime` workflow 负责。推送 `v*` tag 会自动构建 Linux/Windows runtime-minimal App 包，完成 smoke 后上传到对应 GitHub Release：

```bash
git tag v0.1.4
git push origin v0.1.4
```

Actions 页面也可以手动运行 `release-runtime`，填写 `version`，用于重发指定版本。分平台的 `release-linux` 和 `release-windows` 只作为手动诊断入口保留。
