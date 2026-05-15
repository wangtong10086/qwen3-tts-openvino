# 实时推理优化历程

本文记录本仓库从基础 OpenVINO 推理到当前 `fastest` runtime 的主要实验目标、过程和结果。它用于后续排查性能回退和选择优化方向。

## 基线目标

最初目标是把 Qwen3-TTS 迁移成 OpenVINO-only runtime，并在桌面 sidecar 场景中达到可用的实时流式合成：

- VoiceDesign、CustomVoice、VoiceClone 三种模式可用。
- 长文本保持和原始 Python 版本一致的完整上下文自回归。
- 浏览器侧可以连续播放 PCM chunk。
- 热启动 VoiceDesign 尽量达到 `RTF < 1`。
- runtime 不依赖 PyTorch；导出阶段允许使用 PyTorch。

## 阶段 1：OpenVINO runtime 和流式输出

目标：

- 将原始 Qwen3-TTS 的 VoiceDesign 路径跑通到真实 WAV，而不是 smoke test。
- 增加 Python iterator、HTTP/WebSocket sidecar 和 Web Demo。
- 让 decoder 支持 chunk 输出，降低 first audio latency。

过程：

- runtime 增加 `generate_codes_iter()` 和 `stream_voice_design/custom_voice/voice_clone()`。
- exporter/runtime 接入 `speech_decoder_stream_c0_t*`、`speech_decoder_stream_c25_t*` 图。
- WebSocket `/v1/tts/stream` 返回 metadata、PCM binary chunks、final JSON。
- Web Demo 增加 jitter buffer，避免 chunk 间播放队列归零。

结果：

- 流式输出可用，浏览器可以边收边播。
- decoder 不是主要瓶颈，主要耗时集中在 codec/codegen 自回归。
- 切分长文本虽然可以规避上下文长度和显存压力，但会导致音色、语气、节奏明显不连续，后续被判定为不正确路径。

## 阶段 2：INT8、INT8_SYM、unroll 和 profiling

目标：

- 使用 INT8 weight-only 压缩降低 talker 图计算量。
- 尝试 fused cache codegen unroll，减少 Python/OpenVINO 往返。
- 建立 benchmark 子进程隔离，避免 GPU/USM 状态污染结果。

过程：

- 扩展压缩脚本生成 `int8_fused`、`int8_sym_fused` 等 variant。
- 引入 `realtime-int8`、`realtime-int8-sym` profile。
- 尝试 `codegen_unroll=4/8`。
- benchmark 改成 profile x run 子进程隔离，输出 JSON 指标。

结果：

- INT8/INT8_SYM 对默认路径提速有限；瓶颈不只是权重带宽。
- unroll 在当时路径下没有稳定收益，且会增加图和调度复杂度。
- 关键热点逐步定位到 codegen 里的 subcode 推理和 decode step，而不是 Python PCM 转换或 WebSocket 发送。

## 阶段 3：算子与 OpenVINO GenAI 思路

目标：

- 排查 RMS、FullyConnected、attention 等算子耗时。
- 参考 OpenVINO GenAI/vLLM 风格，把自回归循环从 Python 下沉到 native runtime。
- 探索 paged KV cache，支持更长输出和更可控显存。

过程：

- 对 OpenVINO perf count 和 runtime timing 做阶段化拆分。
- 分析 RMS/FC/attention 占比，但直接算子级融合在当前 IR 上收益不稳定。
- 参考 OpenVINO GenAI 的长输出管理思路，新增 native C++ audio pipeline。
- 增加 paged-KV attention、U8 KV cache、GQA seed、split subcode path。

结果：

- 当前生产路径依赖 native C++ pipeline。
- 自回归仍按 frame 从头到尾生成，但 KV 使用 paged-KV 管理，不再依赖固定超长图。
- U8 KV cache 主要降低显存压力，不等于计算量等比例下降。
- 当前 `fastest` profile 在本机 Linux GPU 上可以稳定接近或低于 `RTF < 1`。

## 阶段 4：长文本正确性

目标：

- 解决长文本切分后音色不连续、后半段丢失、噪音等问题。
- 保持和原始 Python 版本相同的数学路径，即完整上下文 full-AR。

过程：

- 移除默认文本切分生成路径。
- 长文本仅把音频输出分块，输入文本和生成上下文保持完整。
- 增加上下文预算、实时上下文使用情况显示、EOS 或显存长度停止策略。
- 对异常音频引入外部 omni 质量评估作为辅助判断。

结果：

- 长文本正确路径是 full-AR，不是分段生成。
- 浏览器不能播放或后半丢失的问题主要来自播放端状态、final flush 和长输出预算，不应通过文本切分解决。

## 阶段 5：Windows GPU+NPU 和 release

目标：

- 区分开发模式和最终用户使用模式。
- Linux/Windows 预编译 runtime 包可直接启动 sidecar。
- Windows 原生机器上验证 GPU+NPU 异构路径。

过程：

- 新增 release server，首次启动自动从 Hugging Face 下载 runtime-minimal IR。
- GitHub Actions 构建 Linux/Windows runtime 包，tag 发布时上传 Releases。
- Windows NPU 作为 decoder/audio offload 实验路径，不在 GitHub Actions 中做 NPU 运行验证。
- 增加 `scripts/benchmark_windows_gpu_npu_release.py` 和 PowerShell benchmark 入口。

结果：

- Windows 纯 GPU release 包已可用。
- GPU+NPU 可作为本地实验路径，主要目标是降低 GPU 音频侧负载；是否提升端到端 RTF 需要按机器实测，不作为默认生产路径。

## 当前生产配置

当前推荐配置以代码里的 `fastest` profile 为准：

- `native_pipeline=require`
- `native_paged_kv=require`
- `native_paged_kv_precision=u8`
- `native_paged_kv_cache_input_precision=f32`
- `native_paged_kv_block_size=16`
- `native_paged_kv_split_subcode=on`
- `native_paged_kv_score_aggregation=on`
- `native_codegen_fusion=split`
- `native_dynamic_quantization_group_size=32`
- `chunk_strategy=smooth`

不要默认启用的实验路径：

- 文本切分长文本生成。
- `paged_split_static_decode`，当前 native static decode 编译失败。
- `u8-all` KV/input 全 U8 路径，当前实测更慢。
- `fastest_native_prompt_cache`，cache 命中但没有稳定降低 TTFT 或 RTF。
- `paged_split_block8`，短样本偶尔 compute RTF 低，但端到端和较长输出不优于默认 block size 16。

## 后续优化方向

优先级从高到低：

1. 继续缩短 subcode inference 单步耗时，这是当前最主要热点。
2. 用 fast-path hard metrics 固化每次 benchmark，防止 fallback 被误认为优化。
3. 在 Windows GPU/NPU 机器上按 profile sweep 自动选择 offload，而不是手动猜 `decoder/audio/all`。
4. 对 native pipeline 做更细粒度 perf counter/trace，定位 subcode 图内部具体算子和内存路径。
5. 只在真实 benchmark 证明有效时再把实验 profile 合入默认配置。
