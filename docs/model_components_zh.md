# 模型组件与优化设计

本页解释公开 IR 中各个子模型的作用，以及当前 `fastest` 生产路径为什么采用这些拆分和优化。它面向需要理解模型产物、排查性能问题、或继续开发 runtime 的用户。

当前 release 使用 `runtime-minimal` IR profile。它只保留经过验证的生产图；源码导出的完整 `openvino/` 目录可能还包含旧实验图、诊断图和不同 profile 的对照图，不代表 release 默认会加载这些图。

## 总体 Pipeline

```text
HTTP / WebSocket request
  -> prompt builder
  -> text_embedding / codec_embedding
  -> native codec generation
       -> paged-KV talker seed graph
       -> subcode_greedy_cached
  -> speech_decoder_stream
  -> PCM chunk / WAV
```

三种模式共享同一套 codegen 和 decoder 结构，差异主要在 prompt 构造：

| 模式 | IR 目录 | Prompt 来源 | 额外组件 |
| --- | --- | --- | --- |
| VoiceDesign | `voice_design/` | `text + instruct + language` | 无 |
| CustomVoice | `custom_voice/` | `text + speaker + optional instruct + language` | speaker token id 表 |
| VoiceClone | `base/` | `text + ref_audio + ref_text + language` | `speech_encoder`、`speaker_encoder`、`code_frame_embedding` |

长文本不会被自动切段。当前路径使用完整上下文从头到尾自回归生成，直到 EOS 或达到显存/上下文预算。

## 子模型职责

| 子模型 / 图 | 作用 | 为什么保留 |
| --- | --- | --- |
| `text_embedding.xml` | 把文本 token 转成 talker 输入 embedding。 | 所有模式必需；prompt 越长，prefill 越依赖它。 |
| `codec_embedding.xml` | 把语言 token、speaker token、历史 codec token 转成 codec embedding。 | codegen 每步需要上一帧 embedding；CustomVoice 的 speaker token 也走这里。 |
| `graph_variants.*.graphs.paged_kv_seed.*` | 当前生产 talker seed 图。它负责主自回归 decode，输出每个 codec frame 的首个 codebook、hidden state 和 KV 更新。 | 这是 `fastest` 路径的核心图；使用 batch/GQA/paged-KV 形态，兼顾单用户和 online batching。公开 `runtime-minimal` 包通常只保留 `int8_sym_batch_fused_gqa` 这一套生产 seed。 |
| `subcode_greedy_cached.xml` | 根据 talker 输出的 `past_hidden + first_code`，补齐同一个 codec frame 内剩余 codebooks，并输出下一步 embedding。 | Qwen3-TTS 一个音频 frame 有多个 codebook。拆出 subcode 图能减小主 paged attention 图，并避免把易受量化影响的 subcode 逻辑全部塞进主图。 |
| `speech_decoder_stream_c0_t8.xml` | 首个音频块 decoder，无左上下文，输入 8 帧 codec code。 | 降低 TTFT，让浏览器尽早收到第一块可播放 PCM。 |
| `speech_decoder_stream_c25_t12.xml` | 有 25 帧左上下文、输出 12 帧 chunk 的 decoder。 | 低延迟策略的稳态 decoder，也作为兼容回退。 |
| `speech_decoder_stream_c25_t24.xml` | 有 25 帧左上下文、输出 24 帧 chunk 的 decoder。 | 当前 `smooth` 默认稳态 chunk，减少块间边界感和服务端调度频率。 |
| `speech_encoder.xml` | VoiceClone 中把参考音频编码成 codec prompt。 | ICL 克隆路径必需；不走它会丢失参考音频中的内容、节奏和风格线索。 |
| `speaker_encoder.xml` | VoiceClone 中从参考音频提取 speaker embedding。 | 提供音色条件；`x_vector_only=true` 时主要依赖它。 |
| `code_frame_embedding.xml` | 把参考音频 codec frames 转成可拼进 prompt 的 embedding。 | VoiceClone ICL 模式需要把 `ref_code` 接入同一条 codegen prompt。 |
| tokenizer 文件 | 文本 tokenization 和上下文预算计算。 | Web Demo 和 sidecar 需要实时显示 token 使用量，并构造正确 prompt。 |

`speech_decoder_stream_*` 只把 codec codes 解码成音频，不参与文本理解和自回归决策。`subcode_greedy_cached` 只生成 codec token，不直接生成 waveform。

## 为什么拆分 Talker 和 Subcode

Qwen3-TTS 的 12 Hz codec generation 不是每步只生成一个普通 token，而是每个音频 frame 需要一组 codebooks。当前模型中 `num_code_groups=16`，可以理解为每一帧有 16 个 codec code。

生产路径采用两阶段：

1. talker/paged-KV 图生成当前 frame 的 `first_code` 和 hidden state。
2. `subcode_greedy_cached` 根据 `first_code + hidden` 生成剩余 subcodes，并汇总出下一步 embedding。

这样做的主要原因：

- 主 talker 图可以专注长上下文 attention 和 KV 更新，图更小，paged-KV 转换更稳定。
- subcode 预测不需要长上下文 paged attention，只需要当前 frame 内的小序列 cache。
- subcode 的量化和 sampling 更容易影响音色细节。拆出来后可以单独选择是否压缩、是否使用 batch-safe 版本。
- online batching 中可以把请求调度放在外层，内层保持固定图和固定数据流，接近 vLLM 的分层方式。

`greedy` 表示 subcode 使用 argmax。它通常比主 talker 的 sampling 对自然度影响小，但能明显减少 host 侧 logits 处理和随机采样开销。当前质量优先点在主 codegen 的采样策略，而不是把每个 subcode 都做 sampling。

## 为什么使用 Paged-KV

早期固定 cache bucket 的做法需要为不同长度导出和编译多套图，长文本容易撞到 bucket 上限，也会增加 release 体积。当前 native backend 使用 OpenVINO paged attention：

- KV cache 按 block 管理，默认 block size 为 16。
- Web Demo 根据显存占比和 tokenizer 结果显示上下文使用情况。
- 生成可以持续到 EOS 或显存/上下文预算，而不是靠切分文本绕过限制。
- 多请求时调度层可以控制 active batch 和 `max_num_batched_tokens`。

默认 KV cache precision 为 U8，用于降低显存占用，让长文本和并发更稳定。这里的 U8 指 KV 数据存储精度；`cache_position`、block table 等索引输入保持整数类型是正常的。

## 为什么使用 INT8_SYM 权重

公开 `runtime-minimal` IR 的生产在线批处理 variant 是 `int8_sym_batch_fused_gqa`。它主要压缩 talker 侧大矩阵权重，减少显存和带宽压力。

源码完整导出目录中可能还存在 `int8_sym_paged_talker_split` 等开发/诊断 variant。它们用于对照 split-subcode、长文本 full-AR 或历史性能实验；是否被选中取决于 manifest、runtime profile 和环境变量。面向普通用户的 release 默认应以 `runtime-minimal` manifest 为准。

不是所有图都强制 INT8，原因是：

- iGPU 上 INT8 不一定总比 FP16 快，取决于 plugin 的 kernel、反量化开销和图融合结果。
- subcode 和 speaker/clone 相关小图对音质漂移更敏感，盲目压缩可能让音色和发音更差。
- release 目标是稳定的默认路径，而不是保留所有实验 variant。

因此 `runtime-minimal` 只发布当前验证过的组合：INT8_SYM batch/GQA talker seed + cached subcode + streaming decoder。更多 PTQ/INT4/NF4 或全 INT8 activation 路径应作为实验分支验证，不能直接进入默认 release。

## 为什么保留多个 Streaming Decoder

公开模型库中常见三个 streaming decoder：

| 文件 | 场景 | 取舍 |
| --- | --- | --- |
| `speech_decoder_stream_c0_t8.xml` | 第一块音频 | 没有左上下文，首包快，但只适合开头。 |
| `speech_decoder_stream_c25_t12.xml` | 低延迟稳态或兼容 fallback | chunk 小，延迟低，调度次数更多。 |
| `speech_decoder_stream_c25_t24.xml` | 默认 `smooth` 稳态 | chunk 大一些，边界更稳，调度开销更低。 |

`c25` 表示使用 25 帧左上下文。左上下文不会重复播放，只用于让 decoder 在 chunk 边界保持连续。删除这些图会导致首包变慢、块间边界更明显，或直接无法匹配当前 runtime 的默认策略。

## Online Batching 的位置

online batching 不是单独的模型文件，而是服务端和 native backend 的调度层。它负责：

- 接收随时到达的请求。
- 分离 prefill 和 decode。
- 按 active batch、prompt 长度和 token budget 调度。
- 在并发时提高 aggregate TPS。

这和 `subcode_greedy_cached.xml`、paged-KV seed graph 的关系是：调度层决定每轮哪些请求参与计算，底层图仍然是固定 OpenVINO IR。单用户和多用户尽量复用同一套 IR，避免同时加载两套模型造成显存压力。

## Release-minimal 与 Full Export

`build-fastest` 的完整导出目录可能含有：

- no-cache talker 图
- fixed cache bucket 图
- unroll 诊断图
- 多种 streaming chunk 对照图
- 旧 quantization/profile sweep 图
- full speech decoder 对照图

这些图对开发调试有用，但不适合作为公开 release 默认下载内容。公开 Hugging Face IR 使用 `runtime-minimal`，只保留 sidecar 当前生产路径需要的文件，减少下载体积、显存压力和用户理解成本。

如果你在本地看到更多图，优先以 `manifest.json` 和 `/health` 中实际命中的字段为准，而不是按目录里的文件名推断运行路径。

## 如何确认当前是否命中生产路径

启动服务后查看：

```bash
curl http://127.0.0.1:17860/health | python -m json.tool
```

重点字段：

| 字段 | 期望 |
| --- | --- |
| `realtime_profile` | `fastest` |
| `graph_variant` | 当前 manifest 支持的生产 variant；公开 `runtime-minimal` 通常为 `int8_sym_batch_fused_gqa` |
| `paged_kv` | `true` |
| `native_paged_kv_precision` | `u8` 或显式配置的 KV profile |
| `online_batching` | `on` |
| `generation_fallback_allowed` | `false`，生产路径不应静默回退 |
| `fast_path_ok` / `fast_path_failure_reason` | runtime 或请求级指标应显示 fast path 成功，失败时先看原因 |
| streaming metadata `segmented` | `no`，长文本不应自动切段 |

Web Demo 的调试面板也会显示 mode、profile、paged-KV、KV cache、chunk strategy、fallback 和上下文使用情况。性能异常时先确认这些字段，再看具体算子 profile。
