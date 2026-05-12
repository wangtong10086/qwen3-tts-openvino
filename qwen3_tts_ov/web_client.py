WEB_CLIENT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-TTS OpenVINO 流式测试</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #eef5f3;
      --text: #16201d;
      --muted: #65716d;
      --line: #dce3e0;
      --accent: #16745f;
      --accent-2: #2357a6;
      --danger: #b83232;
      --warn: #a86200;
      --shadow: 0 12px 30px rgba(22, 32, 29, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    .app {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 32px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
    }

    .status-pill {
      min-width: 148px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      text-align: center;
      white-space: nowrap;
    }

    .status-pill.good { color: var(--accent); border-color: rgba(22, 116, 95, 0.35); }
    .status-pill.bad { color: var(--danger); border-color: rgba(184, 50, 50, 0.35); }

    main {
      display: grid;
      grid-template-columns: minmax(320px, 410px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }

    h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }

    .panel-body { padding: 16px; }

    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      padding: 9px 10px;
      outline: none;
    }

    textarea {
      min-height: 92px;
      resize: vertical;
    }

    input:focus, textarea:focus, select:focus {
      border-color: rgba(35, 87, 166, 0.65);
      box-shadow: 0 0 0 3px rgba(35, 87, 166, 0.12);
    }

    .field { margin-bottom: 13px; }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .check {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-weight: 500;
    }

    .check input {
      width: 16px;
      height: 16px;
      margin: 0;
    }

    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 13px;
      color: #fff;
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button.secondary { background: var(--accent-2); }
    button.ghost {
      color: var(--text);
      background: #eef1f3;
    }
    button.danger { background: var(--danger); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      min-height: 76px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }

    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 20px;
      letter-spacing: 0;
    }

    .meters {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 14px 0;
    }

    .bar {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e7ecea;
    }

    .bar > i {
      display: block;
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 120ms linear;
    }

    .console {
      height: 300px;
      overflow: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111816;
      color: #dcebe5;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }

    .hint {
      color: var(--muted);
      font-size: 12px;
    }

    .hidden { display: none; }

    @media (max-width: 860px) {
      .app { width: min(100vw - 20px, 720px); padding-top: 14px; }
      header { align-items: stretch; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr 1fr; }
    }

    @media (max-width: 520px) {
      .grid-2, .meters { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr; }
      .row button { flex: 1 1 140px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Qwen3-TTS OpenVINO 流式测试</h1>
        <p class="subtitle">浏览器通过 WebSocket 接收 PCM16 音频块，并用 Web Audio 实时播放。</p>
      </div>
      <div id="health" class="status-pill">检查服务中</div>
    </header>

    <main>
      <section>
        <div class="panel-head">
          <h2>请求参数</h2>
          <span class="hint">/v1/tts/stream</span>
        </div>
        <div class="panel-body">
          <div class="field">
            <label for="wsUrl">WebSocket 地址</label>
            <input id="wsUrl" autocomplete="off">
          </div>

          <div class="grid-2">
            <div class="field">
              <label for="mode">模式</label>
              <select id="mode">
                <option value="voice_design">VoiceDesign</option>
                <option value="custom_voice">CustomVoice</option>
                <option value="voice_clone">VoiceClone</option>
              </select>
            </div>
            <div class="field">
              <label for="language">语言</label>
              <select id="language">
                <option value="Chinese">Chinese</option>
                <option value="English">English</option>
                <option value="Auto">Auto</option>
                <option value="Japanese">Japanese</option>
                <option value="Korean">Korean</option>
              </select>
            </div>
          </div>

          <div class="field">
            <label for="text">文本</label>
            <textarea id="text">你好，欢迎使用 OpenVINO 流式语音合成。</textarea>
          </div>

          <div class="field" id="instructField">
            <label for="instruct">Instruct</label>
            <textarea id="instruct">用自然、清晰的中文女声朗读。</textarea>
          </div>

          <div class="field hidden" id="speakerField">
            <label for="speaker">Speaker</label>
            <input id="speaker" placeholder="Vivian">
          </div>

          <div class="hidden" id="cloneFields">
            <div class="field">
              <label for="refAudio">参考音频路径或 URL</label>
              <input id="refAudio" placeholder="/path/to/reference.wav">
            </div>
            <div class="field">
              <label for="refText">参考文本</label>
              <textarea id="refText" placeholder="Reference transcript"></textarea>
            </div>
            <div class="field">
              <label class="check"><input id="xVectorOnly" type="checkbox"> x_vector_only</label>
            </div>
          </div>

          <div class="grid-2">
            <div class="field">
              <label for="maxNewTokens">max_new_tokens</label>
              <input id="maxNewTokens" type="number" min="1" value="48">
            </div>
            <div class="field">
              <label for="minNewTokens">min_new_tokens</label>
              <input id="minNewTokens" type="number" min="0" value="12">
            </div>
            <div class="field">
              <label for="chunkStrategy">chunk_strategy</label>
              <select id="chunkStrategy">
                <option value="realtime">realtime</option>
                <option value="low_latency">low_latency</option>
                <option value="smooth" selected>smooth</option>
                <option value="balanced">balanced</option>
                <option value="stable">stable</option>
              </select>
            </div>
            <div class="field">
              <label for="chunkFrames">chunk_frames</label>
              <input id="chunkFrames" type="number" min="1" value="12">
            </div>
            <div class="field">
              <label for="leftContextFrames">left_context_frames</label>
              <input id="leftContextFrames" type="number" min="0" value="25">
            </div>
          </div>

          <div class="row">
            <button id="startBtn">开始合成</button>
            <button id="stopBtn" class="danger" disabled>停止</button>
            <button id="downloadBtn" class="secondary" disabled>下载 WAV</button>
            <button id="clearBtn" class="ghost">清空日志</button>
          </div>
        </div>
      </section>

      <section>
        <div class="panel-head">
          <h2>流式播放</h2>
          <span id="playState" class="hint">空闲</span>
        </div>
        <div class="panel-body">
          <div class="metrics">
            <div class="metric"><span>首包延迟</span><strong id="firstLatency">-</strong></div>
            <div class="metric"><span>首次出声</span><strong id="firstAudible">-</strong></div>
            <div class="metric"><span>音频块</span><strong id="chunkCount">0</strong></div>
            <div class="metric"><span>音频时长</span><strong id="audioDuration">0.00s</strong></div>
            <div class="metric"><span>总耗时</span><strong id="totalTime">0.00s</strong></div>
            <div class="metric"><span>块间隔</span><strong id="chunkInterval">-</strong></div>
            <div class="metric"><span>播放队列</span><strong id="queueDepth">0ms</strong></div>
            <div class="metric"><span>Underrun</span><strong id="underrunCount">0</strong></div>
            <div class="metric"><span>RTF</span><strong id="rtfValue">-</strong></div>
            <div class="metric"><span>策略</span><strong id="strategyValue">realtime</strong></div>
            <div class="metric"><span>Profile</span><strong id="profileValue">fp16</strong></div>
            <div class="metric"><span>Unroll</span><strong id="unrollValue">1</strong></div>
            <div class="metric"><span>Schedule</span><strong id="scheduleValue">current</strong></div>
          </div>

          <div class="meters">
            <div>
              <label>接收进度</label>
              <div class="bar"><i id="receiveBar"></i></div>
            </div>
            <div>
              <label>播放队列</label>
              <div class="bar"><i id="queueBar"></i></div>
            </div>
          </div>

          <div id="log" class="console"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const els = {
      health: $("health"),
      wsUrl: $("wsUrl"),
      mode: $("mode"),
      language: $("language"),
      text: $("text"),
      instruct: $("instruct"),
      speaker: $("speaker"),
      refAudio: $("refAudio"),
      refText: $("refText"),
      xVectorOnly: $("xVectorOnly"),
      maxNewTokens: $("maxNewTokens"),
      minNewTokens: $("minNewTokens"),
      chunkStrategy: $("chunkStrategy"),
      chunkFrames: $("chunkFrames"),
      leftContextFrames: $("leftContextFrames"),
      instructField: $("instructField"),
      speakerField: $("speakerField"),
      cloneFields: $("cloneFields"),
      startBtn: $("startBtn"),
      stopBtn: $("stopBtn"),
      downloadBtn: $("downloadBtn"),
      clearBtn: $("clearBtn"),
      playState: $("playState"),
      firstLatency: $("firstLatency"),
      firstAudible: $("firstAudible"),
      chunkCount: $("chunkCount"),
      audioDuration: $("audioDuration"),
      totalTime: $("totalTime"),
      chunkInterval: $("chunkInterval"),
      queueDepth: $("queueDepth"),
      underrunCount: $("underrunCount"),
      rtfValue: $("rtfValue"),
      strategyValue: $("strategyValue"),
      profileValue: $("profileValue"),
      unrollValue: $("unrollValue"),
      scheduleValue: $("scheduleValue"),
      receiveBar: $("receiveBar"),
      queueBar: $("queueBar"),
      log: $("log"),
    };

    let ws = null;
    let audioContext = null;
    let player = null;
    let sampleRate = 24000;
    let chunks = [];
    let chunkCount = 0;
    let receivedSamples = 0;
    let startedAt = 0;
    let firstAudioAt = 0;
    let firstAudibleAt = 0;
    let lastAudioAt = 0;
    let lastChunkIntervalMs = 0;
    let streamFinal = false;
    let underrunCount = 0;
    let targetBufferSec = 0.25;
    let minQueueSec = 0.10;
    let maxBufferSec = 2.00;
    let latestRtf = 0;
    let pendingAudioTiming = null;
    let timer = null;
    const strategyDefaults = {
      realtime: { initialChunkFrames: 8, chunkFrames: 12, leftContextFrames: 25 },
      low_latency: { initialChunkFrames: 8, chunkFrames: 12, leftContextFrames: 25 },
      smooth: { initialChunkFrames: 8, chunkFrames: 24, leftContextFrames: 25 },
      balanced: { initialChunkFrames: 12, chunkFrames: 12, leftContextFrames: 25 },
      stable: { initialChunkFrames: 12, chunkFrames: 24, leftContextFrames: 25 },
    };

    function defaultWsUrl() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const host = location.host || "127.0.0.1:17860";
      return `${protocol}//${host}/v1/tts/stream`;
    }

    function log(message) {
      const now = new Date().toLocaleTimeString();
      els.log.textContent += `[${now}] ${message}\n`;
      els.log.scrollTop = els.log.scrollHeight;
    }

    function setHealth(ok, text) {
      els.health.textContent = text;
      els.health.classList.toggle("good", ok);
      els.health.classList.toggle("bad", !ok);
    }

    async function checkHealth() {
      try {
        const res = await fetch("/health", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const errors = data.warmup && data.warmup.errors ? data.warmup.errors : {};
        const errorKeys = Object.keys(errors);
        if (errorKeys.length) {
          setHealth(false, `预热异常 ${data.warmup.status || ""}`);
          log(`health warmup errors: ${JSON.stringify(errors)}`);
        } else {
          setHealth(true, `服务正常 ${data.model_root}`);
        }
        const runtimes = data.runtimes || {};
        const runtime = Object.values(runtimes)[0];
        const profile = data.warmup && data.warmup.realtime_profile
          ? data.warmup.realtime_profile
          : (runtime && runtime.graph_variant === "int8_sym_fused" ? "int8-sym" : (runtime && runtime.graph_variant === "int8_fused" ? "int8" : "fp16"));
        els.profileValue.textContent = profile;
        els.unrollValue.textContent = runtime && runtime.codegen_unroll ? String(runtime.codegen_unroll) : "1";
        els.scheduleValue.textContent = runtime && runtime.codegen_schedule ? String(runtime.codegen_schedule) : "current";
        if (runtime) {
          log(
            `runtime profile=${profile}, mode=${runtime.mode || "-"}, ` +
            `variant=${runtime.graph_variant || "-"}, unroll=${runtime.codegen_unroll || 1}, schedule=${runtime.codegen_schedule || "current"}, ` +
            `unroll_fallback=${runtime.unroll_fallback ? "yes" : "no"}, fused_int8=${runtime.fused_cache_variant_active ? "yes" : "no"}`
          );
        }
      } catch (err) {
        setHealth(false, "服务未连接");
      }
    }

    function updateModeFields() {
      const mode = els.mode.value;
      els.speakerField.classList.toggle("hidden", mode !== "custom_voice");
      els.cloneFields.classList.toggle("hidden", mode !== "voice_clone");
      els.instructField.classList.toggle("hidden", mode === "voice_clone");
    }

    function updateChunkStrategyFields() {
      const strategy = els.chunkStrategy.value;
      const defaults = strategyDefaults[strategy] || strategyDefaults.low_latency;
      els.chunkFrames.value = defaults.chunkFrames;
      els.leftContextFrames.value = defaults.leftContextFrames;
      els.strategyValue.textContent = strategy;
    }

    function requestPayload() {
      const mode = els.mode.value;
      const strategy = els.chunkStrategy.value;
      const defaults = strategyDefaults[strategy] || strategyDefaults.low_latency;
      const payload = {
        mode,
        text: els.text.value,
        language: els.language.value,
        generation: {
          max_new_tokens: Number(els.maxNewTokens.value),
          min_new_tokens: Number(els.minNewTokens.value),
        },
        stream: {
          chunk_strategy: strategy,
          initial_chunk_frames: defaults.initialChunkFrames,
          chunk_frames: Number(els.chunkFrames.value),
          left_context_frames: Number(els.leftContextFrames.value),
          format: "pcm_s16le",
          include_chunk_metadata: true,
        },
      };
      if (mode === "voice_design") {
        payload.instruct = els.instruct.value;
      } else if (mode === "custom_voice") {
        payload.speaker = els.speaker.value;
        payload.instruct = els.instruct.value;
      } else {
        payload.ref_audio = els.refAudio.value;
        payload.ref_text = els.refText.value;
        payload.x_vector_only = els.xVectorOnly.checked;
      }
      return payload;
    }

    function pcm16ToFloat32(arrayBuffer) {
      const input = new Int16Array(arrayBuffer);
      const output = new Float32Array(input.length);
      for (let i = 0; i < input.length; i += 1) {
        output[i] = Math.max(-1, input[i] / 32768);
      }
      return output;
    }

    function resampleLinear(input, fromRate, toRate) {
      if (fromRate === toRate || input.length === 0) return input;
      const ratio = fromRate / toRate;
      const outputLength = Math.max(1, Math.round(input.length / ratio));
      const output = new Float32Array(outputLength);
      for (let i = 0; i < outputLength; i += 1) {
        const pos = i * ratio;
        const left = Math.floor(pos);
        const right = Math.min(input.length - 1, left + 1);
        const frac = pos - left;
        output[i] = input[left] * (1 - frac) + input[right] * frac;
      }
      return output;
    }

    class PcmQueuePlayer {
      constructor(inputSampleRate, initialBufferSec, callbacks) {
        const AudioCtor = window.AudioContext || window.webkitAudioContext;
        try {
          this.context = new AudioCtor({ sampleRate: inputSampleRate });
        } catch (err) {
          this.context = new AudioCtor();
        }
        this.inputSampleRate = inputSampleRate;
        this.outputSampleRate = this.context.sampleRate;
        this.callbacks = callbacks || {};
        this.queue = [];
        this.readOffset = 0;
        this.queuedSamples = 0;
        this.started = false;
        this.stopped = false;
        this.underrunActive = false;
        this.firstOutputReported = false;
        this.targetBufferSamples = Math.max(1, Math.round(initialBufferSec * this.outputSampleRate));
        this.node = this.context.createScriptProcessor(2048, 0, 1);
        this.node.onaudioprocess = (event) => this.process(event.outputBuffer.getChannelData(0));
        this.node.connect(this.context.destination);
      }

      async resume() {
        if (this.context.state !== "running") {
          await this.context.resume();
        }
      }

      setTargetBuffer(seconds) {
        this.targetBufferSamples = Math.max(1, Math.round(seconds * this.outputSampleRate));
      }

      append(arrayBuffer) {
        if (this.context.state !== "running") {
          this.context.resume().catch(() => {});
        }
        const floats = resampleLinear(pcm16ToFloat32(arrayBuffer), this.inputSampleRate, this.outputSampleRate);
        this.queue.push(floats);
        this.queuedSamples += floats.length;
        if (!this.started && this.queuedSamples >= this.targetBufferSamples) {
          this.started = true;
          this.underrunActive = false;
          if (this.callbacks.onStarted) this.callbacks.onStarted();
        }
        return this.queueSeconds();
      }

      queueSeconds() {
        return this.queuedSamples / this.outputSampleRate;
      }

      process(output) {
        output.fill(0);
        if (this.stopped || !this.started) return;
        let wrote = 0;
        for (let i = 0; i < output.length; i += 1) {
          if (!this.queue.length) break;
          const head = this.queue[0];
          output[i] = head[this.readOffset];
          this.readOffset += 1;
          this.queuedSamples -= 1;
          wrote += 1;
          if (this.readOffset >= head.length) {
            this.queue.shift();
            this.readOffset = 0;
          }
        }
        if (wrote > 0) {
          this.underrunActive = false;
          if (!this.firstOutputReported) {
            this.firstOutputReported = true;
            if (this.callbacks.onFirstOutput) this.callbacks.onFirstOutput();
          }
        }
        if (wrote < output.length && !this.underrunActive) {
          this.underrunActive = true;
          if (this.callbacks.onUnderrun) this.callbacks.onUnderrun();
        }
      }

      async stop() {
        this.stopped = true;
        try {
          this.node.disconnect();
        } catch (err) {
          // ignore disconnect races during browser shutdown
        }
        if (this.context && this.context.state !== "closed") {
          await this.context.close();
        }
      }
    }

    function playPcmChunk(arrayBuffer) {
      if (!player) return;
      const queueBefore = player.queueSeconds();
      if (chunkCount > 0 && queueBefore < minQueueSec) {
        targetBufferSec = Math.min(maxBufferSec, targetBufferSec + 0.05);
        player.setTargetBuffer(targetBufferSec);
      }
      const queueSec = player.append(arrayBuffer);
      receivedSamples += arrayBuffer.byteLength / 2;
      els.queueBar.style.width = `${Math.min(100, queueSec * 35)}%`;
    }

    function playerCallbacks() {
      return {
        onFirstOutput: () => {
          if (!firstAudibleAt) {
            firstAudibleAt = performance.now();
            updateMetrics(false);
          }
        },
        onStarted: () => {
          els.playState.textContent = "播放中";
        },
        onUnderrun: () => {
          if (!streamFinal && chunkCount > 0) {
            underrunCount += 1;
            targetBufferSec = Math.min(maxBufferSec, targetBufferSec + 0.05);
            if (player) player.setTargetBuffer(targetBufferSec);
            updateMetrics(false);
          }
        },
      };
    }

    function setBusy(busy) {
      els.startBtn.disabled = busy;
      els.stopBtn.disabled = !busy;
      els.playState.textContent = busy ? "合成中" : "空闲";
    }

    function updateMetrics(final = false) {
      const elapsed = startedAt ? (performance.now() - startedAt) / 1000 : 0;
      els.totalTime.textContent = `${elapsed.toFixed(2)}s`;
      els.chunkCount.textContent = String(chunkCount);
      els.audioDuration.textContent = `${(receivedSamples / sampleRate).toFixed(2)}s`;
      els.firstLatency.textContent = firstAudioAt ? `${((firstAudioAt - startedAt) / 1000).toFixed(2)}s` : "-";
      els.firstAudible.textContent = firstAudibleAt ? `${((firstAudibleAt - startedAt) / 1000).toFixed(2)}s` : "-";
      els.chunkInterval.textContent = lastChunkIntervalMs ? `${lastChunkIntervalMs.toFixed(0)}ms` : "-";
      els.underrunCount.textContent = String(underrunCount);
      els.rtfValue.textContent = latestRtf ? latestRtf.toFixed(2) : "-";
      els.strategyValue.textContent = els.chunkStrategy.value;
      const maxTokens = Math.max(1, Number(els.maxNewTokens.value));
      const expectedChunks = Math.max(1, Math.ceil(maxTokens / Math.max(1, Number(els.chunkFrames.value))));
      els.receiveBar.style.width = `${final ? 100 : Math.min(95, (chunkCount / expectedChunks) * 100)}%`;
      if (player) {
        const queueSec = player.queueSeconds();
        els.queueDepth.textContent = `${Math.round(queueSec * 1000)}ms`;
        els.queueBar.style.width = `${Math.min(100, queueSec * 35)}%`;
      } else {
        els.queueDepth.textContent = "0ms";
      }
    }

    function resetRun() {
      chunks = [];
      chunkCount = 0;
      receivedSamples = 0;
      startedAt = performance.now();
      firstAudioAt = 0;
      firstAudibleAt = 0;
      lastAudioAt = 0;
      lastChunkIntervalMs = 0;
      streamFinal = false;
      underrunCount = 0;
      targetBufferSec = 0.25;
      latestRtf = 0;
      pendingAudioTiming = null;
      sampleRate = 24000;
      els.downloadBtn.disabled = true;
      els.receiveBar.style.width = "0%";
      els.queueBar.style.width = "0%";
      updateMetrics();
    }

    async function start() {
      if (!els.text.value.trim()) {
        log("文本不能为空");
        return;
      }
      resetRun();
      setBusy(true);
      if (player) {
        await player.stop();
        player = null;
        audioContext = null;
      }
      player = new PcmQueuePlayer(sampleRate, targetBufferSec, playerCallbacks());
      audioContext = player.context;
      await player.resume();
      if (audioContext.state !== "running") {
        log(`AudioContext 状态=${audioContext.state}，如果浏览器阻止播放，请再次点击开始合成。`);
      }
      const payload = requestPayload();
      log(`连接 ${els.wsUrl.value}`);
      ws = new WebSocket(els.wsUrl.value);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        log(`发送请求 mode=${payload.mode}, max_new_tokens=${payload.generation.max_new_tokens}`);
        ws.send(JSON.stringify(payload));
      };

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const data = JSON.parse(event.data);
          if (data.type === "metadata") {
            const metadataSampleRate = Number(data.sample_rate || 24000);
            if (metadataSampleRate !== sampleRate) {
              log(`metadata sample_rate=${metadataSampleRate}; 当前播放器按 ${sampleRate}Hz 初始化`);
            }
            sampleRate = metadataSampleRate;
            if (data.recommended_playback_buffer_ms) {
              targetBufferSec = Math.min(maxBufferSec, Math.max(0.05, Number(data.recommended_playback_buffer_ms) / 1000));
              if (player) player.setTargetBuffer(targetBufferSec);
            }
            if (data.chunk_strategy) {
              els.strategyValue.textContent = data.chunk_strategy;
            }
            if (data.realtime_profile || data.graph_variant) {
              els.profileValue.textContent = data.realtime_profile || (data.graph_variant === "int8_sym_fused" ? "int8-sym" : (data.graph_variant === "int8_fused" ? "int8" : "fp16"));
            }
            if (data.codegen_unroll) {
              els.unrollValue.textContent = String(data.codegen_unroll);
            }
            if (data.codegen_schedule) {
              els.scheduleValue.textContent = String(data.codegen_schedule);
            }
            log(
              `metadata sample_rate=${sampleRate}, format=${data.format}, ` +
              `strategy=${data.chunk_strategy || "-"}, initial=${data.initial_chunk_frames || "-"}, chunk=${data.chunk_frames || "-"}, ` +
              `profile=${data.realtime_profile || "-"}, variant=${data.graph_variant || "-"}, unroll=${data.codegen_unroll || 1}, schedule=${data.codegen_schedule || "current"}`
            );
          } else if (data.type === "final") {
            streamFinal = true;
            if (data.timings) {
              latestRtf = Number(data.timings.stream_rtf || data.timings.rtf || latestRtf || 0);
            }
            updateMetrics(true);
            log(
              `final index=${data.index}, elapsed=${Number(data.elapsed || 0).toFixed(3)}s, ` +
              `stream_rtf=${latestRtf ? latestRtf.toFixed(2) : "-"}`
            );
            setBusy(false);
            els.downloadBtn.disabled = chunks.length === 0;
            if (ws) ws.close();
          } else if (data.type === "audio") {
            pendingAudioTiming = data.timings || null;
            if (pendingAudioTiming) {
              latestRtf = Number(pendingAudioTiming.stream_rtf || pendingAudioTiming.rtf || latestRtf || 0);
            }
            log(
              `audio meta index=${data.index}, bytes=${data.byte_length}, ` +
              `path=${pendingAudioTiming && pendingAudioTiming.decode_path ? pendingAudioTiming.decode_path : "-"}, ` +
              `unroll=${pendingAudioTiming && pendingAudioTiming.codegen_unroll ? pendingAudioTiming.codegen_unroll : 1}, ` +
              `schedule=${pendingAudioTiming && pendingAudioTiming.codegen_schedule ? pendingAudioTiming.codegen_schedule : "current"}, ` +
              `unroll_fallback=${pendingAudioTiming && pendingAudioTiming.unroll_fallback ? "yes" : "no"}, ` +
              `chunk_rtf=${pendingAudioTiming && pendingAudioTiming.rtf ? Number(pendingAudioTiming.rtf).toFixed(2) : "-"}, ` +
              `stream_rtf=${pendingAudioTiming && pendingAudioTiming.stream_rtf ? Number(pendingAudioTiming.stream_rtf).toFixed(2) : "-"}`
            );
          } else if (data.type === "error") {
            log(`error ${data.message}`);
            setBusy(false);
          } else {
            log(JSON.stringify(data));
          }
          return;
        }

        const now = performance.now();
        if (!firstAudioAt) firstAudioAt = now;
        if (lastAudioAt) lastChunkIntervalMs = now - lastAudioAt;
        lastAudioAt = now;
        chunks.push(event.data);
        playPcmChunk(event.data);
        const chunkAudioMs = event.data.byteLength / 2 / sampleRate * 1000;
        const chunkElapsedMs = lastChunkIntervalMs || (now - startedAt);
        if (pendingAudioTiming) {
          latestRtf = Number(pendingAudioTiming.stream_rtf || pendingAudioTiming.rtf || latestRtf || 0);
          pendingAudioTiming = null;
        } else {
          latestRtf = chunkAudioMs > 0 ? chunkElapsedMs / chunkAudioMs : 0;
        }
        log(`audio index=${chunkCount}, bytes=${event.data.byteLength}`);
        chunkCount += 1;
        updateMetrics();
      };

      ws.onerror = () => {
        log("WebSocket 连接错误");
        setBusy(false);
      };

      ws.onclose = () => {
        ws = null;
        if (els.stopBtn.disabled === false) setBusy(false);
      };

      clearInterval(timer);
      timer = setInterval(() => updateMetrics(false), 200);
    }

    async function stop() {
      if (ws) {
        ws.close();
        ws = null;
      }
      if (player) {
        await player.stop();
        player = null;
        audioContext = null;
      }
      setBusy(false);
      log("已停止");
    }

    function encodeWav(buffers, sr) {
      const bytes = buffers.reduce((sum, buf) => sum + buf.byteLength, 0);
      const out = new ArrayBuffer(44 + bytes);
      const view = new DataView(out);
      let offset = 0;
      function writeString(text) {
        for (let i = 0; i < text.length; i += 1) {
          view.setUint8(offset, text.charCodeAt(i));
          offset += 1;
        }
      }
      writeString("RIFF");
      view.setUint32(offset, 36 + bytes, true); offset += 4;
      writeString("WAVE");
      writeString("fmt ");
      view.setUint32(offset, 16, true); offset += 4;
      view.setUint16(offset, 1, true); offset += 2;
      view.setUint16(offset, 1, true); offset += 2;
      view.setUint32(offset, sr, true); offset += 4;
      view.setUint32(offset, sr * 2, true); offset += 4;
      view.setUint16(offset, 2, true); offset += 2;
      view.setUint16(offset, 16, true); offset += 2;
      writeString("data");
      view.setUint32(offset, bytes, true); offset += 4;
      const target = new Uint8Array(out, 44);
      let cursor = 0;
      for (const buf of buffers) {
        target.set(new Uint8Array(buf), cursor);
        cursor += buf.byteLength;
      }
      return out;
    }

    function download() {
      if (!chunks.length) return;
      const wav = encodeWav(chunks, sampleRate);
      const blob = new Blob([wav], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `qwen3-tts-stream-${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    els.wsUrl.value = defaultWsUrl();
    els.mode.addEventListener("change", updateModeFields);
    els.chunkStrategy.addEventListener("change", updateChunkStrategyFields);
    els.startBtn.addEventListener("click", start);
    els.stopBtn.addEventListener("click", stop);
    els.downloadBtn.addEventListener("click", download);
    els.clearBtn.addEventListener("click", () => { els.log.textContent = ""; });
    updateModeFields();
    updateChunkStrategyFields();
    checkHealth();
    log("页面已加载，点击开始合成后浏览器会直接播放收到的 PCM chunk。");
  </script>
</body>
</html>
"""
