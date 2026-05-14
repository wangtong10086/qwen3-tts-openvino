WEB_CLIENT_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-TTS OpenVINO 控制台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --surface-tint: #f0f6f4;
      --text: #17201d;
      --muted: #62706b;
      --line: #dbe3e0;
      --line-strong: #c5d0cc;
      --green: #16745f;
      --green-soft: #e3f2ed;
      --blue: #2357a6;
      --blue-soft: #e7eefb;
      --amber: #9a6700;
      --amber-soft: #fff3d6;
      --red: #b83232;
      --red-soft: #fde8e8;
      --shadow: 0 14px 36px rgba(23, 32, 29, 0.08);
      --radius: 8px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #eef3f5 0, #f4f6f8 260px),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    .app {
      width: min(1320px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 34px;
    }

    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 16px;
    }

    .brand h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .brand p {
      margin: 7px 0 0;
      color: var(--muted);
      max-width: 760px;
    }

    .health-card {
      min-width: 300px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      box-shadow: var(--shadow);
    }

    .health-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 88px;
      min-height: 28px;
      padding: 5px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface-soft);
      color: var(--muted);
      font-weight: 700;
      white-space: nowrap;
    }

    .status-pill.good {
      color: var(--green);
      border-color: rgba(22, 116, 95, 0.35);
      background: var(--green-soft);
    }

    .status-pill.bad {
      color: var(--red);
      border-color: rgba(184, 50, 50, 0.35);
      background: var(--red-soft);
    }

    .health-detail {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(560px, 1fr) minmax(340px, 420px);
      gap: 16px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 52px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--surface-soft);
    }

    h2, h3 {
      margin: 0;
      letter-spacing: 0;
    }

    h2 { font-size: 15px; }
    h3 { font-size: 13px; }

    .panel-body { padding: 16px; }

    .section-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 17px 0 9px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }

    .section-label:first-child { margin-top: 0; }

    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
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
      min-height: 116px;
      resize: vertical;
    }

    #text {
      min-height: 260px;
    }

    #instruct {
      min-height: 96px;
    }

    #refText {
      min-height: 88px;
    }

    input:focus, textarea:focus, select:focus {
      border-color: rgba(35, 87, 166, 0.68);
      box-shadow: 0 0 0 3px rgba(35, 87, 166, 0.13);
    }

    input:disabled, select:disabled {
      background: #eef2f3;
      color: #7d8985;
    }

    .field { margin-bottom: 13px; }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .segmented {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #eef2f3;
    }

    .segmented button {
      min-height: 36px;
      padding: 8px 9px;
      border-radius: 6px;
      color: var(--muted);
      background: transparent;
      font-weight: 750;
    }

    .segmented button.active {
      color: var(--text);
      background: var(--surface);
      box-shadow: 0 1px 3px rgba(23, 32, 29, 0.12);
    }

    .check {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-weight: 600;
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
      background: var(--green);
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }

    button.secondary { background: var(--blue); }
    button.ghost {
      color: var(--text);
      background: #eef2f3;
    }
    button.danger { background: var(--red); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    details {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface-soft);
      margin-top: 12px;
    }

    summary {
      cursor: pointer;
      padding: 11px 12px;
      color: var(--text);
      font-weight: 750;
    }

    details .details-body {
      padding: 0 12px 12px;
    }

    .helper {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }

    .helper-item {
      min-height: 48px;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface-soft);
    }

    .helper-item.warn {
      border-color: rgba(154, 103, 0, 0.32);
      background: var(--amber-soft);
    }

    .helper-item.bad {
      border-color: rgba(184, 50, 50, 0.32);
      background: var(--red-soft);
    }

    .helper-item span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }

    .helper-item strong {
      display: block;
      margin-top: 4px;
      font-size: 14px;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .tag.good {
      color: var(--green);
      border-color: rgba(22, 116, 95, 0.32);
      background: var(--green-soft);
    }

    .tag.warn {
      color: var(--amber);
      border-color: rgba(154, 103, 0, 0.32);
      background: var(--amber-soft);
    }

    .tag.bad {
      color: var(--red);
      border-color: rgba(184, 50, 50, 0.32);
      background: var(--red-soft);
    }

    .action-bar {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 15px;
    }

    .action-bar button { min-height: 42px; }

    .secondary-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }

    .output-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }

    .metric {
      min-height: 58px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface-soft);
    }

    .metric.primary {
      background: var(--surface-tint);
      border-color: rgba(22, 116, 95, 0.26);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }

    .metric strong {
      display: block;
      margin-top: 5px;
      font-size: 16px;
      line-height: 1.05;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }

    .status-strip {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }

    .status-box {
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--surface);
      min-height: 50px;
    }

    .status-box span {
      display: block;
      color: var(--muted);
      font-size: 11px;
    }

    .status-box strong {
      display: block;
      margin-top: 5px;
      font-size: 13px;
      letter-spacing: 0;
      overflow-wrap: anywhere;
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
      background: #e5ebe9;
    }

    .bar > i {
      display: block;
      height: 100%;
      width: 0%;
      background: var(--green);
      transition: width 120ms linear;
    }

    .console {
      height: 220px;
      overflow: auto;
      padding: 12px;
      border: 1px solid #1c2824;
      border-radius: var(--radius);
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

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }

    .hidden { display: none !important; }

    @media (max-width: 1060px) {
      .workspace { grid-template-columns: 1fr; }
      .topbar { grid-template-columns: 1fr; align-items: stretch; }
      .health-card { min-width: 0; }
    }

    @media (max-width: 740px) {
      .app { width: min(100vw - 20px, 720px); padding-top: 14px; }
      .output-grid, .status-strip { grid-template-columns: 1fr 1fr; }
      .helper { grid-template-columns: 1fr 1fr; }
      .grid-2, .grid-3, .meters { grid-template-columns: 1fr; }
    }

    @media (max-width: 520px) {
      .output-grid, .status-strip, .helper, .action-bar, .secondary-actions { grid-template-columns: 1fr; }
      .segmented { grid-template-columns: 1fr; }
      .brand h1 { font-size: 22px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>Qwen3-TTS OpenVINO 控制台</h1>
        <p>面向本地 sidecar 的流式合成测试台。配置请求、监听 WebSocket、播放 PCM chunk，并实时观察 RTF、队列和长文本 full-AR 状态。</p>
      </div>
      <div class="health-card">
        <div class="health-row">
          <div>
            <strong>服务状态</strong>
            <div id="healthDetail" class="health-detail">正在检查 /health</div>
          </div>
          <div id="health" class="status-pill">检查中</div>
        </div>
      </div>
    </header>

    <main class="workspace">
      <section class="panel">
        <div class="panel-head">
          <h2>请求构建</h2>
          <span class="tag">/v1/tts/stream</span>
        </div>
        <div class="panel-body">
          <div class="section-label"><span>Endpoint</span><span id="modelRootLine" class="hint">-</span></div>
          <div class="field">
            <label for="wsUrl">WebSocket 地址</label>
            <input id="wsUrl" autocomplete="off" spellcheck="false">
          </div>

          <div class="section-label"><span>模式</span><span id="longModeBadge" class="tag">short_ar</span></div>
          <div class="field">
            <div id="modeButtons" class="segmented">
              <button type="button" data-mode="voice_design" class="active">VoiceDesign</button>
              <button type="button" data-mode="custom_voice">CustomVoice</button>
              <button type="button" data-mode="voice_clone">VoiceClone</button>
            </div>
            <select id="mode" class="hidden" aria-label="mode">
              <option value="voice_design">VoiceDesign</option>
              <option value="custom_voice">CustomVoice</option>
              <option value="voice_clone">VoiceClone</option>
            </select>
          </div>

          <div class="grid-2">
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
            <div class="field">
              <label for="presetText">样例</label>
              <select id="presetText">
                <option value="">不替换当前文本</option>
                <option value="short_zh">短中文</option>
                <option value="long_zh">长中文 full-AR</option>
                <option value="english">English</option>
                <option value="desktop">桌面应用测试</option>
              </select>
            </div>
          </div>

          <div class="field">
            <label for="text">文本</label>
            <textarea id="text">你好，欢迎使用 OpenVINO 流式语音合成。</textarea>
            <div id="textStats" class="hint">-</div>
          </div>

          <div class="helper">
            <div class="helper-item"><span>文本 tokens</span><strong id="textUnitsValue">0</strong></div>
            <div class="helper-item"><span>Prompt tokens</span><strong id="effectiveTokens">-</strong></div>
            <div class="helper-item"><span>最大 tokens</span><strong id="budgetValue">auto</strong></div>
            <div class="helper-item"><span>预算状态</span><strong id="requestKind">short_ar</strong></div>
          </div>

          <div class="field" id="instructField">
            <label for="instruct">Instruct</label>
            <textarea id="instruct">用自然、清晰的中文女声朗读。</textarea>
          </div>

          <div class="field hidden" id="speakerField">
            <label for="speaker">Speaker</label>
            <input id="speaker" list="speakerOptions" placeholder="Vivian">
            <datalist id="speakerOptions"></datalist>
          </div>

          <div class="hidden" id="cloneFields">
            <div class="field">
              <label for="refAudioUpload">上传参考音频</label>
              <input id="refAudioUpload" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg">
              <div class="row" style="margin-top: 8px;">
                <span id="refAudioInfo" class="hint">未选择文件；可使用下方路径或 URL。</span>
                <button id="clearRefAudioBtn" type="button" class="ghost hidden">清除上传</button>
              </div>
            </div>
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

          <div class="section-label"><span>生成参数</span><span class="hint">默认长文本采样</span></div>
          <div class="grid-2">
            <div class="field">
              <label for="maxNewTokens">max_new_tokens</label>
              <input id="maxNewTokens" type="number" min="1" value="48">
            </div>
            <div class="field">
              <label for="minNewTokens">min_new_tokens</label>
              <input id="minNewTokens" type="number" min="0" value="12">
            </div>
          </div>
          <div class="field">
            <label for="maxVramPercent">最大显存占比 <span id="maxVramPercentValue" class="mono">80%</span></label>
            <input id="maxVramPercent" type="range" min="20" max="100" step="5" value="80">
            <div class="hint">服务端会按该比例计算本次请求可用的最大 prompt token 数。</div>
          </div>

          <details>
            <summary>流式与调试参数</summary>
            <div class="details-body">
              <div class="grid-3">
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
                  <input id="chunkFrames" type="number" min="1" value="24">
                </div>
                <div class="field">
                  <label for="leftContextFrames">left_context</label>
                  <input id="leftContextFrames" type="number" min="0" value="25">
                </div>
              </div>
              <div class="row">
                <label class="check"><input id="verboseLog" type="checkbox"> 详细日志</label>
              </div>
            </div>
          </details>

          <div class="action-bar">
            <button id="startBtn">开始合成</button>
            <button id="stopBtn" class="danger" disabled>停止</button>
          </div>
          <div class="secondary-actions">
            <button id="downloadBtn" class="secondary" disabled>下载 WAV</button>
            <button id="copyBtn" class="ghost">复制请求</button>
            <button id="clearBtn" class="ghost">清空日志</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>播放与运行状态</h2>
          <span id="playState" class="tag">空闲</span>
        </div>
        <div class="panel-body">
          <div class="status-strip">
            <div class="status-box"><span>Runtime</span><strong id="profileValue">fastest</strong></div>
            <div class="status-box"><span>Graph</span><strong id="graphValue">-</strong></div>
            <div class="status-box"><span>Schedule</span><strong id="scheduleValue">current</strong></div>
            <div class="status-box"><span>Unroll</span><strong id="unrollValue">1</strong></div>
          </div>

          <div class="output-grid">
            <div class="metric primary"><span>首包延迟</span><strong id="firstLatency">-</strong></div>
            <div class="metric primary"><span>首次出声</span><strong id="firstAudible">-</strong></div>
            <div class="metric primary"><span>RTF</span><strong id="rtfValue">-</strong></div>
            <div class="metric"><span>播放队列</span><strong id="queueDepth">0ms</strong></div>
            <div class="metric"><span>音频块</span><strong id="chunkCount">0</strong></div>
            <div class="metric"><span>音频时长</span><strong id="audioDuration">0.00s</strong></div>
            <div class="metric"><span>总耗时</span><strong id="totalTime">0.00s</strong></div>
            <div class="metric"><span>块间隔</span><strong id="chunkInterval">-</strong></div>
            <div class="metric"><span>Underrun</span><strong id="underrunCount">0</strong></div>
            <div class="metric"><span>策略</span><strong id="strategyValue">smooth</strong></div>
            <div class="metric"><span>连续性</span><strong id="continuityValue">-</strong></div>
            <div class="metric"><span>采样</span><strong id="samplingValue">-</strong></div>
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

          <div class="section-label"><span>事件日志</span><span id="runtimeLine" class="hint">等待请求</span></div>
          <div id="log" class="console"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const els = {
      health: $("health"),
      healthDetail: $("healthDetail"),
      modelRootLine: $("modelRootLine"),
      wsUrl: $("wsUrl"),
      mode: $("mode"),
      modeButtons: $("modeButtons"),
      language: $("language"),
      presetText: $("presetText"),
      text: $("text"),
      instruct: $("instruct"),
      speaker: $("speaker"),
      speakerOptions: $("speakerOptions"),
      refAudioUpload: $("refAudioUpload"),
      refAudioInfo: $("refAudioInfo"),
      clearRefAudioBtn: $("clearRefAudioBtn"),
      refAudio: $("refAudio"),
      refText: $("refText"),
      xVectorOnly: $("xVectorOnly"),
      maxNewTokens: $("maxNewTokens"),
      minNewTokens: $("minNewTokens"),
      maxVramPercent: $("maxVramPercent"),
      maxVramPercentValue: $("maxVramPercentValue"),
      chunkStrategy: $("chunkStrategy"),
      chunkFrames: $("chunkFrames"),
      leftContextFrames: $("leftContextFrames"),
      verboseLog: $("verboseLog"),
      instructField: $("instructField"),
      speakerField: $("speakerField"),
      cloneFields: $("cloneFields"),
      startBtn: $("startBtn"),
      stopBtn: $("stopBtn"),
      downloadBtn: $("downloadBtn"),
      copyBtn: $("copyBtn"),
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
      graphValue: $("graphValue"),
      unrollValue: $("unrollValue"),
      scheduleValue: $("scheduleValue"),
      continuityValue: $("continuityValue"),
      samplingValue: $("samplingValue"),
      receiveBar: $("receiveBar"),
      queueBar: $("queueBar"),
      textStats: $("textStats"),
      textUnitsValue: $("textUnitsValue"),
      effectiveTokens: $("effectiveTokens"),
      budgetValue: $("budgetValue"),
      requestKind: $("requestKind"),
      longModeBadge: $("longModeBadge"),
      runtimeLine: $("runtimeLine"),
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
    let endedAt = 0;
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
    let forcedChunkStrategy = null;
    let activeMaxNewTokens = 48;
    let synthesisDone = false;
    let serverPromptBudget = 2048;
    let serverPromptBudgetConfig = "auto";
    let serverPromptBudgetPolicy = "auto_gpu";
    let serverMaxVramPercent = 80;
    let tokenBudgetState = null;
    let tokenBudgetTimer = null;
    let tokenBudgetSeq = 0;
    let uploadedRefAudio = null;
    const autoSegmentUnits = 64;
    const settingsKey = "qwen3-tts-ov-web-demo-v2";
    const strategyDefaults = {
      realtime: { initialChunkFrames: 8, chunkFrames: 12, leftContextFrames: 25 },
      low_latency: { initialChunkFrames: 8, chunkFrames: 12, leftContextFrames: 25 },
      smooth: { initialChunkFrames: 8, chunkFrames: 24, leftContextFrames: 25 },
      balanced: { initialChunkFrames: 12, chunkFrames: 12, leftContextFrames: 25 },
      stable: { initialChunkFrames: 12, chunkFrames: 24, leftContextFrames: 25 },
    };
    const presets = {
      short_zh: {
        language: "Chinese",
        text: "你好，欢迎使用 OpenVINO 流式语音合成。",
        instruct: "用自然、清晰的中文女声朗读。",
        maxNewTokens: 48,
      },
      long_zh: {
        language: "Chinese",
        text: "这是一段用于验证长文本完整上下文自回归合成的测试内容。它不会被拆成多个独立 prompt，而是保持同一条生成链路，从而尽可能维持音色、语气和韵律的一致性。请用稳定、自然、清晰的语气连续朗读完整内容。",
        instruct: "用自然、清晰、连贯的中文女声朗读。",
        maxNewTokens: 512,
      },
      english: {
        language: "English",
        text: "This is a streaming text to speech test running through the OpenVINO sidecar.",
        instruct: "Read in a calm and clear voice.",
        maxNewTokens: 96,
      },
      desktop: {
        language: "Chinese",
        text: "这是一个面向 Windows 桌面应用集成的流式语音合成测试。浏览器会持续接收 PCM 音频块，并观察播放队列是否稳定。",
        instruct: "用适合桌面助手的自然语气朗读。",
        maxNewTokens: 160,
      },
    };

    function defaultWsUrl() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const host = location.host || "127.0.0.1:17860";
      return `${protocol}//${host}/v1/tts/stream`;
    }

    function speechTextUnitCount(text) {
      const matches = String(text || "").match(/\s+|[A-Za-z0-9]+|[\u3400-\u9fff]|[^\s]/g) || [];
      let count = 0;
      for (const token of matches) {
        if (!token || /^\s+$/.test(token)) continue;
        if (/^[A-Za-z0-9]+$/.test(token)) {
          count += 2;
        } else if (/^[\u3400-\u9fff]$/.test(token)) {
          count += 1;
        } else if (!"。！？!?；;，,、.:：".includes(token)) {
          count += 1;
        }
      }
      return count;
    }

    function estimatedFullContextCodecFrames(text, requested) {
      const units = speechTextUnitCount(text);
      const estimate = Math.ceil(Math.max(48, units * 4.0 + 128));
      return Math.min(2048, Math.max(Number(requested || 48), estimate));
    }

    function log(message, level = "info") {
      if (level === "debug" && !els.verboseLog.checked) return;
      const now = new Date().toLocaleTimeString();
      els.log.textContent += `[${now}] ${message}\n`;
      els.log.scrollTop = els.log.scrollHeight;
    }

    function setHealth(ok, text, detail) {
      els.health.textContent = text;
      els.health.classList.toggle("good", ok);
      els.health.classList.toggle("bad", !ok);
      if (detail) els.healthDetail.textContent = detail;
    }

    function setPlayState(text, tone = "") {
      els.playState.textContent = text;
      els.playState.classList.toggle("good", tone === "good");
      els.playState.classList.toggle("warn", tone === "warn");
      els.playState.classList.toggle("bad", tone === "bad");
    }

    function updateBudgetFromObject(data) {
      const budget = Number(data.effective_max_continuous_prompt_tokens || data.max_continuous_prompt_tokens || 0);
      if (Number.isFinite(budget) && budget >= 0) serverPromptBudget = budget;
      if (data.max_continuous_prompt_tokens_config !== undefined) {
        serverPromptBudgetConfig = String(data.max_continuous_prompt_tokens_config);
      }
      if (data.long_text_budget_policy) {
        serverPromptBudgetPolicy = String(data.long_text_budget_policy);
      }
      const vramPercent = Number(data.max_vram_percent);
      if (Number.isFinite(vramPercent) && vramPercent > 0) {
        serverMaxVramPercent = vramPercent;
        if (els.maxVramPercent && document.activeElement !== els.maxVramPercent) {
          els.maxVramPercent.value = String(Math.round(vramPercent));
        }
      }
      updateTextStats(false);
    }

    async function checkHealth() {
      try {
        const res = await fetch("/health", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const errors = data.warmup && data.warmup.errors ? data.warmup.errors : {};
        const errorKeys = Object.keys(errors);
        const modelRoot = data.model_root || "-";
        els.modelRootLine.textContent = modelRoot;
        if (data.memory) updateBudgetFromObject(data.memory);
        if (errorKeys.length) {
          setHealth(false, "预热异常", `warmup=${data.warmup.status || "-"} model=${modelRoot}`);
          log(`health warmup errors: ${JSON.stringify(errors)}`);
        } else {
          setHealth(true, "在线", `model=${modelRoot}`);
        }
        const runtimes = data.runtimes || {};
        const runtime = Object.values(runtimes)[0];
        const warmup = data.warmup || {};
        const profile = warmup.realtime_profile || (runtime && runtime.graph_variant) || "fastest";
        els.profileValue.textContent = profile;
        els.graphValue.textContent = runtime && runtime.graph_variant ? runtime.graph_variant : (warmup.graph_variant || "-");
        els.unrollValue.textContent = runtime && runtime.codegen_unroll ? String(runtime.codegen_unroll) : String(warmup.codegen_unroll || 1);
        els.scheduleValue.textContent = runtime && runtime.codegen_schedule ? String(runtime.codegen_schedule) : String(warmup.codegen_schedule || "current");
        const memory = data.memory || {};
        const kvProfile = memory.kv_cache_profile || warmup.kv_cache_profile || "-";
        const kvRelative = Number(memory.kv_cache_relative_to_fp16 || warmup.kv_cache_relative_to_fp16 || 0);
        const kvLabel = kvRelative ? `${kvProfile}/${kvRelative.toFixed(2)}x` : kvProfile;
        els.runtimeLine.textContent =
          `profile=${profile}, kv=${kvLabel}, budget=${serverPromptBudgetConfig}/${serverPromptBudget}, vram=${Math.round(serverMaxVramPercent)}%`;
        if (runtime) {
          log(
            `runtime profile=${profile}, mode=${runtime.mode || "-"}, variant=${runtime.graph_variant || "-"}, ` +
            `unroll=${runtime.codegen_unroll || 1}, schedule=${runtime.codegen_schedule || "current"}, ` +
            `kv=${runtime.kv_cache_profile || kvProfile}, fallback=${runtime.unroll_fallback ? "yes" : "no"}`,
            "debug"
          );
        }
        const serverForcedStrategy = warmup.forced_stream_strategy || null;
        if (serverForcedStrategy && strategyDefaults[serverForcedStrategy]) {
          forcedChunkStrategy = serverForcedStrategy;
          els.chunkStrategy.value = forcedChunkStrategy;
          els.chunkStrategy.disabled = true;
          updateChunkStrategyFields();
          log(`server locked chunk_strategy=${forcedChunkStrategy}`);
        } else {
          forcedChunkStrategy = null;
          els.chunkStrategy.disabled = false;
        }
      } catch (err) {
        setHealth(false, "离线", "无法连接 /health");
      }
    }

    async function loadVoices() {
      try {
        const res = await fetch("/v1/audio/voices", { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        const voices = Array.isArray(data.voices) ? data.voices : [];
        els.speakerOptions.innerHTML = "";
        for (const voice of voices) {
          const option = document.createElement("option");
          option.value = voice;
          els.speakerOptions.appendChild(option);
        }
      } catch (err) {
        // optional endpoint
      }
    }

    function setMode(mode) {
      els.mode.value = mode;
      for (const button of els.modeButtons.querySelectorAll("button")) {
        button.classList.toggle("active", button.dataset.mode === mode);
      }
      updateModeFields();
      updateTextStats();
      saveSettings();
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
      updateTextStats();
      saveSettings();
    }

    function currentMaxVramPercent() {
      const percent = Number(els.maxVramPercent.value || serverMaxVramPercent || 80);
      return Number.isFinite(percent) ? Math.min(100, Math.max(1, percent)) : 80;
    }

    function updateMaxVramLabel() {
      const percent = currentMaxVramPercent();
      els.maxVramPercentValue.textContent = `${Math.round(percent)}%`;
    }

    function tokenBudgetFingerprint() {
      return JSON.stringify({
        mode: els.mode.value,
        text: els.text.value,
        instruct: els.instruct.value,
        speaker: els.speaker.value,
        refText: els.refText.value,
        language: els.language.value,
        maxNewTokens: els.maxNewTokens.value,
        minNewTokens: els.minNewTokens.value,
        maxVramPercent: currentMaxVramPercent(),
      });
    }

    function updateTextStats(refreshTokens = true) {
      const text = els.text.value;
      const units = speechTextUnitCount(text);
      const chars = [...String(text || "")].length;
      const requested = Number(els.maxNewTokens.value || 48);
      const fullContext = els.mode.value === "voice_design" && units > autoSegmentUnits;
      const effectiveFrames = fullContext ? estimatedFullContextCodecFrames(text, requested) : requested;
      const instructUnits = els.mode.value === "voice_clone" ? 0 : speechTextUnitCount(els.instruct.value);
      const promptEstimate = units + instructUnits + 16;
      const exact = tokenBudgetState && tokenBudgetState.tokenizer_exact && tokenBudgetState.fingerprint === tokenBudgetFingerprint();
      const textTokens = exact ? Number(tokenBudgetState.text_tokens || 0) : units;
      const promptTokens = exact ? Number(tokenBudgetState.prompt_len || promptEstimate) : promptEstimate;
      const budgetValue = exact ? Number(tokenBudgetState.effective_max_continuous_prompt_tokens || serverPromptBudget) : serverPromptBudget;
      const budgetText = budgetValue === 0 ? "disabled" : `${budgetValue}`;
      const overBudget = budgetValue > 0 && promptTokens > budgetValue;
      const exactLabel = exact ? "tokenizer" : "estimate";
      updateMaxVramLabel();
      els.textStats.textContent =
        `${chars} chars, ${exactLabel} prompt=${promptTokens} tokens, max_new_tokens=${effectiveFrames || requested}`;
      els.textUnitsValue.textContent = exact ? String(textTokens) : `${textTokens}*`;
      els.effectiveTokens.textContent = exact ? String(promptTokens) : `${promptTokens}*`;
      els.budgetValue.textContent = `${budgetText} (${Math.round(currentMaxVramPercent())}%)`;
      els.requestKind.textContent = overBudget ? "超出预算" : (fullContext ? "full_ar" : "short_ar");
      els.longModeBadge.textContent = fullContext ? "full_ar" : "short_ar";
      els.longModeBadge.classList.toggle("good", fullContext);
      els.longModeBadge.classList.toggle("warn", overBudget);
      els.budgetValue.parentElement.classList.toggle("bad", overBudget);
      els.budgetValue.parentElement.classList.toggle("warn", !overBudget && fullContext);
      activeMaxNewTokens = effectiveFrames || requested;
      if (refreshTokens) scheduleTokenBudgetRefresh();
    }

    function applyPreset(name) {
      const preset = presets[name];
      if (!preset) return;
      els.language.value = preset.language;
      els.text.value = preset.text;
      els.instruct.value = preset.instruct;
      els.maxNewTokens.value = String(preset.maxNewTokens);
      els.presetText.value = "";
      updateTextStats();
      saveSettings();
    }

    function generationBudget(textUnits, instructUnits) {
      const raw = Math.max(512, textUnits + instructUnits + 96);
      if (serverPromptBudget === 0) return raw;
      return Math.min(serverPromptBudget || 2048, raw);
    }

    function tokenBudgetPayload() {
      const mode = els.mode.value;
      const payload = {
        mode,
        text: els.text.value,
        language: els.language.value,
        max_vram_ratio: currentMaxVramPercent(),
        generation: {
          max_new_tokens: Number(els.maxNewTokens.value || 48),
          min_new_tokens: Number(els.minNewTokens.value || 0),
        },
      };
      if (mode === "voice_design") {
        payload.instruct = els.instruct.value;
      } else if (mode === "custom_voice") {
        payload.speaker = els.speaker.value;
        payload.instruct = els.instruct.value;
      } else {
        payload.ref_text = els.refText.value;
        payload.x_vector_only = els.xVectorOnly.checked;
      }
      return payload;
    }

    function scheduleTokenBudgetRefresh() {
      if (tokenBudgetTimer) clearTimeout(tokenBudgetTimer);
      tokenBudgetTimer = setTimeout(refreshTokenBudget, 260);
    }

    async function refreshTokenBudget() {
      const seq = ++tokenBudgetSeq;
      const fingerprint = tokenBudgetFingerprint();
      const payload = tokenBudgetPayload();
      try {
        const res = await fetch("/v1/tts/tokenize", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (seq !== tokenBudgetSeq || fingerprint !== tokenBudgetFingerprint()) return;
        data.fingerprint = fingerprint;
        tokenBudgetState = data;
        updateBudgetFromObject(data);
      } catch (err) {
        if (seq !== tokenBudgetSeq) return;
        tokenBudgetState = null;
        updateTextStats(false);
        log(`tokenizer 计数不可用，使用本地估算：${err && err.message ? err.message : err}`, "debug");
      }
    }

    function requestPayload(includeRefAudio = true) {
      const mode = els.mode.value;
      const strategy = forcedChunkStrategy || els.chunkStrategy.value;
      const defaults = strategyDefaults[strategy] || strategyDefaults.low_latency;
      const textUnits = speechTextUnitCount(els.text.value);
      const instructUnits = mode === "voice_clone" ? 0 : speechTextUnitCount(els.instruct.value);
      const requestedMaxNewTokens = Number(els.maxNewTokens.value);
      const fullContext = mode === "voice_design" && textUnits > autoSegmentUnits;
      const effectiveMaxNewTokens = fullContext ? estimatedFullContextCodecFrames(els.text.value, requestedMaxNewTokens) : requestedMaxNewTokens;
      const payload = {
        mode,
        text: els.text.value,
        language: els.language.value,
        max_vram_ratio: currentMaxVramPercent(),
        generation: {
          max_new_tokens: effectiveMaxNewTokens,
          min_new_tokens: Number(els.minNewTokens.value),
          max_prompt_tokens: tokenBudgetState && Number(tokenBudgetState.effective_max_continuous_prompt_tokens) > 0
            ? Number(tokenBudgetState.effective_max_continuous_prompt_tokens)
            : generationBudget(textUnits, instructUnits),
        },
        auto_segment_text: false,
        auto_segment_units: autoSegmentUnits,
        auto_segment_append_prefix_to_prompt: false,
        auto_segment_isolate_native_runner: true,
        force_auto_segment_text: false,
        full_context_text: fullContext,
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
        payload.ref_audio = includeRefAudio && uploadedRefAudio ? uploadedRefAudio.dataUrl : els.refAudio.value;
        payload.ref_text = els.refText.value;
        payload.x_vector_only = els.xVectorOnly.checked;
        if (includeRefAudio && uploadedRefAudio) {
          payload.ref_audio_name = uploadedRefAudio.name;
        }
      }
      activeMaxNewTokens = effectiveMaxNewTokens;
      if (fullContext) {
        log(`长文本 full-AR，预计 ${activeMaxNewTokens} codec frames，prompt budget=${serverPromptBudgetConfig}/${serverPromptBudget}`);
      }
      return payload;
    }

    async function copyRequest() {
      const payload = requestPayload();
      const text = JSON.stringify(payload, null, 2);
      try {
        await navigator.clipboard.writeText(text);
        log("请求 JSON 已复制");
      } catch (err) {
        log(text);
      }
    }

    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      if (value < 1024) return `${value} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
      return `${(value / 1024 / 1024).toFixed(1)} MB`;
    }

    function clearUploadedRefAudio() {
      uploadedRefAudio = null;
      els.refAudioUpload.value = "";
      els.refAudioInfo.textContent = "未选择文件；可使用下方路径或 URL。";
      els.clearRefAudioBtn.classList.add("hidden");
      saveSettings();
    }

    function handleRefAudioUpload(file) {
      if (!file) {
        clearUploadedRefAudio();
        return;
      }
      els.refAudioInfo.textContent = `正在读取 ${file.name} (${formatBytes(file.size)})`;
      const reader = new FileReader();
      reader.onload = () => {
        uploadedRefAudio = {
          name: file.name,
          size: file.size,
          type: file.type || "audio/wav",
          dataUrl: String(reader.result || ""),
        };
        els.refAudioInfo.textContent = `将使用上传文件：${file.name} (${formatBytes(file.size)})`;
        els.clearRefAudioBtn.classList.remove("hidden");
        if (file.size > 20 * 1024 * 1024) {
          log(`参考音频较大：${formatBytes(file.size)}，WebSocket 请求可能变慢`);
        }
      };
      reader.onerror = () => {
        uploadedRefAudio = null;
        els.refAudioInfo.textContent = "读取参考音频失败，请重试或使用路径/URL。";
        els.clearRefAudioBtn.classList.add("hidden");
        log("读取参考音频失败");
      };
      reader.readAsDataURL(file);
    }

    function saveSettings() {
      try {
        const data = {
          wsUrl: els.wsUrl.value,
          mode: els.mode.value,
          language: els.language.value,
          text: els.text.value,
          instruct: els.instruct.value,
          speaker: els.speaker.value,
          refAudio: els.refAudio.value,
          refText: els.refText.value,
          xVectorOnly: els.xVectorOnly.checked,
          maxNewTokens: els.maxNewTokens.value,
          minNewTokens: els.minNewTokens.value,
          maxVramPercent: els.maxVramPercent.value,
          chunkStrategy: els.chunkStrategy.value,
          verboseLog: els.verboseLog.checked,
        };
        localStorage.setItem(settingsKey, JSON.stringify(data));
      } catch (err) {
        // localStorage can be disabled in hardened browsers
      }
    }

    function loadSettings() {
      try {
        const raw = localStorage.getItem(settingsKey);
        if (!raw) return;
        const data = JSON.parse(raw);
        for (const [key, value] of Object.entries(data)) {
          if (!Object.prototype.hasOwnProperty.call(els, key)) continue;
          const el = els[key];
          if (!el) continue;
          if (el.type === "checkbox") el.checked = Boolean(value);
          else el.value = String(value);
        }
      } catch (err) {
        // ignore stale settings
      }
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
        this.pending = [];
        this.pendingSamples = 0;
        this.sources = new Set();
        this.nextStartTime = 0;
        this.started = false;
        this.stopped = false;
        this.underrunActive = false;
        this.firstOutputReported = false;
        this.targetBufferSamples = Math.max(1, Math.round(initialBufferSec * this.outputSampleRate));
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
        if (floats.length === 0) return this.queueSeconds();
        if (!this.started) {
          this.pending.push(floats);
          this.pendingSamples += floats.length;
          if (this.pendingSamples >= this.targetBufferSamples) {
            this.startBuffered();
          }
        } else {
          if (this.queueSeconds() < 0.04 && !this.underrunActive) {
            this.underrunActive = true;
            if (this.callbacks.onUnderrun) this.callbacks.onUnderrun();
          }
          this.schedule(floats);
        }
        return this.queueSeconds();
      }

      startBuffered() {
        if (this.started || this.pendingSamples <= 0) return;
        this.started = true;
        this.underrunActive = false;
        this.nextStartTime = Math.max(this.context.currentTime + 0.06, this.nextStartTime || 0);
        const items = this.pending;
        this.pending = [];
        this.pendingSamples = 0;
        for (const floats of items) this.schedule(floats);
        if (this.callbacks.onStarted) this.callbacks.onStarted();
      }

      queueSeconds() {
        if (!this.started) return this.pendingSamples / this.outputSampleRate;
        return Math.max(0, this.nextStartTime - this.context.currentTime);
      }

      schedule(floats) {
        if (this.stopped || floats.length === 0) return;
        const buffer = this.context.createBuffer(1, floats.length, this.outputSampleRate);
        buffer.copyToChannel(floats, 0);
        const source = this.context.createBufferSource();
        source.buffer = buffer;
        source.connect(this.context.destination);
        const startAt = Math.max(this.nextStartTime || 0, this.context.currentTime + 0.02);
        this.nextStartTime = startAt + buffer.duration;
        this.sources.add(source);
        source.onended = () => {
          try {
            source.disconnect();
          } catch (err) {
            // ignore disconnect races during browser shutdown
          }
          this.sources.delete(source);
          if (this.sources.size === 0 && this.started && !this.stopped) {
            this.underrunActive = false;
            if (this.callbacks.onEnded) this.callbacks.onEnded();
          }
        };
        source.start(startAt);
        this.underrunActive = false;
        if (!this.firstOutputReported) {
          this.firstOutputReported = true;
          const delayMs = Math.max(0, (startAt - this.context.currentTime) * 1000);
          setTimeout(() => {
            if (!this.stopped && this.callbacks.onFirstOutput) this.callbacks.onFirstOutput();
          }, delayMs);
        }
      }

      async stop() {
        this.stopped = true;
        for (const source of Array.from(this.sources)) {
          try {
            source.stop();
            source.disconnect();
          } catch (err) {
            // ignore stop races during browser shutdown
          }
        }
        this.sources.clear();
        this.pending = [];
        this.pendingSamples = 0;
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
          setPlayState("播放中", "good");
        },
        onUnderrun: () => {
          if (!streamFinal && chunkCount > 0) {
            underrunCount += 1;
            targetBufferSec = Math.min(maxBufferSec, targetBufferSec + 0.05);
            if (player) player.setTargetBuffer(targetBufferSec);
            setPlayState("缓冲中", "warn");
            updateMetrics(false);
          }
        },
        onEnded: () => {
          if (streamFinal && synthesisDone) {
            setPlayState("播放完成", "good");
            updateMetrics(true);
          }
        },
      };
    }

    function setBusy(busy) {
      els.startBtn.disabled = busy;
      els.stopBtn.disabled = !busy;
      if (busy) {
        setPlayState("合成中", "warn");
      } else if (player && player.started && player.queueSeconds() > 0.05) {
        setPlayState("播放中", "good");
      } else {
        setPlayState("空闲");
      }
    }

    function stopMetricsTimer() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    function finishRun(final = false) {
      if (!endedAt) endedAt = performance.now();
      synthesisDone = Boolean(final);
      stopMetricsTimer();
      updateMetrics(final);
      setBusy(false);
    }

    function updateMetrics(final = false) {
      const clockNow = endedAt || performance.now();
      const elapsed = startedAt ? (clockNow - startedAt) / 1000 : 0;
      els.totalTime.textContent = `${elapsed.toFixed(2)}s`;
      els.chunkCount.textContent = String(chunkCount);
      els.audioDuration.textContent = `${(receivedSamples / sampleRate).toFixed(2)}s`;
      els.firstLatency.textContent = firstAudioAt ? `${((firstAudioAt - startedAt) / 1000).toFixed(2)}s` : "-";
      els.firstAudible.textContent = firstAudibleAt ? `${((firstAudibleAt - startedAt) / 1000).toFixed(2)}s` : "-";
      els.chunkInterval.textContent = lastChunkIntervalMs ? `${lastChunkIntervalMs.toFixed(0)}ms` : "-";
      els.underrunCount.textContent = String(underrunCount);
      els.rtfValue.textContent = latestRtf ? latestRtf.toFixed(2) : "-";
      els.strategyValue.textContent = els.chunkStrategy.value;
      const maxTokens = Math.max(1, Number(activeMaxNewTokens || els.maxNewTokens.value));
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
      endedAt = 0;
      firstAudioAt = 0;
      firstAudibleAt = 0;
      lastAudioAt = 0;
      lastChunkIntervalMs = 0;
      streamFinal = false;
      underrunCount = 0;
      targetBufferSec = 0.25;
      latestRtf = 0;
      pendingAudioTiming = null;
      synthesisDone = false;
      sampleRate = 24000;
      activeMaxNewTokens = Number(els.maxNewTokens.value);
      els.downloadBtn.disabled = true;
      els.receiveBar.style.width = "0%";
      els.queueBar.style.width = "0%";
      els.continuityValue.textContent = "-";
      els.samplingValue.textContent = "-";
      updateMetrics();
    }

    async function start() {
      if (!els.text.value.trim()) {
        log("文本不能为空");
        return;
      }
      saveSettings();
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
            updateBudgetFromObject(data);
            if (data.recommended_playback_buffer_ms) {
              targetBufferSec = Math.min(maxBufferSec, Math.max(0.05, Number(data.recommended_playback_buffer_ms) / 1000));
              if (player) player.setTargetBuffer(targetBufferSec);
            }
            if (data.chunk_strategy) els.strategyValue.textContent = data.chunk_strategy;
            if (data.realtime_profile || data.graph_variant) {
              els.profileValue.textContent = data.realtime_profile || data.graph_variant || "-";
              els.graphValue.textContent = data.graph_variant || "-";
            }
            if (data.codegen_unroll) els.unrollValue.textContent = String(data.codegen_unroll);
            if (data.codegen_schedule) els.scheduleValue.textContent = String(data.codegen_schedule);
            els.continuityValue.textContent = data.continuous_long_output ? "full_ar" : "short_ar";
            els.samplingValue.textContent = data.long_ar_do_sample ? "sample" : "default";
            const kvProfile = data.kv_cache_profile || "-";
            const kvRelative = Number(data.kv_cache_relative_to_fp16 || 0);
            const kvLabel = kvRelative ? `${kvProfile}/${kvRelative.toFixed(2)}x` : kvProfile;
            els.runtimeLine.textContent =
              `decode=${data.graph_variant || "-"}, kv=${kvLabel}, budget=${serverPromptBudgetConfig}/${serverPromptBudget}, ` +
              `prompt=${data.prompt_len || data.prompt_tokens_estimate || "-"}, vram=${Math.round(serverMaxVramPercent)}%`;
            log(
              `metadata sample_rate=${sampleRate}, strategy=${data.chunk_strategy || "-"}, ` +
              `profile=${data.realtime_profile || "-"}, variant=${data.graph_variant || "-"}, ` +
              `long=${data.long_text_mode || "-"}, segmented=${data.segmented ? "yes" : "no"}, ` +
              `sample=${data.long_ar_do_sample ? "yes" : "no"}, paged_kv=${data.paged_kv ? "yes" : "no"}, ` +
              `kv=${kvLabel}, prompt_tokens=${data.prompt_len || data.prompt_tokens_estimate || "-"}, max_tokens=${serverPromptBudget}`
            );
          } else if (data.type === "final") {
            streamFinal = true;
            if (player) player.startBuffered();
            if (data.timings) {
              latestRtf = Number(data.timings.stream_rtf || data.timings.rtf || latestRtf || 0);
            }
            updateMetrics(true);
            log(
              `final index=${data.index}, elapsed=${Number(data.elapsed || 0).toFixed(3)}s, ` +
              `stream_rtf=${latestRtf ? latestRtf.toFixed(2) : "-"}`
            );
            finishRun(true);
            els.downloadBtn.disabled = chunks.length === 0;
            if (ws) ws.close();
          } else if (data.type === "audio") {
            pendingAudioTiming = data.timings || null;
            if (pendingAudioTiming) {
              latestRtf = Number(pendingAudioTiming.stream_rtf || pendingAudioTiming.rtf || latestRtf || 0);
              els.continuityValue.textContent = pendingAudioTiming.continuous_long_output ? "full_ar" : (pendingAudioTiming.long_text_mode || "-");
              els.samplingValue.textContent = pendingAudioTiming.long_ar_do_sample ? "sample" : "-";
            }
            log(
              `audio meta index=${data.index}, bytes=${data.byte_length}, ` +
              `path=${pendingAudioTiming && pendingAudioTiming.decode_path ? pendingAudioTiming.decode_path : "-"}, ` +
              `rtf=${pendingAudioTiming && pendingAudioTiming.stream_rtf ? Number(pendingAudioTiming.stream_rtf).toFixed(2) : "-"}`,
              "debug"
            );
          } else if (data.type === "error") {
            log(`error ${data.message}`);
            setPlayState("错误", "bad");
            finishRun(false);
          } else {
            log(JSON.stringify(data), "debug");
          }
          return;
        }

        const now = performance.now();
        if (!firstAudioAt) firstAudioAt = now;
        if (lastAudioAt) lastChunkIntervalMs = now - lastAudioAt;
        lastAudioAt = now;
        chunks.push(event.data);
        try {
          playPcmChunk(event.data);
        } catch (err) {
          log(`audio playback error ${err && err.message ? err.message : err}`);
          setPlayState("播放错误", "bad");
          finishRun(false);
          if (ws) ws.close();
          return;
        }
        const chunkAudioMs = event.data.byteLength / 2 / sampleRate * 1000;
        const chunkElapsedMs = lastChunkIntervalMs || (now - startedAt);
        if (pendingAudioTiming) {
          latestRtf = Number(pendingAudioTiming.stream_rtf || pendingAudioTiming.rtf || latestRtf || 0);
          pendingAudioTiming = null;
        } else {
          latestRtf = chunkAudioMs > 0 ? chunkElapsedMs / chunkAudioMs : 0;
        }
        log(`audio index=${chunkCount}, bytes=${event.data.byteLength}`, "debug");
        chunkCount += 1;
        updateMetrics();
      };

      ws.onerror = () => {
        log("WebSocket 连接错误");
        setPlayState("连接错误", "bad");
        finishRun(false);
      };

      ws.onclose = () => {
        ws = null;
        if (els.stopBtn.disabled === false) finishRun(false);
      };

      stopMetricsTimer();
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
      finishRun(false);
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

    function bindFormPersistence() {
      const ids = [
        "wsUrl", "language", "text", "instruct", "speaker", "refAudio", "refText",
        "xVectorOnly", "maxNewTokens", "minNewTokens", "maxVramPercent", "chunkStrategy", "verboseLog",
      ];
      for (const id of ids) {
        const el = els[id];
        if (!el) continue;
        const eventName = el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input";
        el.addEventListener(eventName, () => {
          if (id === "chunkStrategy") updateChunkStrategyFields();
          else updateTextStats();
          saveSettings();
        });
      }
      els.refAudioUpload.addEventListener("change", () => {
        handleRefAudioUpload(els.refAudioUpload.files && els.refAudioUpload.files[0]);
      });
      els.clearRefAudioBtn.addEventListener("click", clearUploadedRefAudio);
    }

    els.wsUrl.value = defaultWsUrl();
    loadSettings();
    if (!els.wsUrl.value) els.wsUrl.value = defaultWsUrl();
    els.modeButtons.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-mode]");
      if (!button) return;
      setMode(button.dataset.mode);
    });
    els.mode.addEventListener("change", () => setMode(els.mode.value));
    els.presetText.addEventListener("change", () => applyPreset(els.presetText.value));
    els.startBtn.addEventListener("click", start);
    els.stopBtn.addEventListener("click", stop);
    els.downloadBtn.addEventListener("click", download);
    els.copyBtn.addEventListener("click", copyRequest);
    els.clearBtn.addEventListener("click", () => { els.log.textContent = ""; });
    bindFormPersistence();
    setMode(els.mode.value || "voice_design");
    updateChunkStrategyFields();
    updateTextStats();
    checkHealth();
    loadVoices();
    log("页面已加载。点击开始合成后，浏览器会播放收到的 PCM chunk。");
  </script>
</body>
</html>
"""
