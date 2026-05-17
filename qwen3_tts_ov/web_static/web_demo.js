    const $ = (id) => document.getElementById(id);
    const els = {
      health: $("health"),
      healthDetail: $("healthDetail"),
      modelRootLine: $("modelRootLine"),
      modelTitle: $("modelTitle"),
      modelSubtitle: $("modelSubtitle"),
      wsUrl: $("wsUrl"),
      mode: $("mode"),
      modeButtons: $("modeButtons"),
      modelManager: $("modelManager"),
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
      requestCount: $("requestCount"),
      requestStaggerMs: $("requestStaggerMs"),
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
      copyCurlBtn: $("copyCurlBtn"),
      exportSummaryBtn: $("exportSummaryBtn"),
      clearBtn: $("clearBtn"),
      requestPreviewDetails: $("requestPreviewDetails"),
      requestPreview: $("requestPreview"),
      customJsonEnabled: $("customJsonEnabled"),
      customRequestJson: $("customRequestJson"),
      customJsonStatus: $("customJsonStatus"),
      fillCustomJsonBtn: $("fillCustomJsonBtn"),
      formatCustomJsonBtn: $("formatCustomJsonBtn"),
      openaiPreview: $("openaiPreview"),
      multiPanel: $("multiPanel"),
      multiSummary: $("multiSummary"),
      multiResults: $("multiResults"),
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
      contextUsage: $("contextUsage"),
      contextGenerated: $("contextGenerated"),
      receiveBar: $("receiveBar"),
      queueBar: $("queueBar"),
      contextBar: $("contextBar"),
      contextLine: $("contextLine"),
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
    let serverMaxTotalTokens = 0;
    let serverMaxNewTokensForBudget = 0;
    let serverKvBlocks = 0;
    let serverKvBudgetMiB = 0;
    let serverKvLimitSource = "fallback";
    let contextPromptTokens = 0;
    let contextGeneratedTokens = 0;
    let contextUsedTokens = 0;
    let contextLimitTokens = 0;
    let contextRemainingTokens = 0;
    let contextUsagePercent = 0;
    let contextGenerationLimitTokens = 0;
    let tokenBudgetState = null;
    let tokenBudgetTimer = null;
    let tokenBudgetSeq = 0;
    let uploadedRefAudio = null;
    let lastRunSummary = null;
    let multiAbort = false;
    const multiSockets = new Set();
    let activeMultiRows = null;
    let availableModes = {
      voice_design: true,
      custom_voice: true,
      voice_clone: true,
    };
    let modeAvailabilityDetails = {};
    let modelDownloadPollTimer = null;
    const autoSegmentUnits = 64;
    const settingsKey = "qwen3-tts-ov-web-demo-v3";
    const settingsVoiceCloneDefaultsVersion = 2;
    const modeLabels = {
      voice_design: "VoiceDesign",
      custom_voice: "CustomVoice",
      voice_clone: "VoiceClone",
    };
    const modeSubtitles = {
      voice_design: "text + instruct + language",
      custom_voice: "text + speaker + optional instruct + language",
      voice_clone: "text + ref_audio + ref_text + language",
    };
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
      clone_en: {
        language: "English",
        text: "This sentence should follow the speaking style of the reference audio.",
        instruct: "",
        maxNewTokens: 128,
      },
      custom_short_zh: {
        language: "Chinese",
        text: "你好，这是 CustomVoice 短文本测试。",
        instruct: "用自然、稳定的语气朗读。",
        maxNewTokens: 64,
      },
      custom_long_zh: {
        language: "Chinese",
        text: "这是一段 CustomVoice 长文本测试，用来观察同一说话人在完整上下文自回归生成中的音色、语气和节奏是否保持稳定。请连续、自然地读完整段内容，不要显得分裂或突然换声线。",
        instruct: "用自然、稳定、连贯的语气朗读。",
        maxNewTokens: 512,
      },
      clone_long_en: {
        language: "English",
        text: "This is a longer voice clone test. The generated speech should preserve the reference voice style across the whole sentence without switching tone between chunks.",
        instruct: "",
        maxNewTokens: 256,
      },
      multi_zh: {
        language: "Chinese",
        text: "这是一个并发请求，用来观察 online batching 的首包延迟和吞吐表现。",
        instruct: "用自然、稳定的中文语气朗读。",
        maxNewTokens: 64,
      },
    };

    function defaultWsUrl() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const host = location.host || "127.0.0.1:17860";
      return `${protocol}//${host}/v1/tts/stream`;
    }

    function httpBaseUrl() {
      const wsValue = els.wsUrl.value || defaultWsUrl();
      if (wsValue.startsWith("wss://")) return `https://${wsValue.slice("wss://".length).replace(/\/v1\/tts\/stream$/, "")}`;
      if (wsValue.startsWith("ws://")) return `http://${wsValue.slice("ws://".length).replace(/\/v1\/tts\/stream$/, "")}`;
      return `${location.protocol}//${location.host || "127.0.0.1:17860"}`;
    }

    function shellQuote(value) {
      return `'${String(value).replace(/'/g, "'\"'\"'")}'`;
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

    function isModeAvailable(mode) {
      return availableModes[mode] !== false;
    }

    function firstAvailableMode() {
      for (const mode of ["voice_design", "custom_voice", "voice_clone"]) {
        if (isModeAvailable(mode)) return mode;
      }
      return "voice_design";
    }

    function modeUnavailableReason(mode) {
      const detail = modeAvailabilityDetails[mode] || {};
      return detail.reason || `${modeLabels[mode] || mode} 模型 IR 不可用`;
    }

    function availableModeSummary() {
      const labels = [];
      for (const mode of ["voice_design", "custom_voice", "voice_clone"]) {
        if (isModeAvailable(mode)) labels.push(modeLabels[mode] || mode);
      }
      return labels.length ? labels.join("/") : "none";
    }

    function downloadStatusTone(status, available) {
      if (available || status === "downloaded" || status === "local") return "good";
      if (status === "queued" || status === "downloading") return "warn";
      return "bad";
    }

    function downloadStatusLabel(status, available) {
      if (available || status === "downloaded" || status === "local") return "已就绪";
      if (status === "queued") return "排队中";
      if (status === "downloading") return "下载中";
      if (status === "failed") return "下载失败";
      return "缺失";
    }

    function renderModelManager() {
      if (!els.modelManager) return;
      els.modelManager.innerHTML = "";
      for (const mode of ["voice_design", "custom_voice", "voice_clone"]) {
        const entry = modeAvailabilityDetails[mode] || {};
        const download = entry.download || {};
        const available = entry.available !== false;
        const status = String(download.status || (available ? "local" : "missing"));
        const tone = downloadStatusTone(status, available);
        const row = document.createElement("div");
        row.className = "model-row";

        const name = document.createElement("strong");
        name.textContent = modeLabels[mode] || mode;
        row.appendChild(name);

        const state = document.createElement("span");
        state.className = `mini-status ${tone}`;
        state.textContent = downloadStatusLabel(status, available);
        row.appendChild(state);

        const path = document.createElement("span");
        path.className = "model-path";
        path.textContent = available
          ? (entry.ir_dir || download.target_dir || "-")
          : (download.target_manifest || entry.required_manifest || entry.expected_ir_dir || "-");
        path.title = path.textContent;
        row.appendChild(path);

        const button = document.createElement("button");
        button.type = "button";
        button.className = available ? "ghost" : "secondary";
        button.dataset.downloadMode = mode;
        const canDownload = download.can_download !== false;
        const busy = status === "queued" || status === "downloading";
        button.disabled = available || busy || !canDownload;
        button.textContent = available ? "已下载" : (busy ? "下载中" : (canDownload ? "下载" : "未配置源"));
        button.title = canDownload
          ? `${download.repo_id || ""}/${download.subdir || ""}`
          : "服务端未配置该模式的 Hugging Face 下载源";
        row.appendChild(button);

        els.modelManager.appendChild(row);
      }
    }

    function applyModeAvailability(modes) {
      if (!modes || typeof modes !== "object") return;
      modeAvailabilityDetails = modes;
      for (const mode of ["voice_design", "custom_voice", "voice_clone"]) {
        const entry = modes[mode] || {};
        const available = entry.available !== false;
        availableModes[mode] = available;
        const title = available
          ? `${modeLabels[mode] || mode} 可用${entry.ir_dir ? `: ${entry.ir_dir}` : ""}`
          : modeUnavailableReason(mode);
        const button = els.modeButtons.querySelector(`button[data-mode="${mode}"]`);
        if (button) {
          button.disabled = !available;
          button.title = title;
          button.classList.toggle("active", els.mode.value === mode && available);
        }
        const option = els.mode.querySelector(`option[value="${mode}"]`);
        if (option) {
          option.disabled = !available;
          option.title = title;
        }
      }
      if (!isModeAvailable(els.mode.value)) {
        const fallback = firstAvailableMode();
        log(`${modeLabels[els.mode.value] || els.mode.value} 当前不可用：${modeUnavailableReason(els.mode.value)}，已切换到 ${modeLabels[fallback] || fallback}`);
        setMode(fallback);
      }
      renderModelManager();
      scheduleModelDownloadPolling();
    }

    function scheduleModelDownloadPolling() {
      if (modelDownloadPollTimer) clearTimeout(modelDownloadPollTimer);
      const hasBusyDownload = Object.values(modeAvailabilityDetails).some((entry) => {
        const status = entry && entry.download ? String(entry.download.status || "") : "";
        return status === "queued" || status === "downloading";
      });
      if (!hasBusyDownload) return;
      modelDownloadPollTimer = setTimeout(async () => {
        await checkHealth();
        scheduleModelDownloadPolling();
      }, 1800);
    }

    async function downloadModel(mode) {
      const detail = modeAvailabilityDetails[mode] || {};
      const download = detail.download || {};
      if (detail.available !== false) {
        log(`${modeLabels[mode] || mode} 已就绪，无需下载`);
        return;
      }
      if (download.can_download === false) {
        log(`${modeLabels[mode] || mode} 未配置下载源`);
        return;
      }
      log(`开始下载 ${modeLabels[mode] || mode}：${download.repo_id || "-"} / ${download.subdir || "-"}`);
      try {
        const res = await fetch("/v1/models/download", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ mode }),
          cache: "no-store",
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        if (data.available_modes) applyModeAvailability(data.available_modes);
        if (data.job) {
          log(`${modeLabels[mode] || mode} 下载任务：${data.job.status}`);
          if (data.job.error) log(`${modeLabels[mode] || mode} 下载错误：${data.job.error}`);
        }
        scheduleModelDownloadPolling();
      } catch (err) {
        log(`${modeLabels[mode] || mode} 下载启动失败：${err && err.message ? err.message : err}`);
      }
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
      const totalTokens = Number(data.effective_max_total_tokens || 0);
      serverMaxTotalTokens = Number.isFinite(totalTokens) && totalTokens > 0 ? totalTokens : 0;
      if (data.max_new_tokens_for_budget !== undefined) {
        const maxNewTokensForBudget = Number(data.max_new_tokens_for_budget || 0);
        serverMaxNewTokensForBudget = Number.isFinite(maxNewTokensForBudget) && maxNewTokensForBudget > 0
          ? maxNewTokensForBudget
          : 0;
      }
      const kvBlocks = Number(data.preallocated_kv_blocks || 0);
      serverKvBlocks = Number.isFinite(kvBlocks) && kvBlocks > 0 ? kvBlocks : 0;
      const kvBudgetBytes = Number(data.kv_cache_budget_bytes || 0);
      serverKvBudgetMiB = Number.isFinite(kvBudgetBytes) && kvBudgetBytes > 0
        ? kvBudgetBytes / (1024 * 1024)
        : 0;
      if (data.kv_cache_limit_source) {
        serverKvLimitSource = String(data.kv_cache_limit_source);
      }
      updateContextUsageFromObject(data, false);
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
        if (data.available_modes) applyModeAvailability(data.available_modes);
        const modesLine = `modes=${availableModeSummary()}`;
        if (errorKeys.length) {
          setHealth(false, "预热异常", `warmup=${data.warmup.status || "-"} model=${modelRoot} ${modesLine}`);
          log(`health warmup errors: ${JSON.stringify(errors)}`);
        } else {
          setHealth(true, "在线", `model=${modelRoot} ${modesLine}`);
        }
        const runtimes = data.runtimes || {};
        const runtime = Object.values(runtimes)[0];
        const warmup = data.warmup || {};
        const profile = warmup.realtime_profile || (runtime && runtime.graph_variant) || "fastest";
        const productionGraph = warmup.online_batch_graph_variant || (data.online_batching && data.online_batching.graph_variant) || (runtime && runtime.graph_variant) || warmup.graph_variant || "-";
        els.profileValue.textContent = profile;
        els.graphValue.textContent = productionGraph;
        els.unrollValue.textContent = runtime && runtime.codegen_unroll ? String(runtime.codegen_unroll) : String(warmup.codegen_unroll || 1);
        els.scheduleValue.textContent = runtime && runtime.codegen_schedule ? String(runtime.codegen_schedule) : String(warmup.codegen_schedule || "current");
        const memory = data.memory || {};
        const kvProfile = memory.kv_cache_profile || warmup.kv_cache_profile || "-";
        const kvRelative = Number(memory.kv_cache_relative_to_fp16 || warmup.kv_cache_relative_to_fp16 || 0);
        const kvLabel = kvRelative ? `${kvProfile}/${kvRelative.toFixed(2)}x` : kvProfile;
        const totalLabel = serverMaxTotalTokens ? `, total=${serverMaxTotalTokens}` : "";
        const blocksLabel = serverKvBlocks ? `, blocks=${serverKvBlocks}` : "";
        const kvBudgetLabel = serverKvBudgetMiB ? `, kv_budget=${Math.round(serverKvBudgetMiB)}MiB` : "";
        els.runtimeLine.textContent =
          `profile=${profile}, kv=${kvLabel}, budget=${serverPromptBudgetConfig}/${serverPromptBudget}${totalLabel}${blocksLabel}${kvBudgetLabel}, vram=${Math.round(serverMaxVramPercent)}%`;
        if (runtime) {
          log(
            `runtime profile=${profile}, mode=${runtime.mode || "-"}, variant=${runtime.graph_variant || "-"}, ` +
            `production_variant=${productionGraph}, ` +
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
        if (data.available_modes) applyModeAvailability(data.available_modes);
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
      if (!isModeAvailable(mode)) {
        const fallback = firstAvailableMode();
        log(`${modeLabels[mode] || mode} 当前不可用：${modeUnavailableReason(mode)}，已切换到 ${modeLabels[fallback] || fallback}`);
        mode = fallback;
      }
      els.mode.value = mode;
      for (const button of els.modeButtons.querySelectorAll("button")) {
        button.classList.toggle("active", button.dataset.mode === mode && !button.disabled);
      }
      updateModeFields();
      updateTextStats();
      saveSettings();
    }

    function updateModeFields() {
      const mode = els.mode.value;
      els.modelTitle.textContent = `${modeLabels[mode] || mode}`;
      els.modelSubtitle.textContent = modeSubtitles[mode] || "";
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

    function limitSourceLabel(source) {
      const labels = {
        model_context_limit: "模型上下文",
        kv_cache_memory: "显存预算",
        kv_cache_blocks: "KV blocks",
        kv_cache_max_blocks: "KV blocks",
        static_kv_blocks: "固定 KV",
        configured_prompt_limit: "手动 Prompt",
        prompt_limit: "Prompt 配置",
        fallback: "保守默认",
      };
      return labels[String(source || "")] || "自动预算";
    }

    function setBudgetDisplay(lines) {
      els.budgetValue.textContent = "";
      for (const [index, line] of lines.entries()) {
        const node = document.createElement("span");
        node.textContent = line;
        if (index > 0) node.className = "subline";
        els.budgetValue.appendChild(node);
      }
    }

    function updateBudgetDisplay(budgetValue, totalTokens, maxNewTokensForBudget, promptTokens, kvBlocks, kvLimitSource) {
      const generationLimit = totalTokens > 0 && promptTokens > 0
        ? Math.max(0, Math.floor(totalTokens - promptTokens - 1))
        : Math.max(0, Number(maxNewTokensForBudget || 0));
      const lines = [];
      if (generationLimit > 0) {
        lines.push(`可生成 ${generationLimit} tokens`);
      } else {
        lines.push("可生成 -");
      }

      const contextParts = [];
      if (totalTokens > 0) contextParts.push(`总上下文 ${totalTokens}`);
      if (promptTokens > 0) contextParts.push(`当前 Prompt ${promptTokens}`);
      if (contextParts.length > 0) lines.push(contextParts.join("，"));

      const requestParts = [];
      if (maxNewTokensForBudget > 0) requestParts.push(`运行上限 ${maxNewTokensForBudget}`);
      requestParts.push(budgetValue === 0 ? "Prompt 不限" : `Prompt 上限 ${budgetValue}`);
      lines.push(requestParts.join("，"));

      const resourceParts = [`显存 ${Math.round(currentMaxVramPercent())}%`];
      if (kvBlocks > 0) resourceParts.push(`KV ${kvBlocks} blocks`);
      resourceParts.push(`限制来自 ${limitSourceLabel(kvLimitSource)}`);
      lines.push(resourceParts.join("，"));
      setBudgetDisplay(lines);
    }

    function updateContextDisplay() {
      const limit = Number(contextLimitTokens || serverMaxTotalTokens || 0);
      const used = Number(contextUsedTokens || 0);
      const generated = Number(contextGeneratedTokens || 0);
      const remaining = Number(contextRemainingTokens || 0);
      const percent = limit > 0
        ? Math.min(100, Math.max(0, Number(contextUsagePercent || (used / limit * 100))))
        : 0;
      els.contextUsage.textContent = limit > 0 ? `${used} / ${limit}` : "-";
      els.contextGenerated.textContent = contextGenerationLimitTokens > 0
        ? `${generated} / ${contextGenerationLimitTokens}`
        : String(generated || 0);
      els.contextLine.textContent = limit > 0
        ? `上下文使用 ${percent.toFixed(1)}%，剩余 ${remaining} tokens`
        : "上下文使用";
      els.contextBar.style.width = `${percent.toFixed(2)}%`;
    }

    function updateContextUsageFromObject(data, force = false) {
      if (!data) return;
      const source = data.timings && typeof data.timings === "object" ? data.timings : data;
      const hasContext =
        source.context_used_tokens !== undefined ||
        source.context_generated_tokens !== undefined ||
        source.prompt_len !== undefined ||
        source.effective_max_total_tokens !== undefined ||
        source.max_generation_tokens_available !== undefined ||
        source.max_new_tokens !== undefined;
      if (!force && !hasContext) return;

      const prompt = Number(source.context_prompt_tokens ?? source.prompt_len ?? source.prompt_tokens_estimate ?? contextPromptTokens ?? 0);
      const generated = Number(source.context_generated_tokens ?? source.emitted_frames ?? contextGeneratedTokens ?? 0);
      const limit = Number(source.context_limit_tokens ?? source.effective_max_total_tokens ?? source.model_context_tokens ?? contextLimitTokens ?? serverMaxTotalTokens ?? 0);
      const generationLimit = Number(
        source.context_generation_limit_tokens ??
        source.max_generation_tokens_available ??
        source.effective_max_new_tokens ??
        source.max_new_tokens ??
        contextGenerationLimitTokens ??
        0
      );
      const used = Number(source.context_used_tokens ?? (prompt + generated));
      const remaining = Number(
        source.context_remaining_tokens ??
        (limit > 0 ? Math.max(0, limit - used - 1) : contextRemainingTokens || 0)
      );
      contextPromptTokens = Number.isFinite(prompt) ? Math.max(0, Math.floor(prompt)) : contextPromptTokens;
      contextGeneratedTokens = Number.isFinite(generated) ? Math.max(0, Math.floor(generated)) : contextGeneratedTokens;
      contextLimitTokens = Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : contextLimitTokens;
      contextGenerationLimitTokens = Number.isFinite(generationLimit) ? Math.max(0, Math.floor(generationLimit)) : contextGenerationLimitTokens;
      contextUsedTokens = Number.isFinite(used) ? Math.max(0, Math.floor(used)) : (contextPromptTokens + contextGeneratedTokens);
      contextRemainingTokens = Number.isFinite(remaining) ? Math.max(0, Math.floor(remaining)) : contextRemainingTokens;
      contextUsagePercent = Number(source.context_usage_percent);
      if (!Number.isFinite(contextUsagePercent)) {
        contextUsagePercent = contextLimitTokens > 0 ? (contextUsedTokens / contextLimitTokens * 100) : 0;
      }
      updateContextDisplay();
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
      const fullContext = units > autoSegmentUnits;
      const instructUnits = els.mode.value === "voice_clone" ? 0 : speechTextUnitCount(els.instruct.value);
      const promptEstimate = units + instructUnits + 16;
      const exact = tokenBudgetState && tokenBudgetState.tokenizer_exact && tokenBudgetState.fingerprint === tokenBudgetFingerprint();
      const textTokens = exact ? Number(tokenBudgetState.text_tokens || 0) : units;
      const promptTokens = exact ? Number(tokenBudgetState.prompt_len || promptEstimate) : promptEstimate;
      const budgetValue = exact ? Number(tokenBudgetState.effective_max_continuous_prompt_tokens || serverPromptBudget) : serverPromptBudget;
      const totalTokens = exact ? Number(tokenBudgetState.effective_max_total_tokens || serverMaxTotalTokens) : serverMaxTotalTokens;
      const generationLimitKnown = exact
        ? tokenBudgetState.max_generation_tokens_available !== undefined
        : totalTokens > 0;
      const maxGenerationTokensAvailable = exact
        ? Number(tokenBudgetState.max_generation_tokens_available || 0)
        : (totalTokens > 0 ? Math.max(0, totalTokens - promptTokens - 1) : 0);
      const effectiveFrames = fullContext
        ? Math.max(1, Number((tokenBudgetState && tokenBudgetState.max_new_tokens) || maxGenerationTokensAvailable || requested))
        : requested;
      const runtimeMaxNewTokens = exact ? Number(tokenBudgetState.max_new_tokens || effectiveFrames) : effectiveFrames;
      const kvBlocks = exact ? Number(tokenBudgetState.preallocated_kv_blocks || serverKvBlocks) : serverKvBlocks;
      const kvLimitSource = exact ? String(tokenBudgetState.kv_cache_limit_source || serverKvLimitSource) : serverKvLimitSource;
      const overPromptBudget = budgetValue > 0 && promptTokens > budgetValue;
      const overGenerationBudget = generationLimitKnown && runtimeMaxNewTokens > maxGenerationTokensAvailable;
      const overBudget = overPromptBudget || overGenerationBudget;
      const exactLabel = exact ? "tokenizer" : "estimate";
      updateMaxVramLabel();
      els.textStats.textContent =
        `${chars} chars, ${exactLabel} prompt=${promptTokens} tokens, runtime_max_new=${runtimeMaxNewTokens || requested}, max_generatable=${maxGenerationTokensAvailable || "-"}`;
      els.textUnitsValue.textContent = exact ? String(textTokens) : `${textTokens}*`;
      els.effectiveTokens.textContent = exact ? String(promptTokens) : `${promptTokens}*`;
      updateBudgetDisplay(budgetValue, totalTokens, runtimeMaxNewTokens, promptTokens, kvBlocks, kvLimitSource);
      els.requestKind.textContent = overBudget ? "超出预算" : (fullContext ? "full_ar" : "short_ar");
      els.longModeBadge.textContent = fullContext ? "full_ar" : "short_ar";
      els.longModeBadge.classList.toggle("good", fullContext);
      els.longModeBadge.classList.toggle("warn", overBudget);
      els.budgetValue.parentElement.classList.toggle("bad", overBudget);
      els.budgetValue.parentElement.classList.toggle("warn", !overBudget && fullContext);
      activeMaxNewTokens = effectiveFrames || requested;
      updateRequestPreviews();
      if (refreshTokens) scheduleTokenBudgetRefresh();
    }

    function applyPreset(name) {
      if (name === "mode_short") {
        const mode = els.mode.value;
        name = mode === "custom_voice" ? "custom_short_zh" : (mode === "voice_clone" ? "clone_en" : "short_zh");
      } else if (name === "mode_long") {
        const mode = els.mode.value;
        name = mode === "custom_voice" ? "custom_long_zh" : (mode === "voice_clone" ? "clone_long_en" : "long_zh");
      }
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
      const textUnits = speechTextUnitCount(els.text.value);
      const requestedMaxNewTokens = Number(els.maxNewTokens.value || 48);
      const fullContext = textUnits > autoSegmentUnits;
      const payload = {
        mode,
        text: els.text.value,
        language: els.language.value,
        max_vram_ratio: currentMaxVramPercent(),
        generation: {
          max_new_tokens: requestedMaxNewTokens,
          min_new_tokens: Number(els.minNewTokens.value || 0),
        },
        full_context_text: fullContext,
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

    function requestPayload(includeRefAudio = true, options = {}) {
      const silent = options && options.silent === true;
      const mode = els.mode.value;
      const strategy = forcedChunkStrategy || els.chunkStrategy.value;
      const defaults = strategyDefaults[strategy] || strategyDefaults.low_latency;
      const textUnits = speechTextUnitCount(els.text.value);
      const instructUnits = mode === "voice_clone" ? 0 : speechTextUnitCount(els.instruct.value);
      const requestedMaxNewTokens = Number(els.maxNewTokens.value);
      const fullContext = textUnits > autoSegmentUnits;
      const effectiveMaxNewTokens = requestedMaxNewTokens;
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
      activeMaxNewTokens = fullContext && tokenBudgetState && Number(tokenBudgetState.max_new_tokens) > 0
        ? Number(tokenBudgetState.max_new_tokens)
        : effectiveMaxNewTokens;
      if (fullContext && !silent) {
        log(`长文本 full-AR，将生成到 EOS 或上下文上限，当前运行上限=${activeMaxNewTokens} codec frames`);
      }
      return payload;
    }

    function openaiSpeechPayload() {
      const mode = els.mode.value;
      const payload = {
        model: "qwen3-tts-openvino",
        voice: mode === "custom_voice" ? (els.speaker.value || "default") : "default",
        input: els.text.value,
        language: els.language.value,
        task_type: mode,
        stream: true,
        response_format: "pcm",
        chunk_strategy: forcedChunkStrategy || els.chunkStrategy.value,
        max_new_tokens: Number(els.maxNewTokens.value),
        min_new_tokens: Number(els.minNewTokens.value),
      };
      if (mode === "voice_design") {
        payload.instructions = els.instruct.value;
      } else if (mode === "custom_voice") {
        payload.instructions = els.instruct.value;
      } else {
        payload.ref_audio = uploadedRefAudio ? "<uploaded audio data URL omitted from preview>" : els.refAudio.value;
        payload.ref_text = els.refText.value;
        payload.x_vector_only_mode = els.xVectorOnly.checked;
      }
      return payload;
    }

    function updateRequestPreviews() {
      try {
        const generated = requestPayload(false, { silent: true });
        if (els.requestPreview) {
          els.requestPreview.value = JSON.stringify(generated, null, 2);
        }
        if (els.customRequestJson && !els.customRequestJson.value.trim()) {
          els.customRequestJson.value = JSON.stringify(generated, null, 2);
        }
        if (els.openaiPreview) {
          els.openaiPreview.value = JSON.stringify(openaiSpeechPayload(), null, 2);
        }
        validateCustomJson();
      } catch (err) {
        // Preview is diagnostic only.
      }
    }

    function parseCustomRequestJson() {
      const text = els.customRequestJson.value.trim();
      if (!text) throw new Error("自定义 JSON 为空");
      const payload = JSON.parse(text);
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("自定义 JSON 必须是对象");
      }
      return payload;
    }

    function effectiveRequestPayload(includeRefAudio = true, options = {}) {
      if (els.customJsonEnabled.checked) {
        return parseCustomRequestJson();
      }
      return requestPayload(includeRefAudio, options);
    }

    function validateCustomJson() {
      if (!els.customJsonEnabled.checked) {
        els.customJsonStatus.textContent = "未启用自定义 JSON。";
        els.customJsonStatus.className = "hint";
        return true;
      }
      try {
        const payload = parseCustomRequestJson();
        const mode = payload.mode || "voice_design";
        els.customJsonStatus.textContent = `自定义 JSON 有效：mode=${mode}`;
        els.customJsonStatus.className = "hint good-text";
        return true;
      } catch (err) {
        els.customJsonStatus.textContent = `自定义 JSON 无效：${err && err.message ? err.message : err}`;
        els.customJsonStatus.className = "hint bad-text";
        return false;
      }
    }

    function fillCustomJsonFromCurrent() {
      els.customRequestJson.value = JSON.stringify(requestPayload(false, { silent: true }), null, 2);
      validateCustomJson();
      saveSettings();
    }

    function formatCustomJson() {
      try {
        const payload = parseCustomRequestJson();
        els.customRequestJson.value = JSON.stringify(payload, null, 2);
        validateCustomJson();
        saveSettings();
      } catch (err) {
        validateCustomJson();
      }
    }

    async function copyRequest() {
      let payload;
      try {
        payload = effectiveRequestPayload(true, { silent: true });
      } catch (err) {
        log(`请求 JSON 无效：${err && err.message ? err.message : err}`);
        return;
      }
      const text = JSON.stringify(payload, null, 2);
      try {
        await navigator.clipboard.writeText(text);
        log("请求 JSON 已复制");
      } catch (err) {
        log(text);
      }
    }

    async function copyCurl() {
      let body;
      try {
        body = JSON.stringify(effectiveRequestPayload(false, { silent: true }));
      } catch (err) {
        log(`请求 JSON 无效：${err && err.message ? err.message : err}`);
        return;
      }
      const cmd = [
        "curl -N",
        shellQuote(`${httpBaseUrl().replace(/\/$/, "")}/v1/tts/stream`),
        "-H",
        shellQuote("content-type: application/json"),
        "-d",
        shellQuote(body),
      ].join(" ");
      try {
        await navigator.clipboard.writeText(cmd);
        log("curl 命令已复制");
      } catch (err) {
        log(cmd);
      }
    }

    function exportSummary() {
      if (!lastRunSummary) return;
      const blob = new Blob([JSON.stringify(lastRunSummary, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `qwen3-tts-summary-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function renderMultiRows(rows) {
      els.multiResults.innerHTML = "";
      for (const row of rows) {
        const tr = document.createElement("tr");
        const statusClass = row.error ? "bad" : (row.done ? "ok" : "warn");
        const cells = [
          String(row.index + 1),
          row.error ? "error" : (row.done ? "done" : "running"),
          row.firstAudioMs == null ? "-" : `${row.firstAudioMs.toFixed(0)}ms`,
          row.rtf == null ? "-" : row.rtf.toFixed(2),
          row.audioSec == null ? "-" : `${row.audioSec.toFixed(2)}s`,
          row.error || "",
        ];
        for (const [cellIndex, value] of cells.entries()) {
          const td = document.createElement("td");
          td.textContent = value;
          if (cellIndex === 1) td.className = statusClass;
          tr.appendChild(td);
        }
        els.multiResults.appendChild(tr);
      }
    }

    function summarizeMulti(rows, wallMs) {
      const completed = rows.filter((row) => row.done && !row.error);
      const failed = rows.length - completed.length;
      const ttfts = completed.map((row) => row.firstAudioMs).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      const rtfs = completed.map((row) => row.rtf).filter((value) => Number.isFinite(value));
      const audioSec = completed.reduce((sum, row) => sum + Number(row.audioSec || 0), 0);
      const tokens = completed.reduce((sum, row) => sum + Number(row.generatedTokens || 0), 0);
      const p50 = ttfts.length ? ttfts[Math.floor(ttfts.length / 2)] : null;
      const avgRtf = rtfs.length ? rtfs.reduce((sum, value) => sum + value, 0) / rtfs.length : null;
      const wallSec = Math.max(0.001, wallMs / 1000);
      const realtimeThroughput = audioSec / wallSec;
      const tokenTps = tokens > 0 ? tokens / wallSec : null;
      els.multiSummary.textContent =
        `完成 ${completed.length}/${rows.length}，失败 ${failed}，` +
        `TTFT p50=${p50 == null ? "-" : `${p50.toFixed(0)}ms`}，` +
        `avg RTF=${avgRtf == null ? "-" : avgRtf.toFixed(2)}，` +
        `总吞吐=${realtimeThroughput.toFixed(2)}x realtime，` +
        `tokens/s=${tokenTps == null ? "-" : tokenTps.toFixed(1)}`;
    }

    function cloneRequestForMulti(payload, index) {
      const request = JSON.parse(JSON.stringify(payload));
      if (index > 0 && request.text) {
        request.text = `${request.text} [request ${index + 1}]`;
      }
      request.generation = request.generation || {};
      request.stream = request.stream || {};
      request.stream.include_chunk_metadata = true;
      return request;
    }

    function runBackgroundRequest(index, payload) {
      return new Promise((resolve) => {
        const row = {
          index,
          done: false,
          error: "",
          firstAudioMs: null,
          rtf: null,
          audioSec: 0,
          generatedTokens: 0,
        };
        const started = performance.now();
        let localSampleRate = 24000;
        let bytes = 0;
        let finished = false;
        const backgroundWs = new WebSocket(els.wsUrl.value);
        backgroundWs.binaryType = "arraybuffer";
        multiSockets.add(backgroundWs);
        const finish = (error = "") => {
          if (finished) return;
          finished = true;
          multiSockets.delete(backgroundWs);
          row.done = !error;
          row.error = error;
          row.audioSec = bytes / 2 / localSampleRate;
          try {
            backgroundWs.close();
          } catch (err) {
            // ignore close races
          }
          resolve(row);
        };
        const timeout = setTimeout(() => finish("timeout"), 300000);
        backgroundWs.onopen = () => {
          backgroundWs.send(JSON.stringify(cloneRequestForMulti(payload, index)));
        };
        backgroundWs.onmessage = (event) => {
          if (typeof event.data !== "string") {
            if (row.firstAudioMs == null) row.firstAudioMs = performance.now() - started;
            bytes += event.data.byteLength;
            return;
          }
          const data = JSON.parse(event.data);
          if (data.type === "metadata") {
            localSampleRate = Number(data.sample_rate || localSampleRate);
          } else if (data.type === "audio" && data.timings) {
            row.rtf = Number(data.timings.stream_rtf || data.timings.rtf || row.rtf || 0) || null;
            row.generatedTokens = Number(data.timings.context_generated_tokens || data.timings.emitted_frames || row.generatedTokens || 0);
          } else if (data.type === "final") {
            if (data.timings) {
              row.rtf = Number(data.timings.stream_rtf || data.timings.rtf || row.rtf || 0) || null;
              row.generatedTokens = Number(data.timings.context_generated_tokens || data.timings.emitted_frames || row.generatedTokens || 0);
            }
            clearTimeout(timeout);
            finish("");
          } else if (data.type === "error") {
            clearTimeout(timeout);
            finish(String(data.message || "error"));
          }
        };
        backgroundWs.onerror = () => {
          clearTimeout(timeout);
          finish("websocket error");
        };
        backgroundWs.onclose = () => {
          clearTimeout(timeout);
          if (!finished) finish("closed");
        };
      });
    }

    function setupMultiRows(total) {
      const rows = Array.from(
        { length: total },
        (_, index) => ({ index, done: false, error: "", firstAudioMs: null, rtf: null, audioSec: null })
      );
      els.multiPanel.classList.toggle("hidden", total <= 1);
      renderMultiRows(rows);
      els.multiSummary.textContent = total > 1 ? `运行中：同时请求 ${total}` : "单请求";
      return rows;
    }

    async function launchBackgroundRequests(basePayload, rows, started) {
      const total = rows.length;
      const staggerMs = Math.max(0, Number(els.requestStaggerMs.value || 0));
      for (let index = 1; index < total; index += 1) {
        if (multiAbort) break;
        setTimeout(() => {
          if (multiAbort) {
            rows[index] = { ...rows[index], done: false, error: "stopped" };
            renderMultiRows(rows);
            return;
          }
          runBackgroundRequest(index, basePayload)
            .then((row) => {
              rows[index] = row;
              renderMultiRows(rows);
              summarizeMulti(rows, performance.now() - started);
            });
        }, index * staggerMs);
      }
    }

    function stopBackgroundRequests() {
      multiAbort = true;
      for (const socket of Array.from(multiSockets)) {
        try {
          socket.close();
        } catch (err) {
          // ignore close races
        }
      }
    }

    function finalizePrimaryMultiRow(rows) {
      if (!rows || !rows.length) return;
      rows[0] = {
        index: 0,
        done: streamFinal,
        error: streamFinal ? "" : "stopped",
        firstAudioMs: firstAudioAt ? firstAudioAt - startedAt : null,
        rtf: latestRtf || null,
        audioSec: receivedSamples / sampleRate,
        generatedTokens: contextGeneratedTokens,
      };
      renderMultiRows(rows);
      summarizeMulti(rows, performance.now() - startedAt);
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
          voiceCloneDefaultsVersion: settingsVoiceCloneDefaultsVersion,
          maxNewTokens: els.maxNewTokens.value,
          minNewTokens: els.minNewTokens.value,
          requestCount: els.requestCount.value,
          requestStaggerMs: els.requestStaggerMs.value,
          maxVramPercent: els.maxVramPercent.value,
          chunkStrategy: els.chunkStrategy.value,
          verboseLog: els.verboseLog.checked,
          customJsonEnabled: els.customJsonEnabled.checked,
          customRequestJson: els.customRequestJson.value,
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
        const migrateVoiceCloneDefaults =
          Number(data.voiceCloneDefaultsVersion || 0) < settingsVoiceCloneDefaultsVersion;
        for (const [key, value] of Object.entries(data)) {
          if (!Object.prototype.hasOwnProperty.call(els, key)) continue;
          const el = els[key];
          if (!el) continue;
          if (key === "xVectorOnly" && migrateVoiceCloneDefaults) {
            el.checked = false;
            continue;
          }
          if (el.type === "checkbox") el.checked = Boolean(value);
          else el.value = String(value);
        }
        if (migrateVoiceCloneDefaults) {
          data.xVectorOnly = false;
          data.voiceCloneDefaultsVersion = settingsVoiceCloneDefaultsVersion;
          localStorage.setItem(settingsKey, JSON.stringify(data));
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
      updateContextDisplay();
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
      contextPromptTokens = 0;
      contextGeneratedTokens = 0;
      contextUsedTokens = 0;
      contextLimitTokens = 0;
      contextRemainingTokens = 0;
      contextUsagePercent = 0;
      contextGenerationLimitTokens = 0;
      lastRunSummary = null;
      stopBackgroundRequests();
      activeMultiRows = null;
      multiAbort = false;
      els.downloadBtn.disabled = true;
      els.exportSummaryBtn.disabled = true;
      els.multiPanel.classList.add("hidden");
      els.multiResults.innerHTML = "";
      els.multiSummary.textContent = "等待请求";
      els.receiveBar.style.width = "0%";
      els.queueBar.style.width = "0%";
      els.contextBar.style.width = "0%";
      els.contextUsage.textContent = "-";
      els.contextGenerated.textContent = "0";
      els.contextLine.textContent = "上下文使用";
      els.continuityValue.textContent = "-";
      els.samplingValue.textContent = "-";
      updateMetrics();
    }

    async function start() {
      if (!els.customJsonEnabled.checked && !els.text.value.trim()) {
        log("文本不能为空");
        return;
      }
      let payload;
      try {
        payload = effectiveRequestPayload(true);
      } catch (err) {
        log(`请求 JSON 无效：${err && err.message ? err.message : err}`);
        setPlayState("请求错误", "bad");
        return;
      }
      const requestMode = payload.mode || els.mode.value;
      if (!isModeAvailable(requestMode)) {
        log(`${modeLabels[requestMode] || requestMode} 当前不可用：${modeUnavailableReason(requestMode)}`);
        setPlayState("模型缺失", "bad");
        return;
      }
      saveSettings();
      resetRun();
      const requestCount = Math.max(1, Math.min(16, Number(els.requestCount.value || 1)));
      activeMultiRows = setupMultiRows(requestCount);
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
      if (requestCount > 1) {
        log(`同时发送 ${requestCount} 个请求；第 1 个请求用于播放，其余请求只统计性能。`);
        launchBackgroundRequests(payload, activeMultiRows, startedAt);
      }
      log(`连接 ${els.wsUrl.value}`);
      ws = new WebSocket(els.wsUrl.value);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        log(`发送请求 mode=${payload.mode || "-"}, max_new_tokens=${payload.generation && payload.generation.max_new_tokens ? payload.generation.max_new_tokens : "-"}`);
        ws.send(JSON.stringify(cloneRequestForMulti(payload, 0)));
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
            updateContextUsageFromObject(data, true);
            if (Number(data.max_new_tokens || 0) > 0) {
              activeMaxNewTokens = Number(data.max_new_tokens);
            }
            if (data.recommended_playback_buffer_ms) {
              targetBufferSec = Math.min(maxBufferSec, Math.max(0.05, Number(data.recommended_playback_buffer_ms) / 1000));
              if (player) player.setTargetBuffer(targetBufferSec);
            }
            if (data.chunk_strategy) els.strategyValue.textContent = data.chunk_strategy;
            const metadataGraph = data.online_batch_graph_variant || data.graph_variant || "-";
            if (data.realtime_profile || data.graph_variant || data.online_batch_graph_variant) {
              els.profileValue.textContent = data.realtime_profile || data.graph_variant || "-";
              els.graphValue.textContent = metadataGraph;
            }
            if (data.codegen_unroll) els.unrollValue.textContent = String(data.codegen_unroll);
            if (data.codegen_schedule) els.scheduleValue.textContent = String(data.codegen_schedule);
            els.continuityValue.textContent = data.continuous_long_output ? "full_ar" : "short_ar";
            els.samplingValue.textContent = data.long_ar_do_sample ? "sample" : "default";
            const kvProfile = data.kv_cache_profile || "-";
            const kvRelative = Number(data.kv_cache_relative_to_fp16 || 0);
            const kvLabel = kvRelative ? `${kvProfile}/${kvRelative.toFixed(2)}x` : kvProfile;
            const totalLabel = serverMaxTotalTokens ? `, total=${serverMaxTotalTokens}` : "";
            const blocksLabel = serverKvBlocks ? `, blocks=${serverKvBlocks}` : "";
            const kvBudgetLabel = serverKvBudgetMiB ? `, kv_budget=${Math.round(serverKvBudgetMiB)}MiB` : "";
            els.runtimeLine.textContent =
              `decode=${metadataGraph}, kv=${kvLabel}, budget=${serverPromptBudgetConfig}/${serverPromptBudget}${totalLabel}${blocksLabel}${kvBudgetLabel}, ` +
              `prompt=${data.prompt_len || data.prompt_tokens_estimate || "-"}, vram=${Math.round(serverMaxVramPercent)}%`;
            log(
              `metadata sample_rate=${sampleRate}, strategy=${data.chunk_strategy || "-"}, ` +
              `profile=${data.realtime_profile || "-"}, variant=${data.graph_variant || "-"}, production_variant=${metadataGraph}, ` +
              `long=${data.long_text_mode || "-"}, segmented=${data.segmented ? "yes" : "no"}, ` +
              `sample=${data.long_ar_do_sample ? "yes" : "no"}, paged_kv=${data.paged_kv ? "yes" : "no"}, ` +
              `kv=${kvLabel}, prompt_tokens=${data.prompt_len || data.prompt_tokens_estimate || "-"}, ` +
              `max_generation_tokens=${data.max_generation_tokens_available || "-"}, ` +
              `max_prompt_tokens=${serverPromptBudget}, max_total_tokens=${serverMaxTotalTokens || "-"}, kv_blocks=${serverKvBlocks || "-"}`
            );
          } else if (data.type === "final") {
            streamFinal = true;
            if (player) player.startBuffered();
            if (data.timings) {
              latestRtf = Number(data.timings.stream_rtf || data.timings.rtf || latestRtf || 0);
              updateContextUsageFromObject(data.timings, true);
            }
            lastRunSummary = {
              request: effectiveRequestPayload(false, { silent: true }),
              final: data,
              metrics: {
                first_audio_ms: firstAudioAt ? firstAudioAt - startedAt : null,
                first_audible_ms: firstAudibleAt ? firstAudibleAt - startedAt : null,
                chunks: chunkCount,
                audio_seconds: receivedSamples / sampleRate,
                underrun_count: underrunCount,
                stream_rtf: latestRtf || null,
              },
              runtime: {
                profile: els.profileValue.textContent,
                graph: els.graphValue.textContent,
                schedule: els.scheduleValue.textContent,
                continuity: els.continuityValue.textContent,
              },
            };
            finalizePrimaryMultiRow(activeMultiRows);
            updateMetrics(true);
            log(
              `final index=${data.index}, elapsed=${Number(data.elapsed || 0).toFixed(3)}s, ` +
              `stream_rtf=${latestRtf ? latestRtf.toFixed(2) : "-"}`
            );
            finishRun(true);
            els.downloadBtn.disabled = chunks.length === 0;
            els.exportSummaryBtn.disabled = !lastRunSummary;
            if (ws) ws.close();
          } else if (data.type === "audio") {
            pendingAudioTiming = data.timings || null;
            if (pendingAudioTiming) {
              latestRtf = Number(pendingAudioTiming.stream_rtf || pendingAudioTiming.rtf || latestRtf || 0);
              els.continuityValue.textContent = pendingAudioTiming.continuous_long_output ? "full_ar" : (pendingAudioTiming.long_text_mode || "-");
              els.samplingValue.textContent = pendingAudioTiming.long_ar_do_sample ? "sample" : "-";
              updateContextUsageFromObject(pendingAudioTiming, true);
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
          updateContextUsageFromObject(pendingAudioTiming, true);
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
      stopBackgroundRequests();
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
        "xVectorOnly", "maxNewTokens", "minNewTokens", "requestCount", "requestStaggerMs",
        "maxVramPercent", "chunkStrategy", "verboseLog",
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
      if (button.disabled) {
        log(`${modeLabels[button.dataset.mode] || button.dataset.mode} 当前不可用：${modeUnavailableReason(button.dataset.mode)}`);
        return;
      }
      setMode(button.dataset.mode);
    });
    els.modelManager.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-download-mode]");
      if (!button || button.disabled) return;
      downloadModel(button.dataset.downloadMode);
    });
    els.mode.addEventListener("change", () => setMode(els.mode.value));
    els.presetText.addEventListener("change", () => applyPreset(els.presetText.value));
    els.startBtn.addEventListener("click", start);
    els.stopBtn.addEventListener("click", stop);
    els.downloadBtn.addEventListener("click", download);
    els.copyBtn.addEventListener("click", copyRequest);
    els.copyCurlBtn.addEventListener("click", copyCurl);
    els.exportSummaryBtn.addEventListener("click", exportSummary);
    els.customJsonEnabled.addEventListener("change", () => {
      validateCustomJson();
      updateRequestPreviews();
      saveSettings();
    });
    els.customRequestJson.addEventListener("input", () => {
      validateCustomJson();
      saveSettings();
    });
    els.fillCustomJsonBtn.addEventListener("click", fillCustomJsonFromCurrent);
    els.formatCustomJsonBtn.addEventListener("click", formatCustomJson);
    els.clearBtn.addEventListener("click", () => { els.log.textContent = ""; });
    bindFormPersistence();
    setMode(els.mode.value || "voice_design");
    updateChunkStrategyFields();
    updateTextStats();
    checkHealth();
    loadVoices();
    log("页面已加载。点击开始合成后，浏览器会播放收到的 PCM chunk。");
