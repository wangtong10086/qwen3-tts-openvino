from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .native_codegen import NativeCodegenRunner
from .runtime import OpenVINOQwen3TTS, paged_kv_seed_uses_gqa, usable_ov_cache_dir

PREFILL_MODE_CHOICES = {"serial", "dynamic_ragged", "bucketed_padded", "auto"}


@dataclass
class OnlineBatchConfig:
    max_batch_size: int = 8
    wait_ms: float = 2.0
    max_queue_delay_ms: float = 0.0
    max_cache_blocks: int = 2048
    max_events: int = 32
    scheduler: str = "layered"
    max_num_batched_tokens: int = 16
    prefill_seq_buckets: str = "128,256,512,1024"
    prefill_batch_buckets: str = "1,2,4,8"
    decode_batch_buckets: str = "1,2,4,8,16"
    graph_variant: str = "int8_sym_batch_fused_gqa"
    subcode_mode: str = "cached"
    sampled_batch_subcode: str = "off"
    sampled_subcode_parallel_rows: bool = False
    kv_precision: str = "u8"
    block_size: int = 16
    head_dim: int = 128
    continuous_policy: str = "layered_vllm"
    prefill_mode: str = "serial"
    batch_prefill: bool = False
    disable_fused_decode: bool = True
    continuous_batch_subcode: bool = False


@dataclass
class OnlineBatchRequest:
    sequence: np.ndarray
    tts_pad_embed: np.ndarray
    max_new_tokens: int
    min_new_tokens: int
    repetition_penalty: float
    vocab_size: int
    num_code_groups: int
    eos_token_id: int
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    seed: int = 0
    submitted_at: float = field(default_factory=time.time)
    native_id: int | None = None
    output: queue.Queue[object] = field(default_factory=lambda: queue.Queue(maxsize=16))
    cancelled: threading.Event = field(default_factory=threading.Event)


class OnlineBatchScheduler:
    """Single-runtime online codegen scheduler.

    The scheduler batches codec autoregressive steps only. Prompt construction
    and speech decoding stay per request so VoiceDesign/CustomVoice/VoiceClone
    can share this path once they have produced prompt embeddings.
    """

    def __init__(self, runtime: OpenVINOQwen3TTS, config: OnlineBatchConfig | None = None):
        self.runtime = runtime
        self.config = config or OnlineBatchConfig()
        self.continuous_batch_subcode_reason: str | None = None
        self._incoming: queue.Queue[OnlineBatchRequest] = queue.Queue()
        self._requests: dict[int, OnlineBatchRequest] = {}
        self._lock = threading.Lock()
        self._runner_lock = threading.Lock()
        self._stop = threading.Event()
        self._runner: NativeCodegenRunner | None = None
        self._last_stats: dict = {"ready": False, "active": 0, "pending": 0}
        self._stats_accumulator: dict = self._new_stats_accumulator()
        self._thread = threading.Thread(target=self._loop, name="qwen3-tts-online-batch", daemon=True)
        self._thread.start()

    @staticmethod
    def _new_stats_accumulator() -> dict:
        return {
            "scheduler_step_count": 0,
            "prefill_modes_seen": [],
            "prefill_batch_buckets_seen": [],
            "prefill_seq_buckets_seen": [],
            "decode_batch_buckets_seen": [],
            "batch_prefill_step_count": 0,
            "max_prefill_batch_bucket": 0,
            "max_decode_batch_bucket": 0,
            "batch_fused_decode_step_count": 0,
            "batch_fused_decode_token_count": 0,
            "batch_single_decode_step_count": 0,
            "batch_single_decode_token_count": 0,
            "batch_fused_decode_active1_bypass_count": 0,
            "batch_fused_decode_logits_bypass_count": 0,
            "batch_subcode_used_count": 0,
            "host_prepare_ms": 0.0,
            "tensor_bind_ms": 0.0,
            "codegen_infer_ms": 0.0,
            "codegen_prefill_infer_ms": 0.0,
            "codegen_decode_infer_ms": 0.0,
            "codegen_subcode_infer_ms": 0.0,
            "subcode_bind_ms": 0.0,
            "subcode_output_read_ms": 0.0,
            "subcode_next_embed_ms": 0.0,
            "sampling_ms": 0.0,
            "host_copy_ms": 0.0,
            "decode_step_prebind_update_ms": 0.0,
            "total_step_elapsed_ms": 0.0,
            "subcode_host_copy_bytes": 0,
            "subcode_host_copy_fallback_count": 0,
            "split_subcode_hidden_direct_bind_count": 0,
            "split_subcode_hidden_bind_fallback_count": 0,
            "split_subcode_hidden_copy_bytes": 0,
            "split_subcode_remote_next_embed_fallback_count": 0,
            "split_subcode_next_embed_host_read_count": 0,
            "active_batch_histogram": [],
            "sampled_batch_subcode_policy_seen": [],
            "sampled_batch_subcode_used_count": 0,
            "sampled_batch_subcode_verified_count": 0,
            "sampled_batch_subcode_fallback_count": 0,
            "sampled_batch_subcode_mismatch_count": 0,
            "sampled_batch_subcode_code_mismatch_count": 0,
            "sampled_batch_subcode_embed_mismatch_count": 0,
            "sampled_batch_subcode_max_abs_diff": 0.0,
            "sampled_batch_subcode_fallback_reasons": [],
        }

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def stats(self) -> dict:
        stats = dict(self._last_stats)
        with self._lock:
            stats["python_tracked_requests"] = len(self._requests)
        return stats

    def ensure_ready(self) -> None:
        self._ensure_runner()

    def warmup(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        *,
        batch_size: int | None = None,
        max_new_tokens: int = 4,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
        reset_after: bool = True,
    ) -> dict:
        """Run a short synthetic online batch to warm OpenVINO infer requests.

        The native online state is reset afterwards, so the warmup does not
        consume cache blocks or leak requests into user-visible statistics.
        """

        self.ensure_ready()
        warmup_batch = max(1, min(int(batch_size or self.config.max_batch_size), int(self.config.max_batch_size)))
        iterators = [
            self.submit(
                sequence,
                tts_pad_embed,
                max_new_tokens=max(1, int(max_new_tokens)),
                min_new_tokens=max(0, int(min_new_tokens)),
                repetition_penalty=float(repetition_penalty),
                do_sample=bool(do_sample),
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                seed=int(seed) + index,
            )
            for index in range(warmup_batch)
        ]
        frames = 0
        started = time.perf_counter()
        for iterator in iterators:
            for _code in iterator:
                frames += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        runner = self._ensure_runner()
        with self._lock:
            if self._requests:
                raise RuntimeError("online batch warmup ended with active requests")
            self._stats_accumulator = self._new_stats_accumulator()
        if reset_after:
            runner.online_batch_reset(self.config.max_cache_blocks)
        self._last_stats = self._merge_step_stats(runner.online_batch_stats(), None, count_step=False)
        return {"batch_size": warmup_batch, "frames": frames, "elapsed_ms": elapsed_ms, "reset_after": bool(reset_after)}

    def _merge_step_stats(self, stats: dict, timing: dict | None = None, *, count_step: bool = True) -> dict:
        merged = dict(stats)
        timing = dict(timing or {})
        if count_step:
            self._stats_accumulator["scheduler_step_count"] += 1

        def append_unique_list(key: str, value) -> None:
            if value is None or value == 0:
                return
            values = self._stats_accumulator[key]
            if value not in values:
                values.append(value)

        def add_histogram(key: str, values) -> None:
            if not isinstance(values, list):
                return
            histogram = self._stats_accumulator[key]
            if len(histogram) < len(values):
                histogram.extend([0] * (len(values) - len(histogram)))
            for index, value in enumerate(values):
                if value is None:
                    continue
                try:
                    histogram[index] += int(str(value))
                except (TypeError, ValueError):
                    continue

        prefill_mode = stats.get("prefill_mode") or timing.get("prefill_mode")
        append_unique_list("prefill_modes_seen", prefill_mode)
        prefill_batch_bucket = int(timing.get("prefill_batch_bucket") or stats.get("last_prefill_batch_bucket") or 0)
        prefill_seq_bucket = int(timing.get("prefill_seq_bucket") or stats.get("last_prefill_seq_bucket") or 0)
        decode_batch_bucket = int(timing.get("decode_batch_bucket") or stats.get("last_decode_batch_bucket") or 0)
        append_unique_list("prefill_batch_buckets_seen", prefill_batch_bucket)
        append_unique_list("prefill_seq_buckets_seen", prefill_seq_bucket)
        append_unique_list("decode_batch_buckets_seen", decode_batch_bucket)
        self._stats_accumulator["max_prefill_batch_bucket"] = max(
            int(self._stats_accumulator["max_prefill_batch_bucket"]),
            prefill_batch_bucket,
        )
        self._stats_accumulator["max_decode_batch_bucket"] = max(
            int(self._stats_accumulator["max_decode_batch_bucket"]),
            decode_batch_bucket,
        )
        if bool(timing.get("batch_prefill_enabled", False)):
            self._stats_accumulator["batch_prefill_step_count"] += 1
        if bool(timing.get("batch_subcode_enabled", False)) or bool(stats.get("last_batch_subcode_used", False)):
            self._stats_accumulator["batch_subcode_used_count"] += 1
        for key in (
            "batch_fused_decode_step_count",
            "batch_fused_decode_token_count",
            "batch_single_decode_step_count",
            "batch_single_decode_token_count",
            "batch_fused_decode_active1_bypass_count",
            "batch_fused_decode_logits_bypass_count",
            "subcode_host_copy_bytes",
            "subcode_host_copy_fallback_count",
            "split_subcode_hidden_direct_bind_count",
            "split_subcode_hidden_bind_fallback_count",
            "split_subcode_hidden_copy_bytes",
            "split_subcode_remote_next_embed_fallback_count",
            "split_subcode_next_embed_host_read_count",
            "sampled_batch_subcode_fallback_count",
            "sampled_batch_subcode_mismatch_count",
            "sampled_batch_subcode_code_mismatch_count",
            "sampled_batch_subcode_embed_mismatch_count",
        ):
            try:
                self._stats_accumulator[key] += int(timing.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        add_histogram("active_batch_histogram", timing.get("active_batch_histogram") or stats.get("last_active_batch_histogram"))
        sampled_policy = timing.get("sampled_batch_subcode_policy") or stats.get("sampled_batch_subcode_policy")
        append_unique_list("sampled_batch_subcode_policy_seen", sampled_policy)
        if bool(timing.get("sampled_batch_subcode_used", False)):
            self._stats_accumulator["sampled_batch_subcode_used_count"] += 1
        if bool(timing.get("sampled_batch_subcode_verified", False)):
            self._stats_accumulator["sampled_batch_subcode_verified_count"] += 1
        sampled_reason = timing.get("sampled_batch_subcode_fallback_reason") or stats.get(
            "sampled_batch_subcode_fallback_reason"
        )
        append_unique_list("sampled_batch_subcode_fallback_reasons", sampled_reason)
        try:
            self._stats_accumulator["sampled_batch_subcode_max_abs_diff"] = max(
                float(self._stats_accumulator["sampled_batch_subcode_max_abs_diff"]),
                float(timing.get("sampled_batch_subcode_max_abs_diff", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            pass
        for key in (
            "host_prepare_ms",
            "tensor_bind_ms",
            "codegen_infer_ms",
            "codegen_prefill_infer_ms",
            "codegen_decode_infer_ms",
            "codegen_subcode_infer_ms",
            "subcode_bind_ms",
            "subcode_output_read_ms",
            "subcode_next_embed_ms",
            "sampling_ms",
            "host_copy_ms",
            "decode_step_prebind_update_ms",
        ):
            try:
                self._stats_accumulator[key] += float(timing.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        try:
            self._stats_accumulator["total_step_elapsed_ms"] += float(timing.get("total_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        merged.update(self._stats_accumulator)
        return merged

    def submit(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> Iterator[np.ndarray]:
        request = OnlineBatchRequest(
            sequence=np.ascontiguousarray(sequence, dtype=np.float32),
            tts_pad_embed=np.ascontiguousarray(tts_pad_embed, dtype=np.float32),
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            repetition_penalty=float(repetition_penalty),
            vocab_size=int(self.runtime.ids["vocab_size"]),
            num_code_groups=int(self.runtime.num_code_groups),
            eos_token_id=int(self.runtime.ids["codec_eos_token_id"]),
            do_sample=bool(do_sample),
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            seed=int(seed),
        )
        self._incoming.put(request)

        def iterator():
            try:
                while True:
                    item = request.output.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield np.asarray(item, dtype=np.int64)
            finally:
                request.cancelled.set()

        return iterator()

    def _select_graphs(self) -> tuple[Path, Path, Path | None, Path | None, int]:
        manifest = self.runtime.manifest
        graphs = manifest.get("graphs") or {}
        variant_graphs = ((manifest.get("graph_variants") or {}).get(self.config.graph_variant) or {}).get("graphs") or {}

        def select_graph(*keys: str) -> str | None:
            for key in keys:
                graph = variant_graphs.get(key) or graphs.get(key)
                if graph and (self.runtime.ir_dir / graph).exists():
                    return graph
            return None

        paged_seed_graphs = dict(graphs.get("paged_kv_seed") or {})
        paged_seed_graphs.update(variant_graphs.get("paged_kv_seed") or {})
        seed_key = "talker_stateful_batch_gqa" if paged_seed_graphs.get("talker_stateful_batch_gqa") else "talker_stateful_batch"
        if not paged_seed_graphs.get(seed_key):
            raise RuntimeError("online batching requires paged_kv_seed.talker_stateful_batch_gqa or talker_stateful_batch")
        fused_key = "fused_cache_step_batch_gqa" if paged_kv_seed_uses_gqa(seed_key) else "fused_cache_step_batch"
        fused_graph = paged_seed_graphs.get(fused_key)
        if not self.config.disable_fused_decode and not fused_graph:
            raise RuntimeError(f"online batching requires paged_kv_seed.{fused_key}")
        subcode_mode = self.config.subcode_mode
        if subcode_mode == "cached_exact":
            subcode = select_graph("subcode_greedy_cached_exact")
            batch_subcode = select_graph("subcode_greedy_cached_exact_batch")
        elif subcode_mode == "recompute_exact":
            subcode = select_graph("subcode_greedy_exact")
            batch_subcode = None
        elif subcode_mode == "recompute":
            subcode = select_graph("subcode_greedy")
            batch_subcode = None
        else:
            subcode = select_graph("subcode_greedy_cached", "subcode_greedy")
            batch_subcode = select_graph("subcode_greedy_cached_batch")
        if not subcode:
            raise RuntimeError("online batching requires a subcode_greedy graph")
        heads = 8 if paged_kv_seed_uses_gqa(seed_key) else 16
        return (
            self.runtime.ir_dir / paged_seed_graphs[seed_key],
            self.runtime.ir_dir / subcode,
            self.runtime.ir_dir / batch_subcode if batch_subcode else None,
            self.runtime.ir_dir / fused_graph if fused_graph else None,
            heads,
        )

    def _effective_prefill_mode(self) -> str:
        mode = str(self.config.prefill_mode or "serial").strip().replace("-", "_").lower()
        if mode not in PREFILL_MODE_CHOICES:
            raise ValueError(
                "OnlineBatchConfig.prefill_mode must be one of "
                "serial, dynamic_ragged, bucketed_padded, auto"
            )
        if mode == "auto":
            mode = "serial"
        if mode == "serial" and self.config.batch_prefill:
            mode = "dynamic_ragged"
        return mode

    def _ensure_runner(self) -> NativeCodegenRunner:
        with self._runner_lock:
            if self._runner is not None:
                return self._runner
            prefill_graph, subcode_graph, batch_subcode_graph, fused_graph, heads = self._select_graphs()
            if batch_subcode_graph is not None:
                os.environ["QWEN3_TTS_OV_NATIVE_BATCH_SUBCODE_XML"] = str(batch_subcode_graph)
            else:
                os.environ.pop("QWEN3_TTS_OV_NATIVE_BATCH_SUBCODE_XML", None)
            scheduler = str(self.config.scheduler or "layered").strip().replace("-", "_").lower()
            if scheduler != "layered":
                raise ValueError("OnlineBatchConfig.scheduler must be layered")
            continuous_policy = str(self.config.continuous_policy or "").strip().replace("-", "_").lower()
            if not continuous_policy:
                continuous_policy = "layered_vllm"
            os.environ["QWEN3_TTS_OV_NATIVE_SCHEDULER"] = scheduler
            os.environ["QWEN3_TTS_OV_NATIVE_PREFILL_SEQ_BUCKETS"] = str(self.config.prefill_seq_buckets)
            os.environ["QWEN3_TTS_OV_NATIVE_PREFILL_BATCH_BUCKETS"] = str(self.config.prefill_batch_buckets)
            os.environ["QWEN3_TTS_OV_NATIVE_DECODE_BATCH_BUCKETS"] = str(self.config.decode_batch_buckets)
            os.environ["QWEN3_TTS_OV_NATIVE_MAX_NUM_BATCHED_TOKENS"] = str(
                max(1, int(self.config.max_num_batched_tokens))
            )
            os.environ["QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_POLICY"] = continuous_policy
            os.environ.setdefault("QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH", "1")
            prefill_mode = self._effective_prefill_mode()
            os.environ["QWEN3_TTS_OV_NATIVE_PREFILL_MODE"] = prefill_mode
            os.environ["QWEN3_TTS_OV_NATIVE_BATCH_PREFILL"] = "1" if prefill_mode != "serial" else "0"
            os.environ["QWEN3_TTS_OV_NATIVE_BATCH_PREFILL_SUBCODE"] = (
                "1" if self.config.continuous_batch_subcode else "0"
            )
            os.environ["QWEN3_TTS_OV_NATIVE_CONTINUOUS_BATCH_SUBCODE"] = (
                "1" if self.config.continuous_batch_subcode else "0"
            )
            if self.config.disable_fused_decode or fused_graph is None:
                os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_FUSED_BATCH_DECODE_XML", None)
            else:
                os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_FUSED_BATCH_DECODE_XML"] = str(fused_graph)
            os.environ["QWEN3_TTS_OV_NATIVE_SAMPLED_BATCH_SUBCODE"] = str(self.config.sampled_batch_subcode)
            os.environ["QWEN3_TTS_OV_NATIVE_SAMPLED_SUBCODE_PARALLEL_ROWS"] = (
                "1" if self.config.sampled_subcode_parallel_rows else "0"
            )
            cache_dir = None if self.runtime.disable_ov_cache else usable_ov_cache_dir(self.runtime.cache_dir)
            runner = NativeCodegenRunner(
                prefill_graph=prefill_graph,
                decode_graph=prefill_graph,
                device=str(os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE") or self.runtime.device),
                cache_dir=cache_dir,
                cache_mode=str(self.runtime.compile_config.get("CACHE_MODE", "OPTIMIZE_SPEED")),
                paged_kv=True,
                kv_cache_precision=self.config.kv_precision,
                kv_cache_heads=heads,
                kv_cache_block_size=self.config.block_size,
                kv_cache_head_dim=self.config.head_dim,
                paged_kv_subcode_graph=subcode_graph,
            )
            runner.online_batch_reset(self.config.max_cache_blocks)
            self._runner = runner
            self._last_stats = runner.online_batch_stats()
            return runner

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                runner = self._ensure_runner()
                drained: list[OnlineBatchRequest] = []
                with self._lock:
                    has_active_requests = bool(self._requests)
                admission_wait_ms = 0.0 if has_active_requests else max(0.0, self.config.wait_ms)
                deadline = time.time() + admission_wait_ms / 1000.0
                while len(drained) < self.config.max_batch_size:
                    if drained and self.config.max_queue_delay_ms > 0:
                        oldest_deadline = drained[0].submitted_at + max(0.0, self.config.max_queue_delay_ms) / 1000.0
                        deadline = min(deadline, oldest_deadline)
                    timeout = max(0.0, deadline - time.time())
                    try:
                        drained.append(self._incoming.get(timeout=timeout))
                    except queue.Empty:
                        break
                for request in drained:
                    if request.cancelled.is_set():
                        request.output.put(None)
                        continue
                    native_id = runner.online_batch_add_sequence(
                        request.sequence,
                        request.tts_pad_embed,
                        max_new_tokens=request.max_new_tokens,
                        min_new_tokens=request.min_new_tokens,
                        repetition_penalty=request.repetition_penalty,
                        vocab_size=request.vocab_size,
                        num_code_groups=request.num_code_groups,
                        eos_token_id=request.eos_token_id,
                        do_sample=request.do_sample,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        temperature=request.temperature,
                        seed=request.seed,
                    )
                    request.native_id = native_id
                    with self._lock:
                        self._requests[native_id] = request

                with self._lock:
                    active_ids = list(self._requests)
                for native_id in active_ids:
                    request = self._requests.get(native_id)
                    if request and request.cancelled.is_set():
                        runner.online_batch_cancel(native_id)
                        with self._lock:
                            self._requests.pop(native_id, None)

                if not drained and not self._requests:
                    self._last_stats = self._merge_step_stats(runner.online_batch_stats(), None, count_step=False)
                    time.sleep(max(0.001, self.config.wait_ms / 1000.0))
                    continue

                result = runner.online_batch_step(
                    max_decode_batch=self.config.max_batch_size,
                    max_events=self.config.max_events,
                    num_code_groups=self.runtime.num_code_groups,
                )
                ids = result["ids"].tolist()
                kinds = result["kinds"].tolist()
                codes = result["codes"]
                for row, native_id in enumerate(ids):
                    request = self._requests.get(int(native_id))
                    if request is None:
                        continue
                    kind = int(kinds[row])
                    if kind in {1, 3}:
                        request.output.put(np.asarray(codes[row], dtype=np.int64))
                    if kind in {2, 3}:
                        request.output.put(None)
                        with self._lock:
                            self._requests.pop(int(native_id), None)
                self._last_stats = self._merge_step_stats(runner.online_batch_stats(), result.get("timing"))
            except BaseException as exc:
                with self._lock:
                    requests = list(self._requests.values())
                    self._requests.clear()
                while True:
                    try:
                        requests.append(self._incoming.get_nowait())
                    except queue.Empty:
                        break
                for request in requests:
                    request.output.put(exc)
                    request.output.put(None)
                time.sleep(0.05)
