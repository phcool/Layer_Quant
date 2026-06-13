from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_int4_cache_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    packed_stride_b,
    packed_stride_h,
    packed_stride_t,
    packed_stride_p,
    scale_stride_b,
    scale_stride_h,
    scale_stride_t,
    scale_stride_g,
    out_stride_b,
    out_stride_h,
    out_stride_t,
    out_stride_d,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
):
    bh = tl.program_id(0)
    token = tl.program_id(1)
    group = tl.program_id(2)

    d_start = group * group_size
    d_offsets = tl.arange(0, BLOCK_GROUP)
    d_mask = d_offsets < group_size
    pair = (d_start + d_offsets) // 2
    byte = tl.load(
        packed_ptr
        + bh * packed_stride_h
        + token * packed_stride_t
        + pair * packed_stride_p,
        mask=d_mask & (token < seq_len),
        other=0,
    ).to(tl.int32)
    nibble = tl.where(((d_start + d_offsets) & 1) == 0, byte & 15, (byte >> 4) & 15)
    signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    scale = tl.load(
        scale_ptr + bh * scale_stride_h + token * scale_stride_t + group * scale_stride_g,
        mask=token < seq_len,
        other=0.0,
    ).to(tl.float32)
    value = signed * scale
    tl.store(
        out_ptr + bh * out_stride_h + token * out_stride_t + (d_start + d_offsets) * out_stride_d,
        value,
        mask=d_mask & (token < seq_len) & ((d_start + d_offsets) < head_dim),
    )


def dequant_int4_cache_to_temp(layer_cache, seq_len: int, tmp_k: torch.Tensor, tmp_v: torch.Tensor, group_size: int) -> None:
    if seq_len <= 0:
        return
    batch, kv_heads, _, packed_dim = layer_cache.k_packed.shape
    head_dim = packed_dim * 2
    groups = head_dim // group_size
    grid = (batch * kv_heads, seq_len, groups)
    block_group = triton.next_power_of_2(group_size)
    k_packed = layer_cache.k_packed[:, :, :seq_len]
    v_packed = layer_cache.v_packed[:, :, :seq_len]
    k_scale = layer_cache.k_scale[:, :, :seq_len]
    v_scale = layer_cache.v_scale[:, :, :seq_len]
    k_out = tmp_k[:, :, :seq_len]
    v_out = tmp_v[:, :, :seq_len]
    _dequant_int4_cache_kernel[grid](
        k_packed,
        k_scale,
        k_out,
        *k_packed.stride(),
        *k_scale.stride(),
        *k_out.stride(),
        seq_len,
        head_dim,
        group_size,
        BLOCK_GROUP=block_group,
    )
    _dequant_int4_cache_kernel[grid](
        v_packed,
        v_scale,
        v_out,
        *v_packed.stride(),
        *v_scale.stride(),
        *v_out.stride(),
        seq_len,
        head_dim,
        group_size,
        BLOCK_GROUP=block_group,
    )


@dataclass
class EventSpan:
    start: torch.cuda.Event
    end: torch.cuda.Event

    def elapsed_ms(self) -> float:
        return float(self.start.elapsed_time(self.end))


@dataclass
class StageRecord:
    step_idx: int
    attn_layer_id: int
    seq_len: int
    ready: torch.cuda.Event
    done: torch.cuda.Event
    start: torch.cuda.Event
    end: torch.cuda.Event
    stage_target: str = "KV"

    def elapsed_ms(self) -> float:
        return float(self.start.elapsed_time(self.end))


@dataclass
class AttentionRecord:
    step_idx: int
    attn_layer_id: int
    prev_attn_layer_id: int | None
    gap: EventSpan
    attn: EventSpan
    wait: EventSpan | None
    stage: StageRecord | None
    staging_mode: str
    stage_target: str
    batch_size: int
    context_length: int
    decode_position: int


@dataclass
class LayerRecord:
    step_idx: int
    layer_id: int
    layer_type: str
    span: EventSpan
    staging_mode: str
    batch_size: int
    context_length: int
    decode_position: int


@dataclass
class KVStagingProfiler:
    attention_layer_ids: list[int]
    layer_types: dict[int, str]
    staging_mode: str
    batch_size: int
    context_length: int
    kv_group_size: int
    enable_nvtx: bool = False
    current_step_idx: int = -1
    current_decode_position: int = -1
    cache_params: object | None = None
    device: torch.device | None = None
    staging_stream: torch.cuda.Stream | None = None
    active: bool = False
    step_start_event: torch.cuda.Event | None = None
    last_attention_end_event: torch.cuda.Event | None = None
    last_attention_layer_id: int | None = None
    scheduled_stages: dict[int, StageRecord] = field(default_factory=dict)
    tmp_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    attention_records: list[AttentionRecord] = field(default_factory=list)
    layer_records: list[LayerRecord] = field(default_factory=list)
    debug_stage_records: list[StageRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.staging_mode not in {"none", "dequant"}:
            raise ValueError(f"Unsupported staging_mode={self.staging_mode!r}")
        self.next_attention = {
            layer_id: self.attention_layer_ids[idx + 1] if idx + 1 < len(self.attention_layer_ids) else None
            for idx, layer_id in enumerate(self.attention_layer_ids)
        }
        self.previous_attention = {
            layer_id: self.attention_layer_ids[idx - 1] if idx > 0 else None
            for idx, layer_id in enumerate(self.attention_layer_ids)
        }

    def event(self) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def dependency_event(self) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=False)

    def nvtx_push(self, name: str) -> None:
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(name)

    def nvtx_pop(self) -> None:
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()

    def start_step(self, step_idx: int, decode_position: int, cache_params, device: torch.device) -> None:
        self.active = True
        self.current_step_idx = step_idx
        self.current_decode_position = decode_position
        self.cache_params = cache_params
        self.device = device
        self.step_start_event = self.event()
        self.step_start_event.record(torch.cuda.current_stream(device))
        self.last_attention_end_event = self.step_start_event
        self.last_attention_layer_id = None
        self.scheduled_stages = {}
        if self.staging_mode == "dequant":
            if self.staging_stream is None:
                self.staging_stream = torch.cuda.Stream(device=device)
            self.schedule_stage(self.attention_layer_ids[0])
        if self.enable_nvtx:
            self.nvtx_push(f"decode_step_{step_idx}")
            self.nvtx_push(f"gap_before_attn_{self.attention_layer_ids[0]}")

    def end_step(self) -> None:
        if self.enable_nvtx:
            self.nvtx_pop()
        self.cache_params = None
        self.active = False

    def cache_for_layer(self, attn_layer_id: int):
        caches = getattr(self.cache_params, "_int4_kv_layer_caches", None)
        if caches is None or attn_layer_id not in caches:
            return None
        return caches[attn_layer_id]

    def allocate_stage_buffers(self, cache_params, max_seq_len: int, dtype: torch.dtype = torch.bfloat16) -> None:
        if self.staging_mode != "dequant":
            return
        self.cache_params = cache_params
        for attn_layer_id in self.attention_layer_ids:
            layer_cache = self.cache_for_layer(attn_layer_id)
            if layer_cache is None:
                continue
            batch, kv_heads, _, packed_dim = layer_cache.k_packed.shape
            head_dim = packed_dim * 2
            tmp_k = torch.empty(
                (batch, kv_heads, max_seq_len, head_dim),
                device=layer_cache.k_packed.device,
                dtype=dtype,
            )
            tmp_v = torch.empty_like(tmp_k)
            self.tmp_buffers[attn_layer_id] = (tmp_k, tmp_v)

    def temp_buffers(self, attn_layer_id: int, layer_cache, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, kv_heads, _, packed_dim = layer_cache.k_packed.shape
        current = self.tmp_buffers.get(attn_layer_id)
        if current is None:
            raise RuntimeError(f"Missing preallocated staging buffers for attention layer {attn_layer_id}")
        tmp_k, tmp_v = current
        if tmp_k.shape[0] != batch or tmp_k.shape[1] != kv_heads or tmp_k.shape[2] < seq_len:
            raise RuntimeError(
                f"Staging buffer for layer {attn_layer_id} has shape {tuple(tmp_k.shape)}, "
                f"but needs batch={batch}, kv_heads={kv_heads}, seq_len>={seq_len}, packed_dim={packed_dim}"
            )
        return current

    def schedule_stage(self, attn_layer_id: int | None) -> StageRecord | None:
        if attn_layer_id is None or self.staging_mode != "dequant":
            return None
        layer_cache = self.cache_for_layer(attn_layer_id)
        if layer_cache is None:
            return None
        seq_len = int(layer_cache.length)
        if seq_len <= 0:
            return None
        tmp_k, tmp_v = self.temp_buffers(attn_layer_id, layer_cache, seq_len)
        device = layer_cache.k_packed.device
        main_stream = torch.cuda.current_stream(device)
        stream = self.staging_stream
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.staging_stream = stream
        ready = self.dependency_event()
        done = self.dependency_event()
        start = self.event()
        end = self.event()
        ready.record(main_stream)
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            if self.enable_nvtx:
                torch.cuda.nvtx.range_push(f"stage_attn_{attn_layer_id}")
            start.record(stream)
            dequant_int4_cache_to_temp(layer_cache, seq_len, tmp_k, tmp_v, self.kv_group_size)
            end.record(stream)
            done.record(stream)
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()
        record = StageRecord(
            step_idx=self.current_step_idx,
            attn_layer_id=attn_layer_id,
            seq_len=seq_len,
            ready=ready,
            done=done,
            start=start,
            end=end,
        )
        self.scheduled_stages[attn_layer_id] = record
        return record

    def debug_stage_repeated(self, attn_layer_id: int, repeats: int) -> None:
        if self.staging_mode != "dequant" or repeats <= 0:
            return
        for repeat in range(repeats):
            record = self.schedule_stage(attn_layer_id)
            if record is not None:
                record.step_idx = -1000 - repeat
                self.debug_stage_records.append(record)

    def begin_layer(self, layer_id: int) -> torch.cuda.Event:
        start = self.event()
        start.record(torch.cuda.current_stream(self.device))
        self.nvtx_push(f"layer_{layer_id}_{self.layer_types[layer_id]}")
        return start

    def end_layer(self, layer_id: int, start: torch.cuda.Event) -> None:
        end = self.event()
        end.record(torch.cuda.current_stream(self.device))
        self.nvtx_pop()
        self.layer_records.append(
            LayerRecord(
                step_idx=self.current_step_idx,
                layer_id=layer_id,
                layer_type=self.layer_types[layer_id],
                span=EventSpan(start, end),
                staging_mode=self.staging_mode,
                batch_size=self.batch_size,
                context_length=self.context_length,
                decode_position=self.current_decode_position,
            )
        )

    def before_attention_layer(self, attn_layer_id: int) -> tuple[torch.cuda.Event, EventSpan | None, StageRecord | None]:
        if self.enable_nvtx:
            self.nvtx_pop()
        gap_end = self.event()
        gap_end.record(torch.cuda.current_stream(self.device))
        gap = EventSpan(self.last_attention_end_event, gap_end)
        stage = self.scheduled_stages.get(attn_layer_id)
        wait = None
        if stage is not None:
            wait_start = self.event()
            wait_end = self.event()
            if self.enable_nvtx:
                self.nvtx_push(f"wait_attn_{attn_layer_id}")
            wait_start.record(torch.cuda.current_stream(self.device))
            torch.cuda.current_stream(self.device).wait_event(stage.done)
            wait_end.record(torch.cuda.current_stream(self.device))
            if self.enable_nvtx:
                self.nvtx_pop()
            wait = EventSpan(wait_start, wait_end)
        layer_start = self.begin_layer(attn_layer_id)
        self._pending_attention_gap = gap
        self._pending_attention_wait = wait
        self._pending_attention_stage = stage
        return layer_start, wait, stage

    def attention_mixer(self, attn_layer_id: int, fn, *args, **kwargs):
        attn_start = self.event()
        attn_end = self.event()
        self.nvtx_push(f"attn_{attn_layer_id}")
        attn_start.record(torch.cuda.current_stream(self.device))
        out = fn(*args, **kwargs)
        attn_end.record(torch.cuda.current_stream(self.device))
        self.nvtx_pop()
        self.attention_records.append(
            AttentionRecord(
                step_idx=self.current_step_idx,
                attn_layer_id=attn_layer_id,
                prev_attn_layer_id=self.last_attention_layer_id,
                gap=self._pending_attention_gap,
                attn=EventSpan(attn_start, attn_end),
                wait=self._pending_attention_wait,
                stage=self._pending_attention_stage,
                staging_mode=self.staging_mode,
                stage_target="KV" if self._pending_attention_stage is not None else "NONE",
                batch_size=self.batch_size,
                context_length=self.context_length,
                decode_position=self.current_decode_position,
            )
        )
        self.last_attention_end_event = attn_end
        self.last_attention_layer_id = attn_layer_id
        next_attn = self.next_attention.get(attn_layer_id)
        self.schedule_stage(next_attn)
        if self.enable_nvtx and next_attn is not None:
            self.nvtx_push(f"gap_before_attn_{next_attn}")
        return out

    def attention_rows(
        self,
        gap_slowdown_by_step: dict[tuple[int, int], dict[str, float | None]] | None = None,
    ) -> list[dict]:
        rows = []
        for rec in self.attention_records:
            gap_ms = rec.gap.elapsed_ms()
            stage_ms = rec.stage.elapsed_ms() if rec.stage is not None else None
            wait_ms = rec.wait.elapsed_ms() if rec.wait is not None else None
            hideability = None if not stage_ms else gap_ms / stage_ms
            overlap_ratio = None if not stage_ms or wait_ms is None else max(0.0, stage_ms - wait_ms) / stage_ms
            slowdown = None if gap_slowdown_by_step is None else gap_slowdown_by_step.get((rec.step_idx, rec.attn_layer_id))
            rows.append(
                {
                    "staging_mode": rec.staging_mode,
                    "batch_size": rec.batch_size,
                    "context_length": rec.context_length,
                    "decode_step": rec.step_idx,
                    "decode_position": rec.decode_position,
                    "attn_layer_id": rec.attn_layer_id,
                    "prev_attn_layer_id": rec.prev_attn_layer_id,
                    "T_gap_ms": gap_ms,
                    "T_stage_ms": stage_ms,
                    "T_attn_ms": rec.attn.elapsed_ms(),
                    "wait_before_attention_ms": wait_ms,
                    "hideability": hideability,
                    "overlap_ratio_wait_based": overlap_ratio,
                    "stage_target": rec.stage_target,
                    "gap_slowdown_ms": None if slowdown is None else slowdown.get("gap_slowdown_ms"),
                    "gap_slowdown_pct": None if slowdown is None else slowdown.get("gap_slowdown_pct"),
                    "net_stage_cost_ms": None if slowdown is None else slowdown.get("net_stage_cost_ms"),
                    "effective_overlap_wall_clock": None if slowdown is None else slowdown.get("effective_overlap"),
                }
            )
        return rows

    def layer_rows(self) -> list[dict]:
        return [
            {
                "staging_mode": rec.staging_mode,
                "batch_size": rec.batch_size,
                "context_length": rec.context_length,
                "decode_step": rec.step_idx,
                "decode_position": rec.decode_position,
                "layer_id": rec.layer_id,
                "layer_type": rec.layer_type,
                "elapsed_ms": rec.span.elapsed_ms(),
            }
            for rec in self.layer_records
        ]

    def debug_stage_rows(self) -> list[dict]:
        return [
            {
                "staging_mode": self.staging_mode,
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "attn_layer_id": rec.attn_layer_id,
                "repeat": -1000 - rec.step_idx,
                "seq_len": rec.seq_len,
                "T_stage_ms": rec.elapsed_ms(),
                "stage_target": rec.stage_target,
            }
            for rec in self.debug_stage_records
        ]


def patch_profiled_blocks(model, profiler: KVStagingProfiler) -> None:
    for layer_id, block in enumerate(model.backbone.layers):
        block_type = getattr(block, "block_type", None)

        def forward_profiled(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
            _layer_id=layer_id,
            _block_type=block_type,
        ):
            if not profiler.active:
                residual = hidden_states
                hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                if _block_type == "mamba":
                    hidden_states = self.mixer(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                    )
                elif _block_type == "attention":
                    hidden_states = self.mixer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_value=cache_params,
                        use_cache=cache_params is not None,
                        cache_position=cache_position,
                    )[0]
                elif _block_type == "mlp":
                    hidden_states = self.mixer(hidden_states)
                else:
                    raise ValueError(f"Invalid block_type: {_block_type}")
                return residual + hidden_states

            if _block_type == "attention":
                layer_start, _, _ = profiler.before_attention_layer(_layer_id)
            else:
                layer_start = profiler.begin_layer(_layer_id)

            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if _block_type == "mamba":
                hidden_states = self.mixer(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            elif _block_type == "attention":
                hidden_states = profiler.attention_mixer(
                    _layer_id,
                    self.mixer,
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=cache_params,
                    use_cache=cache_params is not None,
                    cache_position=cache_position,
                )[0]
            elif _block_type == "mlp":
                hidden_states = self.mixer(hidden_states)
            else:
                raise ValueError(f"Invalid block_type: {_block_type}")

            out = residual + hidden_states
            profiler.end_layer(_layer_id, layer_start)
            return out

        block.forward = MethodType(forward_profiled, block)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
