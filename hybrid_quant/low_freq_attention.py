from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType

import torch


@dataclass
class LowFreqAttentionLayerState:
    last_delta: torch.Tensor | None = None
    last_run_step: int = -1
    run_count: int = 0
    skip_count: int = 0
    hidden_delta_sum: float = 0.0
    hidden_delta_count: int = 0
    hidden_deltas: list[float] = field(default_factory=list)
    prev_hidden: torch.Tensor | None = None


@dataclass
class LowFreqAttentionController:
    attention_layers: list[int]
    mode: str = "periodic"
    interval: int = 1
    layer_intervals: dict[int, int] = field(default_factory=dict)
    run_layers: set[int] | None = None
    reuse_correction: bool = False
    correction_decay: float = 1.0
    replay_schedule: dict[int, set[int]] | None = None
    max_skip: int | None = None
    collect_hidden_delta: bool = False
    step_idx: int = 0
    states: dict[int, LowFreqAttentionLayerState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.attention_layers = list(self.attention_layers)
        self.states = {idx: LowFreqAttentionLayerState() for idx in self.attention_layers}
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")

    def reset(self) -> None:
        self.step_idx = 0
        for idx in self.attention_layers:
            self.states[idx] = LowFreqAttentionLayerState()

    def begin_decode_step(self, step_idx: int) -> None:
        self.step_idx = step_idx

    def layer_interval(self, layer_idx: int) -> int:
        return max(1, int(self.layer_intervals.get(layer_idx, self.interval)))

    def should_run(self, layer_idx: int) -> bool:
        state = self.states[layer_idx]
        if self.run_layers is not None and layer_idx not in self.run_layers:
            return False
        if self.replay_schedule is not None:
            scheduled = layer_idx in self.replay_schedule.get(self.step_idx, set())
            if self.max_skip is not None and state.last_run_step >= 0 and self.step_idx - state.last_run_step > self.max_skip:
                return True
            return scheduled
        interval = self.layer_interval(layer_idx)
        return self.step_idx % interval == 0

    def observe_hidden(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.collect_hidden_delta:
            return
        state = self.states[layer_idx]
        detached = hidden_states.detach()
        if state.prev_hidden is not None:
            diff = (detached.float() - state.prev_hidden.float()).pow(2).mean().sqrt()
            value = float(diff.cpu())
            state.hidden_delta_sum += value
            state.hidden_delta_count += 1
            state.hidden_deltas.append(value)
        state.prev_hidden = detached.clone()

    def record_run(self, layer_idx: int, delta: torch.Tensor) -> None:
        state = self.states[layer_idx]
        state.last_delta = delta.detach()
        state.last_run_step = self.step_idx
        state.run_count += 1

    def skipped_delta(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
        state = self.states[layer_idx]
        state.skip_count += 1
        if not self.reuse_correction or state.last_delta is None:
            return None
        age = max(0, self.step_idx - state.last_run_step)
        if self.correction_decay == 1.0:
            return state.last_delta
        return state.last_delta * (self.correction_decay**age)

    def summary_rows(self) -> list[dict]:
        rows = []
        for layer_idx in self.attention_layers:
            state = self.states[layer_idx]
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "runs": state.run_count,
                    "skips": state.skip_count,
                    "hidden_delta_mean": state.hidden_delta_sum / state.hidden_delta_count
                    if state.hidden_delta_count
                    else float("nan"),
                }
            )
        return rows


def patch_low_freq_attention_blocks(model, controller: LowFreqAttentionController) -> None:
    for layer_idx, block in enumerate(model.backbone.layers):
        if getattr(block, "_low_freq_attention_patched", False):
            block._low_freq_attention_controller = controller
            continue

        def forward_low_freq_attention(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            ctrl: LowFreqAttentionController = self._low_freq_attention_controller
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "mamba":
                delta = self.mixer(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
                return residual + delta
            if self.block_type == "attention":
                layer_id = self.layer_idx
                if hidden_states.shape[1] != 1:
                    delta = self.mixer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_value=cache_params,
                        use_cache=cache_params is not None,
                        cache_position=cache_position,
                    )[0]
                    return residual + delta
                ctrl.observe_hidden(layer_id, hidden_states)
                if ctrl.should_run(layer_id):
                    delta = self.mixer(
                        hidden_states,
                        attention_mask=attention_mask,
                        past_key_value=cache_params,
                        use_cache=cache_params is not None,
                        cache_position=cache_position,
                    )[0]
                    ctrl.record_run(layer_id, delta)
                    return residual + delta
                skipped = ctrl.skipped_delta(layer_id, hidden_states)
                if skipped is None:
                    return residual
                return residual + skipped.to(dtype=residual.dtype)
            if self.block_type == "mlp":
                return residual + self.mixer(hidden_states)
            raise ValueError(f"Invalid block_type: {self.block_type}")

        block.layer_idx = layer_idx
        block._low_freq_attention_controller = controller
        block.forward = MethodType(forward_low_freq_attention, block)
        block._low_freq_attention_patched = True


def build_periodic_schedule(attention_layers: list[int], decode_steps: int, interval: int) -> dict[int, set[int]]:
    return {step: set(attention_layers) if step % interval == 0 else set() for step in range(decode_steps)}


def build_layerwise_schedule(
    attention_layers: list[int],
    decode_steps: int,
    layer_intervals: dict[int, int],
    default_interval: int,
) -> dict[int, set[int]]:
    schedule: dict[int, set[int]] = {}
    for step in range(decode_steps):
        active = set()
        for layer_idx in attention_layers:
            interval = max(1, int(layer_intervals.get(layer_idx, default_interval)))
            if step % interval == 0:
                active.add(layer_idx)
        schedule[step] = active
    return schedule


def build_delta_quantile_schedule(
    stats: dict[int, list[float]],
    attention_layers: list[int],
    decode_steps: int,
    keep_fraction: float,
    max_skip: int,
) -> dict[int, set[int]]:
    values = [value for layer_values in stats.values() for value in layer_values]
    if not values:
        return build_periodic_schedule(attention_layers, decode_steps, max(1, max_skip))
    ordered = sorted(values)
    threshold_index = int(max(0, min(len(ordered) - 1, round((1.0 - keep_fraction) * (len(ordered) - 1)))))
    threshold = ordered[threshold_index]
    schedule: dict[int, set[int]] = {}
    last_run = {layer_idx: -1 for layer_idx in attention_layers}
    for step in range(decode_steps):
        active = set()
        for layer_idx in attention_layers:
            layer_values = stats.get(layer_idx, [])
            value = layer_values[step - 1] if 0 <= step - 1 < len(layer_values) else threshold
            if step == 0 or value >= threshold or step - last_run[layer_idx] >= max_skip:
                active.add(layer_idx)
                last_run[layer_idx] = step
        schedule[step] = active
    return schedule
