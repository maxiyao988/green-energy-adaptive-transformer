# DEBUG-ONLY COPY.
# NEVER USE THIS SCRIPT TO PRODUCE MAIN V2 FORMAL EVIDENCE.
# The frozen formal implementation is panel_baseline_swim.py.

"""Isolated SWiM numerical diagnosis script.

This debug-only copy preserves the frozen SWiM architecture and training
mechanics for diagnostic tracing, while intentionally removing all executable
formal-write paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.python.panel_baseline_swim as formal

BATCH_SIZE = formal.BATCH_SIZE
D_MODEL = formal.D_MODEL
DIM_FEEDFORWARD = formal.DIM_FEEDFORWARD
DROPOUT = formal.DROPOUT
EARLY_STOPPING_PATIENCE = formal.EARLY_STOPPING_PATIENCE
FIXED_MAX_LEN = formal.FIXED_MAX_LEN
FREQUENCY_SETTINGS = formal.FREQUENCY_SETTINGS
HORIZONS = formal.HORIZONS
LEARNING_RATE = formal.LEARNING_RATE
MAX_EPOCHS = formal.MAX_EPOCHS
NHEAD = formal.NHEAD
NUM_BLOCKS = formal.NUM_BLOCKS
SEED = formal.SEED
WINDOW_CONFIG = formal.WINDOW_CONFIG
PositionalEncoding = formal.PositionalEncoding
apply_normalizer = formal.apply_normalizer
assert_frozen_panel_checks = formal.assert_frozen_panel_checks
compute_pad_len = formal.compute_pad_len
count_trainable_parameters = formal.count_trainable_parameters
fit_train_only_normalizer = formal.fit_train_only_normalizer
format_elapsed = formal.format_elapsed
load_frozen_panel = formal.load_frozen_panel
set_seed = formal.set_seed
window_partition_1d = formal.window_partition_1d
window_reverse_1d = formal.window_reverse_1d
build_window_masks = formal.build_window_masks

DEBUG_NAN_MODE = False
DEBUG_NAN_CONTEXT = {
    "frequency": None,
    "horizon": None,
    "epoch": None,
    "batch_index": None,
    "device": None,
}


class TraceFailure(RuntimeError):
    def __init__(self, stage: str, **details):
        super().__init__(stage)
        self.stage = stage
        self.details = details


def _trace_epochs_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 20:
        raise argparse.ArgumentTypeError("--trace-epochs must be an integer in [1, 20].")
    return parsed


def _step_audit_batches_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 10:
        raise argparse.ArgumentTypeError("--step-audit-batches must be an integer in [1, 10].")
    return parsed


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite_stats(x) -> dict[str, float]:
    arr = _to_numpy(x)
    flat = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(flat)
    finite_fraction = float(finite_mask.mean()) if finite_mask.size else 1.0
    finite_vals = flat[finite_mask]
    if finite_vals.size == 0:
        return {
            "finite_fraction": finite_fraction,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "max_abs": float("nan"),
        }
    return {
        "finite_fraction": finite_fraction,
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "mean": float(np.mean(finite_vals)),
        "std": float(np.std(finite_vals)),
        "max_abs": float(np.max(np.abs(finite_vals))),
    }


def _print_summary(label: str, x) -> None:
    stats = _finite_stats(x)
    print(
        f"[TRACE_SUMMARY] name={label} min={stats['min']:.12e} max={stats['max']:.12e} "
        f"mean={stats['mean']:.12e} std={stats['std']:.12e} max_abs={stats['max_abs']:.12e} "
        f"finite_fraction={stats['finite_fraction']:.12f}"
    )


def _print_scalar(name: str, value) -> None:
    scalar = float(value) if value is not None else float("nan")
    print(f"[TRACE_VALUE] {name}={scalar:.12e}")


def _sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha256_model_state(model: nn.Module) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous()
        arr = tensor.numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _sha256_gradient_state(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            h.update(f"{name}:GRAD_NONE".encode("utf-8"))
            continue
        arr = grad.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _gradient_state_summary(model: nn.Module) -> dict[str, object]:
    grad_sq_sum = 0.0
    grad_max_abs = -1.0
    grad_max_name = None
    nonfinite_names = []
    all_finite = True
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            continue
        grad_cpu = grad.detach().float().cpu()
        finite_mask = torch.isfinite(grad_cpu)
        if not bool(finite_mask.all()):
            all_finite = False
            nonfinite_names.append(name)
        finite_grad = grad_cpu[finite_mask]
        if finite_grad.numel() > 0:
            grad_sq_sum += float((finite_grad ** 2).sum().item())
            local_max = float(finite_grad.abs().max().item())
            if local_max > grad_max_abs:
                grad_max_abs = local_max
                grad_max_name = name
    gradient_l2 = float(math.sqrt(grad_sq_sum)) if all_finite else float("nan")
    return {
        "gradient_sha256": _sha256_gradient_state(model),
        "gradient_l2": gradient_l2,
        "gradient_max_abs": float(grad_max_abs) if grad_max_name is not None else float("nan"),
        "gradient_max_parameter": grad_max_name,
        "all_gradients_finite": all_finite,
        "nonfinite_gradient_parameters": nonfinite_names,
    }


def _sha256_optimizer_state(optimizer: torch.optim.Optimizer, model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        state = optimizer.state.get(param, None)
        if state is None:
            h.update(f"{name}:STATE_NONE".encode("utf-8"))
            continue
        for state_key in ["step", "exp_avg", "exp_avg_sq"]:
            if state_key not in state:
                h.update(f"{name}:{state_key}:MISSING".encode("utf-8"))
                continue
            value = state[state_key]
            if torch.is_tensor(value):
                arr = value.detach().cpu().contiguous().numpy()
                h.update(name.encode("utf-8"))
                h.update(state_key.encode("utf-8"))
                h.update(str(arr.dtype).encode("utf-8"))
                h.update(str(tuple(arr.shape)).encode("utf-8"))
                h.update(arr.tobytes())
            else:
                h.update(name.encode("utf-8"))
                h.update(state_key.encode("utf-8"))
                h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def _sha256_named_tensors(named_tensors: list[tuple[str, Optional[torch.Tensor]]]) -> str:
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        h.update(name.encode("utf-8"))
        if tensor is None:
            h.update(b"NONE")
            continue
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _grad_finite_fraction_and_max_abs(grad: Optional[torch.Tensor]) -> tuple[float, float, bool]:
    if grad is None:
        return float("nan"), float("nan"), False
    grad_cpu = grad.detach().float().cpu()
    finite_mask = torch.isfinite(grad_cpu)
    finite_fraction = float(finite_mask.float().mean().item())
    if not bool(finite_mask.any()):
        return finite_fraction, float("nan"), False
    finite_vals = grad_cpu[finite_mask]
    max_abs = float(finite_vals.abs().max().item())
    return finite_fraction, max_abs, bool(finite_mask.all())


def _extract_mha_core_state_dict(attn: nn.MultiheadAttention) -> dict[str, torch.Tensor]:
    state = attn.state_dict()
    keys = ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]
    return {key: state[key].detach().cpu().clone() for key in keys}


def _run_mha_backward_variant(
    variant: str,
    device: torch.device,
    xw_cpu: torch.Tensor,
    attn_state_cpu: dict[str, torch.Tensor],
    attn_mask_cpu: Optional[torch.Tensor],
) -> torch.Tensor:
    set_seed(SEED)
    attn = nn.MultiheadAttention(D_MODEL, NHEAD, dropout=DROPOUT, batch_first=True).to(device)
    attn.load_state_dict(attn_state_cpu)
    attn.train()
    attn.zero_grad(set_to_none=True)

    xw = xw_cpu.to(device)
    mask = attn_mask_cpu.to(device) if attn_mask_cpu is not None else None
    attn_out, _ = attn(xw, xw, xw, attn_mask=mask, need_weights=False)
    all_output_finite = bool(torch.isfinite(attn_out).all())
    objective = attn_out.square().mean()

    out_cpu = attn_out.detach().cpu()
    out_stats = _finite_stats(out_cpu)
    print("[MHA_AUDIT_FORWARD]")
    print(f"variant={variant}")
    print(f"parameter_sha256={_sha256_named_tensors([(k, v) for k, v in sorted(attn_state_cpu.items())])}")
    print(f"output_sha256={_sha256_tensor(out_cpu)}")
    print(f"output_sum={float(_to_numpy(out_cpu).sum()):.12e}")
    print(f"output_mean={out_stats['mean']:.12e}")
    print(f"output_std={out_stats['std']:.12e}")
    print(f"output_max_abs={out_stats['max_abs']:.12e}")
    print(f"objective={float(objective.detach().cpu().item()):.12e}")
    print(f"all_output_finite={all_output_finite}")

    objective.backward()

    grad_named = [
        ("in_proj_weight", attn.in_proj_weight.grad),
        ("in_proj_bias", attn.in_proj_bias.grad),
        ("out_proj.weight", attn.out_proj.weight.grad),
        ("out_proj.bias", attn.out_proj.bias.grad),
    ]
    grad_sha = _sha256_named_tensors(grad_named)

    nonfinite_parameters = []
    print("[MHA_AUDIT_BACKWARD]")
    print(f"variant={variant}")
    for name, grad in grad_named:
        finite_fraction, max_abs, all_finite = _grad_finite_fraction_and_max_abs(grad)
        key = name.replace(".", "_")
        print(f"{key}_finite={all_finite}")
        print(f"{key}_finite_fraction={finite_fraction:.12f}")
        print(f"{key}_max_abs={max_abs:.12e}")
        if not all_finite:
            nonfinite_parameters.append(name)
    print(f"gradient_sha256={grad_sha}")
    if nonfinite_parameters:
        print("nonfinite_gradient_parameters=" + ",".join(nonfinite_parameters))

    return out_cpu


def _print_basic_fingerprint(prefix: str, tensor: torch.Tensor) -> None:
    stats = _finite_stats(tensor)
    print(prefix)
    print(f"output_sum={float(_to_numpy(tensor).sum()):.12e}")
    print(f"output_mean={stats['mean']:.12e}")
    print(f"output_std={stats['std']:.12e}")
    print(f"output_max_abs={stats['max_abs']:.12e}")
    print(f"output_sha256={_sha256_tensor(tensor)}")


def _print_step_audit_failure(batch_idx: int, stage: str) -> None:
    print("[STEP_AUDIT_FAILURE]")
    print(f"batch={batch_idx}")
    print(f"stage={stage}")


def _gradient_diagnostics(model: nn.Module) -> dict[str, object]:
    grad_sq_sum = 0.0
    grad_has_nonfinite = False
    grad_max_abs = -1.0
    grad_max_name = None
    missing = []
    nan_names = []
    inf_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            missing.append(name)
            continue
        grad_cpu = grad.detach().float().cpu()
        grad_finite = torch.isfinite(grad_cpu)
        if not bool(grad_finite.all()):
            grad_has_nonfinite = True
            if bool(torch.isnan(grad_cpu).any()):
                nan_names.append(name)
            if bool(torch.isinf(grad_cpu).any()):
                inf_names.append(name)
        finite_grad = grad_cpu[grad_finite]
        if finite_grad.numel() > 0:
            grad_sq_sum += float((finite_grad ** 2).sum().item())
            local_max = float(finite_grad.abs().max().item())
            if local_max > grad_max_abs:
                grad_max_abs = local_max
                grad_max_name = name
    raw_l2 = float(math.sqrt(grad_sq_sum)) if not grad_has_nonfinite else float("nan")
    return {
        "raw_l2": raw_l2,
        "max_abs": float(grad_max_abs) if grad_max_name is not None else float("nan"),
        "max_abs_name": grad_max_name,
        "missing": missing,
        "nan_names": nan_names,
        "inf_names": inf_names,
    }


def _parameter_diagnostics(model: nn.Module) -> dict[str, object]:
    max_abs = -1.0
    max_abs_name = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        local = float(param.detach().abs().max().item())
        if local > max_abs:
            max_abs = local
            max_abs_name = name
    return {
        "max_abs": float(max_abs) if max_abs_name is not None else float("nan"),
        "max_abs_name": max_abs_name,
    }


def _emit_trace_failure(stage: str, **details) -> None:
    print("[TRACE_FAILURE]")
    print(f"epoch={details.get('epoch')}")
    print(f"batch={details.get('batch')}")
    print(f"frequency={details.get('frequency')}")
    print(f"horizon={details.get('horizon')}d")
    print(f"device={details.get('device')}")
    print(f"stage={stage}")
    for key in ["parameter", "block_type", "tensor_name", "window_index", "query_row", "reason"]:
        if details.get(key) is not None:
            print(f"{key}={details[key]}")
    if details.get("summaries"):
        for label, tensor in details["summaries"]:
            _print_summary(label, tensor)
    if details.get("loss_value") is not None:
        _print_scalar("loss", details["loss_value"])
    grad_diag = details.get("grad_diag")
    if grad_diag is not None:
        _print_scalar("raw_grad_l2_norm", grad_diag["raw_l2"])
        _print_scalar("max_abs_gradient", grad_diag["max_abs"])
        if grad_diag.get("max_abs_name") is not None:
            print(f"max_abs_gradient_parameter={grad_diag['max_abs_name']}")
        if grad_diag.get("missing"):
            print("missing_gradients=" + ",".join(grad_diag["missing"]))
        if grad_diag.get("nan_names"):
            print("nan_gradients=" + ",".join(grad_diag["nan_names"]))
        if grad_diag.get("inf_names"):
            print("inf_gradients=" + ",".join(grad_diag["inf_names"]))
    param_diag = details.get("param_diag")
    if param_diag is not None:
        _print_scalar("max_abs_parameter", param_diag["max_abs"])
        if param_diag.get("max_abs_name") is not None:
            print(f"max_abs_parameter_name={param_diag['max_abs_name']}")


def _trace_fail(stage: str, **details) -> None:
    context = {
        "epoch": DEBUG_NAN_CONTEXT["epoch"],
        "batch": DEBUG_NAN_CONTEXT["batch_index"],
        "frequency": DEBUG_NAN_CONTEXT["frequency"],
        "horizon": DEBUG_NAN_CONTEXT["horizon"],
        "device": DEBUG_NAN_CONTEXT["device"],
    }
    context.update(details)
    _emit_trace_failure(stage, **context)
    raise TraceFailure(stage, **context)


class DebugSWiMWindowBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int, shift_size: int, dim_feedforward: int, dropout: float):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.window_size = window_size
        self.shift_size = shift_size
        self.block_type = "local" if shift_size == 0 else "shifted"
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _hard_zero(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.view(1, -1, 1).expand_as(x)
        return torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

    def _inspect_attention_mask(self, attn_mask_win: torch.Tensor, valid_mask: torch.Tensor) -> None:
        shifted_valid = torch.roll(valid_mask, shifts=-self.shift_size, dims=0) if self.shift_size > 0 else valid_mask
        query_valid = shifted_valid.view(attn_mask_win.shape[0], attn_mask_win.shape[1])
        is_zero = attn_mask_win == 0.0
        is_inf = torch.isinf(attn_mask_win)
        is_negative = attn_mask_win < 0
        is_neginf = is_inf & is_negative
        legal = is_zero | is_neginf
        if not bool(legal.all()):
            bad = torch.nonzero(~legal, as_tuple=False)
            win_idx, q_idx, _ = [int(v) for v in bad[0].tolist()]
            _trace_fail(
                "attention_mask",
                block_type=self.block_type,
                window_index=win_idx,
                query_row=q_idx,
                reason="mask contains values other than 0.0 or -inf",
                summaries=[("attn_mask_win", attn_mask_win)],
            )
        for win_idx in range(attn_mask_win.shape[0]):
            for q_idx in range(attn_mask_win.shape[1]):
                if bool(query_valid[win_idx, q_idx]):
                    row = attn_mask_win[win_idx, q_idx]
                    if not bool(torch.isfinite(row).any()):
                        _trace_fail(
                            "attention_mask",
                            block_type=self.block_type,
                            window_index=win_idx,
                            query_row=q_idx,
                            reason="real query row has no finite allowed key",
                            summaries=[("attn_mask_row", row), ("attn_mask_win", attn_mask_win)],
                        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, t_pad, _ = x.shape
        window_size = self.window_size
        n_win = t_pad // window_size

        if DEBUG_NAN_MODE and not bool(torch.isfinite(x).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="block_input", summaries=[("block_input", x)])

        attn_mask_win = build_window_masks(valid_mask, window_size, self.shift_size)
        if DEBUG_NAN_MODE:
            self._inspect_attention_mask(attn_mask_win, valid_mask)
        attn_mask = attn_mask_win.unsqueeze(0).expand(batch_size, n_win, window_size, window_size).reshape(batch_size * n_win, window_size, window_size)
        attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)

        residual = x
        x_shift = torch.roll(x, shifts=-self.shift_size, dims=1) if self.shift_size > 0 else x
        if DEBUG_NAN_MODE and not bool(torch.isfinite(x_shift).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="shifted_input", summaries=[("shifted_input", x_shift)])

        xw = window_partition_1d(x_shift, window_size)
        if DEBUG_NAN_MODE and not bool(torch.isfinite(xw).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="partitioned_windows", summaries=[("partitioned_windows", xw)])

        attn_out, _ = self.attn(xw, xw, xw, attn_mask=attn_mask, need_weights=False)
        if DEBUG_NAN_MODE and not bool(torch.isfinite(attn_out).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="attention_output", summaries=[("attention_output", attn_out)])

        attn_out = window_reverse_1d(attn_out, window_size, batch_size, t_pad)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=self.shift_size, dims=1)
        if DEBUG_NAN_MODE and not bool(torch.isfinite(attn_out).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="post_reverse_attention_output", summaries=[("post_reverse_attention_output", attn_out)])
        attn_out = self._hard_zero(attn_out, valid_mask)

        x = self.norm1(residual + self.dropout1(attn_out))
        if DEBUG_NAN_MODE and not bool(torch.isfinite(x).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="norm1_output", summaries=[("norm1_output", x)])
        x = self._hard_zero(x, valid_mask)

        ffn_out = self._hard_zero(self.ffn(x), valid_mask)
        if DEBUG_NAN_MODE and not bool(torch.isfinite(ffn_out).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="ffn_output", summaries=[("ffn_output", ffn_out)])
        x = self.norm2(x + self.dropout2(ffn_out))
        if DEBUG_NAN_MODE and not bool(torch.isfinite(x).all()):
            _trace_fail("forward", block_type=self.block_type, tensor_name="norm2_output", summaries=[("norm2_output", x)])
        x = self._hard_zero(x, valid_mask)
        return x


class DebugSWiMBranch(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        frequency_key: str,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if frequency_key not in WINDOW_CONFIG:
            raise ValueError(f"Unknown frequency_key: {frequency_key}")
        cfg = WINDOW_CONFIG[frequency_key]
        self.window_size = cfg["window_size"]
        self.shift_size = cfg["shift_size"]
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=FIXED_MAX_LEN, dropout=dropout)
        assert NUM_BLOCKS == 2, "SWiM v1 requires exactly two blocks: local + shifted-window."
        self.blocks = nn.ModuleList(
            [
                DebugSWiMWindowBlock(d_model, nhead, self.window_size, 0, dim_feedforward, dropout),
                DebugSWiMWindowBlock(d_model, nhead, self.window_size, self.shift_size, dim_feedforward, dropout),
            ]
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        batch_size, timesteps, _ = x.shape
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        pad_len = compute_pad_len(timesteps, self.window_size)
        t_pad = timesteps + pad_len
        valid_mask = torch.ones(t_pad, dtype=torch.bool, device=x.device)
        if pad_len > 0:
            zeros = torch.zeros(batch_size, pad_len, h.shape[-1], device=h.device, dtype=h.dtype)
            h = torch.cat([h, zeros], dim=1)
            valid_mask[timesteps:] = False
        for blk in self.blocks:
            h = blk(h, valid_mask)
        if return_sequence:
            return h, valid_mask
        return h.sum(dim=1) / float(timesteps)


class SingleFrequencySWiMRegressor(nn.Module):
    def __init__(self, feature_dim: int, frequency_key: str):
        super().__init__()
        self.branch = DebugSWiMBranch(feature_dim=feature_dim, frequency_key=frequency_key)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.branch(x)
        return self.head(rep).squeeze(-1)


class DualFrequencySWiMRegressor(nn.Module):
    def __init__(self, daily_feature_dim: int, weekly_feature_dim: int):
        super().__init__()
        self.daily_branch = DebugSWiMBranch(feature_dim=daily_feature_dim, frequency_key="daily")
        self.weekly_branch = DebugSWiMBranch(feature_dim=weekly_feature_dim, frequency_key="weekly")
        self.fusion = nn.Sequential(
            nn.Linear(2 * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        daily_vec = self.daily_branch(x_daily)
        weekly_vec = self.weekly_branch(x_weekly)
        fused = torch.cat([daily_vec, weekly_vec], dim=1)
        return self.fusion(fused).squeeze(-1)


def make_model(frequency: str, daily_feature_dim: int, weekly_feature_dim: int) -> nn.Module:
    if frequency == "daily_only":
        return SingleFrequencySWiMRegressor(feature_dim=daily_feature_dim, frequency_key="daily")
    if frequency == "weekly_only":
        return SingleFrequencySWiMRegressor(feature_dim=weekly_feature_dim, frequency_key="weekly")
    if frequency == "daily_weekly":
        return DualFrequencySWiMRegressor(daily_feature_dim=daily_feature_dim, weekly_feature_dim=weekly_feature_dim)
    raise ValueError(f"Unknown frequency setting: {frequency}")


def _trace_rng(epoch: int) -> None:
    print("[TRACE_RNG]")
    print(f"epoch={epoch}")
    print(formal._trace_rng_fingerprint_summary("torch_rng_state"))


def _model_parameter_fingerprint(model: nn.Module) -> None:
    total, param_sum, param_sq_sum = formal._model_parameter_fingerprint(model)
    print(f"[TRACE] model_total_parameters={total}")
    print(f"[TRACE] model_param_sum={param_sum:.12f}")
    print(f"[TRACE] model_param_sq_sum={param_sq_sum:.12f}")


def _first_batch_fingerprint(batch_idx: int, x: torch.Tensor, y: torch.Tensor) -> None:
    formal._trace_first_batch_fingerprint(batch_idx, x, y)


def _handle_input_failure(stage: str, frequency: str, horizon: int, batch_idx: int, x, y, device: torch.device) -> None:
    DEBUG_NAN_CONTEXT["frequency"] = frequency
    DEBUG_NAN_CONTEXT["horizon"] = horizon
    DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
    DEBUG_NAN_CONTEXT["device"] = device.type.upper()
    summaries = []
    if isinstance(x, tuple):
        for idx, xx in enumerate(x):
            summaries.append((f"input_{idx}", xx))
    else:
        summaries.append(("input", x))
    summaries.append(("target", y))
    _trace_fail(stage, summaries=summaries)


def _prepare_cpu_frequency_tensors(ds, frequency: str, horizon: int, stats: dict):
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"

    if frequency == "daily_only":
        x_all = apply_normalizer(ds, stats, "daily_only")
        x_train = torch.tensor(x_all[train_mask], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        return {
            "frequency": frequency,
            "x_train": x_train,
            "x_val": x_val,
            "y_train": y_train,
            "y_val": y_val,
            "feature_dim": x_all.shape[-1],
        }
    if frequency == "weekly_only":
        x_all = apply_normalizer(ds, stats, "weekly_only")
        x_train = torch.tensor(x_all[train_mask], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        return {
            "frequency": frequency,
            "x_train": x_train,
            "x_val": x_val,
            "y_train": y_train,
            "y_val": y_val,
            "feature_dim": x_all.shape[-1],
        }
    if frequency == "daily_weekly":
        x_daily_all, x_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        x_d_train = torch.tensor(x_daily_all[train_mask], dtype=torch.float32)
        x_d_val = torch.tensor(x_daily_all[val_mask], dtype=torch.float32)
        x_w_train = torch.tensor(x_weekly_all[train_mask], dtype=torch.float32)
        x_w_val = torch.tensor(x_weekly_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        return {
            "frequency": frequency,
            "x_d_train": x_d_train,
            "x_d_val": x_d_val,
            "x_w_train": x_w_train,
            "x_w_val": x_w_val,
            "y_train": y_train,
            "y_val": y_val,
            "daily_feature_dim": x_daily_all.shape[-1],
            "weekly_feature_dim": x_weekly_all.shape[-1],
        }
    raise ValueError(f"Unknown frequency setting: {frequency}")


def _make_formal_model_for_frequency(prepared: dict, frequency: str, device: torch.device) -> nn.Module:
    if frequency == "daily_only":
        return formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    if frequency == "weekly_only":
        return formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="weekly").to(device)
    if frequency == "daily_weekly":
        return formal.DualFrequencySWiMRegressor(
            daily_feature_dim=prepared["daily_feature_dim"],
            weekly_feature_dim=prepared["weekly_feature_dim"],
        ).to(device)
    raise ValueError(f"Unknown frequency setting: {frequency}")


def _make_train_loader_from_prepared(prepared: dict, frequency: str) -> DataLoader:
    if frequency == "daily_only" or frequency == "weekly_only":
        return DataLoader(TensorDataset(prepared["x_train"], prepared["y_train"]), batch_size=BATCH_SIZE, shuffle=True)
    return DataLoader(TensorDataset(prepared["x_d_train"], prepared["x_w_train"], prepared["y_train"]), batch_size=BATCH_SIZE, shuffle=True)


def _print_loader_batch_fingerprint(batch_idx: int, batch, frequency: str) -> None:
    print("[RNG_AUDIT_DATALOADER]")
    print(f"batch={batch_idx}")
    if frequency == "daily_weekly":
        x_d_batch, x_w_batch, y_batch = batch
        x_d = _to_numpy(x_d_batch)
        x_w = _to_numpy(x_w_batch)
        y = _to_numpy(y_batch)
        print(f"daily_x_sum={float(x_d.sum()):.12e}")
        print(f"daily_x_mean={float(x_d.mean()):.12e}")
        print(f"daily_x_std={float(x_d.std()):.12e}")
        print(f"daily_x_max_abs={float(np.max(np.abs(x_d))):.12e}")
        print(f"weekly_x_sum={float(x_w.sum()):.12e}")
        print(f"weekly_x_mean={float(x_w.mean()):.12e}")
        print(f"weekly_x_std={float(x_w.std()):.12e}")
        print(f"weekly_x_max_abs={float(np.max(np.abs(x_w))):.12e}")
    else:
        x_batch, y_batch = batch
        x = _to_numpy(x_batch)
        y = _to_numpy(y_batch)
        print(f"x_sum={float(x.sum()):.12e}")
        print(f"x_mean={float(x.mean()):.12e}")
        print(f"x_std={float(x.std()):.12e}")
        print(f"x_max_abs={float(np.max(np.abs(x))):.12e}")
    print(f"y_sum={float(y.sum()):.12e}")
    print(f"y_mean={float(y.mean()):.12e}")
    print(f"y_std={float(y.std()):.12e}")
    print(f"y_first5={y[:5].tolist()}")


def _get_fixed_forward_inputs(prepared: dict, frequency: str):
    if frequency == "daily_weekly":
        return (
            prepared["x_d_train"][:BATCH_SIZE],
            prepared["x_w_train"][:BATCH_SIZE],
            prepared["y_train"][:BATCH_SIZE],
        )
    return prepared["x_train"][:BATCH_SIZE], prepared["y_train"][:BATCH_SIZE]


def _print_fixed_forward_input_fingerprint(fixed_inputs, frequency: str) -> None:
    print("[RNG_AUDIT_FIXED_FORWARD_INPUT]")
    if frequency == "daily_weekly":
        x_d, x_w, _ = fixed_inputs
        x_d_arr = _to_numpy(x_d)
        x_w_arr = _to_numpy(x_w)
        print(f"daily_x_sum={float(x_d_arr.sum()):.12e}")
        print(f"daily_x_mean={float(x_d_arr.mean()):.12e}")
        print(f"daily_x_std={float(x_d_arr.std()):.12e}")
        print(f"daily_x_max_abs={float(np.max(np.abs(x_d_arr))):.12e}")
        print(f"daily_x_sha256={_sha256_tensor(x_d)}")
        print(f"weekly_x_sum={float(x_w_arr.sum()):.12e}")
        print(f"weekly_x_mean={float(x_w_arr.mean()):.12e}")
        print(f"weekly_x_std={float(x_w_arr.std()):.12e}")
        print(f"weekly_x_max_abs={float(np.max(np.abs(x_w_arr))):.12e}")
        print(f"weekly_x_sha256={_sha256_tensor(x_w)}")
    else:
        x, _ = fixed_inputs
        x_arr = _to_numpy(x)
        print(f"x_sum={float(x_arr.sum()):.12e}")
        print(f"x_mean={float(x_arr.mean()):.12e}")
        print(f"x_std={float(x_arr.std()):.12e}")
        print(f"x_max_abs={float(np.max(np.abs(x_arr))):.12e}")
        print(f"x_sha256={_sha256_tensor(x)}")


def _deterministic_dropout_input() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=32 * 90 * 64, dtype=torch.float32).reshape(32, 90, 64)


def _print_step_audit_input(batch_idx: int, batch, frequency: str) -> None:
    print("[STEP_AUDIT_INPUT]")
    print(f"batch={batch_idx}")
    if frequency == "daily_weekly":
        x_d_batch, x_w_batch, y_batch = batch
        x_d_arr = _to_numpy(x_d_batch)
        x_w_arr = _to_numpy(x_w_batch)
        y_arr = _to_numpy(y_batch)
        print(f"daily_input_sha256={_sha256_tensor(x_d_batch)}")
        print(f"daily_input_sum={float(x_d_arr.sum()):.12e}")
        print(f"daily_input_mean={float(x_d_arr.mean()):.12e}")
        print(f"daily_input_std={float(x_d_arr.std()):.12e}")
        print(f"daily_input_max_abs={float(np.max(np.abs(x_d_arr))):.12e}")
        print(f"weekly_input_sha256={_sha256_tensor(x_w_batch)}")
        print(f"weekly_input_sum={float(x_w_arr.sum()):.12e}")
        print(f"weekly_input_mean={float(x_w_arr.mean()):.12e}")
        print(f"weekly_input_std={float(x_w_arr.std()):.12e}")
        print(f"weekly_input_max_abs={float(np.max(np.abs(x_w_arr))):.12e}")
    else:
        x_batch, y_batch = batch
        x_arr = _to_numpy(x_batch)
        y_arr = _to_numpy(y_batch)
        print(f"input_sha256={_sha256_tensor(x_batch)}")
        print(f"input_sum={float(x_arr.sum()):.12e}")
        print(f"input_mean={float(x_arr.mean()):.12e}")
        print(f"input_std={float(x_arr.std()):.12e}")
        print(f"input_max_abs={float(np.max(np.abs(x_arr))):.12e}")
    print(f"target_sha256={_sha256_tensor(y_batch)}")
    print(f"target_sum={float(y_arr.sum()):.12e}")
    print(f"target_mean={float(y_arr.mean()):.12e}")
    print(f"target_std={float(y_arr.std()):.12e}")
    print(f"target_max_abs={float(np.max(np.abs(y_arr))):.12e}")


# Step audit intentionally fingerprints tensors/gradients/optimizer state by
# copying them to CPU. On MPS this can synchronize execution and may itself
# change numerical behavior. That is intentional: this audit tests whether the
# instrumented state transitions are reproducible across independent launches.
def run_step_audit(ds, frequency: str, horizon: int, stats: dict, step_audit_batches: int) -> None:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    model = _make_formal_model_for_frequency(prepared, frequency, device)
    train_loader = _make_train_loader_from_prepared(prepared, frequency)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= step_audit_batches:
            break
        optimizer.zero_grad()
        _print_step_audit_input(batch_idx, batch, frequency)

        print("[STEP_AUDIT_PRE_PARAMS]")
        print(f"batch={batch_idx}")
        print(f"parameter_sha256={_sha256_model_state(model)}")

        if frequency == "daily_weekly":
            x_d_batch, x_w_batch, y_batch = batch
            logits = model(x_d_batch.to(device), x_w_batch.to(device))
            target = y_batch.to(device)
        else:
            x_batch, y_batch = batch
            logits = model(x_batch.to(device))
            target = y_batch.to(device)

        print("[STEP_AUDIT_FORWARD]")
        print(f"batch={batch_idx}")
        print(f"output_sum={float(_to_numpy(logits.detach().cpu()).sum()):.12e}")
        print(f"output_mean={_finite_stats(logits.detach().cpu())['mean']:.12e}")
        print(f"output_std={_finite_stats(logits.detach().cpu())['std']:.12e}")
        print(f"output_max_abs={_finite_stats(logits.detach().cpu())['max_abs']:.12e}")
        print(f"output_sha256={_sha256_tensor(logits.detach().cpu())}")

        loss = criterion(logits, target)
        loss_finite = bool(torch.isfinite(loss))
        print("[STEP_AUDIT_LOSS]")
        print(f"batch={batch_idx}")
        print(f"loss={loss.item()}")
        print(f"loss_finite={loss_finite}")
        if not loss_finite:
            _print_step_audit_failure(batch_idx, "loss")
            return

        loss.backward()
        raw_grad = _gradient_state_summary(model)
        print("[STEP_AUDIT_RAW_GRAD]")
        print(f"batch={batch_idx}")
        print(f"gradient_sha256={raw_grad['gradient_sha256']}")
        print(f"gradient_l2={raw_grad['gradient_l2']}")
        print(f"gradient_max_abs={raw_grad['gradient_max_abs']}")
        print(f"gradient_max_parameter={raw_grad['gradient_max_parameter']}")
        print(f"all_gradients_finite={raw_grad['all_gradients_finite']}")
        if raw_grad["nonfinite_gradient_parameters"]:
            print("nonfinite_gradient_parameters=" + ",".join(raw_grad["nonfinite_gradient_parameters"]))
        if not raw_grad["all_gradients_finite"]:
            _print_step_audit_failure(batch_idx, "raw_gradient")
            return

        clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        clipped_grad = _gradient_state_summary(model)
        clip_return_finite = bool(torch.isfinite(clip_return))
        print("[STEP_AUDIT_CLIPPED_GRAD]")
        print(f"batch={batch_idx}")
        print(f"clip_return={float(clip_return.item())}")
        print(f"gradient_sha256={clipped_grad['gradient_sha256']}")
        print(f"gradient_l2={clipped_grad['gradient_l2']}")
        print(f"gradient_max_abs={clipped_grad['gradient_max_abs']}")
        print(f"gradient_max_parameter={clipped_grad['gradient_max_parameter']}")
        print(f"all_gradients_finite={clipped_grad['all_gradients_finite']}")
        if clipped_grad["nonfinite_gradient_parameters"]:
            print("nonfinite_gradient_parameters=" + ",".join(clipped_grad["nonfinite_gradient_parameters"]))
        if (not clip_return_finite) or (not clipped_grad["all_gradients_finite"]):
            _print_step_audit_failure(batch_idx, "clipped_gradient")
            return

        optimizer.step()
        param_diag = _parameter_diagnostics(model)
        all_parameters_finite = True
        for _, param in model.named_parameters():
            if param.requires_grad and not bool(torch.isfinite(param).all()):
                all_parameters_finite = False
                break
        print("[STEP_AUDIT_POST_STEP_PARAMS]")
        print(f"batch={batch_idx}")
        print(f"parameter_sha256={_sha256_model_state(model)}")
        print(f"parameter_max_abs={param_diag['max_abs']}")
        print(f"all_parameters_finite={all_parameters_finite}")
        print(f"optimizer_state_sha256={_sha256_optimizer_state(optimizer, model)}")
        if not all_parameters_finite:
            _print_step_audit_failure(batch_idx, "post_step_parameters")
            return

    print("[STEP_AUDIT_RESULT]")
    print("ALL_REQUESTED_BATCHES_FINITE")
    print(f"batches={step_audit_batches}")


def run_rng_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    train_loader = _make_train_loader_from_prepared(prepared, frequency)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:
            break
        _print_loader_batch_fingerprint(batch_idx, batch, frequency)

    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    model = _make_formal_model_for_frequency(prepared, frequency, device)
    param_total, param_sum, param_sq_sum = formal._model_parameter_fingerprint(model)
    param_diag = _parameter_diagnostics(model)
    print("[RNG_AUDIT_MODEL_INIT]")
    print(f"device={device.type.upper()}")
    print(f"parameter_count={param_total}")
    print(f"parameter_sum={param_sum:.12e}")
    print(f"parameter_sq_sum={param_sq_sum:.12e}")
    print(f"parameter_max_abs={param_diag['max_abs']:.12e}")
    print(f"parameter_sha256={_sha256_model_state(model)}")

    set_seed(SEED)
    cpu_dropout = nn.Dropout(p=DROPOUT)
    cpu_dropout.train()
    cpu_dropout_out = cpu_dropout(_deterministic_dropout_input())
    _print_basic_fingerprint("[RNG_AUDIT_CPU_DROPOUT]", cpu_dropout_out)

    print("[RNG_AUDIT_MPS_DROPOUT]")
    if torch.backends.mps.is_available():
        set_seed(SEED)
        mps_dropout = nn.Dropout(p=DROPOUT).to("mps")
        mps_dropout.train()
        mps_input = _deterministic_dropout_input().to("mps")
        mps_dropout_out = mps_dropout(mps_input).detach().cpu()
        print(f"output_sum={float(_to_numpy(mps_dropout_out).sum()):.12e}")
        print(f"output_mean={_finite_stats(mps_dropout_out)['mean']:.12e}")
        print(f"output_std={_finite_stats(mps_dropout_out)['std']:.12e}")
        print(f"output_max_abs={_finite_stats(mps_dropout_out)['max_abs']:.12e}")
        print(f"output_sha256={_sha256_tensor(mps_dropout_out)}")
    else:
        print("status=MPS_UNAVAILABLE")

    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    model = _make_formal_model_for_frequency(prepared, frequency, device)
    fixed_inputs = _get_fixed_forward_inputs(prepared, frequency)
    _print_fixed_forward_input_fingerprint(fixed_inputs, frequency)
    model.eval()
    with torch.no_grad():
        if frequency == "daily_weekly":
            x_d_batch, x_w_batch, _ = fixed_inputs
            eval_out = model(x_d_batch.to(device), x_w_batch.to(device)).detach().cpu()
        else:
            x_batch, _ = fixed_inputs
            eval_out = model(x_batch.to(device)).detach().cpu()
    print("[RNG_AUDIT_FORMAL_EVAL_FORWARD]")
    print(f"device={device.type.upper()}")
    print(f"output_sum={float(_to_numpy(eval_out).sum()):.12e}")
    print(f"output_mean={_finite_stats(eval_out)['mean']:.12e}")
    print(f"output_std={_finite_stats(eval_out)['std']:.12e}")
    print(f"output_max_abs={_finite_stats(eval_out)['max_abs']:.12e}")
    print(f"output_sha256={_sha256_tensor(eval_out)}")

    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    model = _make_formal_model_for_frequency(prepared, frequency, device)
    fixed_inputs = _get_fixed_forward_inputs(prepared, frequency)
    _print_fixed_forward_input_fingerprint(fixed_inputs, frequency)
    model.train()
    with torch.no_grad():
        if frequency == "daily_weekly":
            x_d_batch, x_w_batch, _ = fixed_inputs
            train_out = model(x_d_batch.to(device), x_w_batch.to(device)).detach().cpu()
        else:
            x_batch, _ = fixed_inputs
            train_out = model(x_batch.to(device)).detach().cpu()
    print("[RNG_AUDIT_FORMAL_TRAIN_FORWARD]")
    print(f"device={device.type.upper()}")
    print(f"output_sum={float(_to_numpy(train_out).sum()):.12e}")
    print(f"output_mean={_finite_stats(train_out)['mean']:.12e}")
    print(f"output_std={_finite_stats(train_out)['std']:.12e}")
    print(f"output_max_abs={_finite_stats(train_out)['max_abs']:.12e}")
    print(f"output_sha256={_sha256_tensor(train_out)}")

    print("============================================================")
    print("RNG AUDIT COMPLETE")
    print(f"seed={SEED}")
    print(f"frequency={frequency}")
    print(f"horizon={horizon}d")
    print(f"device={device.type.upper()}")
    print("============================================================")


def run_mha_backward_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[MHA_AUDIT_ERROR]")
        print("This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    formal_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    formal_model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(formal_device)
    formal_model.train()

    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")
    x_batch_cpu, _ = next(iter(train_loader))
    input_sha = _sha256_tensor(x_batch_cpu)

    x_batch = x_batch_cpu.to(formal_device)
    branch = formal_model.branch
    h = branch.input_proj(x_batch)
    h = branch.positional_encoding(h)

    batch_size, timesteps, _ = x_batch.shape
    window_size = branch.window_size
    pad_len = compute_pad_len(timesteps, window_size)
    t_pad = timesteps + pad_len
    valid_mask = torch.ones(t_pad, dtype=torch.bool, device=formal_device)
    if pad_len > 0:
        zeros = torch.zeros(batch_size, pad_len, h.shape[-1], device=h.device, dtype=h.dtype)
        h = torch.cat([h, zeros], dim=1)
        valid_mask[timesteps:] = False

    xw = window_partition_1d(h, window_size)
    xw_cpu = xw.detach().cpu()
    xw_stats = _finite_stats(xw_cpu)

    print("[MHA_AUDIT_PRE_MHA]")
    print(f"input_batch_sha256={input_sha}")
    print(f"xw_sha256={_sha256_tensor(xw_cpu)}")
    print(f"xw_sum={float(_to_numpy(xw_cpu).sum()):.12e}")
    print(f"xw_mean={xw_stats['mean']:.12e}")
    print(f"xw_std={xw_stats['std']:.12e}")
    print(f"xw_max_abs={xw_stats['max_abs']:.12e}")

    local_attn = formal_model.branch.blocks[0].attn
    attn_state_cpu = _extract_mha_core_state_dict(local_attn)

    mask_win = build_window_masks(valid_mask, window_size, shift_size=0)
    mask_stats = _finite_stats(mask_win)
    finite_count = int(torch.isfinite(mask_win).sum().item())
    zero_count = int((mask_win == 0.0).sum().item())
    neginf_count = int((torch.isinf(mask_win) & (mask_win < 0)).sum().item())
    print("[MHA_AUDIT_MASK]")
    print(f"shape={tuple(mask_win.shape)}")
    print(f"finite_count={finite_count}")
    print(f"zero_count={zero_count}")
    print(f"neginf_count={neginf_count}")
    print(f"min={mask_stats['min']:.12e}")
    print(f"max={mask_stats['max']:.12e}")

    n_win = t_pad // window_size
    attn_mask = mask_win.unsqueeze(0).expand(batch_size, n_win, window_size, window_size).reshape(batch_size * n_win, window_size, window_size)
    attn_mask = attn_mask.repeat_interleave(local_attn.num_heads, dim=0)
    attn_mask_cpu = attn_mask.detach().cpu()

    # Copying MPS tensors to CPU for hashing/statistics can synchronize execution
    # and perturb timing. This audit accepts that limitation for operator-level
    # backward isolation and intentionally avoids explicit mps.synchronize calls.
    outputs: dict[str, torch.Tensor] = {}
    if torch.backends.mps.is_available():
        outputs["MPS_MASK"] = _run_mha_backward_variant("A_MPS_MASK", torch.device("mps"), xw_cpu, attn_state_cpu, attn_mask_cpu)
        outputs["MPS_NOMASK"] = _run_mha_backward_variant("B_MPS_NOMASK", torch.device("mps"), xw_cpu, attn_state_cpu, None)
    else:
        print("[MHA_AUDIT_VARIANT]")
        print("variant=A_MPS_MASK")
        print("status=MPS_UNAVAILABLE")
        print("[MHA_AUDIT_VARIANT]")
        print("variant=B_MPS_NOMASK")
        print("status=MPS_UNAVAILABLE")

    outputs["CPU_MASK"] = _run_mha_backward_variant("C_CPU_MASK", torch.device("cpu"), xw_cpu, attn_state_cpu, attn_mask_cpu)
    outputs["CPU_NOMASK"] = _run_mha_backward_variant("D_CPU_NOMASK", torch.device("cpu"), xw_cpu, attn_state_cpu, None)

    if "MPS_MASK" in outputs and "MPS_NOMASK" in outputs:
        mps_diff = float((outputs["MPS_MASK"] - outputs["MPS_NOMASK"]).abs().max().item())
        mps_allclose = bool(torch.allclose(outputs["MPS_MASK"], outputs["MPS_NOMASK"], rtol=1e-6, atol=1e-7))
        print("[MHA_AUDIT_MASK_COMPARISON]")
        print("device=MPS")
        print(f"masked_vs_unmasked_max_abs_diff={mps_diff:.12e}")
        print(f"masked_vs_unmasked_allclose={mps_allclose}")

    cpu_diff = float((outputs["CPU_MASK"] - outputs["CPU_NOMASK"]).abs().max().item())
    cpu_allclose = bool(torch.allclose(outputs["CPU_MASK"], outputs["CPU_NOMASK"], rtol=1e-6, atol=1e-7))
    print("[MHA_AUDIT_MASK_COMPARISON]")
    print("device=CPU")
    print(f"masked_vs_unmasked_max_abs_diff={cpu_diff:.12e}")
    print(f"masked_vs_unmasked_allclose={cpu_allclose}")


def run_backward_boundary_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[BOUNDARY_AUDIT_FAILURE]")
        print("stage=configuration")
        print("message=This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    x_batch_cpu, y_batch_cpu = next(iter(train_loader))
    input_sha = _sha256_tensor(x_batch_cpu)
    x_batch = x_batch_cpu.to(device)
    y_batch = y_batch_cpu.to(device)

    retained: dict[str, torch.Tensor] = {}
    handles = []

    def _retain(name: str, tensor: torch.Tensor) -> None:
        if name in retained:
            return
        retained[name] = tensor
        if tensor.requires_grad:
            tensor.retain_grad()

    def _hook_input(name: str):
        def _inner(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                _retain(name, inputs[0])
            return None

        return _inner

    def _hook_output(name: str):
        def _inner(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                _retain(name, output)
            elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                _retain(name, output[0])
            return None

        return _inner

    branch = model.branch
    block0 = branch.blocks[0]
    block1 = branch.blocks[1]

    handles.append(branch.positional_encoding.register_forward_hook(_hook_output("positional_encoding_output")))
    handles.append(block0.register_forward_pre_hook(_hook_input("block0_input")))
    handles.append(block0.register_forward_hook(_hook_output("block0_output")))
    handles.append(block0.attn.register_forward_pre_hook(_hook_input("block0_mha_input")))
    handles.append(block0.attn.register_forward_hook(_hook_output("block0_mha_output")))
    handles.append(block1.register_forward_pre_hook(_hook_input("block1_input")))
    handles.append(block1.register_forward_hook(_hook_output("block1_output")))
    handles.append(block1.attn.register_forward_pre_hook(_hook_input("block1_mha_input")))
    handles.append(block1.attn.register_forward_hook(_hook_output("block1_mha_output")))
    handles.append(model.head.register_forward_pre_hook(_hook_input("head_input")))

    try:
        logits = model(x_batch)
        _retain("logits", logits)
        loss = criterion(logits, y_batch)
        print("[BOUNDARY_AUDIT_FORWARD]")
        print(f"input_sha256={input_sha}")
        print("status=FORWARD_COMPLETED_DIAGNOSTICS_DEFERRED")

        try:
            loss.backward()
        except Exception as exc:  # pragma: no cover - diagnostic pathway
            print("[BOUNDARY_AUDIT_FAILURE]")
            print("stage=backward_exception")
            print(f"exception={repr(exc)}")
            return

        output_finite = bool(torch.isfinite(logits).all())
        loss_finite = bool(torch.isfinite(loss))
        print("[BOUNDARY_AUDIT_FORWARD_POST_BACKWARD]")
        print(f"output_finite={output_finite}")
        print(f"loss={float(loss.detach().cpu().item()):.12e}")
        print(f"loss_finite={loss_finite}")
    finally:
        for handle in handles:
            handle.remove()

    boundary_names = [
        "positional_encoding_output",
        "block0_input",
        "block0_mha_input",
        "block0_mha_output",
        "block0_output",
        "block1_input",
        "block1_mha_input",
        "block1_mha_output",
        "block1_output",
        "head_input",
        "logits",
    ]
    for name in boundary_names:
        tensor = retained.get(name)
        print("[BOUNDARY_AUDIT_GRAD]")
        print(f"name={name}")
        if tensor is None:
            print("shape=MISSING")
            print("forward_tensor_finite=False")
            print("grad_present=False")
            print("grad_finite=False")
            print("grad_finite_fraction=nan")
            print("grad_max_abs=nan")
            print("grad_sha256=NONE")
            continue

        print(f"shape={tuple(tensor.shape)}")
        forward_finite = bool(torch.isfinite(tensor).all())
        grad = tensor.grad
        grad_present = grad is not None
        print(f"forward_tensor_finite={forward_finite}")
        print(f"grad_present={grad_present}")
        if not grad_present:
            print("grad_finite=False")
            print("grad_finite_fraction=nan")
            print("grad_max_abs=nan")
            print("grad_sha256=NONE")
            continue

        grad_cpu = grad.detach().cpu()
        finite_mask = torch.isfinite(grad_cpu)
        grad_finite = bool(finite_mask.all())
        grad_finite_fraction = float(finite_mask.float().mean().item())
        if bool(finite_mask.any()):
            grad_max_abs = float(grad_cpu[finite_mask].abs().max().item())
        else:
            grad_max_abs = float("nan")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        print(f"grad_sha256={_sha256_tensor(grad_cpu)}")

    print("[BOUNDARY_AUDIT_PARAMETER_GRADS]")
    parameter_names = [
        "branch.blocks.0.attn.in_proj_weight",
        "branch.blocks.0.attn.in_proj_bias",
        "branch.blocks.0.attn.out_proj.weight",
        "branch.blocks.0.attn.out_proj.bias",
        "branch.blocks.1.attn.in_proj_weight",
        "branch.blocks.1.attn.in_proj_bias",
        "branch.blocks.1.attn.out_proj.weight",
        "branch.blocks.1.attn.out_proj.bias",
    ]
    by_name = dict(model.named_parameters())
    nonfinite_parameter_gradients = []
    for name in parameter_names:
        param = by_name.get(name)
        grad = None if param is None else param.grad
        grad_finite_fraction, grad_max_abs, grad_finite = _grad_finite_fraction_and_max_abs(grad)
        print(f"name={name}")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        if (param is None) or (not grad_finite):
            nonfinite_parameter_gradients.append(name)
    print(
        "nonfinite_parameter_gradients="
        + ("NONE" if not nonfinite_parameter_gradients else ",".join(nonfinite_parameter_gradients))
    )


def run_bare_backward_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[BARE_BACKWARD_FAILURE]")
        print("stage=configuration")
        print("message=This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    x_batch_cpu, y_batch_cpu = next(iter(train_loader))
    print("[BARE_BACKWARD_INPUT]")
    print(f"input_sha256={_sha256_tensor(x_batch_cpu)}")
    print(f"target_sha256={_sha256_tensor(y_batch_cpu)}")

    x_batch = x_batch_cpu.to(device)
    y_batch = y_batch_cpu.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    try:
        loss.backward()
    except Exception as exc:  # pragma: no cover - diagnostic pathway
        print("[BARE_BACKWARD_FAILURE]")
        print("stage=backward_exception")
        print(f"exception={repr(exc)}")
        return

    output_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss))
    print("[BARE_BACKWARD_POST]")
    print(f"output_finite={output_finite}")
    print(f"loss={float(loss.detach().cpu().item()):.12e}")
    print(f"loss_finite={loss_finite}")

    print("[BARE_BACKWARD_PARAMETER_GRADS]")
    nonfinite_parameter_gradients = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        grad_present = grad is not None
        grad_finite_fraction, grad_max_abs, grad_finite = _grad_finite_fraction_and_max_abs(grad)
        print(f"name={name}")
        print(f"grad_present={grad_present}")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        if (not grad_present) or (not grad_finite):
            nonfinite_parameter_gradients.append(name)

    print(
        "nonfinite_parameter_gradients="
        + ("NONE" if not nonfinite_parameter_gradients else ",".join(nonfinite_parameter_gradients))
    )
    print(f"gradient_sha256={_sha256_gradient_state(model)}")


def run_loss_item_backward_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[LOSS_ITEM_AUDIT_FAILURE]")
        print("stage=configuration")
        print("message=This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    x_batch_cpu, y_batch_cpu = next(iter(train_loader))
    print("[LOSS_ITEM_AUDIT_INPUT]")
    print(f"input_sha256={_sha256_tensor(x_batch_cpu)}")
    print(f"target_sha256={_sha256_tensor(y_batch_cpu)}")

    x_batch = x_batch_cpu.to(device)
    y_batch = y_batch_cpu.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    pre_backward_loss_value = loss.item()
    print("[LOSS_ITEM_AUDIT_PRE_BACKWARD]")
    print(f"loss_item={pre_backward_loss_value:.12e}")

    try:
        loss.backward()
    except Exception as exc:  # pragma: no cover - diagnostic pathway
        print("[LOSS_ITEM_AUDIT_FAILURE]")
        print("stage=backward_exception")
        print(f"exception={repr(exc)}")
        return

    output_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss))
    print("[LOSS_ITEM_AUDIT_POST]")
    print(f"output_finite={output_finite}")
    print(f"loss={float(loss.detach().cpu().item()):.12e}")
    print(f"loss_finite={loss_finite}")

    print("[LOSS_ITEM_AUDIT_PARAMETER_GRADS]")
    nonfinite_parameter_gradients = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        grad_present = grad is not None
        grad_finite_fraction, grad_max_abs, grad_finite = _grad_finite_fraction_and_max_abs(grad)
        print(f"name={name}")
        print(f"grad_present={grad_present}")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        if (not grad_present) or (not grad_finite):
            nonfinite_parameter_gradients.append(name)

    print(
        "nonfinite_parameter_gradients="
        + ("NONE" if not nonfinite_parameter_gradients else ",".join(nonfinite_parameter_gradients))
    )
    print(f"gradient_sha256={_sha256_gradient_state(model)}")


def run_logits_cpu_backward_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[LOGITS_CPU_AUDIT_FAILURE]")
        print("stage=configuration")
        print("message=This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    model.train()
    criterion = nn.HuberLoss(delta=1.0)
    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")

    x_batch_cpu, y_batch_cpu = next(iter(train_loader))
    print("[LOGITS_CPU_AUDIT_INPUT]")
    print(f"input_sha256={_sha256_tensor(x_batch_cpu)}")
    print(f"target_sha256={_sha256_tensor(y_batch_cpu)}")

    x_batch = x_batch_cpu.to(device)
    y_batch = y_batch_cpu.to(device)

    logits = model(x_batch)
    pre_backward_logits_cpu = logits.detach().cpu()
    loss = criterion(logits, y_batch)
    pre_backward_loss_value = loss.item()
    print("[LOGITS_CPU_AUDIT_PRE_BACKWARD]")
    print(f"loss_item={pre_backward_loss_value:.12e}")

    try:
        loss.backward()
    except Exception as exc:  # pragma: no cover - diagnostic pathway
        print("[LOGITS_CPU_AUDIT_FAILURE]")
        print("stage=backward_exception")
        print(f"exception={repr(exc)}")
        return

    output_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss))
    print("[LOGITS_CPU_AUDIT_POST]")
    print(f"output_finite={output_finite}")
    print(f"loss={float(loss.detach().cpu().item()):.12e}")
    print(f"loss_finite={loss_finite}")
    print(f"pre_backward_logits_cpu_sha256={_sha256_tensor(pre_backward_logits_cpu)}")

    print("[LOGITS_CPU_AUDIT_PARAMETER_GRADS]")
    nonfinite_parameter_gradients = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        grad_present = grad is not None
        grad_finite_fraction, grad_max_abs, grad_finite = _grad_finite_fraction_and_max_abs(grad)
        print(f"name={name}")
        print(f"grad_present={grad_present}")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        if (not grad_present) or (not grad_finite):
            nonfinite_parameter_gradients.append(name)

    print(
        "nonfinite_parameter_gradients="
        + ("NONE" if not nonfinite_parameter_gradients else ",".join(nonfinite_parameter_gradients))
    )
    print(f"gradient_sha256={_sha256_gradient_state(model)}")


def run_step_prebackward_audit(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[STEP_PREBACKWARD_FAILURE]")
        print("stage=configuration")
        print("message=This diagnostic-only mode currently supports only --frequency daily_only --horizon 10d.")
        return

    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    prepared = _prepare_cpu_frequency_tensors(ds, "daily_only", 10, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    train_loader = _make_train_loader_from_prepared(prepared, "daily_only")
    criterion = nn.HuberLoss(delta=1.0)
    model.train()

    x_batch_cpu, y_batch_cpu = next(iter(train_loader))
    _print_step_audit_input(0, (x_batch_cpu, y_batch_cpu), "daily_only")
    input_sha = _sha256_tensor(x_batch_cpu)
    target_sha = _sha256_tensor(y_batch_cpu)

    print("[STEP_PREBACKWARD_PRE_PARAMS]")
    print("batch=0")
    parameter_sha256_pre_forward = _sha256_model_state(model)
    print(f"parameter_sha256={parameter_sha256_pre_forward}")

    logits = model(x_batch_cpu.to(device))
    target = y_batch_cpu.to(device)
    print("[STEP_PREBACKWARD_FORWARD]")
    print("batch=0")
    print(f"output_sum={float(_to_numpy(logits.detach().cpu()).sum()):.12e}")
    print(f"output_mean={_finite_stats(logits.detach().cpu())['mean']:.12e}")
    print(f"output_std={_finite_stats(logits.detach().cpu())['std']:.12e}")
    print(f"output_max_abs={_finite_stats(logits.detach().cpu())['max_abs']:.12e}")
    output_sha256 = _sha256_tensor(logits.detach().cpu())
    print(f"output_sha256={output_sha256}")

    loss = criterion(logits, target)
    loss_finite = bool(torch.isfinite(loss))
    loss_value = loss.item()
    print("[STEP_PREBACKWARD_LOSS]")
    print("batch=0")
    print(f"loss={loss_value}")
    print(f"loss_finite={loss_finite}")
    if not loss_finite:
        print("[STEP_PREBACKWARD_FAILURE]")
        print("stage=loss")
        return

    try:
        loss.backward()
    except Exception as exc:  # pragma: no cover - diagnostic pathway
        print("[STEP_PREBACKWARD_FAILURE]")
        print("stage=backward_exception")
        print(f"exception={repr(exc)}")
        return

    print("[STEP_PREBACKWARD_PARAMETER_GRADS]")
    nonfinite_parameter_gradients = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        grad_present = grad is not None
        grad_finite_fraction, grad_max_abs, grad_finite = _grad_finite_fraction_and_max_abs(grad)
        print(f"name={name}")
        print(f"grad_present={grad_present}")
        print(f"grad_finite={grad_finite}")
        print(f"grad_finite_fraction={grad_finite_fraction:.12f}")
        print(f"grad_max_abs={grad_max_abs:.12e}")
        if (not grad_present) or (not grad_finite):
            nonfinite_parameter_gradients.append(name)

    nonfinite_summary = "NONE" if not nonfinite_parameter_gradients else ",".join(nonfinite_parameter_gradients)
    print(f"nonfinite_parameter_gradients={nonfinite_summary}")
    gradient_sha256 = _sha256_gradient_state(model)
    print(f"gradient_sha256={gradient_sha256}")

    print("[STEP_PREBACKWARD_RESULT]")
    print(f"input_sha256={input_sha}")
    print(f"target_sha256={target_sha}")
    print(f"parameter_sha256_pre_forward={parameter_sha256_pre_forward}")
    print(f"output_sha256={output_sha256}")
    print(f"loss={loss_value}")
    print(f"nonfinite_parameter_gradients={nonfinite_summary}")


def train_and_trace_frequency(ds, frequency: str, horizon: int, stats: dict, trace_epochs: int, trace_mode: str) -> None:
    global DEBUG_NAN_MODE
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    single_model_cls = formal.SingleFrequencySWiMRegressor if trace_mode == "minimal" else SingleFrequencySWiMRegressor
    dual_model_cls = formal.DualFrequencySWiMRegressor if trace_mode == "minimal" else DualFrequencySWiMRegressor

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"

    if frequency == "daily_only":
        x_all = apply_normalizer(ds, stats, "daily_only")
        x_train = torch.tensor(x_all[train_mask], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        model = single_model_cls(feature_dim=x_all.shape[-1], frequency_key="daily").to(device)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "weekly_only":
        x_all = apply_normalizer(ds, stats, "weekly_only")
        x_train = torch.tensor(x_all[train_mask], dtype=torch.float32)
        x_val = torch.tensor(x_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        model = single_model_cls(feature_dim=x_all.shape[-1], frequency_key="weekly").to(device)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_batch, y_batch):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            return model(x_batch), y_batch

    elif frequency == "daily_weekly":
        x_daily_all, x_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        x_d_train = torch.tensor(x_daily_all[train_mask], dtype=torch.float32)
        x_d_val = torch.tensor(x_daily_all[val_mask], dtype=torch.float32)
        x_w_train = torch.tensor(x_weekly_all[train_mask], dtype=torch.float32)
        x_w_val = torch.tensor(x_weekly_all[val_mask], dtype=torch.float32)
        y_train = torch.tensor(ds.y[horizon][train_mask], dtype=torch.float32)
        y_val = torch.tensor(ds.y[horizon][val_mask], dtype=torch.float32)
        model = dual_model_cls(
            daily_feature_dim=x_daily_all.shape[-1],
            weekly_feature_dim=x_weekly_all.shape[-1],
        ).to(device)
        train_loader = DataLoader(TensorDataset(x_d_train, x_w_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_d_val, x_w_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        def batched_forward(x_daily, x_weekly, y_batch):
            x_daily = x_daily.to(device)
            x_weekly = x_weekly.to(device)
            y_batch = y_batch.to(device)
            return model(x_daily, x_weekly), y_batch

    else:
        raise ValueError(f"Unknown frequency setting: {frequency}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)

    if trace_mode == "minimal":
        DEBUG_NAN_MODE = False
        for epoch in range(1, trace_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_weight = 0

            if frequency == "daily_weekly":
                for x_d_batch, x_w_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                    loss = criterion(logits, target)
                    batch_size_value = y_batch.shape[0]
                    train_loss_sum += float(loss.item()) * batch_size_value
                    train_weight += int(batch_size_value)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    logits, target = batched_forward(x_batch, y_batch)
                    loss = criterion(logits, target)
                    batch_size_value = y_batch.shape[0]
                    train_loss_sum += float(loss.item()) * batch_size_value
                    train_weight += int(batch_size_value)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            model.eval()
            val_loss_sum = 0.0
            val_weight = 0
            with torch.no_grad():
                if frequency == "daily_weekly":
                    for x_d_batch, x_w_batch, y_batch in val_loader:
                        logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                        batch_loss = criterion(logits, target)
                        batch_size_value = y_batch.shape[0]
                        val_loss_sum += float(batch_loss.item()) * batch_size_value
                        val_weight += int(batch_size_value)
                else:
                    for x_batch, y_batch in val_loader:
                        logits, target = batched_forward(x_batch, y_batch)
                        batch_loss = criterion(logits, target)
                        batch_size_value = y_batch.shape[0]
                        val_loss_sum += float(batch_loss.item()) * batch_size_value
                        val_weight += int(batch_size_value)

            train_loss = train_loss_sum / train_weight if train_weight else float("nan")
            val_loss = val_loss_sum / val_weight if val_weight else float("nan")
            train_loss_finite = bool(np.isfinite(train_loss))
            val_loss_finite = bool(np.isfinite(val_loss))

            print("[MINIMAL_TRACE_EPOCH]")
            print(f"epoch={epoch}")
            print(f"train_loss={train_loss}")
            print(f"val_loss={val_loss}")
            print(f"train_loss_finite={train_loss_finite}")
            print(f"val_loss_finite={val_loss_finite}")

            if (not train_loss_finite) or (not val_loss_finite):
                print("[MINIMAL_TRACE_FAILURE]")
                print(f"epoch={epoch}")
                print(f"frequency={frequency}")
                print(f"horizon={horizon}d")
                print(f"device={device.type.upper()}")
                print(f"train_loss={train_loss}")
                print(f"val_loss={val_loss}")
                return
        return

    print(f"[TRACE] torch.initial_seed()={torch.initial_seed()}")
    print(f"[TRACE] selected device={device.type.upper()}")
    print("[TRACE] DEBUG_NAN_MODE = True")
    _model_parameter_fingerprint(model)
    print(f"[TRACE] {formal._trace_rng_fingerprint_summary('torch_rng_state_before_train_loader')}")

    DEBUG_NAN_MODE = True
    try:
        for epoch in range(1, trace_epochs + 1):
            DEBUG_NAN_CONTEXT["frequency"] = frequency
            DEBUG_NAN_CONTEXT["horizon"] = horizon
            DEBUG_NAN_CONTEXT["epoch"] = epoch
            DEBUG_NAN_CONTEXT["batch_index"] = None
            DEBUG_NAN_CONTEXT["device"] = device.type.upper()
            _trace_rng(epoch)
            model.train()
            train_loss_sum = 0.0
            train_weight = 0

            if frequency == "daily_weekly":
                for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(train_loader):
                    DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
                    optimizer.zero_grad()
                    if not bool(torch.isfinite(x_d_batch).all()):
                        _handle_input_failure("input", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                    if not bool(torch.isfinite(x_w_batch).all()):
                        _handle_input_failure("input", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                    if not bool(torch.isfinite(y_batch).all()):
                        _handle_input_failure("target", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                    if batch_idx < 3:
                        _first_batch_fingerprint(batch_idx, x_d_batch, y_batch)
                    logits = None
                    loss = None
                    try:
                        logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                    except TraceFailure:
                        raise
                    if not bool(torch.isfinite(logits).all()):
                        _trace_fail("forward", summaries=[("input_daily", x_d_batch), ("input_weekly", x_w_batch), ("target", y_batch), ("logits", logits)])
                    loss = criterion(logits, target)
                    if not bool(torch.isfinite(loss)):
                        _trace_fail("loss", summaries=[("input_daily", x_d_batch), ("input_weekly", x_w_batch), ("target", y_batch), ("logits", logits)], loss_value=loss.item())
                    batch_size_value = y_batch.shape[0]
                    train_loss_sum += float(loss.item()) * batch_size_value
                    train_weight += int(batch_size_value)
                    loss.backward()
                    grad_diag = _gradient_diagnostics(model)
                    if grad_diag["missing"] or grad_diag["nan_names"] or grad_diag["inf_names"]:
                        _trace_fail(
                            "backward",
                            parameter=grad_diag["nan_names"][0] if grad_diag["nan_names"] else grad_diag["inf_names"][0] if grad_diag["inf_names"] else grad_diag["missing"][0],
                            summaries=[("input_daily", x_d_batch), ("input_weekly", x_w_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not bool(torch.isfinite(grad_norm)):
                        _trace_fail(
                            "gradient_clipping",
                            summaries=[("input_daily", x_d_batch), ("input_weekly", x_w_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                        )
                    optimizer.step()
                    bad_param = None
                    for name, param in model.named_parameters():
                        if param.requires_grad and not bool(torch.isfinite(param).all()):
                            bad_param = name
                            break
                    if bad_param is not None:
                        _trace_fail(
                            "optimizer_step",
                            parameter=bad_param,
                            summaries=[("input_daily", x_d_batch), ("input_weekly", x_w_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                            param_diag=_parameter_diagnostics(model),
                        )
            else:
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
                    optimizer.zero_grad()
                    if not bool(torch.isfinite(x_batch).all()):
                        _handle_input_failure("input", frequency, horizon, batch_idx, x_batch, y_batch, device)
                    if not bool(torch.isfinite(y_batch).all()):
                        _handle_input_failure("target", frequency, horizon, batch_idx, x_batch, y_batch, device)
                    if batch_idx < 3:
                        _first_batch_fingerprint(batch_idx, x_batch, y_batch)
                    logits = None
                    loss = None
                    try:
                        logits, target = batched_forward(x_batch, y_batch)
                    except TraceFailure:
                        raise
                    if not bool(torch.isfinite(logits).all()):
                        _trace_fail("forward", summaries=[("input", x_batch), ("target", y_batch), ("logits", logits)])
                    loss = criterion(logits, target)
                    if not bool(torch.isfinite(loss)):
                        _trace_fail("loss", summaries=[("input", x_batch), ("target", y_batch), ("logits", logits)], loss_value=loss.item())
                    batch_size_value = y_batch.shape[0]
                    train_loss_sum += float(loss.item()) * batch_size_value
                    train_weight += int(batch_size_value)
                    loss.backward()
                    grad_diag = _gradient_diagnostics(model)
                    if grad_diag["missing"] or grad_diag["nan_names"] or grad_diag["inf_names"]:
                        _trace_fail(
                            "backward",
                            parameter=grad_diag["nan_names"][0] if grad_diag["nan_names"] else grad_diag["inf_names"][0] if grad_diag["inf_names"] else grad_diag["missing"][0],
                            summaries=[("input", x_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not bool(torch.isfinite(grad_norm)):
                        _trace_fail(
                            "gradient_clipping",
                            summaries=[("input", x_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                        )
                    optimizer.step()
                    bad_param = None
                    for name, param in model.named_parameters():
                        if param.requires_grad and not bool(torch.isfinite(param).all()):
                            bad_param = name
                            break
                    if bad_param is not None:
                        _trace_fail(
                            "optimizer_step",
                            parameter=bad_param,
                            summaries=[("input", x_batch), ("target", y_batch), ("logits", logits)],
                            loss_value=loss.item(),
                            grad_diag=grad_diag,
                            param_diag=_parameter_diagnostics(model),
                        )

            model.eval()
            val_loss_sum = 0.0
            val_weight = 0
            with torch.no_grad():
                if frequency == "daily_weekly":
                    for batch_idx, (x_d_batch, x_w_batch, y_batch) in enumerate(val_loader):
                        DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
                        if not bool(torch.isfinite(x_d_batch).all()):
                            _handle_input_failure("validation_input", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                        if not bool(torch.isfinite(x_w_batch).all()):
                            _handle_input_failure("validation_input", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                        if not bool(torch.isfinite(y_batch).all()):
                            _handle_input_failure("validation_target", frequency, horizon, batch_idx, (x_d_batch, x_w_batch), y_batch, device)
                        logits, target = batched_forward(x_d_batch, x_w_batch, y_batch)
                        if not bool(torch.isfinite(logits).all()):
                            _trace_fail("validation_forward", summaries=[("val_input_daily", x_d_batch), ("val_input_weekly", x_w_batch), ("val_target", y_batch), ("val_logits", logits)])
                        batch_loss = criterion(logits, target)
                        if not bool(torch.isfinite(batch_loss)):
                            _trace_fail("validation_loss", summaries=[("val_input_daily", x_d_batch), ("val_input_weekly", x_w_batch), ("val_target", y_batch), ("val_logits", logits)], loss_value=batch_loss.item())
                        batch_size_value = y_batch.shape[0]
                        val_loss_sum += float(batch_loss.item()) * batch_size_value
                        val_weight += int(batch_size_value)
                else:
                    for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                        DEBUG_NAN_CONTEXT["batch_index"] = batch_idx
                        if not bool(torch.isfinite(x_batch).all()):
                            _handle_input_failure("validation_input", frequency, horizon, batch_idx, x_batch, y_batch, device)
                        if not bool(torch.isfinite(y_batch).all()):
                            _handle_input_failure("validation_target", frequency, horizon, batch_idx, x_batch, y_batch, device)
                        logits, target = batched_forward(x_batch, y_batch)
                        if not bool(torch.isfinite(logits).all()):
                            _trace_fail("validation_forward", summaries=[("val_input", x_batch), ("val_target", y_batch), ("val_logits", logits)])
                        batch_loss = criterion(logits, target)
                        if not bool(torch.isfinite(batch_loss)):
                            _trace_fail("validation_loss", summaries=[("val_input", x_batch), ("val_target", y_batch), ("val_logits", logits)], loss_value=batch_loss.item())
                        batch_size_value = y_batch.shape[0]
                        val_loss_sum += float(batch_loss.item()) * batch_size_value
                        val_weight += int(batch_size_value)

            train_loss = train_loss_sum / train_weight if train_weight else float("nan")
            val_loss = val_loss_sum / val_weight if val_weight else float("nan")
            if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)):
                _trace_fail("loss", loss_value=train_loss if not np.isfinite(train_loss) else val_loss)
            print("[TRACE_EPOCH]")
            print(f"epoch={epoch}")
            print(f"train_loss={train_loss:.12f}")
            print(f"val_loss={val_loss:.12f}")
            print("status=FINITE")

        print("[TRACE_RESULT]")
        print("ALL_REQUESTED_EPOCHS_FINITE")
        print(f"trace_epochs={trace_epochs}")
    except TraceFailure:
        return
    finally:
        DEBUG_NAN_MODE = False
        DEBUG_NAN_CONTEXT["frequency"] = None
        DEBUG_NAN_CONTEXT["horizon"] = None
        DEBUG_NAN_CONTEXT["epoch"] = None
        DEBUG_NAN_CONTEXT["batch_index"] = None
        DEBUG_NAN_CONTEXT["device"] = None


def _serialize_cpu_validation_trajectory(rows: list[tuple[int, float, float]]) -> str:
    parts = []
    for epoch, train_loss, val_loss in rows:
        parts.append(f"{epoch},{train_loss:.17e},{val_loss:.17e}")
    return "\n".join(parts)


def _cpu_validation_trajectory_sha(rows: list[tuple[int, float, float]]) -> str:
    payload = _serialize_cpu_validation_trajectory(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_cpu_validation(ds, frequency: str, horizon: int, stats: dict) -> None:
    if frequency != "daily_only" or horizon != 10:
        print("[CPU_VALIDATION_FAILURE]")
        print("stage=configuration")
        print("message=This mode supports only --frequency daily_only --horizon 10d.")
        return

    device = torch.device("cpu")
    set_seed(SEED)
    prepared = _prepare_cpu_frequency_tensors(ds, frequency, horizon, stats)
    model = formal.SingleFrequencySWiMRegressor(feature_dim=prepared["feature_dim"], frequency_key="daily").to(device)
    train_loader = DataLoader(TensorDataset(prepared["x_train"], prepared["y_train"]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(prepared["x_val"], prepared["y_val"]), batch_size=BATCH_SIZE, shuffle=False)

    rng_state_before_fingerprint = torch.get_rng_state().clone()
    initial_x_batch, initial_y_batch = next(iter(train_loader))
    expected_first_input_sha = _sha256_tensor(initial_x_batch)
    expected_first_target_sha = _sha256_tensor(initial_y_batch)
    torch.set_rng_state(rng_state_before_fingerprint)

    print("[CPU_VALIDATION_CONFIG]")
    print("device=CPU")
    print("frequency=daily_only")
    print("horizon=10d")
    print("seed=42")
    print("[CPU_VALIDATION_INITIAL]")
    print(f"input_first_shuffled_batch_sha256={expected_first_input_sha}")
    print(f"target_first_shuffled_batch_sha256={expected_first_target_sha}")
    print(f"initial_model_parameter_sha256={_sha256_model_state(model)}")
    print(f"parameter_count={count_trainable_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss(delta=1.0)
    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    trajectory: list[tuple[int, float, float]] = []
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            if epoch == 1 and batch_idx == 0:
                actual_first_input_sha = _sha256_tensor(x_batch)
                actual_first_target_sha = _sha256_tensor(y_batch)
                print("[CPU_VALIDATION_FIRST_BATCH_MATCH]")
                print(f"input_match={actual_first_input_sha == expected_first_input_sha}")
                print(f"target_match={actual_first_target_sha == expected_first_target_sha}")
                if actual_first_input_sha != expected_first_input_sha or actual_first_target_sha != expected_first_target_sha:
                    print("[CPU_VALIDATION_FAILURE]")
                    print("epoch=1")
                    print("batch=0")
                    print("stage=first_batch_reproducibility")
                    return

            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            target = y_batch
            loss = criterion(logits, target)
            if not bool(torch.isfinite(loss)):
                print("[CPU_VALIDATION_FAILURE]")
                print(f"epoch={epoch}")
                print(f"batch={batch_idx}")
                print("stage=train_loss")
                return

            batch_size_value = y_batch.shape[0]
            train_loss_sum += float(loss.item()) * batch_size_value
            train_weight += int(batch_size_value)

            loss.backward()
            if epoch == 1 and batch_idx == 0:
                bad_grad_names = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.grad is None:
                        bad_grad_names.append(name)
                        continue
                    if not bool(torch.isfinite(param.grad).all()):
                        bad_grad_names.append(name)
                if bad_grad_names:
                    print("[CPU_VALIDATION_FIRST_BACKWARD]")
                    print("all_raw_gradients_finite=False")
                    for name in bad_grad_names:
                        print(f"parameter={name}")
                    print("[CPU_VALIDATION_FAILURE]")
                    print(f"epoch={epoch}")
                    print(f"batch={batch_idx}")
                    print("stage=gradient")
                    print(f"parameter={bad_grad_names[0]}")
                    return
                print("[CPU_VALIDATION_FIRST_BACKWARD]")
                print("all_raw_gradients_finite=True")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not bool(torch.isfinite(grad_norm)):
                print("[CPU_VALIDATION_FAILURE]")
                print(f"epoch={epoch}")
                print(f"batch={batch_idx}")
                print("stage=gradient_or_clip_if_checked")
                return

            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad and not bool(torch.isfinite(param).all()):
                    print("[CPU_VALIDATION_FAILURE]")
                    print(f"epoch={epoch}")
                    print(f"batch={batch_idx}")
                    print("stage=parameter")
                    print(f"parameter={name}")
                    return

        model.eval()
        val_loss_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(val_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                if not bool(torch.isfinite(logits).all()):
                    print("[CPU_VALIDATION_FAILURE]")
                    print(f"epoch={epoch}")
                    print(f"batch={batch_idx}")
                    print("stage=validation_loss")
                    return
                batch_loss = criterion(logits, y_val)
                if not bool(torch.isfinite(batch_loss)):
                    print("[CPU_VALIDATION_FAILURE]")
                    print(f"epoch={epoch}")
                    print(f"batch={batch_idx}")
                    print("stage=validation_loss")
                    return
                batch_size_value = y_val.shape[0]
                val_loss_sum += float(batch_loss.item()) * batch_size_value
                val_weight += int(batch_size_value)

        train_loss = train_loss_sum / train_weight if train_weight else float("nan")
        val_loss = val_loss_sum / val_weight if val_weight else float("nan")
        if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)):
            print("[CPU_VALIDATION_FAILURE]")
            print(f"epoch={epoch}")
            print("stage=train_loss" if not np.isfinite(train_loss) else "stage=validation_loss")
            return

        print("[CPU_VALIDATION_EPOCH]")
        print(f"epoch={epoch}")
        print(f"train_loss={train_loss:.12f}")
        print(f"val_loss={val_loss:.12f}")
        print(f"train_loss_finite={bool(np.isfinite(train_loss))}")
        print(f"val_loss_finite={bool(np.isfinite(val_loss))}")

        trajectory.append((epoch, float(train_loss), float(val_loss)))
        epochs_trained = epoch

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        print("[CPU_VALIDATION_FAILURE]")
        print("stage=training")
        print("message=No valid training state found.")
        return

    model.load_state_dict(best_state)
    model.eval()
    restored_finite = True
    for name, param in model.named_parameters():
        if param.requires_grad and not bool(torch.isfinite(param).all()):
            restored_finite = False
            print("[CPU_VALIDATION_FAILURE]")
            print(f"epoch={best_epoch}")
            print("stage=parameter")
            print(f"parameter={name}")
            return

    print("[CPU_VALIDATION_RESULT]")
    print("status=PASS")
    print("device=CPU")
    print("frequency=daily_only")
    print("horizon=10d")
    print("seed=42")
    print(f"epochs_trained={epochs_trained}")
    print(f"best_epoch={best_epoch}")
    print(f"best_val_loss={best_val_loss:.12f}")
    print(f"final_restored_parameter_sha256={_sha256_model_state(model)}")
    print(f"all_parameters_finite={restored_finite}")
    print(f"training_trajectory_sha256={_cpu_validation_trajectory_sha(trajectory)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWiM numerical diagnosis debug-only copy")
    parser.add_argument("--trace-formal-nan", action="store_true", help="Run frozen-path multi-epoch numerical tracing without writing formal evidence.")
    parser.add_argument("--rng-audit", action="store_true", help="Run reproducibility fingerprints without training or writing formal evidence.")
    parser.add_argument("--step-audit", action="store_true", help="Run first-N training batch state-transition fingerprints without validation or formal writes.")
    parser.add_argument("--mha-backward-audit", action="store_true", help="Run isolated local MultiheadAttention backward audit without full-model training or formal writes.")
    parser.add_argument("--backward-boundary-audit", action="store_true", help="Run full-graph backward boundary diagnostics with formal model hooks only.")
    parser.add_argument("--bare-backward-audit", action="store_true", help="Run least-instrumented full-model backward reproduction audit.")
    parser.add_argument("--loss-item-backward-audit", action="store_true", help="Run bare backward audit with one pre-backward loss.item() materialization.")
    parser.add_argument("--logits-cpu-backward-audit", action="store_true", help="Run loss-item audit plus one pre-backward logits.detach().cpu() materialization.")
    parser.add_argument("--step-prebackward-audit", action="store_true", help="Run step-audit pre-backward instrumentation without optimizer/zero_grad.")
    parser.add_argument("--cpu-validation", action="store_true", help="Run a debug-only CPU training gate for the exact frozen Daily-only / 10d / seed-42 SWiM path without writing evidence.")
    parser.add_argument("--trace-mode", choices=["minimal", "full"], default="full", help="Trace instrumentation level: minimal or full. Default: full.")
    parser.add_argument("--trace-epochs", type=_trace_epochs_arg, default=3, help="Trace epoch count for --trace-formal-nan only. Range: 1..20. Default: 3.")
    parser.add_argument("--step-audit-batches", type=_step_audit_batches_arg, default=3, help="Number of training batches for --step-audit only. Range: 1..10. Default: 3.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, help="Tracing frequency selection: daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], help="Tracing horizon selection: 5d | 10d | 20d")
    parser.add_argument("--train", action="store_true", help="Rejected in debug copy.")
    parser.add_argument("--all", action="store_true", help="Rejected in debug copy.")
    parser.add_argument("--finalize-only", action="store_true", help="Rejected in debug copy.")
    args = parser.parse_args()

    if args.train:
        parser.error("--train is disabled in panel_baseline_swim_debug.py.")
    if args.all:
        parser.error("--all is disabled in panel_baseline_swim_debug.py.")
    if args.finalize_only:
        parser.error("--finalize-only is disabled in panel_baseline_swim_debug.py.")
    selected_modes = [
        args.trace_formal_nan,
        args.rng_audit,
        args.step_audit,
        args.mha_backward_audit,
        args.backward_boundary_audit,
        args.bare_backward_audit,
        args.loss_item_backward_audit,
        args.logits_cpu_backward_audit,
        args.step_prebackward_audit,
        args.cpu_validation,
    ]
    if sum(1 for flag in selected_modes if flag) > 1:
        parser.error("Use only one of --trace-formal-nan, --rng-audit, --step-audit, --mha-backward-audit, --backward-boundary-audit, --bare-backward-audit, --loss-item-backward-audit, --logits-cpu-backward-audit, --step-prebackward-audit, or --cpu-validation.")
    if (
        not args.trace_formal_nan
        and not args.rng_audit
        and not args.step_audit
        and not args.mha_backward_audit
        and not args.backward_boundary_audit
        and not args.bare_backward_audit
        and not args.loss_item_backward_audit
        and not args.logits_cpu_backward_audit
        and not args.step_prebackward_audit
        and not args.cpu_validation
    ):
        parser.error("panel_baseline_swim_debug.py is diagnostic-only; use --trace-formal-nan, --rng-audit, --step-audit, --mha-backward-audit, --backward-boundary-audit, --bare-backward-audit, --loss-item-backward-audit, --logits-cpu-backward-audit, --step-prebackward-audit, or --cpu-validation.")
    if args.trace_formal_nan and (args.frequency is None or args.horizon is None):
        parser.error("--trace-formal-nan requires both --frequency and --horizon.")
    if args.rng_audit and (args.frequency is None or args.horizon is None):
        parser.error("--rng-audit requires both --frequency and --horizon.")
    if args.step_audit and (args.frequency is None or args.horizon is None):
        parser.error("--step-audit requires both --frequency and --horizon.")
    if args.mha_backward_audit and (args.frequency is None or args.horizon is None):
        parser.error("--mha-backward-audit requires both --frequency and --horizon.")
    if args.mha_backward_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--mha-backward-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.backward_boundary_audit and (args.frequency is None or args.horizon is None):
        parser.error("--backward-boundary-audit requires both --frequency and --horizon.")
    if args.backward_boundary_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--backward-boundary-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.bare_backward_audit and (args.frequency is None or args.horizon is None):
        parser.error("--bare-backward-audit requires both --frequency and --horizon.")
    if args.bare_backward_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--bare-backward-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.loss_item_backward_audit and (args.frequency is None or args.horizon is None):
        parser.error("--loss-item-backward-audit requires both --frequency and --horizon.")
    if args.loss_item_backward_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--loss-item-backward-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.logits_cpu_backward_audit and (args.frequency is None or args.horizon is None):
        parser.error("--logits-cpu-backward-audit requires both --frequency and --horizon.")
    if args.logits_cpu_backward_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--logits-cpu-backward-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.step_prebackward_audit and (args.frequency is None or args.horizon is None):
        parser.error("--step-prebackward-audit requires both --frequency and --horizon.")
    if args.step_prebackward_audit and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--step-prebackward-audit currently supports only --frequency daily_only --horizon 10d.")
    if args.cpu_validation and (args.frequency is None or args.horizon is None):
        parser.error("--cpu-validation requires both --frequency and --horizon.")
    if args.cpu_validation and (args.frequency != "daily_only" or args.horizon != "10d"):
        parser.error("--cpu-validation currently supports only --frequency daily_only --horizon 10d.")
    return args


def main() -> None:
    args = parse_args()
    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    formal.validate_frozen_dataset_contract(ds)
    stats = fit_train_only_normalizer(ds)
    frequency = args.frequency
    horizon = int(args.horizon[:-1])
    if args.rng_audit:
        run_rng_audit(ds, frequency, horizon, stats)
        return
    if args.step_audit:
        run_step_audit(ds, frequency, horizon, stats, step_audit_batches=args.step_audit_batches)
        return
    if args.mha_backward_audit:
        run_mha_backward_audit(ds, frequency, horizon, stats)
        return
    if args.backward_boundary_audit:
        run_backward_boundary_audit(ds, frequency, horizon, stats)
        return
    if args.bare_backward_audit:
        run_bare_backward_audit(ds, frequency, horizon, stats)
        return
    if args.loss_item_backward_audit:
        run_loss_item_backward_audit(ds, frequency, horizon, stats)
        return
    if args.logits_cpu_backward_audit:
        run_logits_cpu_backward_audit(ds, frequency, horizon, stats)
        return
    if args.step_prebackward_audit:
        run_step_prebackward_audit(ds, frequency, horizon, stats)
        return
    if args.cpu_validation:
        run_cpu_validation(ds, frequency, horizon, stats)
        return
    train_and_trace_frequency(ds, frequency, horizon, stats, trace_epochs=args.trace_epochs, trace_mode=args.trace_mode)


if __name__ == "__main__":
    main()
