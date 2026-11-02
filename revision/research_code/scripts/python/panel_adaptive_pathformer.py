"""ADAPTED panel PathFormer architecture for the frozen 17-stock green-energy panel.

This is an ADAPTED panel PathFormer architecture.

Core retained PathFormer-style mechanisms:
    multi-scale patch experts
    intra-patch attention
    inter-patch attention
    adaptive scale weighting

Project-specific adaptation:
    frequency-specific branches
    within-frequency scale routing
    late Daily/Weekly fusion
    frozen panel data/evaluation contract

This is not a byte-for-byte reproduction of the historical FSLR PathFormer
implementation or its full-frequency experiment setup. The design intentionally
supports the scientific architecture principle mandated by the advisor:

    PathFormer solves the MULTI-SCALE problem WITHIN each frequency.
    Late fusion solves the INFORMATION COMPLEMENTARITY problem BETWEEN
    different frequencies.

Therefore, the implementation contains frequency-specific scale experts and
within-frequency routers only, with no global Daily-vs-Weekly router.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import (
    AUDIT_DIR,
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
)

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_NORM = 1.0
HUBER_DELTA = 1.0

INPUT_DIM = 5
EXPERT_D_MODEL = 32
NHEAD = 4
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
ROUTER_HIDDEN = 32
DAILY_PATCH_SIZES = [5, 10, 20, 30]
WEEKLY_PATCH_SIZES = [2, 4, 8, 13]
FREQUENCY_SETTINGS = ["daily_only", "weekly_only", "daily_weekly"]
SCALE_MODES = ["single", "fixed", "static", "adaptive"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding with a fixed max length."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-(math.log(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds positional encoding capacity {self.pe.size(1)}.")
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute masked mean over the requested dimension.

    The valid mask is shaped like the input tensor without the feature dimension.
    We explicitly expand the denominator to match the feature dimension so that
    padded elements are ignored with correct broadcasting on arbitrary batch sizes.
    """
    valid_mask = valid_mask.to(dtype=torch.bool)
    masked = values * valid_mask.unsqueeze(-1).to(values.dtype)
    denom = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1).to(values.dtype)
    numerator = masked.sum(dim=dim)
    denom_for_broadcast = denom.squeeze(dim).unsqueeze(-1)
    return numerator / denom_for_broadcast


class DualAttentionScaleExpert(nn.Module):
    """PathFormer-style scale expert that uses intra-patch and inter-patch attention.

    Input: projected sequence [B, T, D], D=32.
    Output: expert representation z_k [B, D].
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        d_model: int = EXPERT_D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert input_dim == d_model, "Shared input projection output must match expert dimension."
        assert d_model % nhead == 0, "d_model must be divisible by nhead."
        assert patch_size > 0, "Patch size must be positive."

        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_size = int(patch_size)
        self.nhead = nhead

        self.intra_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.intra_norm = nn.LayerNorm(d_model)

        self.patch_positional_encoding = PositionalEncoding(d_model=d_model, max_len=1024, dropout=0.0)

        self.inter_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.inter_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.inter_norm2 = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def patchify(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad to a multiple of patch_size and reshape to [B, n_patches, p, D].

        If `valid_mask` is provided, it must describe the actual valid positions in
        `h` and can include padded tail slots that should be ignored. This supports
        the formal non-divisible sequence padding contract and the smoke test that
        changes only padded values while leaving real observations fixed.
        """
        B, T, D = h.shape
        p = self.patch_size
        if valid_mask is None:
            valid = torch.ones(B, T, device=h.device, dtype=torch.bool)
        else:
            if valid_mask.shape != (B, T):
                raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match input shape {(B, T)}.")
            valid = valid_mask.to(device=h.device, dtype=torch.bool)

        padded_T = int(math.ceil(T / p) * p)
        pad_len = padded_T - T
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, D, device=h.device, dtype=h.dtype)
            h_padded = torch.cat([h, pad], dim=1)
            valid = torch.cat([valid, torch.zeros(B, pad_len, device=h.device, dtype=torch.bool)], dim=1)
        else:
            h_padded = h

        # Explicitly zero all invalid padded positions before attention. This keeps
        # padded-tail values from influencing the true temporal content even when
        # the caller passes a nonzero value for the padded region.
        h_padded = h_padded * valid.unsqueeze(-1).to(h_padded.dtype)

        n_patches = padded_T // p
        patches = h_padded.reshape(B, n_patches, p, D)
        valid_patch = valid.reshape(B, n_patches, p)
        return patches, valid_patch

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Patchify the projected sequence into [B, n_patches, p, D].
        patches, valid_patch = self.patchify(h, valid_mask=valid_mask)
        B, n_patches, p, D = patches.shape

        flat_patches = patches.reshape(B * n_patches, p, D)
        flat_valid = valid_patch.reshape(B * n_patches, p)

        # Intra-patch self-attention with padding mask: padded tail tokens do not contribute.
        intra_out, _ = self.intra_attn(
            flat_patches,
            flat_patches,
            flat_patches,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        flat_patches = self.intra_norm(flat_patches + intra_out)
        flat_patches = flat_patches * flat_valid.unsqueeze(-1).to(flat_patches.dtype)

        # Masked mean in each patch to collapse the within-patch sequence to one token.
        patch_tokens = masked_mean(flat_patches, flat_valid, dim=1)
        patch_tokens = patch_tokens.reshape(B, n_patches, D)
        patch_tokens = self.patch_positional_encoding(patch_tokens)

        # Inter-patch self-attention across the patch-token sequence.
        inter_out, _ = self.inter_attn(
            patch_tokens,
            patch_tokens,
            patch_tokens,
            need_weights=False,
        )
        patch_tokens = self.inter_norm1(patch_tokens + inter_out)
        patch_tokens = self.inter_norm2(patch_tokens + self.ffn(patch_tokens))

        # Masked mean pooling across valid patches; padded tail patch contributions are ignored.
        valid_patch_any = valid_patch.any(dim=-1)
        expert_rep = masked_mean(patch_tokens, valid_patch_any, dim=1)
        expert_rep = self.output_norm(expert_rep)
        return expert_rep


class ScaleAggregator(nn.Module):
    """Aggregate multiple scale experts into one branch-level representation.

    Returns both branch_representation and per-scale weights. A consistent output
    interface is kept for later Experiment 4 interpretability analysis.
    """

    def __init__(
        self,
        scale_mode: str,
        patch_sizes: list[int],
        d_model: int = EXPERT_D_MODEL,
        router_hidden: int = ROUTER_HIDDEN,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        assert scale_mode in SCALE_MODES, f"Unsupported scale_mode={scale_mode}"
        self.scale_mode = scale_mode
        self.patch_sizes = list(patch_sizes)
        self.d_model = d_model
        self.K = len(self.patch_sizes)
        self.selected_patch_size = selected_patch_size

        if scale_mode == "single":
            if selected_patch_size is None:
                raise ValueError("scale_mode='single' requires explicit selected_patch_size.")
            self.selected_patch_size = int(selected_patch_size)
            if self.selected_patch_size not in self.patch_sizes:
                raise ValueError(f"selected_patch_size={selected_patch_size} not in {self.patch_sizes}.")
        elif scale_mode == "static":
            self.scale_logits = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))
        elif scale_mode == "adaptive":
            self.router = nn.Sequential(
                nn.Linear(self.K * d_model, router_hidden),
                nn.ReLU(),
                nn.Linear(router_hidden, self.K),
            )

    def _single_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones(batch_size, 1, device=device, dtype=torch.float32)

    def _fixed_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        w = torch.full((batch_size, self.K), 1.0 / self.K, device=device, dtype=torch.float32)
        return w

    def _static_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        alpha = torch.softmax(self.scale_logits, dim=0)
        return alpha.view(1, -1).expand(batch_size, -1)

    def _adaptive_weights(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(expert_outputs, dim=-1)
        logits = self.router(merged)
        weights = torch.softmax(logits, dim=-1)
        return weights

    def forward(self, expert_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not expert_outputs:
            raise ValueError("No expert outputs available for scale aggregation.")

        batch_size = expert_outputs[0].shape[0]
        device = expert_outputs[0].device
        if self.scale_mode == "single":
            z = expert_outputs[0]
            weights = self._single_weights(batch_size, device)
            return z, weights

        if self.scale_mode == "fixed":
            weights = self._fixed_weights(batch_size, device)
            z = sum(w * expert for w, expert in zip(weights[0], expert_outputs))
            return z, weights

        if self.scale_mode == "static":
            weights = self._static_weights(batch_size, device)
            z = sum(w * expert for w, expert in zip(weights[0], expert_outputs))
            return z, weights

        if self.scale_mode == "adaptive":
            weights = self._adaptive_weights(expert_outputs)
            z = sum(w.unsqueeze(-1) * expert for w, expert in zip(weights.T, expert_outputs))
            return z, weights

        raise ValueError(f"Unsupported scale_mode={self.scale_mode}")


class FrequencyAdaptivePathFormerBranch(nn.Module):
    """One frequency branch with shared input projection and K scale experts."""

    def __init__(
        self,
        frequency: str,
        input_dim: int,
        patch_sizes: list[int],
        scale_mode: str,
        d_model: int = EXPERT_D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        assert frequency in {"daily", "weekly"}
        assert len(patch_sizes) == 4, "Each frequency branch must have four scientific scale candidates."
        self.frequency = frequency
        self.input_dim = input_dim
        self.patch_sizes = list(patch_sizes)
        self.scale_mode = scale_mode
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=1024, dropout=0.0)

        if scale_mode == "single":
            if selected_patch_size is None:
                raise ValueError("scale_mode='single' requires an explicit selected_patch_size.")
            if int(selected_patch_size) not in self.patch_sizes:
                raise ValueError(f"selected_patch_size={selected_patch_size} not in {self.patch_sizes}.")
            self.selected_patch_size = int(selected_patch_size)
            self.experts = nn.ModuleList(
                [
                    DualAttentionScaleExpert(
                        input_dim=d_model,
                        patch_size=self.selected_patch_size,
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    DualAttentionScaleExpert(
                        input_dim=d_model,
                        patch_size=p,
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                    for p in patch_sizes
                ]
            )
            self.selected_patch_size = None

        self.aggregator = ScaleAggregator(
            scale_mode=scale_mode,
            patch_sizes=patch_sizes,
            d_model=d_model,
            router_hidden=ROUTER_HIDDEN,
            selected_patch_size=selected_patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        expert_outputs = [expert(h) for expert in self.experts]
        z, weights = self.aggregator(expert_outputs)
        return z, weights


class SingleFrequencyAdaptivePathFormerRegressor(nn.Module):
    """Single-frequency regressor for daily_only or weekly_only."""

    def __init__(
        self,
        frequency: str,
        scale_mode: str,
        input_dim: int = INPUT_DIM,
        d_model: int = EXPERT_D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        if frequency == "daily":
            patch_sizes = DAILY_PATCH_SIZES
        elif frequency == "weekly":
            patch_sizes = WEEKLY_PATCH_SIZES
        else:
            raise ValueError(f"Unsupported frequency={frequency}")

        self.frequency = frequency
        self.scale_mode = scale_mode
        self.branch = FrequencyAdaptivePathFormerBranch(
            frequency=frequency,
            input_dim=input_dim,
            patch_sizes=patch_sizes,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=selected_patch_size,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.branch(x)
        pred = self.head(z).squeeze(-1)
        return pred

    def get_router_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, weights = self.branch(x)
        return weights


class DualFrequencyAdaptivePathFormerRegressor(nn.Module):
    """Daily + Weekly regressor with late fusion after branch-level embeddings."""

    def __init__(
        self,
        scale_mode: str,
        daily_selected_patch_size: int | None = None,
        weekly_selected_patch_size: int | None = None,
        input_dim: int = INPUT_DIM,
        d_model: int = EXPERT_D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.scale_mode = scale_mode
        self.daily_branch = FrequencyAdaptivePathFormerBranch(
            frequency="daily",
            input_dim=input_dim,
            patch_sizes=DAILY_PATCH_SIZES,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=daily_selected_patch_size,
        )
        self.weekly_branch = FrequencyAdaptivePathFormerBranch(
            frequency="weekly",
            input_dim=input_dim,
            patch_sizes=WEEKLY_PATCH_SIZES,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=weekly_selected_patch_size,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        z_d, _ = self.daily_branch(x_daily)
        z_w, _ = self.weekly_branch(x_weekly)
        fused = torch.cat([z_d, z_w], dim=-1)
        pred = self.fusion(fused).squeeze(-1)
        return pred

    def get_router_weights(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, daily_weights = self.daily_branch(x_daily)
        _, weekly_weights = self.weekly_branch(x_weekly)
        return daily_weights, weekly_weights


def _make_model(frequency: str, scale_mode: str, selected_daily_patch: int | None = None, selected_weekly_patch: int | None = None):
    if frequency == "daily_only":
        selected_patch = selected_daily_patch if selected_daily_patch is not None else 20
        return SingleFrequencyAdaptivePathFormerRegressor(
            frequency="daily",
            scale_mode=scale_mode,
            selected_patch_size=selected_patch if scale_mode == "single" else None,
        )
    if frequency == "weekly_only":
        selected_patch = selected_weekly_patch if selected_weekly_patch is not None else 4
        return SingleFrequencyAdaptivePathFormerRegressor(
            frequency="weekly",
            scale_mode=scale_mode,
            selected_patch_size=selected_patch if scale_mode == "single" else None,
        )
    if frequency == "daily_weekly":
        return DualFrequencyAdaptivePathFormerRegressor(
            scale_mode=scale_mode,
            daily_selected_patch_size=selected_daily_patch if scale_mode == "single" else None,
            weekly_selected_patch_size=selected_weekly_patch if scale_mode == "single" else None,
        )
    raise ValueError(f"Unsupported frequency={frequency}")


def _run_frozen_panel_invariants() -> None:
    ds = load_frozen_panel()
    assert len(ds.split[ds.split == "train"]) == 31008, f"Unexpected train count: {len(ds.split[ds.split == 'train'])}"
    assert len(ds.split[ds.split == "val"]) == 6647, f"Unexpected val count: {len(ds.split[ds.split == 'val'])}"
    assert len(ds.split[ds.split == "test"]) == 6664, f"Unexpected test count: {len(ds.split[ds.split == 'test'])}"
    assert ds.X_daily.shape[1:] == (90, 5), ds.X_daily.shape
    assert ds.X_weekly.shape[1:] == (26, 5), ds.X_weekly.shape

    stats = fit_train_only_normalizer(ds)
    train_idx = np.where(ds.split == "train")[0][:8]
    x_daily = ds.X_daily[train_idx]
    x_weekly = ds.X_weekly[train_idx]
    norm_daily = apply_normalizer(ds, stats, "daily_only")
    norm_weekly = apply_normalizer(ds, stats, "weekly_only")
    assert np.isfinite(norm_daily[train_idx]).all(), "daily normalized batch has NaN/Inf"
    assert np.isfinite(norm_weekly[train_idx]).all(), "weekly normalized batch has NaN/Inf"
    print("FROZEN PANEL INVARIANTS: OK")


def _run_adaptive_forward_backward_smoke() -> None:
    ds = load_frozen_panel()
    stats = fit_train_only_normalizer(ds)
    normalized = apply_normalizer(ds, stats, "daily_weekly")
    Xd_norm, Xw_norm = normalized

    for frequency in ["daily_only", "weekly_only", "daily_weekly"]:
        if frequency == "daily_only":
            x = torch.from_numpy(Xd_norm[np.where(ds.split == "train")[0][:8]])
            y = torch.from_numpy(ds.y[5][np.where(ds.split == "train")[0][:8]])
            model = _make_model("daily_only", "adaptive")
            pred = model(x)
        elif frequency == "weekly_only":
            x = torch.from_numpy(Xw_norm[np.where(ds.split == "train")[0][:8]])
            y = torch.from_numpy(ds.y[5][np.where(ds.split == "train")[0][:8]])
            model = _make_model("weekly_only", "adaptive")
            pred = model(x)
        else:
            idx = np.where(ds.split == "train")[0][:8]
            x_d = torch.from_numpy(Xd_norm[idx])
            x_w = torch.from_numpy(Xw_norm[idx])
            y = torch.from_numpy(ds.y[5][idx])
            model = _make_model("daily_weekly", "adaptive")
            pred = model(x_d, x_w)

        assert pred.shape == y.shape, f"Unexpected prediction shape for {frequency}: {pred.shape} vs {y.shape}"
        assert torch.isfinite(pred).all(), f"Non-finite predictions for {frequency}"

        loss_fn = nn.HuberLoss(delta=1.0)
        loss = loss_fn(pred, y)
        assert torch.isfinite(loss), f"Non-finite Huber loss for {frequency}"

        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name} in {frequency}"
        print(f"ADAPTIVE FORWARD/BACKWARD {frequency}: OK")


def _run_router_invariants() -> None:
    ds = load_frozen_panel()
    stats = fit_train_only_normalizer(ds)
    Xd_norm, Xw_norm = apply_normalizer(ds, stats, "daily_weekly")
    idx = np.where(ds.split == "train")[0][:8]
    x_d = torch.from_numpy(Xd_norm[idx])
    x_w = torch.from_numpy(Xw_norm[idx])

    model_d = _make_model("daily_only", "adaptive")
    daily_weights = model_d.get_router_weights(x_d)
    assert daily_weights.shape == (x_d.shape[0], 4), daily_weights.shape
    assert torch.isfinite(daily_weights).all()
    assert (daily_weights >= 0).all()
    assert torch.allclose(daily_weights.sum(dim=1), torch.ones(x_d.shape[0]), atol=1e-6, rtol=0.0)

    model_w = _make_model("weekly_only", "adaptive")
    weekly_weights = model_w.get_router_weights(x_w)
    assert weekly_weights.shape == (x_w.shape[0], 4), weekly_weights.shape
    assert torch.isfinite(weekly_weights).all()
    assert (weekly_weights >= 0).all()
    assert torch.allclose(weekly_weights.sum(dim=1), torch.ones(x_w.shape[0]), atol=1e-6, rtol=0.0)

    dual_model = _make_model("daily_weekly", "adaptive")
    daily_w, weekly_w = dual_model.get_router_weights(x_d, x_w)
    assert daily_w.shape == (x_d.shape[0], 4), daily_w.shape
    assert weekly_w.shape == (x_w.shape[0], 4), weekly_w.shape
    assert daily_w.shape != weekly_w.shape or daily_w is not weekly_w
    print("ROUTER INVARIANTS: OK")


def _run_aggregation_mode_tests() -> None:
    ds = load_frozen_panel()
    stats = fit_train_only_normalizer(ds)
    Xd_norm, Xw_norm = apply_normalizer(ds, stats, "daily_weekly")
    idx = np.where(ds.split == "train")[0][:8]
    x_d = torch.from_numpy(Xd_norm[idx])
    x_w = torch.from_numpy(Xw_norm[idx])

    for scale_mode in ["single", "fixed", "static", "adaptive"]:
        if scale_mode == "single":
            daily_branch = FrequencyAdaptivePathFormerBranch(
                frequency="daily",
                input_dim=INPUT_DIM,
                patch_sizes=DAILY_PATCH_SIZES,
                scale_mode="single",
                selected_patch_size=20,
            )
            z, w = daily_branch(x_d)
            assert w.shape == (x_d.shape[0], 1), w.shape
            assert torch.allclose(w, torch.ones_like(w)), f"single mode weights not all ones: {w}"
        elif scale_mode == "fixed":
            daily_branch = FrequencyAdaptivePathFormerBranch(
                frequency="daily",
                input_dim=INPUT_DIM,
                patch_sizes=DAILY_PATCH_SIZES,
                scale_mode="fixed",
            )
            _, w = daily_branch(x_d)
            expected = torch.full_like(w, 0.25)
            assert torch.allclose(w, expected, atol=1e-6, rtol=0.0), w
        elif scale_mode == "static":
            daily_branch = FrequencyAdaptivePathFormerBranch(
                frequency="daily",
                input_dim=INPUT_DIM,
                patch_sizes=DAILY_PATCH_SIZES,
                scale_mode="static",
            )
            _, w = daily_branch(x_d)
            expected = torch.full_like(w, 0.25)
            assert torch.allclose(w, expected, atol=1e-6, rtol=0.0), w
        else:
            daily_branch = FrequencyAdaptivePathFormerBranch(
                frequency="daily",
                input_dim=INPUT_DIM,
                patch_sizes=DAILY_PATCH_SIZES,
                scale_mode="adaptive",
            )
            _, w = daily_branch(x_d)
            assert w.shape == (x_d.shape[0], 4), w.shape
            assert torch.isfinite(w).all()
            assert (w >= 0).all()
            assert torch.allclose(w.sum(dim=1), torch.ones(x_d.shape[0]), atol=1e-6, rtol=0.0)
    print("AGGREGATION MODE TESTS: OK")


def _run_padding_mask_tests() -> None:
    B = 2
    device = torch.device("cpu")
    for seq_len, p in [(90, 20), (26, 4), (26, 8)]:
        padded_len = int(math.ceil(seq_len / p) * p)
        x_base = torch.randn(B, seq_len, INPUT_DIM, device=device)
        proj = torch.randn(INPUT_DIM, EXPERT_D_MODEL, device=device)

        projected0 = x_base @ proj
        projected0_padded = torch.cat([projected0, torch.zeros(B, padded_len - seq_len, EXPERT_D_MODEL, device=device)], dim=1)

        # The real sequence is the same in both cases; only the explicitly padded
        # tail differs. The validity mask ensures those padded slots are ignored.
        projected1 = projected0_padded.clone()
        projected1[:, seq_len:, :] = torch.randn(B, padded_len - seq_len, EXPERT_D_MODEL, device=device) * 1000.0
        valid_mask = torch.ones(B, padded_len, device=device, dtype=torch.bool)
        valid_mask[:, seq_len:] = False

        expert = DualAttentionScaleExpert(input_dim=EXPERT_D_MODEL, patch_size=p, d_model=EXPERT_D_MODEL)
        expert.eval()
        out0 = expert(projected0_padded, valid_mask=valid_mask)
        out1 = expert(projected1, valid_mask=valid_mask)
        assert out0.shape == (B, EXPERT_D_MODEL), out0.shape
        assert torch.isfinite(out0).all(), "expert output has NaN/Inf"
        assert torch.allclose(out0, out1, atol=1e-5, rtol=1e-5), f"padding ignored check failed for p={p}, T={seq_len}"
    print("PADDING / MASK TESTS: OK")


def _run_parameter_budget_check() -> None:
    configs = {
        "daily_only_adaptive": _make_model("daily_only", "adaptive"),
        "weekly_only_adaptive": _make_model("weekly_only", "adaptive"),
        "daily_weekly_adaptive": _make_model("daily_weekly", "adaptive"),
        "daily_only_single": _make_model("daily_only", "single", selected_daily_patch=20),
        "weekly_only_single": _make_model("weekly_only", "single", selected_weekly_patch=4),
        "daily_weekly_single": _make_model("daily_weekly", "single", selected_daily_patch=20, selected_weekly_patch=4),
    }
    refs = {
        "daily_only": 67393,
        "weekly_only": 67393,
        "daily_weekly": 142977,
    }
    print("PARAMETER COUNTS:")
    for name, model in configs.items():
        count = count_trainable_parameters(model)
        if "daily_only" in name:
            ref_key = "daily_only"
        elif "weekly_only" in name:
            ref_key = "weekly_only"
        else:
            ref_key = "daily_weekly"

        ratio = count / refs[ref_key]
        if "single" in name:
            print(f"  {name}: {count}  (single-mode construction check; ratio={ratio:.3f} vs {ref_key})")
        else:
            print(f"  {name}: {count}  ref={refs[ref_key]}  ratio={ratio:.3f}")
            if ratio > 2.0:
                print("PARAMETER BUDGET WARNING")
                raise RuntimeError(f"Model exceeds 2x reference budget for {name}: count={count}, ratio={ratio}")
    print("PARAMETER BUDGET: OK")


def _run_single_mode_construction_checks() -> None:
    daily_model = _make_model("daily_only", "single", selected_daily_patch=20)
    weekly_model = _make_model("weekly_only", "single", selected_weekly_patch=4)
    dual_model = _make_model("daily_weekly", "single", selected_daily_patch=20, selected_weekly_patch=4)

    assert len(daily_model.branch.experts) == 1, f"daily single should have 1 expert, got {len(daily_model.branch.experts)}"
    assert daily_model.branch.experts[0].patch_size == 20, daily_model.branch.experts[0].patch_size
    assert len(weekly_model.branch.experts) == 1, f"weekly single should have 1 expert, got {len(weekly_model.branch.experts)}"
    assert weekly_model.branch.experts[0].patch_size == 4, weekly_model.branch.experts[0].patch_size
    assert len(dual_model.daily_branch.experts) == 1, f"dual daily single should have 1 expert, got {len(dual_model.daily_branch.experts)}"
    assert dual_model.daily_branch.experts[0].patch_size == 20, dual_model.daily_branch.experts[0].patch_size
    assert len(dual_model.weekly_branch.experts) == 1, f"dual weekly single should have 1 expert, got {len(dual_model.weekly_branch.experts)}"
    assert dual_model.weekly_branch.experts[0].patch_size == 4, dual_model.weekly_branch.experts[0].patch_size
    print("SINGLE MODE CONSTRUCTION CHECKS: OK")


def run_smoke_validation() -> None:
    """Default smoke validation path. This is the only validation path executed by default."""
    set_seed(SEED)
    _run_frozen_panel_invariants()
    _run_single_mode_construction_checks()
    _run_adaptive_forward_backward_smoke()
    _run_router_invariants()
    _run_aggregation_mode_tests()
    _run_padding_mask_tests()
    _run_parameter_budget_check()

    if torch.backends.mps.is_available():
        ds = load_frozen_panel()
        stats = fit_train_only_normalizer(ds)
        Xd_norm, Xw_norm = apply_normalizer(ds, stats, "daily_weekly")
        idx = np.where(ds.split == "train")[0][:4]
        device = torch.device("mps")
        x_d = torch.from_numpy(Xd_norm[idx]).to(device)
        x_w = torch.from_numpy(Xw_norm[idx]).to(device)
        y = torch.from_numpy(ds.y[5][idx]).to(device)

        for model_cfg in [
            ("daily_only", _make_model("daily_only", "adaptive").to(device), x_d),
            ("weekly_only", _make_model("weekly_only", "adaptive").to(device), x_w),
            ("daily_weekly", _make_model("daily_weekly", "adaptive").to(device), (x_d, x_w)),
        ]:
            name, model, batch = model_cfg
            model.train()
            if name == "daily_weekly":
                pred = model(*batch)
            else:
                pred = model(batch)
            loss = nn.HuberLoss(delta=1.0)(pred, y)
            model.zero_grad()
            loss.backward()
            assert torch.isfinite(pred).all(), f"MPS non-finite pred for {name}"
            assert torch.isfinite(loss), f"MPS non-finite loss for {name}"
            print(f"MPS SMOKE {name}: OK")
    else:
        print("MPS SMOKE: SKIPPED (MPS unavailable)")

    print("SMOKE VALIDATION COMPLETE")


def build_training_loaders(frequency: str, horizon: int, ds: Any, batch_size: int = BATCH_SIZE):
    """Construct train/val/test loaders using the frozen panel contract and the exact
    pre-transfer ordering required by the proposal: X_daily and X_weekly are used
    as their own frequency tensors and not combined until late fusion.
    """
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    stats = fit_train_only_normalizer(ds)
    if frequency == "daily_only":
        X_train, X_val, X_test = (
            apply_normalizer(ds, stats, "daily_only")[train_mask],
            apply_normalizer(ds, stats, "daily_only")[val_mask],
            apply_normalizer(ds, stats, "daily_only")[test_mask],
        )
        y_train, y_val, y_test = (
            ds.y[horizon][train_mask],
            ds.y[horizon][val_mask],
            ds.y[horizon][test_mask],
        )
    elif frequency == "weekly_only":
        X_train, X_val, X_test = (
            apply_normalizer(ds, stats, "weekly_only")[train_mask],
            apply_normalizer(ds, stats, "weekly_only")[val_mask],
            apply_normalizer(ds, stats, "weekly_only")[test_mask],
        )
        y_train, y_val, y_test = (
            ds.y[horizon][train_mask],
            ds.y[horizon][val_mask],
            ds.y[horizon][test_mask],
        )
    elif frequency == "daily_weekly":
        Xd_norm, Xw_norm = apply_normalizer(ds, stats, "daily_weekly")
        Xd_train, Xd_val, Xd_test = Xd_norm[train_mask], Xd_norm[val_mask], Xd_norm[test_mask]
        Xw_train, Xw_val, Xw_test = Xw_norm[train_mask], Xw_norm[val_mask], Xw_norm[test_mask]
        y_train, y_val, y_test = ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]
    else:
        raise ValueError(f"Unsupported frequency={frequency}")

    if frequency == "daily_weekly":
        train_set = TensorDataset(torch.from_numpy(Xd_train), torch.from_numpy(Xw_train), torch.from_numpy(y_train))
        val_set = TensorDataset(torch.from_numpy(Xd_val), torch.from_numpy(Xw_val), torch.from_numpy(y_val))
        test_set = TensorDataset(torch.from_numpy(Xd_test), torch.from_numpy(Xw_test), torch.from_numpy(y_test))
    else:
        train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _router_weights_for_mode(scale_mode: str, branch: FrequencyAdaptivePathFormerBranch, x: torch.Tensor, frequency: str) -> tuple[torch.Tensor, list[int]]:
    if scale_mode == "single":
        active_patch = branch.experts[0].patch_size
        weights = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)
        return weights, [active_patch]
    if scale_mode == "fixed":
        weights = torch.full((x.shape[0], len(branch.patch_sizes)), 1.0 / len(branch.patch_sizes), device=x.device, dtype=torch.float32)
        return weights, branch.patch_sizes
    if scale_mode == "static":
        alpha = torch.softmax(branch.aggregator.scale_logits, dim=0)
        weights = alpha.view(1, -1).expand(x.shape[0], -1)
        return weights, branch.patch_sizes
    if scale_mode == "adaptive":
        _, weights = branch(x)
        return weights, branch.patch_sizes
    raise ValueError(f"Unsupported scale_mode={scale_mode}")


def _collect_test_router_rows(model: nn.Module, frequency: str, scale_mode: str, x_daily: torch.Tensor, x_weekly: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, ticker: np.ndarray, anchor_date: np.ndarray, horizon: int) -> list[dict]:
    rows: list[dict] = []
    if frequency == "daily_only":
        branch = model.branch
        weights, patch_sizes = _router_weights_for_mode(scale_mode, branch, x_daily, "daily")
        for i in range(x_daily.shape[0]):
            for idx, patch_size in enumerate(patch_sizes):
                rows.append(
                    {
                        "ticker": ticker[i],
                        "anchor_date": str(anchor_date[i]),
                        "horizon": horizon,
                        "frequency_setting": "daily_only",
                        "scale_mode": scale_mode,
                        "frequency": "daily",
                        "patch_size": int(patch_size),
                        "router_weight": float(weights[i, idx] if weights.ndim == 2 else weights[i]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                    }
                )
        return rows
    if frequency == "weekly_only":
        branch = model.branch
        weights, patch_sizes = _router_weights_for_mode(scale_mode, branch, x_weekly, "weekly")
        for i in range(x_weekly.shape[0]):
            for idx, patch_size in enumerate(patch_sizes):
                rows.append(
                    {
                        "ticker": ticker[i],
                        "anchor_date": str(anchor_date[i]),
                        "horizon": horizon,
                        "frequency_setting": "weekly_only",
                        "scale_mode": scale_mode,
                        "frequency": "weekly",
                        "patch_size": int(patch_size),
                        "router_weight": float(weights[i, idx] if weights.ndim == 2 else weights[i]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                    }
                )
        return rows
    if frequency == "daily_weekly":
        daily_weights, daily_patches = _router_weights_for_mode(scale_mode, model.daily_branch, x_daily, "daily")
        weekly_weights, weekly_patches = _router_weights_for_mode(scale_mode, model.weekly_branch, x_weekly, "weekly")
        for i in range(x_daily.shape[0]):
            for idx, patch_size in enumerate(daily_patches):
                rows.append(
                    {
                        "ticker": ticker[i],
                        "anchor_date": str(anchor_date[i]),
                        "horizon": horizon,
                        "frequency_setting": "daily_weekly",
                        "scale_mode": scale_mode,
                        "frequency": "daily",
                        "patch_size": int(patch_size),
                        "router_weight": float(daily_weights[i, idx]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                    }
                )
        for i in range(x_weekly.shape[0]):
            for idx, patch_size in enumerate(weekly_patches):
                rows.append(
                    {
                        "ticker": ticker[i],
                        "anchor_date": str(anchor_date[i]),
                        "horizon": horizon,
                        "frequency_setting": "daily_weekly",
                        "scale_mode": scale_mode,
                        "frequency": "weekly",
                        "patch_size": int(patch_size),
                        "router_weight": float(weekly_weights[i, idx]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                    }
                )
        return rows
    raise ValueError(f"Unsupported frequency={frequency}")


def _replace_existing_config_rows(df: pd.DataFrame, frequency_setting: str, horizon: int, scale_mode: str) -> pd.DataFrame:
    if df.empty or "frequency_setting" not in df.columns or "horizon" not in df.columns or "scale_mode" not in df.columns:
        return df.copy()
    horizon_numeric = pd.to_numeric(df["horizon"], errors="coerce")
    mask = (
        df["frequency_setting"].astype(str).str.strip().eq(str(frequency_setting))
        & (horizon_numeric == int(horizon))
        & df["scale_mode"].astype(str).str.strip().eq(str(scale_mode))
    )
    return df.loc[~mask].copy()


def _write_formal_outputs(summary_df: pd.DataFrame, predictions_df: pd.DataFrame, history_df: pd.DataFrame, rank_ic_df: pd.DataFrame, router_df: pd.DataFrame, summary_txt: str, frequency_setting: str, horizon: int, scale_mode: str) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = AUDIT_DIR / "panel_pathformer_summary_metrics.csv"
    pred_path = AUDIT_DIR / "panel_pathformer_test_predictions.csv"
    hist_path = AUDIT_DIR / "panel_pathformer_training_history.csv"
    rank_ic_path = AUDIT_DIR / "panel_pathformer_rank_ic_by_date.csv"
    router_path = AUDIT_DIR / "panel_pathformer_router_weights.csv"
    summary_txt_path = AUDIT_DIR / "panel_pathformer_final_summary.txt"

    for path, frame in [
        (summary_path, summary_df),
        (pred_path, predictions_df),
        (hist_path, history_df),
        (rank_ic_path, rank_ic_df),
        (router_path, router_df),
    ]:
        if path.exists():
            existing = pd.read_csv(path)
            existing = _replace_existing_config_rows(existing, frequency_setting, horizon, scale_mode)
            pd.concat([existing, frame], ignore_index=True).to_csv(path, index=False)
        else:
            frame.to_csv(path, index=False)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write(summary_txt)


def _forward_with_weights(model: nn.Module, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], frequency: str):
    if frequency == "daily_only":
        z, weights = model.branch(x)
        pred = model.head(z).squeeze(-1)
        return pred, weights
    if frequency == "weekly_only":
        z, weights = model.branch(x)
        pred = model.head(z).squeeze(-1)
        return pred, weights
    if frequency == "daily_weekly":
        x_daily, x_weekly = x
        z_d, daily_w = model.daily_branch(x_daily)
        z_w, weekly_w = model.weekly_branch(x_weekly)
        fused = torch.cat([z_d, z_w], dim=-1)
        pred = model.fusion(fused).squeeze(-1)
        return pred, daily_w, weekly_w
    raise ValueError(f"Unsupported frequency={frequency}")


def _evaluate_and_write_formal_outputs(model: nn.Module, ds: Any, frequency: str, scale_mode: str, horizon: int, horizon_key: int, device: torch.device | None = None):
    if device is None:
        device = _resolve_device()

    test_mask = ds.split == "test"
    ticker = ds.ticker[test_mask]
    anchor_date = ds.anchor_date[test_mask]
    y_true_raw = ds.y[horizon_key][test_mask]
    n_test = int(len(ticker))

    stats = fit_train_only_normalizer(ds)
    model.eval()
    pred_list: list[torch.Tensor] = []
    router_rows: list[dict] = []

    with torch.inference_mode():
        if frequency == "daily_only":
            X_norm = apply_normalizer(ds, stats, "daily_only")
            x_test = torch.from_numpy(X_norm[test_mask])
            y_test = torch.from_numpy(y_true_raw)
            patch_sizes = [model.branch.experts[0].patch_size] if scale_mode == "single" else DAILY_PATCH_SIZES
            loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred_batch, weights_batch = _forward_with_weights(model, x_batch, frequency)
                pred_list.append(pred_batch.detach().cpu())
                w = weights_batch.detach().cpu()
                for row_idx in range(x_batch.shape[0]):
                    for patch_idx, patch_size in enumerate(patch_sizes):
                        router_rows.append({
                            "ticker": ticker[row_idx + batch_idx * BATCH_SIZE],
                            "anchor_date": str(anchor_date[row_idx + batch_idx * BATCH_SIZE]),
                            "horizon": int(horizon),
                            "frequency_setting": frequency,
                            "scale_mode": scale_mode,
                            "frequency": "daily",
                            "patch_size": int(patch_size),
                            "router_weight": float(w[row_idx, patch_idx] if w.ndim == 2 else 1.0),
                            "y_true": float(y_batch[row_idx].cpu().item()),
                            "y_pred": float(pred_batch[row_idx].cpu().item()),
                        })
        elif frequency == "weekly_only":
            X_norm = apply_normalizer(ds, stats, "weekly_only")
            x_test = torch.from_numpy(X_norm[test_mask])
            y_test = torch.from_numpy(y_true_raw)
            patch_sizes = [model.branch.experts[0].patch_size] if scale_mode == "single" else WEEKLY_PATCH_SIZES
            loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred_batch, weights_batch = _forward_with_weights(model, x_batch, frequency)
                pred_list.append(pred_batch.detach().cpu())
                w = weights_batch.detach().cpu()
                for row_idx in range(x_batch.shape[0]):
                    for patch_idx, patch_size in enumerate(patch_sizes):
                        router_rows.append({
                            "ticker": ticker[row_idx + batch_idx * BATCH_SIZE],
                            "anchor_date": str(anchor_date[row_idx + batch_idx * BATCH_SIZE]),
                            "horizon": int(horizon),
                            "frequency_setting": frequency,
                            "scale_mode": scale_mode,
                            "frequency": "weekly",
                            "patch_size": int(patch_size),
                            "router_weight": float(w[row_idx, patch_idx] if w.ndim == 2 else 1.0),
                            "y_true": float(y_batch[row_idx].cpu().item()),
                            "y_pred": float(pred_batch[row_idx].cpu().item()),
                        })
        elif frequency == "daily_weekly":
            Xd_norm, Xw_norm = apply_normalizer(ds, stats, "daily_weekly")
            x_daily_test = torch.from_numpy(Xd_norm[test_mask])
            x_weekly_test = torch.from_numpy(Xw_norm[test_mask])
            y_test = torch.from_numpy(y_true_raw)
            daily_patch_sizes = [model.daily_branch.experts[0].patch_size] if scale_mode == "single" else DAILY_PATCH_SIZES
            weekly_patch_sizes = [model.weekly_branch.experts[0].patch_size] if scale_mode == "single" else WEEKLY_PATCH_SIZES
            loader = DataLoader(TensorDataset(x_daily_test, x_weekly_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
            for batch_idx, (x_daily_batch, x_weekly_batch, y_batch) in enumerate(loader):
                x_daily_batch = x_daily_batch.to(device)
                x_weekly_batch = x_weekly_batch.to(device)
                y_batch = y_batch.to(device)
                pred_batch, daily_w, weekly_w = _forward_with_weights(model, (x_daily_batch, x_weekly_batch), frequency)
                pred_list.append(pred_batch.detach().cpu())
                daily_w_cpu = daily_w.detach().cpu()
                weekly_w_cpu = weekly_w.detach().cpu()
                for row_idx in range(x_daily_batch.shape[0]):
                    base_index = row_idx + batch_idx * BATCH_SIZE
                    for patch_idx, patch_size in enumerate(daily_patch_sizes):
                        router_rows.append({
                            "ticker": ticker[base_index],
                            "anchor_date": str(anchor_date[base_index]),
                            "horizon": int(horizon),
                            "frequency_setting": frequency,
                            "scale_mode": scale_mode,
                            "frequency": "daily",
                            "patch_size": int(patch_size),
                            "router_weight": float(daily_w_cpu[row_idx, patch_idx] if daily_w_cpu.ndim == 2 else 1.0),
                            "y_true": float(y_batch[row_idx].cpu().item()),
                            "y_pred": float(pred_batch[row_idx].cpu().item()),
                        })
                    for patch_idx, patch_size in enumerate(weekly_patch_sizes):
                        router_rows.append({
                            "ticker": ticker[base_index],
                            "anchor_date": str(anchor_date[base_index]),
                            "horizon": int(horizon),
                            "frequency_setting": frequency,
                            "scale_mode": scale_mode,
                            "frequency": "weekly",
                            "patch_size": int(patch_size),
                            "router_weight": float(weekly_w_cpu[row_idx, patch_idx] if weekly_w_cpu.ndim == 2 else 1.0),
                            "y_true": float(y_batch[row_idx].cpu().item()),
                            "y_pred": float(pred_batch[row_idx].cpu().item()),
                        })
        else:
            raise ValueError(f"Unsupported frequency={frequency}")

    if not pred_list:
        raise RuntimeError(f"No test predictions collected for {frequency} / {scale_mode}")
    pred = torch.cat(pred_list, dim=0).numpy().astype(np.float32)
    y_true_cpu = y_true_raw.astype(np.float32)

    assert len(pred) == n_test, f"Pred length mismatch: {len(pred)} vs {n_test}"
    assert len(y_true_cpu) == n_test, f"y_true length mismatch: {len(y_true_cpu)} vs {n_test}"
    assert len(ticker) == n_test, f"ticker length mismatch: {len(ticker)} vs {n_test}"
    assert len(anchor_date) == n_test, f"anchor_date length mismatch: {len(anchor_date)} vs {n_test}"

    pred_df = pd.DataFrame({
        "ticker": ticker,
        "anchor_date": anchor_date,
        "frequency_setting": frequency,
        "scale_mode": scale_mode,
        "horizon": int(horizon),
        "y_true": y_true_cpu,
        "y_pred": pred,
    })

    metrics = evaluate_predictions(y_true_cpu, pred, ticker, anchor_date)
    rank_ic_df = rank_ic_by_date(y_true_cpu, pred, ticker, anchor_date)
    rank_ic_df["frequency_setting"] = frequency
    rank_ic_df["horizon"] = int(horizon)
    rank_ic_df["scale_mode"] = scale_mode
    rank_ic_vals = rank_ic_df["rank_ic"].dropna().to_numpy()
    summary_row = {
        "frequency_setting": frequency,
        "horizon": int(horizon),
        "scale_mode": scale_mode,
        "best_epoch": getattr(model, "best_epoch", None),
        "epochs_trained": getattr(model, "epochs_trained", None),
        "val_loss": getattr(model, "best_val_loss", None),
        "parameter_count": count_trainable_parameters(model),
        "pooled_mse": metrics["pooled_mse"],
        "pooled_mae": metrics["pooled_mae"],
        "pooled_corr": metrics["pooled_corr"],
        "ticker_avg_corr": metrics["ticker_avg_corr"],
        "direction_accuracy": metrics["pooled_da"],
        "pred_std": metrics["pooled_pred_std"],
        "true_std": metrics["pooled_true_std"],
        "pred_std_true_std_ratio": metrics["pooled_pred_std_over_true_std"],
        "cross_sectional_rank_ic_mean": float(np.nanmean(rank_ic_vals)) if len(rank_ic_vals) > 0 else float("nan"),
        "cross_sectional_rank_ic_median": float(np.nanmedian(rank_ic_vals)) if len(rank_ic_vals) > 0 else float("nan"),
        "cross_sectional_rank_ic_std": float(np.nanstd(rank_ic_vals)) if len(rank_ic_vals) > 0 else float("nan"),
        "cross_sectional_rank_ic_positive_ratio": float(np.mean(rank_ic_vals > 0)) if len(rank_ic_vals) > 0 else float("nan"),
    }
    summary_df = pd.DataFrame([summary_row])
    history_df = pd.DataFrame(getattr(model, "training_history", []))
    summary_txt = (
        f"frequency_setting={frequency}\n"
        f"horizon={horizon}\n"
        f"scale_mode={scale_mode}\n"
        f"best_epoch={getattr(model, 'best_epoch', 'n/a')}\n"
        f"epochs_trained={getattr(model, 'epochs_trained', 'n/a')}\n"
        f"best_val_loss={getattr(model, 'best_val_loss', 'n/a')}\n"
        f"parameter_count={count_trainable_parameters(model)}\n"
        f"pooled_mse={summary_row['pooled_mse']}\n"
        f"pooled_mae={summary_row['pooled_mae']}\n"
        f"pooled_corr={summary_row['pooled_corr']}\n"
    )
    router_df = pd.DataFrame(router_rows)
    _write_formal_outputs(summary_df, pred_df, history_df, rank_ic_df, router_df, summary_txt, frequency, horizon, scale_mode)

    return {"metrics": metrics, "summary": summary_row, "predictions": pred_df, "rank_ic": rank_ic_df, "router_weights": router_df}


def train_model(
    frequency: str,
    scale_mode: str,
    horizon: int,
    ds: Any,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device | None = None,
    single_daily_patch: int | None = None,
    single_weekly_patch: int | None = None,
):
    """Formal training contract for later experiment execution.

    This function is implemented but deliberately not executed by the default smoke
    validation path. The default CLI path validates implementation only.
    """
    if device is None:
        device = _resolve_device()

    set_seed(SEED)
    train_loader, val_loader, test_loader = build_training_loaders(frequency=frequency, horizon=horizon, ds=ds, batch_size=batch_size)

    if frequency == "daily_only":
        model = _make_model(frequency, scale_mode, selected_daily_patch=single_daily_patch if scale_mode == "single" else None)
    elif frequency == "weekly_only":
        model = _make_model(frequency, scale_mode, selected_weekly_patch=single_weekly_patch if scale_mode == "single" else None)
    else:
        model = _make_model(
            frequency,
            scale_mode,
            selected_daily_patch=single_daily_patch if scale_mode == "single" else None,
            selected_weekly_patch=single_weekly_patch if scale_mode == "single" else None,
        )
    model = model.to(device)
    if scale_mode == "single":
        if frequency == "daily_only":
            print(f"Formal single selected patch: daily_only -> {single_daily_patch}")
        elif frequency == "weekly_only":
            print(f"Formal single selected patch: weekly_only -> {single_weekly_patch}")
        else:
            print(f"Formal single selected patches: daily={single_daily_patch}, weekly={single_weekly_patch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    min_val_loss = float("inf")
    best_epoch = None
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        epoch_index = epoch + 1
        model.train()
        train_loss_total = 0.0
        for batch_index, batch in enumerate(train_loader):
            if frequency == "daily_weekly":
                x_daily, x_weekly, y_batch = batch
                x_daily = x_daily.to(device)
                x_weekly = x_weekly.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_daily, x_weekly)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)

            loss = loss_fn(pred, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite batch loss: frequency={frequency}, horizon={horizon}, scale_mode={scale_mode}, epoch={epoch}, batch_index={batch_index}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()
            train_loss_total += loss.item() * y_batch.shape[0]

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if frequency == "daily_weekly":
                    x_daily, x_weekly, y_batch = batch
                    x_daily = x_daily.to(device)
                    x_weekly = x_weekly.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_daily, x_weekly)
                else:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_batch)
                val_loss = loss_fn(pred, y_batch)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite val loss: frequency={frequency}, horizon={horizon}, scale_mode={scale_mode}, epoch={epoch}")
                val_loss_total += val_loss.item() * y_batch.shape[0]

        train_loss = train_loss_total / max(len(train_loader.dataset), 1)
        val_loss = val_loss_total / max(len(val_loader.dataset), 1)
        if not np.isfinite(train_loss):
            raise RuntimeError(f"Non-finite train_loss: frequency={frequency}, horizon={horizon}, scale_mode={scale_mode}, epoch={epoch_index}")
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite val_loss: frequency={frequency}, horizon={horizon}, scale_mode={scale_mode}, epoch={epoch_index}")
        is_best = val_loss < min_val_loss - 1e-12
        history.append(
            {
                "frequency_setting": frequency,
                "horizon": int(horizon),
                "scale_mode": scale_mode,
                "epoch": int(epoch_index),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "is_best": bool(is_best),
            }
        )
        if is_best:
            min_val_loss = val_loss
            best_epoch = epoch_index
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    model.training_history = history
    model.best_epoch = best_epoch
    model.epochs_trained = len(history)
    model.best_val_loss = min_val_loss
    return model


def _single_training_run(args: argparse.Namespace) -> None:
    """Run one formal training configuration when explicitly requested by CLI."""
    ds = load_frozen_panel()
    horizon_map = {"5d": 5, "10d": 10, "20d": 20}
    horizon = horizon_map[args.horizon]
    if args.frequency not in FREQUENCY_SETTINGS:
        raise ValueError(f"Unsupported frequency={args.frequency}")
    if args.scale_mode not in SCALE_MODES:
        raise ValueError(f"Unsupported scale_mode={args.scale_mode}")

    device = _resolve_device()
    print(f"Formal configuration: {args.frequency} / {args.scale_mode} / {args.horizon}")
    print(f"Selected patches: daily={args.single_daily_patch}, weekly={args.single_weekly_patch}")
    if args.scale_mode == "single":
        if args.frequency == "daily_only":
            print(f"Single-mode daily patch selected = {args.single_daily_patch}")
        elif args.frequency == "weekly_only":
            print(f"Single-mode weekly patch selected = {args.single_weekly_patch}")
        else:
            print(f"Single-mode daily patch selected = {args.single_daily_patch}; weekly patch selected = {args.single_weekly_patch}")

    model = train_model(
        frequency=args.frequency,
        scale_mode=args.scale_mode,
        horizon=horizon,
        ds=ds,
        batch_size=BATCH_SIZE,
        device=device,
        single_daily_patch=args.single_daily_patch,
        single_weekly_patch=args.single_weekly_patch,
    )
    print(f"Training complete for {args.frequency} / {args.scale_mode} / {args.horizon}")
    print(f"Device: {device}")
    print(f"Parameters: {count_trainable_parameters(model)}")
    _evaluate_and_write_formal_outputs(model, ds, args.frequency, args.scale_mode, horizon, horizon, device)


def _full_benchmark_run(args: argparse.Namespace) -> None:
    """Reserved for later benchmark execution. This is intentionally not launched here."""
    raise NotImplementedError("Formal 9-configuration benchmark is intentionally not implemented for this smoke-only task.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive panel PathFormer smoke validation and later formal training entrypoint.")
    parser.add_argument("--train", action="store_true", help="Run formal training for a single configuration.")
    parser.add_argument("--all", action="store_true", help="Require this flag before launching the formal 9-configuration benchmark.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, default="daily_weekly", help="daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], default="5d")
    parser.add_argument("--scale-mode", choices=SCALE_MODES, default="adaptive", help="single | fixed | static | adaptive")
    parser.add_argument("--single-daily-patch", type=int, default=20, help="SMOKE / DEVELOPMENT DEFAULT ONLY; formal training must pass the explicit selected patch.")
    parser.add_argument("--single-weekly-patch", type=int, default=4, help="SMOKE / DEVELOPMENT DEFAULT ONLY; formal training must pass the explicit selected patch.")
    args = parser.parse_args()
    if args.all and not args.train:
        parser.error("--all requires --train to be set.")
    return args


def main() -> None:
    args = parse_args()
    if args.train:
        if args.all:
            _full_benchmark_run(args)
        else:
            _single_training_run(args)
        return

    # Default behavior: execute smoke validation only, never the formal training pipeline.
    run_smoke_validation()


if __name__ == "__main__":
    main()
