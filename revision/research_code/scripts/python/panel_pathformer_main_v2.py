"""Formal MAIN V2 benchmark for a PathFormer-inspired adaptive multi-scale Transformer.

This script is intentionally a formal execution entrypoint and is distinct from the
historical development prototype at scripts/python/panel_adaptive_pathformer.py.

It is not a literal reproduction of the original repository AMS / PathFormer code.
It is a custom PathFormer-inspired adaptive multi-scale Transformer that preserves
key audited design choices from the development candidate while enforcing the
accepted Dataset Contract V2 and a formal evidence-safe execution contract.

Main scientific contract:
- Daily branch: adaptive multi-scale path experts over [5, 10, 20, 30]
- Weekly branch: adaptive multi-scale path experts over [2, 4, 8, 13]
- within-frequency routing only
- no global Daily-vs-Weekly router
- no early raw-token fusion
- late fusion across frequency branches only

The script enforces formal CPU execution, Dataset V2 invariants, output
provenance, and a strict 9-cell MAIN benchmark definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import (
    FROZEN_TICKERS,
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
)
from scripts.python.panel_baseline_swim import (
    canonicalize_main_v2_lstm_or_vanilla,
    canonicalize_main_v2_naive,
    canonicalize_main_v2_ridge,
)

DATASET_MANIFEST_SHA256 = "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c"
DATASET_MANIFEST_PATH = ROOT / "dataset" / "multiscale_dataset" / "panel" / "panel_build_manifest.csv"
FREQUENCY_SETTINGS = ["daily_only", "weekly_only", "daily_weekly"]
MAIN_HORIZONS = [5, 10, 20]
MAIN_BENCHMARK_MODE = "adaptive"
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HUBER_DELTA = 1.0
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0
INPUT_DIM = 5
D_MODEL = 32
NHEAD = 4
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
ROUTER_HIDDEN = 32
DAILY_PATCH_SIZES = [5, 10, 20, 30]
WEEKLY_PATCH_SIZES = [2, 4, 8, 13]
SCALE_MODES = ["single", "fixed", "static", "adaptive"]
FORMAL_OUTPUT_DIR = ROOT / "dataset" / "audit" / "main_v2" / "pathformer"
FORMAL_SUMMARY_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_summary_metrics.csv"
FORMAL_PRED_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_test_predictions.csv"
FORMAL_RANK_IC_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_rank_ic_by_date.csv"
FORMAL_HISTORY_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_training_history.csv"
FORMAL_ROUTER_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_router_weights.csv"
FORMAL_COMPARISON_PATH = FORMAL_OUTPUT_DIR / "main_v2_pathformer_comparison.csv"
FORMAL_MANIFEST_PATH = FORMAL_OUTPUT_DIR / "main_v2_pathformer_run_manifest.csv"
FORMAL_PROVENANCE_PATH = FORMAL_OUTPUT_DIR / "main_v2_pathformer_training_provenance.csv"
FORMAL_FINAL_SUMMARY_PATH = FORMAL_OUTPUT_DIR / "panel_pathformer_final_summary.txt"
RAW_EVIDENCE_PATHS = {
    "summary": FORMAL_SUMMARY_PATH,
    "predictions": FORMAL_PRED_PATH,
    "rank_ic": FORMAL_RANK_IC_PATH,
    "history": FORMAL_HISTORY_PATH,
    "router": FORMAL_ROUTER_PATH,
}
RAW_EVIDENCE_KEYS = tuple(RAW_EVIDENCE_PATHS.keys())


@dataclass
class FormalCell:
    frequency: str
    horizon: int
    scale_mode: str = "adaptive"


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    This preserves the audited development implementation structure and supports
    the formal dataset contract without introducing any frequency / horizon state.
    """

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


def _resolve_device() -> torch.device:
    """Formal execution must be CPU only for accepted MAIN V2 evidence."""
    return torch.device("cpu")


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute mean only over valid entries along a given dimension.

    This preserves the audited partial-patch behavior: padded observations never
    contribute to the patch representation.
    """
    valid_mask = valid_mask.to(dtype=torch.bool)
    masked = values * valid_mask.unsqueeze(-1).to(values.dtype)
    denom = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1).to(values.dtype)
    numerator = masked.sum(dim=dim)
    denom_for_broadcast = denom.squeeze(dim).unsqueeze(-1)
    return numerator / denom_for_broadcast


def occupancy_weighted_patch_mean(patch_tokens: torch.Tensor, valid_patch_counts: torch.Tensor) -> torch.Tensor:
    """Occupancy-weighted patch mean with exact [B, D] output semantics.

    patch_tokens: [B, P, D]
    valid_patch_counts: [B, P]
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"patch_tokens must have shape [B, P, D], got {tuple(patch_tokens.shape)}")
    if valid_patch_counts.shape != patch_tokens.shape[:2]:
        raise ValueError(
            "valid_patch_counts shape mismatch: "
            f"expected {tuple(patch_tokens.shape[:2])}, got {tuple(valid_patch_counts.shape)}"
        )

    counts = valid_patch_counts.to(patch_tokens.dtype)
    numerator = (patch_tokens * counts.unsqueeze(-1)).sum(dim=1)
    denominator = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    result = numerator / denominator
    if result.ndim != 2:
        raise ValueError(f"occupancy_weighted_patch_mean output must be [B, D], got {tuple(result.shape)}")
    return result


class DualAttentionScaleExpert(nn.Module):
    """PathFormer-inspired scale expert with patch-based local and cross-patch attention.

    This is intentionally not a literal reproduction of the original repository AMS.
    It preserves the development-candidate structure:
      - patchify with right-padding
      - valid-token masking in intra-patch attention
      - valid-only patch mean
      - inter-patch attention
      - residual FFN blocks
      - LayerNorms
      - occupancy-weighted final pooling across patch tokens
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert input_dim == d_model
        assert d_model % nhead == 0
        assert patch_size > 0

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

        # Padded observations are explicitly zeroed before any attention stage.
        h_padded = h_padded * valid.unsqueeze(-1).to(h_padded.dtype)

        n_patches = padded_T // p
        patches = h_padded.reshape(B, n_patches, p, D)
        valid_patch = valid.reshape(B, n_patches, p)
        return patches, valid_patch

    def forward(self, h: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        patches, valid_patch = self.patchify(h, valid_mask=valid_mask)
        B, n_patches, p, D = patches.shape

        flat_patches = patches.reshape(B * n_patches, p, D)
        flat_valid = valid_patch.reshape(B * n_patches, p)

        intra_out, _ = self.intra_attn(
            flat_patches,
            flat_patches,
            flat_patches,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        flat_patches = self.intra_norm(flat_patches + intra_out)
        flat_patches = flat_patches * flat_valid.unsqueeze(-1).to(flat_patches.dtype)

        # Each patch is represented by its valid-only mean, so partial final patches
        # do not receive the same weight as full patches in the final expert output.
        patch_tokens = masked_mean(flat_patches, flat_valid, dim=1)
        patch_tokens = patch_tokens.reshape(B, n_patches, D)
        patch_tokens = self.patch_positional_encoding(patch_tokens)

        inter_out, _ = self.inter_attn(
            patch_tokens,
            patch_tokens,
            patch_tokens,
            need_weights=False,
        )
        patch_tokens = self.inter_norm1(patch_tokens + inter_out)
        patch_tokens = self.inter_norm2(patch_tokens + self.ffn(patch_tokens))

        # Occupancy-weighted final pooling across patch tokens.
        # This is part of the formal MAIN V2 contract: partial patch tokens are weighted
        # by the number of valid observations in that patch, not by a constant 1 for each
        # non-empty patch. This ensures a 2-observation patch does not carry equal mass to
        # an 8-observation full patch when representations are otherwise aligned.
        valid_patch_counts = valid_patch.sum(dim=-1).to(patch_tokens.dtype)
        expert_rep = occupancy_weighted_patch_mean(patch_tokens, valid_patch_counts)
        if expert_rep.shape[1] != patch_tokens.shape[-1]:
            raise ValueError(f"Occupancy pooling output shape mismatch: {tuple(expert_rep.shape)}")
        expert_rep = self.output_norm(expert_rep)
        return expert_rep


class ScaleAggregator(nn.Module):
    """Fixed / static / adaptive aggregation over scale experts.

    The router is sample-dependent, not batch-dependent: it consumes a per-sample
    concatenated expert representation and applies an MLP + softmax per sample.
    There is no BatchNorm, batch reduction, or cross-sample mixing.
    """

    def __init__(
        self,
        scale_mode: str,
        patch_sizes: list[int],
        d_model: int = D_MODEL,
        router_hidden: int = ROUTER_HIDDEN,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        assert scale_mode in SCALE_MODES
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
    """Single frequency branch with K adaptive multi-scale experts."""

    def __init__(
        self,
        frequency: str,
        input_dim: int,
        patch_sizes: list[int],
        scale_mode: str,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        assert frequency in {"daily", "weekly"}
        assert len(patch_sizes) == 4
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
            self.experts = nn.ModuleList([
                DualAttentionScaleExpert(
                    input_dim=d_model,
                    patch_size=self.selected_patch_size,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
            ])
        else:
            self.experts = nn.ModuleList([
                DualAttentionScaleExpert(
                    input_dim=d_model,
                    patch_size=p,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for p in patch_sizes
            ])
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
    def __init__(
        self,
        frequency: str,
        scale_mode: str,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
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

    def forward_with_router(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, weights = self.branch(x)
        pred = self.head(z).squeeze(-1)
        return pred, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.forward_with_router(x)
        return pred

    def get_router_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, weights = self.branch(x)
        return weights


class DualFrequencyAdaptivePathFormerRegressor(nn.Module):
    def __init__(
        self,
        scale_mode: str,
        daily_selected_patch_size: int | None = None,
        weekly_selected_patch_size: int | None = None,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
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

    def forward_with_router(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_d, daily_weights = self.daily_branch(x_daily)
        z_w, weekly_weights = self.weekly_branch(x_weekly)
        fused = torch.cat([z_d, z_w], dim=-1)
        pred = self.fusion(fused).squeeze(-1)
        return pred, daily_weights, weekly_weights

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        pred, _, _ = self.forward_with_router(x_daily, x_weekly)
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


def _formal_check_dataset_contract(ds: Any) -> None:
    """Fail closed if the Dataset Contract V2 invariants are violated."""
    if not DATASET_MANIFEST_PATH.exists():
        raise RuntimeError(f"Dataset V2 manifest is missing: {DATASET_MANIFEST_PATH}")
    manifest_sha = _compute_sha256(DATASET_MANIFEST_PATH)
    if manifest_sha != DATASET_MANIFEST_SHA256:
        raise RuntimeError(
            f"Dataset V2 manifest SHA mismatch: expected {DATASET_MANIFEST_SHA256}, got {manifest_sha}"
        )

    unique_tickers = np.unique(ds.ticker)
    expected_tickers = np.array(FROZEN_TICKERS)
    if set(unique_tickers.tolist()) != set(expected_tickers.tolist()):
        raise RuntimeError(f"Ticker set mismatch: expected {sorted(expected_tickers.tolist())}, got {sorted(unique_tickers.tolist())}")
    if len(unique_tickers) != 17:
        raise RuntimeError(f"Expected 17 unique tickers for Dataset V2, got {len(unique_tickers)}.")

    n_samples = len(ds.ticker)
    length_checks = {
        "len(ds.ticker)": len(ds.ticker),
        "len(ds.anchor_date)": len(ds.anchor_date),
        "len(ds.split)": len(ds.split),
        "ds.X_daily.shape[0]": ds.X_daily.shape[0],
        "ds.X_weekly.shape[0]": ds.X_weekly.shape[0],
        "len(ds.y[5])": len(ds.y[5]),
        "len(ds.y[10])": len(ds.y[10]),
        "len(ds.y[20])": len(ds.y[20]),
    }
    for key, value in length_checks.items():
        if value != 43571:
            raise RuntimeError(f"Dataset V2 sample-count contract mismatch for {key}: expected 43571, got {value}.")
    if n_samples != 43571:
        raise RuntimeError(f"Expected total sample count 43571, got {n_samples}")

    if ds.X_daily.shape[1:] != (90, 5):
        raise RuntimeError(f"Daily input shape mismatch: {ds.X_daily.shape}")
    if ds.X_weekly.shape[1:] != (26, 5):
        raise RuntimeError(f"Weekly input shape mismatch: {ds.X_weekly.shape}")
    train_count = int(np.sum(ds.split == "train"))
    val_count = int(np.sum(ds.split == "val"))
    test_count = int(np.sum(ds.split == "test"))
    if train_count != 30634:
        raise RuntimeError(f"Unexpected train count: {train_count}")
    if val_count != 6290:
        raise RuntimeError(f"Unexpected val count: {val_count}")
    if test_count != 6647:
        raise RuntimeError(f"Unexpected test count: {test_count}")

    train_dates = pd.to_datetime(ds.anchor_date[ds.split == "train"])
    val_dates = pd.to_datetime(ds.anchor_date[ds.split == "val"])
    test_dates = pd.to_datetime(ds.anchor_date[ds.split == "test"])
    if len(np.unique(train_dates)) != 1802:
        raise RuntimeError(f"Unexpected train unique anchor dates: {len(np.unique(train_dates))}")
    if len(np.unique(val_dates)) != 370:
        raise RuntimeError(f"Unexpected val unique anchor dates: {len(np.unique(val_dates))}")
    if len(np.unique(test_dates)) != 391:
        raise RuntimeError(f"Unexpected test unique anchor dates: {len(np.unique(test_dates))}")

    all_unique_dates = pd.to_datetime(np.unique(ds.anchor_date))
    if len(all_unique_dates) != 2563:
        raise RuntimeError(f"Unexpected retained anchor-date universe: {len(all_unique_dates)}")
    date_counts = pd.Series(pd.to_datetime(ds.anchor_date)).value_counts()
    if not (date_counts == 17).all():
        raise RuntimeError("Every retained anchor date must have exactly 17 observations (one per ticker).")

    date_tickers = pd.DataFrame({"anchor_date": ds.anchor_date, "ticker": ds.ticker})
    for anchor_date_value, group in date_tickers.groupby("anchor_date"):
        unique_tickers = set(group["ticker"].unique().tolist())
        if len(unique_tickers) != 17:
            raise RuntimeError(f"Anchor date {anchor_date_value} does not contain 17 unique tickers: {len(unique_tickers)}")
        if unique_tickers != set(FROZEN_TICKERS):
            raise RuntimeError(
                f"Anchor date {anchor_date_value} tickers do not match the frozen set: "
                f"expected {sorted(FROZEN_TICKERS)}, got {sorted(unique_tickers)}"
            )

    train_start = train_dates.min().date()
    train_end = train_dates.max().date()
    val_start = val_dates.min().date()
    val_end = val_dates.max().date()
    test_start = test_dates.min().date()
    test_end = test_dates.max().date()
    if (train_start, train_end) != (pd.Timestamp("2016-01-29").date(), pd.Timestamp("2023-03-27").date()):
        raise RuntimeError(f"Train split bounds mismatch: {(train_start, train_end)}")
    if (val_start, val_end) != (pd.Timestamp("2023-04-26").date(), pd.Timestamp("2024-10-14").date()):
        raise RuntimeError(f"Validation split bounds mismatch: {(val_start, val_end)}")
    if (test_start, test_end) != (pd.Timestamp("2024-11-12").date(), pd.Timestamp("2026-06-05").date()):
        raise RuntimeError(f"Test split bounds mismatch: {(test_start, test_end)}")


def _expected_model_count(frequency: str, scale_mode: str) -> int:
    if frequency == "daily_only" and scale_mode == "adaptive":
        return 56069
    if frequency == "weekly_only" and scale_mode == "adaptive":
        return 56069
    if frequency == "daily_weekly" and scale_mode == "adaptive":
        return 116297
    return -1


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Deterministic binary fingerprint: dtype + shape metadata + raw contiguous CPU bytes."""
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(cpu_tensor.dtype).encode("utf-8"))
    h.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    h.update(cpu_tensor.numpy().tobytes())
    return h.hexdigest()


def _state_dict_sha256(model: nn.Module) -> str:
    """Deterministic binary fingerprint over the full state_dict (params + buffers), sorted by name."""
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state.keys()):
        cpu_tensor = state[name].detach().to("cpu").contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(cpu_tensor.dtype).encode("utf-8"))
        h.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
        h.update(cpu_tensor.numpy().tobytes())
    return h.hexdigest()


def _trajectory_sha256(history: list[dict]) -> str:
    """Deterministic binary fingerprint over the chronological epoch/train_loss/val_loss trajectory."""
    h = hashlib.sha256()
    h.update(struct.pack("<q", len(history)))
    for row in history:
        h.update(struct.pack("<q", int(row["epoch"])))
        h.update(struct.pack("<d", float(row["train_loss"])))
        h.update(struct.pack("<d", float(row["val_loss"])))
    return h.hexdigest()


def _build_current_provenance_record() -> dict:
    script_path = Path(__file__).resolve()
    shared_loader = (ROOT / "scripts" / "python" / "panel_common.py").resolve()
    dev_ref = (ROOT / "scripts" / "python" / "panel_adaptive_pathformer.py").resolve()
    return {
        "run_family": "MAIN_V2",
        "temporal_status": "POST_TEMPORAL_FIX",
        "model_family": "ADAPTIVE_MULTISCALE_TRANSFORMER_PATHFORMER_INSPIRED",
        "raw_model_value": "adaptive_pathformer",
        "dataset_contract_version": 2,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "formal_ticker_count": 17,
        "seed": 42,
        "execution_backend": "CPU",
        "training_script": "scripts/python/panel_pathformer_main_v2.py",
        "training_script_sha256": _compute_sha256(script_path),
        "shared_loader_script": "scripts/python/panel_common.py",
        "shared_loader_script_sha256": _compute_sha256(shared_loader),
        "development_reference_script": "scripts/python/panel_adaptive_pathformer.py",
        "development_reference_script_sha256": _compute_sha256(dev_ref),
        "historical_results_used_for_training": False,
        "historical_results_used_for_model_selection": False,
        "pre_temporal_fix_results_used": False,
    }


def _write_provenance_row(provenance: dict) -> None:
    FORMAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([provenance])
    df.to_csv(FORMAL_PROVENANCE_PATH, index=False)


def _validate_stored_provenance(provenance_df: pd.DataFrame) -> None:
    if provenance_df.empty:
        raise RuntimeError("Formal provenance file is empty.")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Formal provenance must contain exactly one row; found {len(provenance_df)}.")

    current = _build_current_provenance_record()
    for field in [
        "run_family",
        "temporal_status",
        "model_family",
        "raw_model_value",
        "dataset_contract_version",
        "dataset_manifest_sha256",
        "formal_ticker_count",
        "seed",
        "execution_backend",
        "training_script",
        "training_script_sha256",
        "shared_loader_script",
        "shared_loader_script_sha256",
        "development_reference_script",
        "development_reference_script_sha256",
        "historical_results_used_for_training",
        "historical_results_used_for_model_selection",
        "pre_temporal_fix_results_used",
    ]:
        stored_value = provenance_df[field].iloc[0]
        if stored_value != current[field]:
            raise RuntimeError(f"Immutable provenance mismatch for {field}: stored={stored_value}, current={current[field]}")

    manifest_sha = _compute_sha256(DATASET_MANIFEST_PATH)
    if manifest_sha != DATASET_MANIFEST_SHA256:
        raise RuntimeError(
            f"Dataset V2 manifest SHA mismatch: expected {DATASET_MANIFEST_SHA256}, got {manifest_sha}"
        )


def _ensure_training_provenance() -> None:
    if not FORMAL_OUTPUT_DIR.exists():
        if any(path.exists() for path in RAW_EVIDENCE_PATHS.values()):
            raise RuntimeError("Formal evidence exists without provenance; refusing to assign new provenance retroactively.")
        FORMAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_provenance_row(_build_current_provenance_record())
        return

    if not FORMAL_PROVENANCE_PATH.exists():
        if any(path.exists() for path in RAW_EVIDENCE_PATHS.values()):
            raise RuntimeError("Formal evidence exists without provenance; refusing to assign new provenance retroactively.")
        _write_provenance_row(_build_current_provenance_record())
        return

    provenance_df = pd.read_csv(FORMAL_PROVENANCE_PATH)
    _validate_stored_provenance(provenance_df)


def _ensure_clean_namespace() -> None:
    if FORMAL_OUTPUT_DIR.exists():
        if any(FORMAL_OUTPUT_DIR.iterdir()):
            raise RuntimeError("Formal PathFormer namespace is not clean; refusing to overwrite completed evidence.")


def _cell_key_from_summary_row(row: pd.Series) -> tuple[str, str, int, int]:
    return (
        str(row["model"]),
        str(row["frequency_setting"]),
        int(row["horizon"]),
        int(row["seed"]),
    )


def _formal_cell_grid() -> set[tuple[str, str, int, int]]:
    return {
        ("adaptive_pathformer", str(cell.frequency), int(cell.horizon), int(SEED))
        for cell in _full_main_benchmark_cells()
    }


def _normalize_key_set(df: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    normalized: set[tuple[str, ...]] = set()
    for _, row in df[columns].iterrows():
        token: list[str] = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                token.append("NA")
            elif isinstance(value, (np.integer, int)):
                token.append(str(int(value)))
            elif isinstance(value, (np.floating, float)):
                token.append(str(float(value)))
            else:
                token.append(str(value))
        normalized.add(tuple(token))
    return normalized


def _extract_raw_cell_set(df: pd.DataFrame, *, kind: str) -> set[tuple[str, str, int, int]]:
    if df.empty:
        return set()

    if kind == "history":
        required = {"frequency_setting", "horizon", "seed"}
        if not required.issubset(df.columns):
            raise RuntimeError(f"History raw evidence is missing required cell columns: {sorted(required)}")
        cells: set[tuple[str, str, int, int]] = set()
        for _, row in df.iterrows():
            frequency = str(row["frequency_setting"])
            horizon = int(row["horizon"])
            seed = int(row["seed"])
            if frequency not in FREQUENCY_SETTINGS:
                raise RuntimeError(f"Forbidden history frequency in raw evidence: {frequency}")
            if horizon not in MAIN_HORIZONS:
                raise RuntimeError(f"Forbidden history horizon in raw evidence: {horizon}")
            if seed != SEED:
                raise RuntimeError(f"Forbidden history seed in raw evidence: {seed}")
            if "model" in df.columns and str(row["model"]) != "adaptive_pathformer":
                raise RuntimeError(f"History raw evidence contains wrong model: {row['model']}")
            cells.add(("adaptive_pathformer", frequency, horizon, seed))
        return cells

    required = {"model", "frequency_setting", "horizon", "seed"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{kind} raw evidence is missing required cell columns: {sorted(required)}")

    cells: set[tuple[str, str, int, int]] = set()
    for _, row in df.iterrows():
        model = str(row["model"])
        frequency = str(row["frequency_setting"])
        horizon = int(row["horizon"])
        seed = int(row["seed"])
        if model != "adaptive_pathformer":
            raise RuntimeError(f"Forbidden model in {kind} raw evidence: {model}")
        if frequency not in FREQUENCY_SETTINGS:
            raise RuntimeError(f"Forbidden {kind} frequency in raw evidence: {frequency}")
        if horizon not in MAIN_HORIZONS:
            raise RuntimeError(f"Forbidden {kind} horizon in raw evidence: {horizon}")
        if seed != SEED:
            raise RuntimeError(f"Forbidden {kind} seed in raw evidence: {seed}")
        cells.add((model, frequency, horizon, seed))
    return cells


def _assert_cell_not_completed(frequency: str, horizon: int, model: str = "adaptive_pathformer", seed: int = SEED) -> None:
    present_by_file: dict[str, bool] = {}
    for name, path in RAW_EVIDENCE_PATHS.items():
        if not path.exists():
            present_by_file[name] = False
            continue
        df = pd.read_csv(path)
        if name == "history":
            if {"frequency_setting", "horizon", "seed"}.issubset(df.columns):
                subset = df[
                    (df["frequency_setting"].astype(str) == str(frequency))
                    & (df["horizon"].astype(int) == int(horizon))
                    & (df["seed"].astype(int) == int(seed))
                ]
                present_by_file[name] = not subset.empty
            else:
                present_by_file[name] = False
        else:
            if {"model", "frequency_setting", "horizon", "seed"}.issubset(df.columns):
                subset = df[
                    (df["model"].astype(str) == model)
                    & (df["frequency_setting"].astype(str) == str(frequency))
                    & (df["horizon"].astype(int) == int(horizon))
                    & (df["seed"].astype(int) == int(seed))
                ]
                present_by_file[name] = not subset.empty
            else:
                present_by_file[name] = False

    if any(present_by_file.values()):
        if all(present_by_file.values()):
            raise RuntimeError(f"Duplicate completed formal cell detected for {model}/{frequency}/{horizon}/seed={seed}.")
        raise RuntimeError(
            f"Inconsistent partial evidence detected for {model}/{frequency}/{horizon}/seed={seed}; "
            "a formal cell appears in some raw files but not others."
        )


def _combined_partial_loader() -> dict[str, pd.DataFrame]:
    exists = {name: path.exists() for name, path in RAW_EVIDENCE_PATHS.items()}
    if not any(exists.values()):
        return {name: pd.DataFrame() for name in RAW_EVIDENCE_PATHS}
    if any((not exists[name]) for name in RAW_EVIDENCE_PATHS):
        missing = [name for name, present in exists.items() if not present]
        raise RuntimeError(f"Incomplete raw evidence set: missing {missing}.")
    return {name: pd.read_csv(path) for name, path in RAW_EVIDENCE_PATHS.items()}


def _combined_partial_validation() -> None:
    raw = _combined_partial_loader()
    if all(frame.empty for frame in raw.values()):
        return

    for name, frame in raw.items():
        if frame.empty:
            continue
        if name == "summary":
            if frame.duplicated(subset=["model", "frequency_setting", "horizon", "seed"]).any():
                raise RuntimeError("Duplicate summary keys detected in raw evidence.")
        elif name == "predictions":
            if frame.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]).any():
                raise RuntimeError("Duplicate prediction keys detected in raw evidence.")
        elif name == "rank_ic":
            if frame.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "anchor_date"]).any():
                raise RuntimeError("Duplicate Rank-IC keys detected in raw evidence.")
        elif name == "history":
            if frame.duplicated(subset=["frequency_setting", "horizon", "seed", "epoch"]).any():
                raise RuntimeError("Duplicate history keys detected in raw evidence.")
        elif name == "router":
            if frame.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]).any():
                raise RuntimeError("Duplicate router keys detected in raw evidence.")

    summary_cells = _extract_raw_cell_set(raw["summary"], kind="summary")
    pred_cells = _extract_raw_cell_set(raw["predictions"], kind="predictions")
    rank_cells = _extract_raw_cell_set(raw["rank_ic"], kind="rank_ic")
    history_cells = _extract_raw_cell_set(raw["history"], kind="history")
    router_cells = _extract_raw_cell_set(raw["router"], kind="router")

    cell_sets = {
        "summary": summary_cells,
        "predictions": pred_cells,
        "rank_ic": rank_cells,
        "history": history_cells,
        "router": router_cells,
    }
    first_set = next(iter(cell_sets.values()))
    for name, cell_set in cell_sets.items():
        if cell_set != first_set:
            raise RuntimeError(
                "Raw evidence cell sets are not exactly equal: "
                f"summary={summary_cells}, predictions={pred_cells}, rank_ic={rank_cells}, history={history_cells}, router={router_cells}"
            )

    allowed = _formal_cell_grid()
    for name, cell_set in cell_sets.items():
        if not cell_set.issubset(allowed):
            raise RuntimeError(f"Raw evidence contains forbidden cells in {name}: {sorted(cell_set - allowed)}")

    common = first_set
    for cell in sorted(common):
        model, frequency, horizon, seed = cell
        summary_df = raw["summary"][
            (raw["summary"]["model"].astype(str) == model)
            & (raw["summary"]["frequency_setting"].astype(str) == str(frequency))
            & (raw["summary"]["horizon"].astype(int) == int(horizon))
            & (raw["summary"]["seed"].astype(int) == int(seed))
        ].copy()
        pred_df = raw["predictions"][
            (raw["predictions"]["model"].astype(str) == model)
            & (raw["predictions"]["frequency_setting"].astype(str) == str(frequency))
            & (raw["predictions"]["horizon"].astype(int) == int(horizon))
            & (raw["predictions"]["seed"].astype(int) == int(seed))
        ].copy()
        rank_ic_df = raw["rank_ic"][
            (raw["rank_ic"]["model"].astype(str) == model)
            & (raw["rank_ic"]["frequency_setting"].astype(str) == str(frequency))
            & (raw["rank_ic"]["horizon"].astype(int) == int(horizon))
            & (raw["rank_ic"]["seed"].astype(int) == int(seed))
        ].copy()
        history_df = raw["history"][
            (raw["history"]["frequency_setting"].astype(str) == str(frequency))
            & (raw["history"]["horizon"].astype(int) == int(horizon))
            & (raw["history"]["seed"].astype(int) == int(seed))
        ].copy()
        router_df = raw["router"][
            (raw["router"]["model"].astype(str) == model)
            & (raw["router"]["frequency_setting"].astype(str) == str(frequency))
            & (raw["router"]["horizon"].astype(int) == int(horizon))
            & (raw["router"]["seed"].astype(int) == int(seed))
        ].copy()
        _validate_single_cell_evidence(summary_df, pred_df, rank_ic_df, history_df, router_df, frequency, horizon)


def _validate_single_cell_evidence(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    history_df: pd.DataFrame,
    router_df: pd.DataFrame,
    frequency: str,
    horizon: int,
) -> None:
    if len(summary_df) != 1:
        raise RuntimeError(f"Expected exactly 1 summary row for {frequency}/{horizon}, found {len(summary_df)}.")

    row = summary_df.iloc[0]
    if str(row["model"]) != "adaptive_pathformer":
        raise RuntimeError("Summary row model must be adaptive_pathformer.")
    if str(row["frequency_setting"]) != frequency:
        raise RuntimeError(f"Summary row frequency mismatch: expected {frequency}, got {row['frequency_setting']}.")
    if int(row["horizon"]) != int(horizon):
        raise RuntimeError(f"Summary row horizon mismatch: expected {horizon}, got {row['horizon']}.")
    if int(row["seed"]) != SEED:
        raise RuntimeError(f"Summary row seed mismatch: expected {SEED}, got {row['seed']}.")

    expected_params = {"daily_only": 56069, "weekly_only": 56069, "daily_weekly": 116297}
    if int(row["n_trainable_parameters"]) != expected_params[frequency]:
        raise RuntimeError(
            f"Unexpected trainable parameter count for {frequency}: expected {expected_params[frequency]}, got {row['n_trainable_parameters']}"
        )
    if int(row["best_epoch"]) < 1:
        raise RuntimeError(f"best_epoch must be >= 1 for {frequency}/{horizon}.")
    if int(row["epochs_trained"]) < int(row["best_epoch"]):
        raise RuntimeError(f"epochs_trained must be >= best_epoch for {frequency}/{horizon}.")
    if not np.isfinite(float(row["runtime_seconds"])) or float(row["runtime_seconds"]) < 0:
        raise RuntimeError(f"runtime_seconds must be finite and >= 0 for {frequency}/{horizon}.")
    if not np.isfinite(float(row["val_loss"])):
        raise RuntimeError(f"val_loss must be finite for {frequency}/{horizon}.")
    for metric in ["test_mse", "test_mae", "direction_accuracy", "pred_std", "true_std"]:
        if not np.isfinite(float(row[metric])):
            raise RuntimeError(f"Metric {metric} must be finite for {frequency}/{horizon}.")
    if "test_corr" in row and not pd.isna(row["test_corr"]) and not np.isfinite(float(row["test_corr"])):
        raise RuntimeError(f"test_corr must be finite or NaN for {frequency}/{horizon}.")
    for metric in ["mean_rank_ic", "median_rank_ic", "rank_ic_std", "positive_ic_ratio", "rank_icir"]:
        if metric in row and not pd.isna(row[metric]) and not np.isfinite(float(row[metric])):
            raise RuntimeError(f"Metric {metric} must be finite or NaN for {frequency}/{horizon}.")
    if int(row["n_valid_ic_dates"]) != int(rank_ic_df["rank_ic"].notna().sum()):
        raise RuntimeError(f"n_valid_ic_dates mismatch for {frequency}/{horizon}.")

    if len(pred_df) != 6647:
        raise RuntimeError(f"Prediction rows for {frequency}/{horizon} must be exactly 6647.")
    required_pred_cols = {"model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date", "y_true", "y_pred"}
    if not required_pred_cols.issubset(pred_df.columns):
        raise RuntimeError(f"Predictions missing required columns for {frequency}/{horizon}.")
    if not all(pred_df["model"].astype(str) == "adaptive_pathformer"):
        raise RuntimeError("Every prediction row must be model=adaptive_pathformer.")
    if not all(pred_df["frequency_setting"].astype(str) == frequency):
        raise RuntimeError(f"Prediction frequency mismatch for {frequency}/{horizon}.")
    if not all(pred_df["horizon"].astype(int) == int(horizon)):
        raise RuntimeError(f"Prediction horizon mismatch for {frequency}/{horizon}.")
    if not all(pred_df["seed"].astype(int) == SEED):
        raise RuntimeError(f"Prediction seed mismatch for {frequency}/{horizon}.")
    if pred_df["ticker"].nunique() != 17:
        raise RuntimeError(f"Prediction tickers must be 17 unique values for {frequency}/{horizon}.")
    if pred_df["anchor_date"].nunique() != 391:
        raise RuntimeError(f"Prediction anchor dates must be 391 unique values for {frequency}/{horizon}.")
    if pred_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]).any():
        raise RuntimeError(f"Prediction row duplicates detected for {frequency}/{horizon}.")
    if pred_df["y_true"].isna().any() or pred_df["y_pred"].isna().any():
        raise RuntimeError(f"Prediction values must be finite for {frequency}/{horizon}.")
    if not np.isfinite(pred_df["y_true"]).all() or not np.isfinite(pred_df["y_pred"]).all():
        raise RuntimeError(f"Prediction values must be finite for {frequency}/{horizon}.")
    if pred_df.groupby("anchor_date").size().nunique() != 1:
        raise RuntimeError(f"Prediction dates should be uniform-per-date length for {frequency}/{horizon}.")
    if pred_df.groupby("anchor_date").size().tolist() and any(size != 17 for size in pred_df.groupby("anchor_date").size().tolist()):
        raise RuntimeError(f"Every anchor date in predictions must have exactly 17 rows for {frequency}/{horizon}.")
    if pred_df.groupby("anchor_date")["ticker"].nunique().lt(17).any():
        raise RuntimeError(f"Every anchor date in predictions must contain all 17 unique tickers for {frequency}/{horizon}.")
    if pred_df.groupby("anchor_date")["ticker"].nunique().gt(17).any():
        raise RuntimeError(f"Every anchor date in predictions must not have more than 17 unique tickers for {frequency}/{horizon}.")
    if set(pred_df["ticker"].unique()) != set(FROZEN_TICKERS):
        raise RuntimeError(f"Prediction ticker set mismatch for {frequency}/{horizon}: {sorted(pred_df['ticker'].unique())}")

    if len(rank_ic_df) != 391:
        raise RuntimeError(f"Rank-IC rows for {frequency}/{horizon} must be exactly 391.")
    required_rank_cols = {"model", "frequency_setting", "horizon", "seed"}
    if not required_rank_cols.issubset(rank_ic_df.columns):
        raise RuntimeError(f"Rank-IC output missing required metadata columns for {frequency}/{horizon}.")
    if not all(rank_ic_df["model"].astype(str) == "adaptive_pathformer"):
        raise RuntimeError(f"Rank-IC model mismatch for {frequency}/{horizon}.")
    if not all(rank_ic_df["frequency_setting"].astype(str) == frequency):
        raise RuntimeError(f"Rank-IC frequency mismatch for {frequency}/{horizon}.")
    if not all(rank_ic_df["horizon"].astype(int) == int(horizon)):
        raise RuntimeError(f"Rank-IC horizon mismatch for {frequency}/{horizon}.")
    if not all(rank_ic_df["seed"].astype(int) == SEED):
        raise RuntimeError(f"Rank-IC seed mismatch for {frequency}/{horizon}.")
    if rank_ic_df["anchor_date"].nunique() != 391:
        raise RuntimeError(f"Rank-IC dates must be 391 unique values for {frequency}/{horizon}.")
    if rank_ic_df.duplicated(subset=["anchor_date"]).any():
        raise RuntimeError(f"Rank-IC duplicate date rows found for {frequency}/{horizon}.")
    if "n_tickers" in rank_ic_df.columns and not (rank_ic_df["n_tickers"] == 17).all():
        raise RuntimeError(f"Rank-IC n_tickers values must all equal 17 for {frequency}/{horizon}.")

    if len(history_df) < 1:
        raise RuntimeError(f"Training history is empty for {frequency}/{horizon}.")
    required_hist_cols = {"frequency_setting", "horizon", "seed", "epoch", "train_loss", "val_loss"}
    if not required_hist_cols.issubset(history_df.columns):
        raise RuntimeError(f"History missing required columns for {frequency}/{horizon}.")
    if not all(history_df["frequency_setting"].astype(str) == frequency):
        raise RuntimeError(f"History frequency mismatch for {frequency}/{horizon}.")
    if not all(history_df["horizon"].astype(int) == int(horizon)):
        raise RuntimeError(f"History horizon mismatch for {frequency}/{horizon}.")
    if not all(history_df["seed"].astype(int) == SEED):
        raise RuntimeError(f"History seed mismatch for {frequency}/{horizon}.")
    if history_df["epoch"].duplicated().any():
        raise RuntimeError(f"History epochs must be unique for {frequency}/{horizon}.")
    if not np.isfinite(history_df["train_loss"]).all() or not np.isfinite(history_df["val_loss"]).all():
        raise RuntimeError(f"History losses must be finite for {frequency}/{horizon}.")
    summary_epochs = int(row["epochs_trained"])
    if summary_epochs != len(history_df):
        raise RuntimeError(f"epochs_trained must equal history length for {frequency}/{horizon}.")
    if summary_epochs != int(history_df["epoch"].max()):
        raise RuntimeError(f"epochs_trained must equal max(epoch) for {frequency}/{horizon}.")
    expected_epochs = list(range(1, summary_epochs + 1))
    actual_epochs = sorted(history_df["epoch"].astype(int).unique().tolist())
    if actual_epochs != expected_epochs:
        raise RuntimeError(f"History epochs must be exactly 1..epochs_trained for {frequency}/{horizon}: expected {expected_epochs}, got {actual_epochs}.")
    min_val = float(history_df["val_loss"].min())
    if not np.isclose(float(row["val_loss"]), min_val, rtol=1e-6, atol=1e-8):
        raise RuntimeError(f"Summary val_loss must equal minimum history val_loss for {frequency}/{horizon}.")
    best_epoch_set = set(history_df.loc[np.isclose(history_df["val_loss"], min_val, rtol=1e-6, atol=1e-8), "epoch"].astype(int).tolist())
    if int(row["best_epoch"]) not in best_epoch_set:
        raise RuntimeError(f"Summary best_epoch must match an epoch with minimum val_loss for {frequency}/{horizon}.")

    if len(router_df) != 6647:
        raise RuntimeError(f"Router rows for {frequency}/{horizon} must be exactly 6647.")
    required_router_key = {"model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"}
    if not required_router_key.issubset(router_df.columns):
        raise RuntimeError(f"Router missing required key columns for {frequency}/{horizon}.")
    if router_df.duplicated(subset=["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]).any():
        raise RuntimeError(f"Router duplicate keys detected for {frequency}/{horizon}.")

    pred_key_cols = pred_df[["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]].copy()
    pred_key_cols["anchor_date"] = pred_key_cols["anchor_date"].apply(_canonical_anchor_date)
    router_key_cols = router_df[["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"]].copy()
    router_key_cols["anchor_date"] = router_key_cols["anchor_date"].apply(_canonical_anchor_date)

    prediction_key_set = _normalize_key_set(
        pred_key_cols,
        ["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"],
    )
    router_key_set = _normalize_key_set(
        router_key_cols,
        ["model", "frequency_setting", "horizon", "seed", "ticker", "anchor_date"],
    )
    if prediction_key_set != router_key_set:
        raise RuntimeError(f"Prediction and router key sets must match exactly for {frequency}/{horizon}.")

    daily_cols = ["daily_w_5", "daily_w_10", "daily_w_20", "daily_w_30"]
    weekly_cols = ["weekly_w_2", "weekly_w_4", "weekly_w_8", "weekly_w_13"]
    all_router_cols = daily_cols + weekly_cols
    for col in all_router_cols:
        if col not in router_df.columns:
            raise RuntimeError(f"Router column {col} missing for {frequency}/{horizon}.")

    if frequency == "daily_only":
        for col in daily_cols:
            if not np.isfinite(router_df[col].astype(float)).all():
                raise RuntimeError(f"Daily router weights must be finite for {frequency}/{horizon}.")
            if (router_df[col].astype(float) < 0).any():
                raise RuntimeError(f"Daily router weights must be >= 0 for {frequency}/{horizon}.")
        for col in weekly_cols:
            if not router_df[col].isna().all():
                raise RuntimeError(f"Weekly router columns must all be NaN for daily_only {frequency}/{horizon}.")
        daily_row_sums = router_df[daily_cols].sum(axis=1)
        if not np.allclose(daily_row_sums.to_numpy(), np.ones(len(router_df)), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Daily router row sums must be ~1 for {frequency}/{horizon}.")
    elif frequency == "weekly_only":
        for col in weekly_cols:
            if not np.isfinite(router_df[col].astype(float)).all():
                raise RuntimeError(f"Weekly router weights must be finite for {frequency}/{horizon}.")
            if (router_df[col].astype(float) < 0).any():
                raise RuntimeError(f"Weekly router weights must be >= 0 for {frequency}/{horizon}.")
        for col in daily_cols:
            if not router_df[col].isna().all():
                raise RuntimeError(f"Daily router columns must all be NaN for weekly_only {frequency}/{horizon}.")
        weekly_row_sums = router_df[weekly_cols].sum(axis=1)
        if not np.allclose(weekly_row_sums.to_numpy(), np.ones(len(router_df)), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Weekly router row sums must be ~1 for {frequency}/{horizon}.")
    else:
        for col in all_router_cols:
            if not np.isfinite(router_df[col].astype(float)).all():
                raise RuntimeError(f"Router weights must be finite for {frequency}/{horizon}.")
            if (router_df[col].astype(float) < 0).any():
                raise RuntimeError(f"Router weights must be >= 0 for {frequency}/{horizon}.")
        daily_row_sums = router_df[daily_cols].sum(axis=1)
        weekly_row_sums = router_df[weekly_cols].sum(axis=1)
        if not np.allclose(daily_row_sums.to_numpy(), np.ones(len(router_df)), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Daily router row sums must be ~1 for {frequency}/{horizon}.")
        if not np.allclose(weekly_row_sums.to_numpy(), np.ones(len(router_df)), rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Weekly router row sums must be ~1 for {frequency}/{horizon}.")


def _validate_evidence_integrity(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    history_df: pd.DataFrame,
    router_df: pd.DataFrame,
    frequency: str,
    horizon: int,
) -> None:
    _validate_single_cell_evidence(summary_df, pred_df, rank_ic_df, history_df, router_df, frequency, horizon)


def _write_formal_outputs_for_cell(cell: FormalCell, evidence: dict) -> None:
    FORMAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_key = (
        str(evidence["summary"].iloc[0]["model"]),
        str(evidence["summary"].iloc[0]["frequency_setting"]),
        int(evidence["summary"].iloc[0]["horizon"]),
        int(evidence["summary"].iloc[0]["seed"]),
    )

    for file_name, path in RAW_EVIDENCE_PATHS.items():
        df = evidence[file_name]
        if path.exists():
            existing = pd.read_csv(path)
            if file_name == "history":
                required = {"frequency_setting", "horizon", "seed"}
                if not required.issubset(existing.columns):
                    raise RuntimeError(f"History raw evidence missing required columns for append: {sorted(required)}")
                mask = (
                    (existing["frequency_setting"].astype(str) == str(cell.frequency))
                    & (existing["horizon"].astype(int) == int(cell.horizon))
                    & (existing["seed"].astype(int) == int(SEED))
                )
                if mask.any():
                    raise RuntimeError(
                        f"Duplicate formal history cell detected for {cell.frequency}/{cell.horizon}/seed={SEED}."
                    )
            else:
                required = {"model", "frequency_setting", "horizon", "seed"}
                if not required.issubset(existing.columns):
                    raise RuntimeError(f"Raw evidence file {path} is missing required metadata columns; refusing to append.")
                mask = (
                    (existing["model"].astype(str) == summary_key[0])
                    & (existing["frequency_setting"].astype(str) == summary_key[1])
                    & (existing["horizon"].astype(int) == summary_key[2])
                    & (existing["seed"].astype(int) == summary_key[3])
                )
                if mask.any():
                    raise RuntimeError(f"Duplicate formal cell evidence detected while appending raw evidence for {summary_key} to {file_name}.")

        if path.exists():
            existing = pd.read_csv(path)
            pd.concat([existing, df], ignore_index=True).to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)


def _full_main_benchmark_cells() -> list[FormalCell]:
    return [
        FormalCell("daily_only", 5, "adaptive"),
        FormalCell("daily_only", 10, "adaptive"),
        FormalCell("daily_only", 20, "adaptive"),
        FormalCell("weekly_only", 5, "adaptive"),
        FormalCell("weekly_only", 10, "adaptive"),
        FormalCell("weekly_only", 20, "adaptive"),
        FormalCell("daily_weekly", 5, "adaptive"),
        FormalCell("daily_weekly", 10, "adaptive"),
        FormalCell("daily_weekly", 20, "adaptive"),
    ]


def _build_training_loaders(frequency: str, horizon: int, ds: Any, batch_size: int = BATCH_SIZE):
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

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def _run_occupancy_weighted_pooling_validation() -> None:
    """Synthetic validation proving partial patches are not equal-weighted."""
    patch_tokens = torch.tensor([
        [[1.0, 1.0], [3.0, 3.0]],
        [[1.0, 1.0], [3.0, 3.0]],
    ], dtype=torch.float32)
    counts = torch.tensor([
        [8, 2],
        [8, 8],
    ], dtype=torch.float32)
    output = occupancy_weighted_patch_mean(patch_tokens, counts)
    expected_weighted = torch.tensor([[1.4, 1.4], [2.0, 2.0]], dtype=torch.float32)
    if not torch.allclose(output[0], expected_weighted[0], atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Weighted occupancy output mismatch: {output[0]} vs {expected_weighted[0]}")
    if not torch.allclose(output[1], expected_weighted[1], atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Equal-count occupancy output mismatch: {output[1]} vs {expected_weighted[1]}")
    if torch.allclose(output[0], torch.tensor([2.0, 2.0], dtype=torch.float32), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Occupancy-weighted mean was incorrectly reduced to the equal-patch mean.")
    if not torch.allclose(output[1], torch.tensor([2.0, 2.0], dtype=torch.float32), atol=1e-6, rtol=1e-6):
        raise RuntimeError("Equal full-patch counts should reduce to the ordinary equal-patch mean.")


def _run_smoke_validation() -> None:
    """Architecture-only validation. Writes nothing.

    This is the default no-argument execution mode.
    """
    namespace_exists_before = FORMAL_OUTPUT_DIR.exists()
    namespace_children_before = set(p.name for p in FORMAL_OUTPUT_DIR.iterdir()) if namespace_exists_before else set()

    set_seed(SEED)
    ds = load_frozen_panel()
    _formal_check_dataset_contract(ds)

    # Daily patch geometry checks.
    for p, expected_patches in [(5, 18), (10, 9), (20, 5), (30, 3)]:
        T = 90
        padded_T = int(math.ceil(T / p) * p)
        n_patches = padded_T // p
        assert n_patches == expected_patches, (p, n_patches, expected_patches)
    for p, expected_pad in [(20, 10), (8, 6)]:
        T = 90 if p == 20 else 26
        padded_T = int(math.ceil(T / p) * p)
        assert padded_T - T == expected_pad

    # Weekly patch geometry checks.
    for p, expected_patches in [(2, 13), (4, 7), (8, 4), (13, 2)]:
        T = 26
        padded_T = int(math.ceil(T / p) * p)
        n_patches = padded_T // p
        assert n_patches == expected_patches, (p, n_patches, expected_patches)

    # Partial patch behavior check.
    _run_occupancy_weighted_pooling_validation()

    # Batch independence check: sample A router weights must be unchanged if sample B changes.
    Xd_norm, Xw_norm = apply_normalizer(ds, fit_train_only_normalizer(ds), "daily_weekly")
    idx = np.where(ds.split == "train")[0][:4]
    x_d = torch.from_numpy(Xd_norm[idx])
    x_w = torch.from_numpy(Xw_norm[idx])
    model = _make_model("daily_weekly", "adaptive")
    model.eval()
    with torch.no_grad():
        daily_w1, weekly_w1 = model.get_router_weights(x_d, x_w)
        x_d_perturbed = x_d.clone()
        x_d_perturbed[1] = x_d_perturbed[1] * 0.0 + 99.0
        daily_w2, weekly_w2 = model.get_router_weights(x_d_perturbed, x_w)
        assert torch.allclose(daily_w1[0], daily_w2[0], atol=1e-6, rtol=1e-6)
        assert daily_w1.shape == (4, 4)
        assert torch.isfinite(daily_w1).all()
        assert (daily_w1 >= 0).all()
        assert torch.allclose(daily_w1.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6)
        assert weekly_w1.shape == (4, 4)
        assert torch.isfinite(weekly_w1).all()
        assert (weekly_w1 >= 0).all()
        assert torch.allclose(weekly_w1.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6)

        x_w_perturbed = x_w.clone()
        x_w_perturbed[1] = x_w_perturbed[1] * 0.0 + 99.0
        daily_w3, weekly_w3 = model.get_router_weights(x_d, x_w_perturbed)
        assert torch.allclose(weekly_w1[0], weekly_w3[0], atol=1e-6, rtol=1e-6)

    # Forward shape checks.
    for frequency in ["daily_only", "weekly_only", "daily_weekly"]:
        if frequency == "daily_only":
            x = torch.from_numpy(apply_normalizer(ds, fit_train_only_normalizer(ds), "daily_only")[np.where(ds.split == "train")[0][:4]])
            y = torch.from_numpy(ds.y[5][np.where(ds.split == "train")[0][:4]])
            model = _make_model("daily_only", "adaptive")
            pred = model(x)
        elif frequency == "weekly_only":
            x = torch.from_numpy(apply_normalizer(ds, fit_train_only_normalizer(ds), "weekly_only")[np.where(ds.split == "train")[0][:4]])
            y = torch.from_numpy(ds.y[5][np.where(ds.split == "train")[0][:4]])
            model = _make_model("weekly_only", "adaptive")
            pred = model(x)
        else:
            Xd_norm, Xw_norm = apply_normalizer(ds, fit_train_only_normalizer(ds), "daily_weekly")
            idx = np.where(ds.split == "train")[0][:4]
            x_d = torch.from_numpy(Xd_norm[idx])
            x_w = torch.from_numpy(Xw_norm[idx])
            y = torch.from_numpy(ds.y[5][idx])
            model = _make_model("daily_weekly", "adaptive")
            pred = model(x_d, x_w)
        assert pred.shape == y.shape, (frequency, pred.shape, y.shape)
        assert torch.isfinite(pred).all()
        loss = nn.HuberLoss(delta=1.0)(pred, y)
        model.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient: {name}"

    # Parameter count check.
    counts = {
        ("daily_only", "adaptive"): 56069,
        ("weekly_only", "adaptive"): 56069,
        ("daily_weekly", "adaptive"): 116297,
    }
    for (freq, scale_mode), expected in counts.items():
        model = _make_model(freq, scale_mode)
        actual = count_trainable_parameters(model)
        assert actual == expected, (freq, scale_mode, actual, expected)

    namespace_exists_after = FORMAL_OUTPUT_DIR.exists()
    namespace_children_after = set(p.name for p in FORMAL_OUTPUT_DIR.iterdir()) if namespace_exists_after else set()
    if namespace_exists_before != namespace_exists_after or namespace_children_before != namespace_children_after:
        raise RuntimeError("Smoke validation must leave the formal namespace unchanged.")

    print("CPU PATHFORMER MAIN V2 SMOKE VALIDATION PASSED")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_anchor_date(value) -> str:
    """Single canonical formal anchor_date representation: YYYY-MM-DD."""
    if isinstance(value, np.str_):
        value = str(value)

    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _batched_test_inference(model: nn.Module, test_loader: DataLoader, frequency: str, device: torch.device):
    model.eval()
    pred_chunks = []
    daily_chunks = []
    weekly_chunks = []
    with torch.no_grad():
        for batch in test_loader:
            if frequency == "daily_weekly":
                x_daily, x_weekly, _ = batch
                x_daily = x_daily.to(device)
                x_weekly = x_weekly.to(device)
                pred, daily_w, weekly_w = model.forward_with_router(x_daily, x_weekly)
                pred_chunks.append(pred.detach().cpu())
                daily_chunks.append(daily_w.detach().cpu())
                weekly_chunks.append(weekly_w.detach().cpu())
            else:
                x_batch, _ = batch
                x_batch = x_batch.to(device)
                if frequency == "daily_only":
                    pred, daily_w = model.forward_with_router(x_batch)
                    weekly_w = None
                else:
                    pred, weekly_w = model.forward_with_router(x_batch)
                    daily_w = None
                pred_chunks.append(pred.detach().cpu())
                if daily_w is not None:
                    daily_chunks.append(daily_w.detach().cpu())
                if weekly_w is not None:
                    weekly_chunks.append(weekly_w.detach().cpu())

    if len(pred_chunks) == 0:
        raise RuntimeError("No test batches were processed during formal evaluation.")

    pred_tensor = torch.cat(pred_chunks, dim=0)
    if pred_tensor.shape[0] != 6647:
        raise RuntimeError(f"Formal test prediction count mismatch: expected 6647, got {pred_tensor.shape[0]}")

    if daily_chunks:
        daily_tensor = torch.cat(daily_chunks, dim=0)
        if daily_tensor.shape[0] != 6647:
            raise RuntimeError(f"Formal test daily router count mismatch: expected 6647, got {daily_tensor.shape[0]}")
    else:
        daily_tensor = None

    if weekly_chunks:
        weekly_tensor = torch.cat(weekly_chunks, dim=0)
        if weekly_tensor.shape[0] != 6647:
            raise RuntimeError(f"Formal test weekly router count mismatch: expected 6647, got {weekly_tensor.shape[0]}")
    else:
        weekly_tensor = None

    return pred_tensor, daily_tensor, weekly_tensor


def _validate_main_bench_for_frequency(frequency: str, horizon: int) -> None:
    if frequency not in FREQUENCY_SETTINGS:
        raise ValueError(f"Unsupported frequency={frequency}")
    if horizon not in MAIN_HORIZONS:
        raise ValueError(f"Unsupported horizon={horizon}")


def _capture_first_train_batch_trace(
    qualification_trace: dict | None,
    frequency: str,
    epoch: int,
    batch_index: int,
    y_batch: torch.Tensor,
    x_batch: torch.Tensor | None = None,
    x_daily: torch.Tensor | None = None,
    x_weekly: torch.Tensor | None = None,
) -> None:
    """Read-only hash capture of the real epoch-1/batch-0 tensors; never replayed or skipped."""
    if qualification_trace is None or epoch != 1 or batch_index != 0:
        return
    if "first_train_batch_target_sha256" in qualification_trace:
        return
    if frequency == "daily_weekly":
        qualification_trace["first_train_batch_daily_input_sha256"] = _tensor_sha256(x_daily)
        qualification_trace["first_train_batch_weekly_input_sha256"] = _tensor_sha256(x_weekly)
    else:
        qualification_trace["first_train_batch_input_sha256"] = _tensor_sha256(x_batch)
    qualification_trace["first_train_batch_target_sha256"] = _tensor_sha256(y_batch)


def train_one_cell(
    frequency: str,
    horizon: int,
    scale_mode: str = "adaptive",
    qualification_trace: dict | None = None,
) -> dict:
    """Single-cell training stub.

    This function exists to formalize the contract; it does not execute in the
    default path. The actual benchmark is only launched by an explicit CLI call.

    qualification_trace is an optional external mutable dict used only for
    deterministic CPU-qualification fingerprinting (--cpu-validation). When it
    is None, training is mathematically and operationally identical to formal
    training: no extra RNG consumption, no extra forward/backward, no change to
    DataLoader iteration, optimizer state, or checkpoint/stopping behavior.
    """
    _validate_main_bench_for_frequency(frequency, horizon)
    if scale_mode != MAIN_BENCHMARK_MODE:
        raise ValueError(f"Formal MAIN V2 benchmark only permits scale_mode='adaptive'; got {scale_mode}.")

    ds = load_frozen_panel()
    _formal_check_dataset_contract(ds)
    set_seed(SEED)
    train_loader, val_loader, test_loader = _build_training_loaders(frequency, horizon, ds, batch_size=BATCH_SIZE)
    model = _make_model(frequency, scale_mode)
    model = model.to(_resolve_device())
    if qualification_trace is not None:
        qualification_trace["initial_model_parameter_sha256"] = _state_dict_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    min_val_loss = float("inf")
    best_epoch = 0
    history = []
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        for batch_index, batch in enumerate(train_loader):
            if frequency == "daily_weekly":
                x_daily, x_weekly, y_batch = batch
                x_daily = x_daily.to(_resolve_device())
                x_weekly = x_weekly.to(_resolve_device())
                y_batch = y_batch.to(_resolve_device())
                _capture_first_train_batch_trace(
                    qualification_trace, frequency, epoch, batch_index, y_batch, x_daily=x_daily, x_weekly=x_weekly
                )
                pred = model(x_daily, x_weekly)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(_resolve_device())
                y_batch = y_batch.to(_resolve_device())
                _capture_first_train_batch_trace(
                    qualification_trace, frequency, epoch, batch_index, y_batch, x_batch=x_batch
                )
                pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {frequency}/{horizon}, epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Non-finite gradient for {name} at {frequency}/{horizon}, epoch={epoch}")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            train_loss_total += loss.item() * y_batch.shape[0]

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if frequency == "daily_weekly":
                    x_daily, x_weekly, y_batch = batch
                    x_daily = x_daily.to(_resolve_device())
                    x_weekly = x_weekly.to(_resolve_device())
                    y_batch = y_batch.to(_resolve_device())
                    pred = model(x_daily, x_weekly)
                else:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(_resolve_device())
                    y_batch = y_batch.to(_resolve_device())
                    pred = model(x_batch)
                val_loss = loss_fn(pred, y_batch)
                if not torch.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite val loss at {frequency}/{horizon}, epoch={epoch}")
                val_loss_total += val_loss.item() * y_batch.shape[0]

        train_loss = train_loss_total / max(len(train_loader.dataset), 1)
        val_loss = val_loss_total / max(len(val_loader.dataset), 1)
        history.append({
            "frequency_setting": frequency,
            "horizon": int(horizon),
            "seed": SEED,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if val_loss < min_val_loss - 1e-12:
            min_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    if qualification_trace is not None:
        qualification_trace["final_restored_parameter_sha256"] = _state_dict_sha256(model)
    model.eval()
    model.best_epoch = best_epoch
    model.epochs_trained = len(history)
    model.best_val_loss = min_val_loss
    model.training_history = history
    runtime_seconds = time.time() - start_time
    if qualification_trace is not None:
        qualification_trace["epochs_trained"] = int(model.epochs_trained)
        qualification_trace["best_epoch"] = int(best_epoch)
        qualification_trace["best_val_loss"] = float(min_val_loss)
        qualification_trace["history"] = [dict(row) for row in history]
        qualification_trace["trajectory_sha256"] = _trajectory_sha256(history)

    # Evaluate on formal test set.
    test_mask = ds.split == "test"
    ticker = ds.ticker[test_mask]
    anchor_date = ds.anchor_date[test_mask]
    anchor_date_canonical = [_canonical_anchor_date(v) for v in anchor_date]
    y_true = ds.y[horizon][test_mask]
    pred_tensor, daily_router_tensor, weekly_router_tensor = _batched_test_inference(model, test_loader, frequency, _resolve_device())
    pred_np = pred_tensor.numpy().astype(np.float32)

    metrics = evaluate_predictions(y_true, pred_np, ticker, anchor_date)
    rank_ic_df = rank_ic_by_date(y_true, pred_np, ticker, anchor_date)
    rank_ic_df["frequency_setting"] = frequency
    rank_ic_df["horizon"] = int(horizon)
    rank_ic_df["seed"] = SEED
    rank_ic_df["model"] = "adaptive_pathformer"
    rank_ic_df["anchor_date"] = rank_ic_df["anchor_date"].apply(_canonical_anchor_date)
    mean_rank_ic = float(np.nanmean(rank_ic_df["rank_ic"].dropna().to_numpy())) if len(rank_ic_df["rank_ic"].dropna()) else float("nan")
    rank_ic_std = float(np.nanstd(rank_ic_df["rank_ic"].dropna().to_numpy())) if len(rank_ic_df["rank_ic"].dropna()) else float("nan")
    if np.isfinite(rank_ic_std) and rank_ic_std > 0:
        rank_icir = mean_rank_ic / rank_ic_std
    else:
        rank_icir = float("nan")

    summary = {
        "model": "adaptive_pathformer",
        "frequency_setting": frequency,
        "horizon": int(horizon),
        "seed": SEED,
        "n_trainable_parameters": count_trainable_parameters(model),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(model.epochs_trained),
        "runtime_seconds": float(runtime_seconds),
        "val_loss": float(model.best_val_loss),
        "test_mse": float(metrics["pooled_mse"]),
        "test_mae": float(metrics["pooled_mae"]),
        "test_corr": float(metrics["pooled_corr"]),
        "ticker_avg_mse": float(metrics["ticker_avg_mse"]),
        "ticker_avg_mae": float(metrics["ticker_avg_mae"]),
        "ticker_avg_corr": float(metrics["ticker_avg_corr"]),
        "mean_rank_ic": mean_rank_ic,
        "median_rank_ic": float(np.nanmedian(rank_ic_df["rank_ic"].dropna().to_numpy())) if len(rank_ic_df["rank_ic"].dropna()) else float("nan"),
        "rank_ic_std": rank_ic_std,
        "positive_ic_ratio": float(np.mean(rank_ic_df["rank_ic"].dropna().to_numpy() > 0)) if len(rank_ic_df["rank_ic"].dropna()) else float("nan"),
        "rank_icir": float(rank_icir),
        "n_valid_ic_dates": int(rank_ic_df["rank_ic"].notna().sum()),
        "direction_accuracy": float(metrics["pooled_da"]),
        "pred_std": float(metrics["pooled_pred_std"]),
        "true_std": float(metrics["pooled_true_std"]),
        "pred_std_true_std_ratio": float(metrics["pooled_pred_std_over_true_std"]),
    }

    prediction_df = pd.DataFrame({
        "model": "adaptive_pathformer",
        "frequency_setting": frequency,
        "horizon": int(horizon),
        "seed": SEED,
        "ticker": ticker,
        "anchor_date": anchor_date_canonical,
        "y_true": y_true.astype(np.float32),
        "y_pred": pred_np.astype(np.float32),
    })

    router_rows = []
    if frequency == "daily_only":
        if daily_router_tensor is None or daily_router_tensor.shape != (6647, 4):
            raise RuntimeError(f"daily_only router tensor mismatch: got {None if daily_router_tensor is None else tuple(daily_router_tensor.shape)}")
        if weekly_router_tensor is not None:
            raise RuntimeError(f"daily_only expected weekly router tensor to be None, got shape {tuple(weekly_router_tensor.shape)}")
        daily_router_np = daily_router_tensor.numpy()
        for i in range(len(ticker)):
            row = {
                "model": "adaptive_pathformer",
                "frequency_setting": frequency,
                "horizon": int(horizon),
                "seed": SEED,
                "ticker": ticker[i],
                "anchor_date": anchor_date_canonical[i],
                "daily_w_5": float(daily_router_np[i, 0]) if daily_router_np.shape[1] > 0 else float("nan"),
                "daily_w_10": float(daily_router_np[i, 1]) if daily_router_np.shape[1] > 1 else float("nan"),
                "daily_w_20": float(daily_router_np[i, 2]) if daily_router_np.shape[1] > 2 else float("nan"),
                "daily_w_30": float(daily_router_np[i, 3]) if daily_router_np.shape[1] > 3 else float("nan"),
                "weekly_w_2": float("nan"),
                "weekly_w_4": float("nan"),
                "weekly_w_8": float("nan"),
                "weekly_w_13": float("nan"),
            }
            router_rows.append(row)
    elif frequency == "weekly_only":
        if weekly_router_tensor is None or weekly_router_tensor.shape != (6647, 4):
            raise RuntimeError(f"weekly_only router tensor mismatch: got {None if weekly_router_tensor is None else tuple(weekly_router_tensor.shape)}")
        if daily_router_tensor is not None:
            raise RuntimeError(f"weekly_only expected daily router tensor to be None, got shape {tuple(daily_router_tensor.shape)}")
        weekly_router_np = weekly_router_tensor.numpy()
        for i in range(len(ticker)):
            row = {
                "model": "adaptive_pathformer",
                "frequency_setting": frequency,
                "horizon": int(horizon),
                "seed": SEED,
                "ticker": ticker[i],
                "anchor_date": anchor_date_canonical[i],
                "daily_w_5": float("nan"),
                "daily_w_10": float("nan"),
                "daily_w_20": float("nan"),
                "daily_w_30": float("nan"),
                "weekly_w_2": float(weekly_router_np[i, 0]) if weekly_router_np.shape[1] > 0 else float("nan"),
                "weekly_w_4": float(weekly_router_np[i, 1]) if weekly_router_np.shape[1] > 1 else float("nan"),
                "weekly_w_8": float(weekly_router_np[i, 2]) if weekly_router_np.shape[1] > 2 else float("nan"),
                "weekly_w_13": float(weekly_router_np[i, 3]) if weekly_router_np.shape[1] > 3 else float("nan"),
            }
            router_rows.append(row)
    else:
        if daily_router_tensor is None or weekly_router_tensor is None:
            raise RuntimeError("daily_weekly evaluation requires both daily and weekly router tensors.")
        if daily_router_tensor.shape != (6647, 4):
            raise RuntimeError(f"daily_weekly daily router tensor mismatch: got {tuple(daily_router_tensor.shape)}")
        if weekly_router_tensor.shape != (6647, 4):
            raise RuntimeError(f"daily_weekly weekly router tensor mismatch: got {tuple(weekly_router_tensor.shape)}")
        daily_router_np = daily_router_tensor.numpy()
        weekly_router_np = weekly_router_tensor.numpy()
        for i in range(len(ticker)):
            row = {
                "model": "adaptive_pathformer",
                "frequency_setting": frequency,
                "horizon": int(horizon),
                "seed": SEED,
                "ticker": ticker[i],
                "anchor_date": anchor_date_canonical[i],
                "daily_w_5": float(daily_router_np[i, 0]) if daily_router_np.shape[1] > 0 else float("nan"),
                "daily_w_10": float(daily_router_np[i, 1]) if daily_router_np.shape[1] > 1 else float("nan"),
                "daily_w_20": float(daily_router_np[i, 2]) if daily_router_np.shape[1] > 2 else float("nan"),
                "daily_w_30": float(daily_router_np[i, 3]) if daily_router_np.shape[1] > 3 else float("nan"),
                "weekly_w_2": float(weekly_router_np[i, 0]) if weekly_router_np.shape[1] > 0 else float("nan"),
                "weekly_w_4": float(weekly_router_np[i, 1]) if weekly_router_np.shape[1] > 1 else float("nan"),
                "weekly_w_8": float(weekly_router_np[i, 2]) if weekly_router_np.shape[1] > 2 else float("nan"),
                "weekly_w_13": float(weekly_router_np[i, 3]) if weekly_router_np.shape[1] > 3 else float("nan"),
            }
            router_rows.append(row)

    summary_df = pd.DataFrame([summary])
    history_df = pd.DataFrame(history)
    router_df = pd.DataFrame(router_rows)
    rank_ic_df = rank_ic_df.reset_index(drop=True)
    return {
        "summary": summary_df,
        "predictions": prediction_df,
        "rank_ic": rank_ic_df,
        "history": history_df,
        "router": router_df,
    }


def _full_benchmark_run() -> None:
    """Formal benchmark runner for the 9 adaptive-router MAIN V2 cells.

    This is intentionally separated from the historical development script and is the
    only supported formal benchmark path for this file.
    """
    _ensure_clean_namespace()
    ds_preflight = load_frozen_panel()
    _formal_check_dataset_contract(ds_preflight)
    _ensure_training_provenance()
    cells = _full_main_benchmark_cells()

    for cell in cells:
        _combined_partial_validation()
        _assert_cell_not_completed(cell.frequency, cell.horizon, model="adaptive_pathformer", seed=SEED)
        evidence = train_one_cell(cell.frequency, cell.horizon, scale_mode=cell.scale_mode)
        _validate_evidence_integrity(
            evidence["summary"],
            evidence["predictions"],
            evidence["rank_ic"],
            evidence["history"],
            evidence["router"],
            cell.frequency,
            cell.horizon,
        )
        _write_formal_outputs_for_cell(cell, evidence)
        _combined_partial_validation()

    if any(path.exists() for path in RAW_EVIDENCE_PATHS.values()):
        raw = _combined_partial_loader()
        if not raw["summary"].empty:
            completed_cells = set(
                (str(row["model"]), str(row["frequency_setting"]), int(row["horizon"]), int(row["seed"]))
                for _, row in raw["summary"].iterrows()
            )
            if len(completed_cells) != 9:
                raise RuntimeError(f"Expected 9 completed formal cells after raw append, found {len(completed_cells)}.")


def _safe_pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a, ddof=0) <= 1e-12 or np.std(b, ddof=0) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nan_aware_isclose(a: float, b: float, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan and b_nan:
        return True
    if a_nan != b_nan:
        return False
    return bool(np.isclose(a, b, atol=atol, rtol=rtol))


def _reconstruct_cell_summary_metrics(cell_pred_df: pd.DataFrame) -> dict:
    """Independent metric reconstruction, deliberately bypassing evaluate_predictions()."""
    y_true = cell_pred_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = cell_pred_df["y_pred"].to_numpy(dtype=np.float64)

    pooled_mse = float(np.mean((y_pred - y_true) ** 2))
    pooled_mae = float(np.mean(np.abs(y_pred - y_true)))
    pooled_corr = _safe_pearson_corr(y_pred, y_true)
    direction_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    pred_std = float(np.std(y_pred, ddof=0))
    true_std = float(np.std(y_true, ddof=0))
    pred_std_true_std_ratio = float(pred_std / true_std) if true_std > 1e-12 else float("nan")

    ticker_mse_list = []
    ticker_mae_list = []
    ticker_corr_list = []
    for _, group in cell_pred_df.groupby("ticker"):
        g_true = group["y_true"].to_numpy(dtype=np.float64)
        g_pred = group["y_pred"].to_numpy(dtype=np.float64)
        ticker_mse_list.append(float(np.mean((g_pred - g_true) ** 2)))
        ticker_mae_list.append(float(np.mean(np.abs(g_pred - g_true))))
        ticker_corr_list.append(_safe_pearson_corr(g_pred, g_true))

    return {
        "test_mse": pooled_mse,
        "test_mae": pooled_mae,
        "test_corr": pooled_corr,
        "direction_accuracy": direction_accuracy,
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_std_true_std_ratio": pred_std_true_std_ratio,
        "ticker_avg_mse": float(np.mean(ticker_mse_list)),
        "ticker_avg_mae": float(np.mean(ticker_mae_list)),
        "ticker_avg_corr": float(np.nanmean(ticker_corr_list)),
    }


def _reconstruct_cell_rank_ic(cell_pred_df: pd.DataFrame) -> pd.DataFrame:
    """Independent datewise Rank-IC reconstruction: pandas rank() + Pearson, no scipy spearmanr."""
    rows = []
    for anchor_date_value, group in cell_pred_df.groupby("anchor_date"):
        if len(group) != 17 or group["ticker"].nunique() != 17:
            raise RuntimeError(f"Reconstructed Rank-IC requires exactly 17 unique tickers for anchor_date={anchor_date_value}.")
        pred_rank = group["y_pred"].rank(method="average").to_numpy(dtype=np.float64)
        true_rank = group["y_true"].rank(method="average").to_numpy(dtype=np.float64)
        rows.append({"anchor_date": _canonical_anchor_date(anchor_date_value), "rank_ic": _safe_pearson_corr(pred_rank, true_rank)})
    result = pd.DataFrame(rows)
    if len(result) != 391:
        raise RuntimeError(f"Reconstructed Rank-IC must cover exactly 391 anchor dates, found {len(result)}.")
    return result


def _compare_reconstructed_rank_ic(
    reconstructed_df: pd.DataFrame,
    saved_rank_ic_df: pd.DataFrame,
    frequency: str,
    horizon: int,
) -> None:
    saved = saved_rank_ic_df.copy()
    saved["anchor_date"] = saved["anchor_date"].apply(_canonical_anchor_date)
    reconstructed = reconstructed_df.copy()
    reconstructed["anchor_date"] = reconstructed["anchor_date"].apply(_canonical_anchor_date)

    saved_dates = set(saved["anchor_date"])
    reconstructed_dates = set(reconstructed["anchor_date"])
    if saved_dates != reconstructed_dates:
        raise RuntimeError(f"Reconstructed Rank-IC anchor dates mismatch for {frequency}/{horizon}.")

    saved_map = dict(zip(saved["anchor_date"], saved["rank_ic"].astype(float)))
    recon_map = dict(zip(reconstructed["anchor_date"], reconstructed["rank_ic"].astype(float)))
    for anchor_date_value in saved_dates:
        if not _nan_aware_isclose(saved_map[anchor_date_value], recon_map[anchor_date_value]):
            raise RuntimeError(
                f"Reconstructed Rank-IC mismatch for {frequency}/{horizon} at {anchor_date_value}: "
                f"saved={saved_map[anchor_date_value]}, reconstructed={recon_map[anchor_date_value]}"
            )


def _reconstruct_rank_ic_summary_stats(reconstructed_df: pd.DataFrame) -> dict:
    values = reconstructed_df["rank_ic"].astype(float).to_numpy()
    valid = values[~np.isnan(values)]
    n_valid = int(len(valid))
    if n_valid > 0:
        mean_rank_ic = float(np.nanmean(values))
        median_rank_ic = float(np.nanmedian(values))
        rank_ic_std = float(np.nanstd(values, ddof=0))
        positive_ic_ratio = float(np.mean(valid > 0))
    else:
        mean_rank_ic = float("nan")
        median_rank_ic = float("nan")
        rank_ic_std = float("nan")
        positive_ic_ratio = float("nan")
    if np.isfinite(rank_ic_std) and rank_ic_std > 0 and np.isfinite(mean_rank_ic):
        rank_icir = float(mean_rank_ic / rank_ic_std)
    else:
        rank_icir = float("nan")
    return {
        "mean_rank_ic": mean_rank_ic,
        "median_rank_ic": median_rank_ic,
        "rank_ic_std": rank_ic_std,
        "positive_ic_ratio": positive_ic_ratio,
        "rank_icir": rank_icir,
        "n_valid_ic_dates": n_valid,
    }


def _validate_cell_metric_reconstruction(
    summary_row: pd.Series,
    reconstructed_metrics: dict,
    reconstructed_rank_ic_stats: dict,
    frequency: str,
    horizon: int,
) -> None:
    pointwise_fields = [
        "test_mse", "test_mae", "test_corr",
        "ticker_avg_mse", "ticker_avg_mae", "ticker_avg_corr",
        "direction_accuracy", "pred_std", "true_std", "pred_std_true_std_ratio",
    ]
    for field in pointwise_fields:
        saved_value = float(summary_row[field]) if not pd.isna(summary_row[field]) else float("nan")
        reconstructed_value = reconstructed_metrics[field]
        if not _nan_aware_isclose(saved_value, reconstructed_value):
            raise RuntimeError(
                f"Reconstructed {field} mismatch for {frequency}/{horizon}: saved={saved_value}, reconstructed={reconstructed_value}"
            )

    rank_fields = ["mean_rank_ic", "median_rank_ic", "rank_ic_std", "positive_ic_ratio", "rank_icir"]
    for field in rank_fields:
        saved_value = float(summary_row[field]) if not pd.isna(summary_row[field]) else float("nan")
        reconstructed_value = reconstructed_rank_ic_stats[field]
        if not _nan_aware_isclose(saved_value, reconstructed_value):
            raise RuntimeError(
                f"Reconstructed {field} mismatch for {frequency}/{horizon}: saved={saved_value}, reconstructed={reconstructed_value}"
            )

    if int(summary_row["n_valid_ic_dates"]) != int(reconstructed_rank_ic_stats["n_valid_ic_dates"]):
        raise RuntimeError(
            f"Reconstructed n_valid_ic_dates mismatch for {frequency}/{horizon}: "
            f"saved={summary_row['n_valid_ic_dates']}, reconstructed={reconstructed_rank_ic_stats['n_valid_ic_dates']}"
        )


def _validate_cross_frequency_target_identity(pred_df: pd.DataFrame) -> None:
    for horizon in MAIN_HORIZONS:
        frames = {}
        for frequency in FREQUENCY_SETTINGS:
            subset = pred_df[
                (pred_df["frequency_setting"].astype(str) == frequency)
                & (pred_df["horizon"].astype(int) == int(horizon))
            ].copy()
            if subset.empty:
                raise RuntimeError(f"Missing predictions for cross-frequency identity check: {frequency}/{horizon}.")
            subset["key"] = subset["ticker"].astype(str) + "|" + subset["anchor_date"].apply(_canonical_anchor_date)
            if subset["key"].duplicated().any():
                raise RuntimeError(f"Duplicate ticker/anchor_date keys detected for {frequency}/{horizon}.")
            frames[frequency] = subset.set_index("key")["y_true"].astype(np.float64)

        reference_frequency = FREQUENCY_SETTINGS[0]
        reference_keys = set(frames[reference_frequency].index)
        for frequency, series in frames.items():
            if set(series.index) != reference_keys:
                raise RuntimeError(f"Cross-frequency ticker/anchor_date key set mismatch for horizon={horizon} at frequency={frequency}.")

        reference_series = frames[reference_frequency].sort_index()
        for frequency in FREQUENCY_SETTINGS[1:]:
            other_series = frames[frequency].sort_index()
            if not np.array_equal(reference_series.to_numpy(), other_series.to_numpy()):
                raise RuntimeError(f"Cross-frequency y_true mismatch for horizon={horizon}: {reference_frequency} vs {frequency}.")


def _validate_full_main_v2_evidence() -> dict[str, pd.DataFrame]:
    _combined_partial_validation()

    missing = [name for name, path in RAW_EVIDENCE_PATHS.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing raw evidence files for full validation: {missing}")

    summary_df = pd.read_csv(FORMAL_SUMMARY_PATH)
    pred_df = pd.read_csv(FORMAL_PRED_PATH)
    rank_ic_df = pd.read_csv(FORMAL_RANK_IC_PATH)
    history_df = pd.read_csv(FORMAL_HISTORY_PATH)
    router_df = pd.read_csv(FORMAL_ROUTER_PATH)

    if len(summary_df) != 9:
        raise RuntimeError(f"Expected exactly 9 summary rows, found {len(summary_df)}.")
    if len(pred_df) != 59823:
        raise RuntimeError(f"Expected exactly 59823 prediction rows, found {len(pred_df)}.")
    if len(rank_ic_df) != 3519:
        raise RuntimeError(f"Expected exactly 3519 Rank-IC rows, found {len(rank_ic_df)}.")
    if len(router_df) != 59823:
        raise RuntimeError(f"Expected exactly 59823 router rows, found {len(router_df)}.")

    completed_cells = set(
        (str(row["model"]), str(row["frequency_setting"]), int(row["horizon"]), int(row["seed"]))
        for _, row in summary_df.iterrows()
    )
    if len(completed_cells) != 9:
        raise RuntimeError(f"Expected exactly 9 completed formal cells, found {len(completed_cells)}.")
    if completed_cells != _formal_cell_grid():
        raise RuntimeError(f"Completed cell grid mismatch: {sorted(completed_cells)}")

    for _, summary_row in summary_df.iterrows():
        frequency = str(summary_row["frequency_setting"])
        horizon = int(summary_row["horizon"])
        seed = int(summary_row["seed"])
        model = str(summary_row["model"])

        cell_pred_df = pred_df[
            (pred_df["model"].astype(str) == model)
            & (pred_df["frequency_setting"].astype(str) == frequency)
            & (pred_df["horizon"].astype(int) == horizon)
            & (pred_df["seed"].astype(int) == seed)
        ].copy()
        cell_rank_ic_df = rank_ic_df[
            (rank_ic_df["model"].astype(str) == model)
            & (rank_ic_df["frequency_setting"].astype(str) == frequency)
            & (rank_ic_df["horizon"].astype(int) == horizon)
            & (rank_ic_df["seed"].astype(int) == seed)
        ].copy()
        cell_router_df = router_df[
            (router_df["model"].astype(str) == model)
            & (router_df["frequency_setting"].astype(str) == frequency)
            & (router_df["horizon"].astype(int) == horizon)
            & (router_df["seed"].astype(int) == seed)
        ].copy()

        if len(cell_pred_df) != 6647:
            raise RuntimeError(f"Expected exactly 6647 prediction rows for {frequency}/{horizon}, found {len(cell_pred_df)}.")
        if len(cell_rank_ic_df) != 391:
            raise RuntimeError(f"Expected exactly 391 Rank-IC rows for {frequency}/{horizon}, found {len(cell_rank_ic_df)}.")
        if len(cell_router_df) != 6647:
            raise RuntimeError(f"Expected exactly 6647 router rows for {frequency}/{horizon}, found {len(cell_router_df)}.")
        if cell_pred_df["ticker"].nunique() != 17:
            raise RuntimeError(f"Expected exactly 17 tickers in predictions for {frequency}/{horizon}.")
        if cell_pred_df["anchor_date"].nunique() != 391:
            raise RuntimeError(f"Expected exactly 391 anchor dates in predictions for {frequency}/{horizon}.")

        reconstructed_metrics = _reconstruct_cell_summary_metrics(cell_pred_df)
        reconstructed_rank_ic = _reconstruct_cell_rank_ic(cell_pred_df)
        _compare_reconstructed_rank_ic(reconstructed_rank_ic, cell_rank_ic_df, frequency, horizon)
        reconstructed_rank_ic_stats = _reconstruct_rank_ic_summary_stats(reconstructed_rank_ic)
        _validate_cell_metric_reconstruction(summary_row, reconstructed_metrics, reconstructed_rank_ic_stats, frequency, horizon)

    _validate_cross_frequency_target_identity(pred_df)

    return {
        "summary": summary_df,
        "predictions": pred_df,
        "rank_ic": rank_ic_df,
        "history": history_df,
        "router": router_df,
    }


def _load_accepted_main_v2_baselines() -> dict[str, pd.DataFrame]:
    naive_path = ROOT / "dataset" / "audit" / "main_v2" / "naive" / "panel_naive_baseline_summary.csv"
    ridge_path = ROOT / "dataset" / "audit" / "main_v2" / "ridge" / "main_v2_ridge_comparison.csv"
    lstm_path = ROOT / "dataset" / "audit" / "main_v2" / "lstm" / "panel_lstm_summary_metrics.csv"
    vanilla_path = ROOT / "dataset" / "audit" / "main_v2" / "vanilla" / "panel_vanilla_summary_metrics.csv"
    swim_path = ROOT / "dataset" / "audit" / "main_v2" / "swim_cpu" / "panel_swim_summary_metrics.csv"

    required_paths = {
        "naive": naive_path,
        "ridge": ridge_path,
        "lstm": lstm_path,
        "vanilla": vanilla_path,
        "swim": swim_path,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError("Missing accepted MAIN V2 baseline artifacts: " + "; ".join(missing))

    return {
        "naive": canonicalize_main_v2_naive(pd.read_csv(naive_path)),
        "ridge": canonicalize_main_v2_ridge(pd.read_csv(ridge_path)),
        "lstm": canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(lstm_path), label="lstm", model_value="lstm"),
        "vanilla": canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(vanilla_path), label="vanilla", model_value=None),
        "swim": canonicalize_main_v2_lstm_or_vanilla(pd.read_csv(swim_path), label="swim", model_value="swim_transformer"),
    }


def _build_matched_main_v2_comparison(summary_df: pd.DataFrame, baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    naive_df = baselines["naive"]
    ridge_df = baselines["ridge"]
    lstm_df = baselines["lstm"]
    vanilla_df = baselines["vanilla"]
    swim_df = baselines["swim"]

    rows = []
    for _, row in summary_df.sort_values(["frequency_setting", "horizon"]).iterrows():
        frequency = str(row["frequency_setting"])
        horizon = int(row["horizon"])
        horizon_norm = str(horizon)
        horizon_label = f"{horizon}d"

        naive_matched = naive_df[naive_df["horizon"] == horizon_norm].copy()
        naive_matched["model"] = naive_matched["model"].astype(str).str.lower()
        zero_row = naive_matched[naive_matched["model"] == "zero"]
        train_mean_row = naive_matched[naive_matched["model"] == "train_mean"]
        if len(zero_row) != 1 or len(train_mean_row) != 1:
            raise RuntimeError(f"Naive baseline zero/train_mean rows missing or duplicated for horizon={horizon}.")
        zero_mse = float(zero_row["pooled_mse"].iloc[0])
        train_mean_mse = float(train_mean_row["pooled_mse"].iloc[0])
        best_naive_mse = float(min(zero_mse, train_mean_mse))

        ridge_matched = ridge_df[(ridge_df["frequency_setting"] == frequency) & (ridge_df["horizon"] == horizon_norm)]
        if len(ridge_matched) != 1:
            raise RuntimeError(f"Ridge baseline missing or duplicated for {frequency}/{horizon}.")
        ridge_mse = float(ridge_matched["pooled_mse"].iloc[0])
        ridge_mean_rank_ic = float(ridge_matched["mean_rank_ic"].iloc[0])

        lstm_matched = lstm_df[(lstm_df["frequency_setting"] == frequency) & (lstm_df["horizon"] == horizon_norm)]
        if len(lstm_matched) != 1:
            raise RuntimeError(f"LSTM baseline missing or duplicated for {frequency}/{horizon}.")
        lstm_mse = float(lstm_matched["test_mse"].iloc[0])
        lstm_mean_rank_ic = float(lstm_matched["mean_rank_ic"].iloc[0])

        vanilla_matched = vanilla_df[(vanilla_df["frequency_setting"] == frequency) & (vanilla_df["horizon"] == horizon_norm)]
        if len(vanilla_matched) != 1:
            raise RuntimeError(f"Vanilla baseline missing or duplicated for {frequency}/{horizon}.")
        vanilla_mse = float(vanilla_matched["test_mse"].iloc[0])
        vanilla_mean_rank_ic = float(vanilla_matched["mean_rank_ic"].iloc[0])

        swim_matched = swim_df[(swim_df["frequency_setting"] == frequency) & (swim_df["horizon"] == horizon_norm)]
        if len(swim_matched) != 1:
            raise RuntimeError(f"SWiM CPU baseline missing or duplicated for {frequency}/{horizon}.")
        swim_mse = float(swim_matched["test_mse"].iloc[0])
        swim_mean_rank_ic = float(swim_matched["mean_rank_ic"].iloc[0])

        pathformer_mse = float(row["test_mse"])

        rows.append({
            "model": str(row["model"]),
            "frequency_setting": frequency,
            "horizon": horizon_label,
            "seed": int(row["seed"]),
            "n_trainable_parameters": int(row["n_trainable_parameters"]),
            "pathformer_mse": pathformer_mse,
            "pathformer_mae": float(row["test_mae"]),
            "pathformer_corr": float(row["test_corr"]),
            "pathformer_mean_rank_ic": float(row["mean_rank_ic"]),
            "pathformer_direction_accuracy": float(row["direction_accuracy"]),
            "pathformer_pred_std_true_std_ratio": float(row["pred_std_true_std_ratio"]),
            "zero_mse": zero_mse,
            "train_mean_mse": train_mean_mse,
            "best_naive_mse": best_naive_mse,
            "ridge_mse": ridge_mse,
            "ridge_mean_rank_ic": ridge_mean_rank_ic,
            "lstm_mse": lstm_mse,
            "lstm_mean_rank_ic": lstm_mean_rank_ic,
            "vanilla_mse": vanilla_mse,
            "vanilla_mean_rank_ic": vanilla_mean_rank_ic,
            "swim_mse": swim_mse,
            "swim_mean_rank_ic": swim_mean_rank_ic,
            "pathformer_minus_best_naive_mse": pathformer_mse - best_naive_mse,
            "pathformer_minus_ridge_mse": pathformer_mse - ridge_mse,
            "pathformer_minus_lstm_mse": pathformer_mse - lstm_mse,
            "pathformer_minus_vanilla_mse": pathformer_mse - vanilla_mse,
            "pathformer_minus_swim_mse": pathformer_mse - swim_mse,
            "pathformer_beats_best_naive_mse": bool(pathformer_mse < best_naive_mse),
            "pathformer_beats_ridge_mse": bool(pathformer_mse < ridge_mse),
            "pathformer_beats_lstm_mse": bool(pathformer_mse < lstm_mse),
            "pathformer_beats_vanilla_mse": bool(pathformer_mse < vanilla_mse),
            "pathformer_beats_swim_mse": bool(pathformer_mse < swim_mse),
        })

    return pd.DataFrame(rows)


def _validate_comparison_contract(comparison_df: pd.DataFrame) -> None:
    if len(comparison_df) != 9:
        raise RuntimeError(f"Expected exactly 9 comparison rows, found {len(comparison_df)}.")
    if comparison_df.duplicated(subset=["frequency_setting", "horizon"]).any():
        raise RuntimeError("Comparison contains duplicate frequency_setting/horizon rows.")

    expected_grid = {(freq, f"{horizon}d") for freq in FREQUENCY_SETTINGS for horizon in MAIN_HORIZONS}
    observed_grid = set(zip(comparison_df["frequency_setting"].astype(str), comparison_df["horizon"].astype(str)))
    if observed_grid != expected_grid:
        raise RuntimeError(f"Comparison frequency x horizon grid mismatch: {sorted(observed_grid)}")

    required_fields = [
        "model", "frequency_setting", "horizon", "seed", "n_trainable_parameters",
        "pathformer_mse", "pathformer_mae", "pathformer_corr", "pathformer_mean_rank_ic",
        "pathformer_direction_accuracy", "pathformer_pred_std_true_std_ratio",
        "zero_mse", "train_mean_mse", "best_naive_mse",
        "ridge_mse", "ridge_mean_rank_ic",
        "lstm_mse", "lstm_mean_rank_ic",
        "vanilla_mse", "vanilla_mean_rank_ic",
        "swim_mse", "swim_mean_rank_ic",
        "pathformer_minus_best_naive_mse", "pathformer_minus_ridge_mse",
        "pathformer_minus_lstm_mse", "pathformer_minus_vanilla_mse", "pathformer_minus_swim_mse",
        "pathformer_beats_best_naive_mse", "pathformer_beats_ridge_mse",
        "pathformer_beats_lstm_mse", "pathformer_beats_vanilla_mse", "pathformer_beats_swim_mse",
    ]
    missing_fields = [field for field in required_fields if field not in comparison_df.columns]
    if missing_fields:
        raise RuntimeError(f"Comparison is missing required fields: {missing_fields}")


def _build_final_manifest_row(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    history_df: pd.DataFrame,
    router_df: pd.DataFrame,
    provenance_row: pd.Series,
    artifact_shas: dict[str, str],
) -> dict:
    return {
        "run_family": "MAIN_V2",
        "temporal_status": "POST_TEMPORAL_FIX",
        "model_family": "ADAPTIVE_MULTISCALE_TRANSFORMER_PATHFORMER_INSPIRED",
        "raw_model_value": "adaptive_pathformer",
        "dataset_contract_version": 2,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "formal_ticker_count": 17,
        "common_anchor_dates": 2563,
        "total_samples": 43571,
        "train_samples": 30634,
        "validation_samples": 6290,
        "test_samples": 6647,
        "train_dates": 1802,
        "validation_dates": 370,
        "test_dates": 391,
        "seed": 42,
        "execution_backend": "CPU",
        "completed_cells": int(len(summary_df)),
        "expected_summary_rows": 9,
        "actual_summary_rows": int(len(summary_df)),
        "expected_prediction_rows": 59823,
        "actual_prediction_rows": int(len(pred_df)),
        "expected_rank_ic_rows": 3519,
        "actual_rank_ic_rows": int(len(rank_ic_df)),
        "expected_router_weight_rows": 59823,
        "actual_router_weight_rows": int(len(router_df)),
        "actual_training_history_rows": int(len(history_df)),
        "training_script": str(provenance_row["training_script"]),
        "training_script_sha256": str(provenance_row["training_script_sha256"]),
        "shared_loader_script": str(provenance_row["shared_loader_script"]),
        "shared_loader_script_sha256": str(provenance_row["shared_loader_script_sha256"]),
        "development_reference_script": str(provenance_row["development_reference_script"]),
        "development_reference_script_sha256": str(provenance_row["development_reference_script_sha256"]),
        "summary_artifact_sha256": artifact_shas["summary"],
        "predictions_artifact_sha256": artifact_shas["predictions"],
        "rank_ic_artifact_sha256": artifact_shas["rank_ic"],
        "training_history_artifact_sha256": artifact_shas["history"],
        "router_weights_artifact_sha256": artifact_shas["router"],
        "comparison_artifact_sha256": artifact_shas["comparison"],
        "uses_pre_temporal_fix_results": False,
        "pre_temporal_fix_results_used_for_selection": False,
        "historical_results_mixed_into_main_v2": False,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "router_hidden": ROUTER_HIDDEN,
        "daily_patch_sizes": ",".join(str(p) for p in DAILY_PATCH_SIZES),
        "weekly_patch_sizes": ",".join(str(p) for p in WEEKLY_PATCH_SIZES),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "HuberLoss(delta=1.0)",
        "max_epochs": MAX_EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "gradient_clip_max_norm": GRADIENT_CLIP_MAX_NORM,
        "pooling": "occupancy_weighted_final_patch_pooling",
        "router": "sample_dependent_within_frequency_softmax",
        "fusion": "separate_daily_weekly_branches_late_fusion",
    }


def _build_final_summary_text(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    provenance_row: pd.Series,
    artifact_shas: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("")
    lines.append("Adaptive Multi-Scale Transformer (PathFormer-inspired)")
    lines.append("This is not a literal reproduction of the original AMS/PathFormer repository code.")
    lines.append("")
    lines.append("formal backend = CPU")
    lines.append("nominal seed = 42")
    lines.append(f"completed cells = {len(summary_df)} / 9")
    lines.append("")
    lines.append("Architecture:")
    lines.append("  Daily scales: 5, 10, 20, 30")
    lines.append("  Weekly scales: 2, 4, 8, 13")
    lines.append("  within-frequency sample-dependent softmax router")
    lines.append("  separate Daily/Weekly branches")
    lines.append("  late fusion only")
    lines.append("  occupancy-weighted final patch pooling")
    lines.append("  no global frequency router")
    lines.append("")
    lines.append("Parameter counts:")
    lines.append("  daily_only = 56069")
    lines.append("  weekly_only = 56069")
    lines.append("  daily_weekly = 116297")
    lines.append("")
    lines.append("Per-cell test results:")
    for _, row in summary_df.sort_values(["frequency_setting", "horizon"]).iterrows():
        lines.append(
            "  - "
            f"{row['frequency_setting']} / {int(row['horizon'])}d: "
            f"MSE={float(row['test_mse'])}, MAE={float(row['test_mae'])}, "
            f"Corr={float(row['test_corr'])}, MeanRankIC={float(row['mean_rank_ic'])}, "
            f"DirectionAccuracy={float(row['direction_accuracy'])}, "
            f"PredStd/TrueStd={float(row['pred_std_true_std_ratio'])}"
        )
    lines.append("")
    lines.append("Descriptive matched-baseline MSE win counts (seed 42 only):")
    lines.append(f"  vs best naive: {int(comparison_df['pathformer_beats_best_naive_mse'].sum())} / 9")
    lines.append(f"  vs Ridge: {int(comparison_df['pathformer_beats_ridge_mse'].sum())} / 9")
    lines.append(f"  vs LSTM: {int(comparison_df['pathformer_beats_lstm_mse'].sum())} / 9")
    lines.append(f"  vs Vanilla: {int(comparison_df['pathformer_beats_vanilla_mse'].sum())} / 9")
    lines.append(f"  vs SWiM: {int(comparison_df['pathformer_beats_swim_mse'].sum())} / 9")
    lines.append("")
    lines.append("These are purely descriptive seed-42 comparisons.")
    lines.append("No statistical-significance claim.")
    lines.append("No profitability claim.")
    lines.append("No causal claim.")
    lines.append("No multi-seed robustness claim.")
    lines.append("Router weights are saved for later interpretation.")
    lines.append("TEST router weights were not used for model selection.")
    lines.append("")
    lines.append("Provenance:")
    lines.append(f"  dataset_manifest_sha256 = {provenance_row['dataset_manifest_sha256']}")
    lines.append(f"  training_script_sha256 = {provenance_row['training_script_sha256']}")
    lines.append(f"  shared_loader_script_sha256 = {provenance_row['shared_loader_script_sha256']}")
    lines.append(f"  development_reference_script_sha256 = {provenance_row['development_reference_script_sha256']}")
    lines.append("")
    lines.append("Artifact SHA256:")
    for name, sha in artifact_shas.items():
        lines.append(f"  {name} = {sha}")
    return "\n".join(lines) + "\n"


def _finalize_only() -> None:
    """Immutable finalization path after training has completed.

    This never rewrites raw evidence or provenance; it only reads existing
    evidence, reconstructs metrics independently, matches accepted baselines,
    and writes the finalization-only artifacts (comparison, manifest, final summary).
    """
    if not FORMAL_PROVENANCE_PATH.exists():
        raise RuntimeError("Formal provenance is missing; finalization requires immutable provenance to already exist.")
    provenance_df = pd.read_csv(FORMAL_PROVENANCE_PATH)
    if provenance_df.empty:
        raise RuntimeError("Formal provenance file is empty.")
    if len(provenance_df) != 1:
        raise RuntimeError(f"Formal provenance must contain exactly one row; found {len(provenance_df)}.")
    _validate_stored_provenance(provenance_df)
    provenance_row = provenance_df.iloc[0]

    raw = _validate_full_main_v2_evidence()
    summary_df = raw["summary"]
    pred_df = raw["predictions"]
    rank_ic_df = raw["rank_ic"]
    history_df = raw["history"]
    router_df = raw["router"]

    baselines = _load_accepted_main_v2_baselines()
    comparison_df = _build_matched_main_v2_comparison(summary_df, baselines)
    _validate_comparison_contract(comparison_df)

    comparison_df.to_csv(FORMAL_COMPARISON_PATH, index=False)
    if not FORMAL_COMPARISON_PATH.exists():
        raise RuntimeError("Formal comparison artifact was not written successfully.")

    artifact_shas = {
        "summary": _compute_sha256(FORMAL_SUMMARY_PATH),
        "predictions": _compute_sha256(FORMAL_PRED_PATH),
        "rank_ic": _compute_sha256(FORMAL_RANK_IC_PATH),
        "history": _compute_sha256(FORMAL_HISTORY_PATH),
        "router": _compute_sha256(FORMAL_ROUTER_PATH),
        "comparison": _compute_sha256(FORMAL_COMPARISON_PATH),
    }

    manifest_row = _build_final_manifest_row(
        summary_df, pred_df, rank_ic_df, history_df, router_df, provenance_row, artifact_shas
    )
    pd.DataFrame([manifest_row]).to_csv(FORMAL_MANIFEST_PATH, index=False)

    final_summary_text = _build_final_summary_text(summary_df, comparison_df, provenance_row, artifact_shas)
    FORMAL_FINAL_SUMMARY_PATH.write_text(final_summary_text, encoding="utf-8")

    print("PATHFORMER MAIN V2 CPU FINALIZATION PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal CPU PathFormer MAIN V2 benchmark and smoke validation.")
    parser.add_argument("--train", action="store_true", help="Enable formal training/benchmark execution.")
    parser.add_argument("--all", action="store_true", help="Require explicit full 9-cell adaptive benchmark execution.")
    parser.add_argument("--finalize-only", action="store_true", help="Finalize the 9-cell evidence after successful training.")
    parser.add_argument("--frequency", choices=FREQUENCY_SETTINGS, default="daily_weekly", help="daily_only | weekly_only | daily_weekly")
    parser.add_argument("--horizon", choices=["5d", "10d", "20d"], default="5d")
    parser.add_argument("--scale-mode", choices=SCALE_MODES, default="adaptive", help="single | fixed | static | adaptive")
    parser.add_argument("--single-daily-patch", type=int, default=20, help="Explicit single daily patch size for single-scale construction.")
    parser.add_argument("--single-weekly-patch", type=int, default=4, help="Explicit single weekly patch size for single-scale construction.")
    parser.add_argument("--cpu-validation", action="store_true", help="CPU qualification validation only; does not write formal evidence files.")
    args = parser.parse_args()
    if args.finalize_only and args.train:
        parser.error("--finalize-only and --train cannot be combined.")
    if args.all and not args.train:
        parser.error("--all requires --train.")
    if args.all and args.scale_mode != "adaptive":
        parser.error("Formal MAIN V2 benchmark only allows adaptive scale_mode.")
    if args.scale_mode != "adaptive" and args.train:
        parser.error("This formal script only permits adaptive for formal MAIN V2 training/evidence.")
    if args.cpu_validation and args.train:
        parser.error("--cpu-validation cannot be combined with --train.")
    if args.cpu_validation and args.all:
        parser.error("--cpu-validation cannot be combined with --all.")
    if args.cpu_validation and args.finalize_only:
        parser.error("--cpu-validation cannot be combined with --finalize-only.")
    if args.cpu_validation and args.scale_mode != "adaptive":
        parser.error("CPU qualification only permits scale_mode='adaptive'.")
    return args


def _snapshot_formal_namespace() -> dict:
    """Read-only snapshot: namespace existence, child filenames, and SHA256 of existing files."""
    if not FORMAL_OUTPUT_DIR.exists():
        return {"exists": False, "children": {}}
    children = {}
    for child in FORMAL_OUTPUT_DIR.iterdir():
        children[child.name] = _compute_sha256(child) if child.is_file() else None
    return {"exists": True, "children": children}


def _assert_formal_namespace_unchanged(before: dict, after: dict) -> None:
    if before["exists"] != after["exists"]:
        raise RuntimeError("CPU qualification changed formal namespace existence; qualification must not create formal artifacts.")
    if not before["exists"]:
        return
    if set(before["children"].keys()) != set(after["children"].keys()):
        raise RuntimeError("CPU qualification changed formal namespace child filenames.")
    for name, sha in before["children"].items():
        if sha is not None and after["children"][name] != sha:
            raise RuntimeError(f"CPU qualification modified a pre-existing formal artifact: {name}")


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        _finalize_only()
        return

    if args.cpu_validation:
        # CPU qualification: deterministic fingerprinting only, via the same train_one_cell()
        # implementation used for formal training. This is read/compute/print only; it must
        # never create provenance, raw evidence, comparison, manifest, or final-summary
        # artifacts. The fixed qualification cell is daily_only/10d/seed=42/adaptive.
        #
        # Two-fresh-process acceptance semantics: run --cpu-validation in two independent
        # fresh Python processes. Exact reproducibility is accepted only if BOTH runs match
        # on first_train_batch_input_sha256, first_train_batch_target_sha256,
        # initial_model_parameter_sha256, epochs_trained, best_epoch, best_val_loss, every
        # epoch train_loss/val_loss, final_restored_parameter_sha256, and trajectory_sha256.
        # A single run cannot establish this; this script never claims that it has.
        namespace_before = _snapshot_formal_namespace()

        trace: dict = {}
        evidence = train_one_cell("daily_only", 10, scale_mode="adaptive", qualification_trace=trace)
        _validate_single_cell_evidence(
            evidence["summary"],
            evidence["predictions"],
            evidence["rank_ic"],
            evidence["history"],
            evidence["router"],
            "daily_only",
            10,
        )

        namespace_after = _snapshot_formal_namespace()
        _assert_formal_namespace_unchanged(namespace_before, namespace_after)

        summary_row = evidence["summary"].iloc[0]

        print("CPU PATHFORMER MAIN V2 QUALIFICATION")
        print()
        print("frequency = daily_only")
        print("horizon = 10d")
        print("seed = 42")
        print("execution_backend = CPU")
        print(f"n_trainable_parameters = {int(summary_row['n_trainable_parameters'])}")
        print()
        print(f"first_train_batch_input_sha256 = {trace['first_train_batch_input_sha256']}")
        print(f"first_train_batch_target_sha256 = {trace['first_train_batch_target_sha256']}")
        print()
        print(f"initial_model_parameter_sha256 = {trace['initial_model_parameter_sha256']}")
        print()
        for row in trace["history"]:
            print(f"epoch={int(row['epoch'])} train_loss={float(row['train_loss']):.12f} val_loss={float(row['val_loss']):.12f}")
        print()
        print(f"epochs_trained = {trace['epochs_trained']}")
        print(f"best_epoch = {trace['best_epoch']}")
        print(f"best_val_loss = {trace['best_val_loss']:.12f}")
        print()
        print(f"final_restored_parameter_sha256 = {trace['final_restored_parameter_sha256']}")
        print(f"trajectory_sha256 = {trace['trajectory_sha256']}")
        print()
        print(f"test_mse = {float(summary_row['test_mse']):.12f}")
        print(f"test_mae = {float(summary_row['test_mae']):.12f}")
        print(f"test_corr = {float(summary_row['test_corr']):.12f}")
        print(f"mean_rank_ic = {float(summary_row['mean_rank_ic']):.12f}")
        print(f"pred_std_true_std_ratio = {float(summary_row['pred_std_true_std_ratio']):.12f}")
        print()
        print("CPU PATHFORMER MAIN V2 QUALIFICATION COMPLETE")
        return

    if args.train:
        if args.all:
            _full_benchmark_run()
        else:
            horizon_map = {"5d": 5, "10d": 10, "20d": 20}
            frequency = args.frequency
            horizon = horizon_map[args.horizon]
            _validate_main_bench_for_frequency(frequency, horizon)
            if args.scale_mode != MAIN_BENCHMARK_MODE:
                raise ValueError("Formal MAIN V2 runner only permits adaptive scale_mode.")
            ds_preflight = load_frozen_panel()
            _formal_check_dataset_contract(ds_preflight)
            _ensure_training_provenance()
            _combined_partial_validation()
            _assert_cell_not_completed(
                frequency,
                horizon,
                model="adaptive_pathformer",
                seed=SEED,
            )
            evidence = train_one_cell(frequency, horizon, scale_mode=args.scale_mode)
            _validate_evidence_integrity(
                evidence["summary"],
                evidence["predictions"],
                evidence["rank_ic"],
                evidence["history"],
                evidence["router"],
                frequency,
                horizon,
            )
            _write_formal_outputs_for_cell(FormalCell(frequency, horizon, args.scale_mode), evidence)
            _combined_partial_validation()
        return

    _run_smoke_validation()


if __name__ == "__main__":
    main()
