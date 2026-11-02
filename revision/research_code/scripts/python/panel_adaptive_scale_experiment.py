"""
Task (Advisor Pivot) - Panel Adaptive-Scale Selection Experiment (Experiment 1 + 4 skeleton)

Runs the green-energy panel (Daily + Weekly, see panel_universe.py and
panel_build_multiscale_dataset.py) through four model variants to test
whether adaptive, per-sample scale selection (a PathFormer-router-style
mechanism) beats fixed fusion:

  --model_variant daily_only        single-scale baseline (Daily branch only)
  --model_variant weekly_only       single-scale baseline (Weekly branch only)
  --model_variant fixed_multi       multi-scale, fixed/equal fusion (concat)
  --model_variant learnable_weight  multi-scale, global learnable softmax weights
                                     (same weights for every sample)
  --model_variant adaptive_router   multi-scale, PER-SAMPLE adaptive router
                                     (AMS-style noisy top-k gating; gate input is
                                     a small volatility/return summary of each
                                     frequency window, so the router reacts to
                                     current market dynamics)

Each frequency branch is itself a PathFormer backbone (intra-patch + inter-patch
attention, with its own internal patch-size router), so the "adaptive_router"
variant stacks two router mechanisms: the existing intra-frequency patch router
(unchanged, from pathformer.layers.AMS) and a new inter-frequency scale router
defined in this file (ScaleRouter).

Data: pooled across the whole panel (all tickers concatenated), split globally
by anchor_date (70/15/15) so no ticker's train rows are dated after any test
row across the panel (avoids cross-ticker leakage through the shared time
split).

Outputs:
  dataset/audit/panel_adaptive_scale_results.csv       (per horizon x variant metrics)
  dataset/audit/panel_adaptive_scale_summary.txt       (human-readable summary)
  dataset/audit/panel_adaptive_scale_router_weights_<horizon>_<variant>.csv
      (per-sample router gates: ticker, anchor_date, gate_daily, gate_weekly;
       only written for --model_variant adaptive_router, feeds Experiment 4
       interpretability analysis)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pathformer.models.PathFormer import Model as PathFormerBackbone
from scripts.python.panel_universe import GREEN_ENERGY_UNIVERSE

PANEL_DIR = BASE_DIR / "dataset" / "multiscale_dataset" / "panel"
AUDIT_DIR = BASE_DIR / "dataset" / "audit"

FREQ_NAMES = ["daily", "weekly"]
CLOSE_IDX = 3  # feature order: open, high, low, close, volume

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 0.0

SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VARIANTS = ["daily_only", "weekly_only", "fixed_multi", "learnable_weight", "adaptive_router"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    sec = int(max(0, round(seconds)))
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def log_progress(msg: str, log_file: Path | None) -> None:
    print(msg)
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_output_path(name: str) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    return AUDIT_DIR / p


@dataclass
class Metrics:
    mse: float
    mae: float
    mae_ratio: float  # placeholder to keep dataclass extensible; unused for now
    corr: float
    rank_corr: float
    direction_acc: float
    pred_std: float
    true_std: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan")
    if len(y_true) > 1 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        rank_corr = float(spearmanr(y_pred, y_true)[0])
    else:
        rank_corr = float("nan")
    direction_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return Metrics(
        mse=mse,
        mae=mae,
        mae_ratio=float("nan"),
        corr=corr,
        rank_corr=rank_corr,
        direction_acc=direction_acc,
        pred_std=float(np.std(y_pred)),
        true_std=float(np.std(y_true)),
    )


# =====================================================================
# DATA LOADING (pooled panel, global time-ordered split)
# =====================================================================


def load_pooled_panel(tickers: list[str], horizon_key: str) -> dict:
    X_daily_list, X_weekly_list, y_list, ticker_list, date_list = [], [], [], [], []

    for ticker in tickers:
        tdir = PANEL_DIR / ticker
        x_daily_path = tdir / "X_daily.npy"
        x_weekly_path = tdir / "X_weekly.npy"
        y_path = tdir / f"y_{horizon_key}.npy"
        meta_path = tdir / "meta.csv"
        if not (x_daily_path.exists() and x_weekly_path.exists() and y_path.exists() and meta_path.exists()):
            continue

        X_daily = np.load(x_daily_path).astype(np.float32)
        X_weekly = np.load(x_weekly_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        meta = pd.read_csv(meta_path, parse_dates=["anchor_date"])
        n = min(len(X_daily), len(X_weekly), len(y), len(meta))

        X_daily_list.append(X_daily[:n])
        X_weekly_list.append(X_weekly[:n])
        y_list.append(y[:n])
        ticker_list.append(np.full(n, ticker, dtype=object))
        date_list.append(meta["anchor_date"].values[:n])

    if not X_daily_list:
        raise RuntimeError("No panel data found. Run panel_build_multiscale_dataset.py first.")

    return {
        "X_daily": np.concatenate(X_daily_list, axis=0),
        "X_weekly": np.concatenate(X_weekly_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
        "ticker": np.concatenate(ticker_list, axis=0),
        "anchor_date": np.concatenate(date_list, axis=0),
    }


def global_time_split(anchor_date: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-threshold split shared across the whole panel (no cross-ticker leakage)."""
    unique_dates = np.sort(np.unique(anchor_date))
    n_dates = len(unique_dates)
    train_cut = unique_dates[int(n_dates * SPLIT_TRAIN)]
    val_cut = unique_dates[int(n_dates * (SPLIT_TRAIN + SPLIT_VAL))]

    train_mask = anchor_date <= train_cut
    val_mask = (anchor_date > train_cut) & (anchor_date <= val_cut)
    test_mask = anchor_date > val_cut
    return train_mask, val_mask, test_mask


def compute_patch_sizes(seq_len: int, num_scales: int = 4) -> list[int]:
    divisors = [d for d in range(1, seq_len + 1) if seq_len % d == 0]
    divisors = sorted(divisors, reverse=True)
    if len(divisors) >= num_scales:
        idx = np.linspace(0, len(divisors) - 1, num_scales).round().astype(int)
        chosen = [divisors[i] for i in idx]
    else:
        chosen = divisors + [1] * (num_scales - len(divisors))
    return chosen[:num_scales]


# =====================================================================
# MODEL
# =====================================================================


class FrequencyBranchEncoder(nn.Module):
    """PathFormer backbone (intra-patch + inter-patch attention + internal
    patch-size router) for one frequency, projected to a fixed-size vector."""

    def __init__(self, seq_len: int, feature_dim: int, layer_nums: int, d_model: int, d_ff: int, d_branch: int):
        super().__init__()
        patch_sizes = compute_patch_sizes(seq_len, num_scales=4)
        cfg = SimpleNamespace(
            layer_nums=layer_nums,
            num_nodes=feature_dim,
            pred_len=1,
            seq_len=seq_len,
            k=2,
            num_experts_list=[4] * layer_nums,
            patch_size_list=[patch_sizes for _ in range(layer_nums)],
            d_model=d_model,
            d_ff=d_ff,
            residual_connection=1,
            revin=True,
            gpu=0,
            batch_norm=False,
        )
        self.backbone = PathFormerBackbone(cfg)
        self.proj = nn.Linear(feature_dim, d_branch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, aux_loss = self.backbone(x)
        return self.proj(out[:, -1, :]), aux_loss


class ScaleRouter(nn.Module):
    """Adaptive, per-sample scale-selection router.

    Mirrors the noisy top-k gating mechanism used by PathFormer's AMS router
    (pathformer/layers/AMS.py), but operates across FREQUENCY branches instead
    of patch-size experts. The gate input is a small, frequency-invariant
    market-dynamics summary (mean/std of log returns) computed independently
    per branch, so the router can react to the current volatility/trend regime
    rather than to the branch encoder's own output.
    """

    def __init__(self, in_dim: int, num_scales: int, k: int | None = None, noisy_gating: bool = True):
        super().__init__()
        self.num_scales = num_scales
        self.k = num_scales if k is None else min(k, num_scales)
        self.w_gate = nn.Linear(in_dim, num_scales)
        self.w_noise = nn.Linear(in_dim, num_scales)
        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()

    def forward(self, gate_input: torch.Tensor, training: bool) -> torch.Tensor:
        clean_logits = self.w_gate(gate_input)
        if self.noisy_gating and training:
            noise_stddev = self.softplus(self.w_noise(gate_input)) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        if self.k < self.num_scales:
            top_vals, top_idx = logits.topk(self.k, dim=1)
            top_gates = torch.softmax(top_vals, dim=1)
            gates = torch.zeros_like(logits).scatter(1, top_idx, top_gates)
        else:
            gates = torch.softmax(logits, dim=1)
        return gates


def make_router_input(x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
    """Per-frequency [mean_logret, std_logret] summary, concatenated across scales."""

    def summarize(x: torch.Tensor) -> torch.Tensor:
        close = x[:, :, CLOSE_IDX].clamp_min(1e-6)
        log_ret = torch.diff(torch.log(close), dim=1)
        mean_r = log_ret.mean(dim=1, keepdim=True)
        std_r = log_ret.std(dim=1, keepdim=True)
        return torch.cat([mean_r, std_r], dim=1)

    return torch.cat([summarize(x_daily), summarize(x_weekly)], dim=1)


class AdaptiveScaleRegressor(nn.Module):
    def __init__(
        self,
        model_variant: str,
        seq_lens: dict[str, int],
        feature_dim: int,
        layer_nums: int,
        d_model: int,
        d_ff: int,
        d_branch: int,
        router_k: int | None,
        head_init_std: float,
    ):
        super().__init__()
        self.model_variant = model_variant

        if model_variant == "daily_only":
            self.active_freqs = ["daily"]
        elif model_variant == "weekly_only":
            self.active_freqs = ["weekly"]
        else:
            self.active_freqs = ["daily", "weekly"]

        self.encoders = nn.ModuleDict(
            {
                freq: FrequencyBranchEncoder(seq_lens[freq], feature_dim, layer_nums, d_model, d_ff, d_branch)
                for freq in self.active_freqs
            }
        )

        n_active = len(self.active_freqs)
        if model_variant in ("daily_only", "weekly_only"):
            fused_dim = d_branch
        elif model_variant == "fixed_multi":
            fused_dim = d_branch * n_active
        elif model_variant == "learnable_weight":
            fused_dim = d_branch
            self.scale_weight_logits = nn.Parameter(torch.zeros(n_active))
        elif model_variant == "adaptive_router":
            fused_dim = d_branch
            self.router = ScaleRouter(in_dim=2 * n_active, num_scales=n_active, k=router_k)
        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")

        self.neck = nn.LayerNorm(fused_dim)
        self.head = nn.Linear(fused_dim, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_by_freq: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        vecs = {}
        aux_total = 0.0
        for freq in self.active_freqs:
            v, aux = self.encoders[freq](x_by_freq[freq])
            vecs[freq] = v
            aux_total = aux_total + aux

        gates_out = None
        if self.model_variant in ("daily_only", "weekly_only"):
            fused = vecs[self.active_freqs[0]]
        elif self.model_variant == "fixed_multi":
            fused = torch.cat([vecs[f] for f in self.active_freqs], dim=-1)
        elif self.model_variant == "learnable_weight":
            weights = torch.softmax(self.scale_weight_logits, dim=0)
            stacked = torch.stack([vecs[f] for f in self.active_freqs], dim=1)
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        elif self.model_variant == "adaptive_router":
            gate_input = make_router_input(x_by_freq["daily"], x_by_freq["weekly"])
            gates_out = self.router(gate_input, self.training)
            stacked = torch.stack([vecs[f] for f in self.active_freqs], dim=1)
            fused = (stacked * gates_out.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")

        fused = self.neck(fused)
        pred = self.head(fused).squeeze(-1)
        return pred, aux_total, gates_out


# =====================================================================
# TRAIN / EVAL
# =====================================================================


@dataclass
class TrainArtifacts:
    metrics: dict
    epochs_ran: int
    elapsed_sec: float
    router_log: pd.DataFrame | None


def train_one_horizon(
    pooled: dict,
    *,
    model_variant: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    layer_nums: int,
    d_model: int,
    d_ff: int,
    d_branch: int,
    router_k: int | None,
    head_init_std: float,
    y_standardize: bool,
    loss_type: str,
    huber_delta: float,
    aux_loss_weight: float,
    horizon_name: str,
    progress_file: Path | None,
    progress_every: int,
) -> TrainArtifacts:
    X_daily = pooled["X_daily"]
    X_weekly = pooled["X_weekly"]
    y = pooled["y"]
    ticker = pooled["ticker"]
    anchor_date = pooled["anchor_date"]

    train_mask, val_mask, test_mask = global_time_split(anchor_date)

    seq_lens = {"daily": X_daily.shape[1], "weekly": X_weekly.shape[1]}
    feature_dim = X_daily.shape[2]

    def subset(mask: np.ndarray) -> dict:
        return {
            "daily": torch.from_numpy(X_daily[mask]),
            "weekly": torch.from_numpy(X_weekly[mask]),
        }

    X_train, X_val, X_test = subset(train_mask), subset(val_mask), subset(test_mask)
    y_train_np, y_val_np, y_test_np = y[train_mask], y[val_mask], y[test_mask]

    y_mean = float(np.mean(y_train_np))
    y_std = float(np.std(y_train_np))
    if y_std < 1e-8:
        y_std = 1.0

    if y_standardize:
        y_train_use = (y_train_np - y_mean) / y_std
        y_val_use = (y_val_np - y_mean) / y_std
        y_test_use = (y_test_np - y_mean) / y_std
    else:
        y_train_use, y_val_use, y_test_use = y_train_np, y_val_np, y_test_np

    def make_loader(x_dict: dict[str, torch.Tensor], y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = [x_dict["daily"], x_dict["weekly"], torch.from_numpy(y_arr)]
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train_use, shuffle=True)
    train_eval_loader = make_loader(X_train, y_train_use, shuffle=False)
    val_loader = make_loader(X_val, y_val_use, shuffle=False)
    test_loader = make_loader(X_test, y_test_use, shuffle=False)

    model = AdaptiveScaleRegressor(
        model_variant=model_variant,
        seq_lens=seq_lens,
        feature_dim=feature_dim,
        layer_nums=layer_nums,
        d_model=d_model,
        d_ff=d_ff,
        d_branch=d_branch,
        router_k=router_k,
        head_init_std=head_init_std,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=huber_delta) if loss_type == "huber" else nn.MSELoss()

    def batch_to_dict(batch: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        xd, xw, yb = batch
        return {"daily": xd.to(DEVICE), "weekly": xw.to(DEVICE)}, yb.to(DEVICE)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    horizon_start = time.perf_counter()
    running_epoch_secs: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            x_dict, yb = batch_to_dict(batch)
            optimizer.zero_grad()
            pred, aux_loss, _ = model(x_dict)
            loss = criterion(pred, yb) + aux_loss_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_dict, yb = batch_to_dict(batch)
                pred, _, _ = model(x_dict)
                val_losses.append(float(criterion(pred, yb).item()))

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_sec = time.perf_counter() - epoch_start
        running_epoch_secs.append(epoch_sec)
        mean_epoch_sec = float(np.mean(running_epoch_secs))
        eta_sec = max(0, max_epochs - epoch) * mean_epoch_sec

        if (epoch % max(1, progress_every) == 0) or (epoch == 1) or (epoch == max_epochs) or (bad_epochs >= patience):
            log_progress(
                (
                    f"[{horizon_name}/{model_variant}] epoch {epoch:03d}/{max_epochs:03d} "
                    f"val_loss={mean_val:.8f} best_val_loss={best_val:.8f} "
                    f"bad_epochs={bad_epochs}/{patience} "
                    f"epoch_time={format_seconds(epoch_sec)} eta={format_seconds(eta_sec)}"
                ),
                progress_file,
            )

        if bad_epochs >= patience:
            log_progress(f"[{horizon_name}/{model_variant}] early stopping at epoch {epoch}", progress_file)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    def collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        preds, trues, gates_all = [], [], []
        with torch.no_grad():
            for batch in loader:
                x_dict, yb = batch_to_dict(batch)
                pred, _, gates = model(x_dict)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
                if gates is not None:
                    gates_all.append(gates.cpu().numpy())
        gates_arr = np.concatenate(gates_all, axis=0) if gates_all else None
        return np.concatenate(trues), np.concatenate(preds), gates_arr

    ytr_true_use, ytr_pred_use, _ = collect(train_eval_loader)
    yv_true_use, yv_pred_use, _ = collect(val_loader)
    yt_true_use, yt_pred_use, yt_gates = collect(test_loader)

    if y_standardize:
        ytr_true, ytr_pred = ytr_true_use * y_std + y_mean, ytr_pred_use * y_std + y_mean
        yv_true, yv_pred = yv_true_use * y_std + y_mean, yv_pred_use * y_std + y_mean
        yt_true, yt_pred = yt_true_use * y_std + y_mean, yt_pred_use * y_std + y_mean
    else:
        ytr_true, ytr_pred = ytr_true_use, ytr_pred_use
        yv_true, yv_pred = yv_true_use, yv_pred_use
        yt_true, yt_pred = yt_true_use, yt_pred_use

    train_m = regression_metrics(ytr_true, ytr_pred)
    val_m = regression_metrics(yv_true, yv_pred)
    test_m = regression_metrics(yt_true, yt_pred)

    elapsed_sec = time.perf_counter() - horizon_start
    metrics = {
        "train_samples": int(len(y_train_np)),
        "val_samples": int(len(y_val_np)),
        "test_samples": int(len(y_test_np)),
        "train_mse": train_m.mse,
        "train_mae": train_m.mae,
        "train_pred_std": train_m.pred_std,
        "train_true_std": train_m.true_std,
        "val_mse": val_m.mse,
        "val_mae": val_m.mae,
        "val_corr": val_m.corr,
        "val_rank_corr": val_m.rank_corr,
        "val_direction_acc": val_m.direction_acc,
        "val_pred_std": val_m.pred_std,
        "val_true_std": val_m.true_std,
        "test_mse": test_m.mse,
        "test_mae": test_m.mae,
        "test_corr": test_m.corr,
        "test_rank_corr": test_m.rank_corr,
        "test_direction_acc": test_m.direction_acc,
        "test_pred_std": test_m.pred_std,
        "test_true_std": test_m.true_std,
        "y_train_mean": y_mean,
        "y_train_std": y_std,
    }

    router_log = None
    if yt_gates is not None:
        test_ticker = ticker[test_mask]
        test_date = anchor_date[test_mask]
        router_log = pd.DataFrame(
            {
                "ticker": test_ticker,
                "anchor_date": test_date,
                "horizon": horizon_name,
                "gate_daily": yt_gates[:, 0],
                "gate_weekly": yt_gates[:, 1],
                "y_true": yt_true,
                "y_pred": yt_pred,
            }
        )
        metrics["test_mean_gate_daily"] = float(yt_gates[:, 0].mean())
        metrics["test_mean_gate_weekly"] = float(yt_gates[:, 1].mean())

    return TrainArtifacts(metrics=metrics, epochs_ran=len(running_epoch_secs), elapsed_sec=elapsed_sec, router_log=router_log)


# =====================================================================
# CLI / MAIN
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", default=["5d", "10d", "20d"], choices=["5d", "10d", "20d"])
    p.add_argument("--model_variant", type=str, default="adaptive_router", choices=MODEL_VARIANTS)
    p.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers; default = full panel universe")

    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    p.add_argument("--layer_nums", type=int, default=1)
    p.add_argument("--d_model", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=16)
    p.add_argument("--d_branch", type=int, default=16)
    p.add_argument("--head_init_std", type=float, default=0.02)

    p.add_argument("--router_k", type=int, default=None, help="Top-k scales for adaptive_router; default=all (soft gate)")
    p.add_argument("--aux_loss_weight", type=float, default=1.0, help="Weight on internal patch-router balance loss")

    p.add_argument("--loss_type", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.35)

    p.add_argument("--y_standardize", dest="y_standardize", action="store_true")
    p.add_argument("--no_y_standardize", dest="y_standardize", action="store_false")
    p.set_defaults(y_standardize=True)

    p.add_argument("--progress_every", type=int, default=1)
    p.add_argument("--progress_file", type=str, default="panel_adaptive_scale.log")
    p.add_argument("--out_csv_name", type=str, default="panel_adaptive_scale_results.csv")
    p.add_argument("--out_summary_name", type=str, default="panel_adaptive_scale_summary.txt")
    p.add_argument("--router_log_prefix", type=str, default="panel_adaptive_scale_router_weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    progress_file: Path | None
    if str(args.progress_file).lower() == "none":
        progress_file = None
    else:
        progress_file = resolve_output_path(args.progress_file)
        progress_file.write_text("", encoding="utf-8")

    tickers = args.tickers if args.tickers else GREEN_ENERGY_UNIVERSE

    log_progress("PANEL ADAPTIVE-SCALE EXPERIMENT (Experiment 1) RUN START", progress_file)
    log_progress(f"device={DEVICE}", progress_file)
    log_progress(
        (
            f"config: horizons={args.horizons}, model_variant={args.model_variant}, "
            f"n_tickers={len(tickers)}, max_epochs={args.max_epochs}, patience={args.patience}, "
            f"batch_size={args.batch_size}, layer_nums={args.layer_nums}, d_model={args.d_model}, "
            f"d_ff={args.d_ff}, d_branch={args.d_branch}, router_k={args.router_k}, "
            f"aux_loss_weight={args.aux_loss_weight}, lr={args.lr}, weight_decay={args.weight_decay}, "
            f"seed={args.seed}, y_standardize={args.y_standardize}, loss_type={args.loss_type}, "
            f"huber_delta={args.huber_delta}"
        ),
        progress_file,
    )

    rows = []
    horizon_elapsed: list[float] = []

    for i, horizon in enumerate(args.horizons, start=1):
        log_progress(f"\n[{horizon}/{args.model_variant}] loading pooled panel data...", progress_file)
        pooled = load_pooled_panel(tickers, horizon)
        log_progress(
            f"[{horizon}/{args.model_variant}] pooled samples={len(pooled['y'])} "
            f"n_tickers_loaded={len(np.unique(pooled['ticker']))}",
            progress_file,
        )

        result = train_one_horizon(
            pooled,
            model_variant=args.model_variant,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            layer_nums=args.layer_nums,
            d_model=args.d_model,
            d_ff=args.d_ff,
            d_branch=args.d_branch,
            router_k=args.router_k,
            head_init_std=args.head_init_std,
            y_standardize=args.y_standardize,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            aux_loss_weight=args.aux_loss_weight,
            horizon_name=horizon,
            progress_file=progress_file,
            progress_every=args.progress_every,
        )

        metrics = result.metrics
        row = {"horizon": horizon, "model_variant": args.model_variant, **metrics}
        rows.append(row)

        horizon_elapsed.append(result.elapsed_sec)
        mean_horizon_sec = float(np.mean(horizon_elapsed))
        eta_horizons = max(0, len(args.horizons) - i) * mean_horizon_sec

        log_progress(
            f"[{horizon}/{args.model_variant}] train_mse={metrics['train_mse']:.8f} "
            f"test_mse={metrics['test_mse']:.8f} test_mae={metrics['test_mae']:.6f} "
            f"test_corr={metrics['test_corr']:.4f} test_rank_corr={metrics['test_rank_corr']:.4f} "
            f"test_da={metrics['test_direction_acc']:.4f} "
            f"test_pred_std={metrics['test_pred_std']:.6e} test_true_std={metrics['test_true_std']:.6e} "
            f"epochs_ran={result.epochs_ran} elapsed={format_seconds(result.elapsed_sec)} "
            f"global_eta={format_seconds(eta_horizons)}",
            progress_file,
        )

        if result.router_log is not None:
            router_csv = resolve_output_path(f"{args.router_log_prefix}_{horizon}_{args.model_variant}.csv")
            result.router_log.to_csv(router_csv, index=False)
            log_progress(
                f"[{horizon}/{args.model_variant}] router weights saved -> {router_csv.name} "
                f"(mean_gate_daily={metrics.get('test_mean_gate_daily', float('nan')):.4f}, "
                f"mean_gate_weekly={metrics.get('test_mean_gate_weekly', float('nan')):.4f})",
                progress_file,
            )

    df = pd.DataFrame(rows)
    out_csv = resolve_output_path(args.out_csv_name)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        key_cols = ["horizon", "model_variant"]
        merge_key = df[key_cols].apply(tuple, axis=1)
        old_key = old[key_cols].apply(tuple, axis=1)
        keep_old = old[~old_key.isin(merge_key)].copy()
        df = pd.concat([keep_old, df], ignore_index=True)
    df = df.sort_values(["horizon", "model_variant"])
    df.to_csv(out_csv, index=False)

    lines = ["PANEL ADAPTIVE-SCALE EXPERIMENT (Experiment 1)", "=" * 72]
    lines.append(f"device: {DEVICE}")
    lines.append(f"seed: {args.seed}")
    lines.append(f"model_variant: {args.model_variant}")
    lines.append(f"n_tickers: {len(tickers)}")
    lines.append(
        f"epochs<= {args.max_epochs}, patience={args.patience}, batch={args.batch_size}, "
        f"layer_nums={args.layer_nums}, d_model={args.d_model}, d_ff={args.d_ff}, d_branch={args.d_branch}, "
        f"router_k={args.router_k}, aux_loss_weight={args.aux_loss_weight}"
    )
    lines.append(
        f"lr={args.lr}, weight_decay={args.weight_decay}, y_standardize={args.y_standardize}, "
        f"loss_type={args.loss_type}, huber_delta={args.huber_delta}"
    )
    lines.append("")
    for _, r in df.iterrows():
        if r["model_variant"] != args.model_variant:
            continue
        extra = ""
        if "test_mean_gate_daily" in r and not pd.isna(r["test_mean_gate_daily"]):
            extra = f", mean_gate_daily={r['test_mean_gate_daily']:.4f}, mean_gate_weekly={r['test_mean_gate_weekly']:.4f}"
        lines.append(
            f"- {r['horizon']}: train_mse={r['train_mse']:.8f}, "
            f"test_mse={r['test_mse']:.8f}, test_mae={r['test_mae']:.6f}, "
            f"test_corr={r['test_corr']:.4f}, test_rank_corr={r['test_rank_corr']:.4f}, "
            f"test_da={r['test_direction_acc']:.4f}, test_pred_std={r['test_pred_std']:.6e}, "
            f"test_true_std={r['test_true_std']:.6e}{extra}"
        )

    out_summary = resolve_output_path(args.out_summary_name)
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_progress(f"\nSaved results -> {out_csv}", progress_file)
    log_progress(f"Saved summary -> {out_summary}", progress_file)


if __name__ == "__main__":
    main()
