"""First formal panel LSTM baseline for the frozen 17-stock Daily+Weekly panel.

This script follows the exact frozen panel protocol defined in
`panel_common.py` and must remain fully aligned with the Ridge benchmark:

- same 17-stock universe
- same common-date split
- same train/val/test sample index
- same train-only normalization
- same target construction and horizons
- same Rank IC definition (cross-sectional by date)
- no change to the frozen Ridge data pipeline

The first LSTM baseline is intentionally small and clean:

A. daily_only
    Daily sequence -> LSTM -> representation -> linear head -> scalar output
B. weekly_only
    Weekly sequence -> LSTM -> representation -> linear head -> scalar output
C. daily_weekly
    Daily LSTM + Weekly LSTM -> concat -> prediction head -> scalar output

No attention, no router, no ticker embeddings, no architectural complexity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_AUDIT_DIR = ROOT / "dataset" / "audit"
AUDIT_DIR = DEFAULT_AUDIT_DIR

from scripts.python.panel_common import (
    AUDIT_DIR,
    HORIZONS,
    apply_normalizer,
    evaluate_predictions,
    fit_train_only_normalizer,
    load_frozen_panel,
    rank_ic_by_date,
    safe_pearson_corr,
)

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
FREQUENCY_SETTINGS = ["daily_only", "weekly_only", "daily_weekly"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def prediction_scale_looks_normal(pred_std: float, true_std: float) -> bool:
    if not np.isfinite(pred_std) or not np.isfinite(true_std) or true_std <= 1e-12:
        return False
    ratio = pred_std / true_std
    return 0.2 <= ratio <= 2.0


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        final_hidden = out[:, -1, :]
        return self.head(final_hidden).squeeze(-1)


class DualFrequencyLSTM(nn.Module):
    def __init__(
        self,
        daily_input_size: int,
        weekly_input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.daily_lstm = nn.LSTM(
            input_size=daily_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.weekly_lstm = nn.LSTM(
            input_size=weekly_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        d_out, _ = self.daily_lstm(x_daily)
        w_out, _ = self.weekly_lstm(x_weekly)
        daily_hidden = d_out[:, -1, :]
        weekly_hidden = w_out[:, -1, :]
        fused = torch.cat([daily_hidden, weekly_hidden], dim=1)
        return self.mlp(fused).squeeze(-1)


def compute_rank_ic_summary(ic_df: pd.DataFrame) -> dict:
    valid = ic_df["rank_ic"].dropna()
    n_valid = int(len(valid))
    mean_ic = float(valid.mean()) if n_valid else float("nan")
    median_ic = float(valid.median()) if n_valid else float("nan")
    std_ic = float(valid.std()) if n_valid > 1 else float("nan")
    positive_ratio = float((valid > 0).mean()) if n_valid else float("nan")
    icir = (mean_ic / std_ic) if (n_valid > 1 and np.isfinite(std_ic) and abs(std_ic) > 1e-12) else float("nan")
    return {
        "mean_rank_ic": mean_ic,
        "median_rank_ic": median_ic,
        "rank_ic_std": std_ic,
        "positive_ic_ratio": positive_ratio,
        "n_valid_ic_dates": n_valid,
        "rank_icir": icir,
    }


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def build_frequency_data(ds, frequency: str, horizon: int, stats: dict):
    """Return train/val/test arrays for one frequency + horizon."""
    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    if frequency == "daily_only":
        X_all = apply_normalizer(ds, stats, "daily_only")
        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]
        return X_train, X_val, X_test, ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]

    if frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        X_train = X_all[train_mask]
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]
        return X_train, X_val, X_test, ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]

    if frequency == "daily_weekly":
        X_daily_all, X_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        X_d_train = X_daily_all[train_mask]
        X_d_val = X_daily_all[val_mask]
        X_d_test = X_daily_all[test_mask]
        X_w_train = X_weekly_all[train_mask]
        X_w_val = X_weekly_all[val_mask]
        X_w_test = X_weekly_all[test_mask]
        return (
            X_d_train,
            X_d_val,
            X_d_test,
            X_w_train,
            X_w_val,
            X_w_test,
            ds.y[horizon][train_mask],
            ds.y[horizon][val_mask],
            ds.y[horizon][test_mask],
        )

    raise ValueError(f"Unknown frequency setting: {frequency}")


def make_model(frequency: str, input_size_daily: int, input_size_weekly: int) -> nn.Module:
    if frequency == "daily_only":
        return LSTMRegressor(input_size=input_size_daily, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    if frequency == "weekly_only":
        return LSTMRegressor(input_size=input_size_weekly, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    if frequency == "daily_weekly":
        return DualFrequencyLSTM(
            daily_input_size=input_size_daily,
            weekly_input_size=input_size_weekly,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
    raise ValueError(f"Unknown frequency setting: {frequency}")


def train_and_eval_frequency(ds, frequency: str, horizon: int, stats: dict) -> dict:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    experiment_start = time.perf_counter()

    train_mask = ds.split == "train"
    val_mask = ds.split == "val"
    test_mask = ds.split == "test"

    if frequency == "daily_only":
        X_all = apply_normalizer(ds, stats, "daily_only")
        input_size = X_all.shape[-1]
        X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]
        y_train, y_val, y_test = ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        model = LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

        def forward_fn(batch_x, batch_y):
            return model(batch_x.to(device)), batch_y.to(device)

    elif frequency == "weekly_only":
        X_all = apply_normalizer(ds, stats, "weekly_only")
        input_size = X_all.shape[-1]
        X_train, X_val, X_test = X_all[train_mask], X_all[val_mask], X_all[test_mask]
        y_train, y_val, y_test = ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        model = LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

        def forward_fn(batch_x, batch_y):
            return model(batch_x.to(device)), batch_y.to(device)

    elif frequency == "daily_weekly":
        X_daily_all, X_weekly_all = apply_normalizer(ds, stats, "daily_weekly")
        X_d_train, X_d_val, X_d_test = X_daily_all[train_mask], X_daily_all[val_mask], X_daily_all[test_mask]
        X_w_train, X_w_val, X_w_test = X_weekly_all[train_mask], X_weekly_all[val_mask], X_weekly_all[test_mask]
        y_train, y_val, y_test = ds.y[horizon][train_mask], ds.y[horizon][val_mask], ds.y[horizon][test_mask]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_d_train, dtype=torch.float32),
                torch.tensor(X_w_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_d_val, dtype=torch.float32),
                torch.tensor(X_w_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        model = DualFrequencyLSTM(
            daily_input_size=X_daily_all.shape[-1],
            weekly_input_size=X_weekly_all.shape[-1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        ).to(device)

        def forward_fn(batch_x_daily, batch_x_weekly, batch_y):
            return model(batch_x_daily.to(device), batch_x_weekly.to(device)), batch_y.to(device)

    else:
        raise ValueError(f"unknown frequency: {frequency}")

    criterion = nn.HuberLoss(delta=1.0)  # raw-return targets make this behave mostly like quadratic loss here.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history_rows = []
    epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight = 0
        epoch_start = time.perf_counter()

        if frequency == "daily_weekly":
            for batch_x_d, batch_x_w, batch_y in train_loader:
                optimizer.zero_grad()
                logits, target = forward_fn(batch_x_d, batch_x_w, batch_y)
                loss = criterion(logits, target)
                batch_size = batch_y.shape[0]
                train_loss_sum += float(loss.item()) * batch_size
                train_weight += int(batch_size)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits, target = forward_fn(batch_x, batch_y)
                loss = criterion(logits, target)
                batch_size = batch_y.shape[0]
                train_loss_sum += float(loss.item()) * batch_size
                train_weight += int(batch_size)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_weight = 0
        with torch.no_grad():
            if frequency == "daily_weekly":
                for batch_x_d, batch_x_w, batch_y in val_loader:
                    logits, target = forward_fn(batch_x_d, batch_x_w, batch_y)
                    batch_loss = criterion(logits, target)
                    batch_size = batch_y.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size
                    val_weight += int(batch_size)
            else:
                for batch_x, batch_y in val_loader:
                    logits, target = forward_fn(batch_x, batch_y)
                    batch_loss = criterion(logits, target)
                    batch_size = batch_y.shape[0]
                    val_loss_sum += float(batch_loss.item()) * batch_size
                    val_weight += int(batch_size)

        train_loss = train_loss_sum / train_weight if train_weight else float("nan")
        val_loss = val_loss_sum / val_weight if val_weight else float("nan")
        if (not np.isfinite(train_loss)) or (not np.isfinite(val_loss)):
            raise RuntimeError(
                f"Non-finite loss detected for {frequency} / {horizon}d at epoch {epoch}: "
                f"train_loss={train_loss}, val_loss={val_loss}"
            )
        epochs_trained = epoch
        history_rows.append({
            "frequency_setting": frequency,
            "horizon": f"{horizon}d",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": time.perf_counter() - epoch_start,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"best_val={best_val_loss:.6f} | "
            f"best_epoch={best_epoch} | "
            f"patience={patience_counter}/{EARLY_STOPPING_PATIENCE} | "
            f"elapsed={format_elapsed(time.perf_counter() - epoch_start)}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            print(f"Best validation checkpoint = epoch {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"No valid training state for {frequency}, horizon={horizon}d")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if frequency == "daily_weekly":
            test_pred = []
            for batch_x_d, batch_x_w, batch_y in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_d_test, dtype=torch.float32),
                    torch.tensor(X_w_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32),
                ),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
                batch_pred = model(batch_x_d.to(device), batch_x_w.to(device))
                test_pred.append(batch_pred.cpu().numpy())
            y_pred = np.concatenate(test_pred)
        else:
            test_pred = []
            for batch_x, _ in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            ):
                batch_pred = model(batch_x.to(device))
                test_pred.append(batch_pred.cpu().numpy())
            y_pred = np.concatenate(test_pred)

    y_true = y_test.copy()
    if not np.all(np.isfinite(y_pred)):
        raise RuntimeError(f"Non-finite test predictions detected for {frequency} / {horizon}d")

    sanity_metric = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ticker=ds.ticker[test_mask],
        anchor_date=ds.anchor_date[test_mask],
    )

    ic_df = rank_ic_by_date(y_true, y_pred, ds.ticker[test_mask], ds.anchor_date[test_mask])
    rank_summary = compute_rank_ic_summary(ic_df)
    ratio = sanity_metric["pooled_pred_std_over_true_std"]
    no_extreme_scale_pathology = prediction_scale_looks_normal(sanity_metric["pooled_pred_std"], sanity_metric["pooled_true_std"])
    experiment_elapsed = time.perf_counter() - experiment_start

    result = {
        "model": "lstm",
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "seed": SEED,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "runtime_seconds": experiment_elapsed,
        "val_loss": best_val_loss,
        "test_mse": sanity_metric["pooled_mse"],
        "test_mae": sanity_metric["pooled_mae"],
        "test_corr": sanity_metric["pooled_corr"],
        "ticker_avg_mse": sanity_metric["ticker_avg_mse"],
        "ticker_avg_mae": sanity_metric["ticker_avg_mae"],
        "ticker_avg_corr": sanity_metric["ticker_avg_corr"],
        "direction_accuracy": sanity_metric["pooled_da"],
        "pred_std": sanity_metric["pooled_pred_std"],
        "true_std": sanity_metric["pooled_true_std"],
        "pred_std_true_std_ratio": ratio,
        "mean_rank_ic": rank_summary["mean_rank_ic"],
        "median_rank_ic": rank_summary["median_rank_ic"],
        "rank_ic_std": rank_summary["rank_ic_std"],
        "positive_ic_ratio": rank_summary["positive_ic_ratio"],
        "rank_icir": rank_summary["rank_icir"],
        "n_valid_ic_dates": rank_summary["n_valid_ic_dates"],
        "y_pred": y_pred,
        "y_true": y_true,
        "ticker": ds.ticker[test_mask],
        "anchor_date": ds.anchor_date[test_mask],
        "rank_ic_by_date": ic_df,
        "history": history_rows,
        "no_extreme_scale_pathology": no_extreme_scale_pathology,
    }
    return result


def assert_sequence_orientation(ds: object) -> None:
    """Verify the panel contract established in the dataset builder.

    The data windows are constructed in panel_build_multiscale_dataset.py as:
      daily_window = daily.iloc[i - H_DAILY + 1:i + 1][FEATURES]
      weekly_window = weekly_subset.tail(H_WEEKLY)[FEATURES]
    and the parent DataFrames are first sorted by `datetime` before slicing.
    Therefore the resulting arrays are ordered oldest -> newest along axis 1,
    and the 90-day Daily window includes anchor D, so `X[:, -1, :]` is the
    most recent observation.
    """
    if ds.X_daily.ndim != 3 or ds.X_weekly.ndim != 3:
        raise RuntimeError("Panel arrays must have shape (n_samples, window, features)")
    if ds.X_daily.shape[1] < 2 or ds.X_weekly.shape[1] < 2:
        raise RuntimeError("Panel sequences must have at least 2 timesteps along axis 1")
    if not np.isfinite(ds.X_daily).all() or not np.isfinite(ds.X_weekly).all():
        raise RuntimeError("Panel arrays contain non-finite values before training")


def assert_frozen_panel_checks(ds: object) -> None:
    train_n = int(np.sum(ds.split == "train"))
    val_n = int(np.sum(ds.split == "val"))
    test_n = int(np.sum(ds.split == "test"))

    if len(np.unique(ds.ticker)) != 17:
        raise RuntimeError(f"Ticker count mismatch: {len(np.unique(ds.ticker))} != 17")
    if len(np.unique(ds.anchor_date)) != 2563:
        raise RuntimeError(f"Common anchor count mismatch: {len(np.unique(ds.anchor_date))} != 2563")
    if train_n != 30634:
        raise RuntimeError(f"Train sample count mismatch: {train_n} != 30634")
    if val_n != 6290:
        raise RuntimeError(f"Validation sample count mismatch: {val_n} != 6290")
    if test_n != 6647:
        raise RuntimeError(f"Test sample count mismatch: {test_n} != 6647")

    if len(np.unique(ds.anchor_date[ds.split == "train"])) != 1802:
        raise RuntimeError(f"Train date count mismatch: {len(np.unique(ds.anchor_date[ds.split == 'train']))} != 1802")
    if len(np.unique(ds.anchor_date[ds.split == "val"])) != 370:
        raise RuntimeError(f"Validation date count mismatch: {len(np.unique(ds.anchor_date[ds.split == 'val']))} != 370")
    if len(np.unique(ds.anchor_date[ds.split == "test"])) != 391:
        raise RuntimeError(f"Test date count mismatch: {len(np.unique(ds.anchor_date[ds.split == 'test']))} != 391")

    assert_sequence_orientation(ds)

    train_stats = fit_train_only_normalizer(ds)
    for freq in ["daily_only", "weekly_only"]:
        X = apply_normalizer(ds, train_stats, freq)
        if np.isnan(X).any() or np.isinf(X).any():
            raise RuntimeError(f"NaN/Inf in normalized {freq} features")

    Xd, Xw = apply_normalizer(ds, train_stats, "daily_weekly")
    if np.isnan(Xd).any() or np.isinf(Xd).any():
        raise RuntimeError("NaN/Inf in normalized daily features for daily_weekly")
    if np.isnan(Xw).any() or np.isinf(Xw).any():
        raise RuntimeError("NaN/Inf in normalized weekly features for daily_weekly")

    for key in [5, 10, 20]:
        if not np.isfinite(ds.y[key]).all():
            raise RuntimeError(f"Non-finite y values detected for horizon {key}d")

    for ticker in np.unique(ds.ticker):
        mask = ds.ticker == ticker
        ordered_dates = pd.to_datetime(ds.anchor_date[mask])
        if not ordered_dates.is_monotonic_increasing:
            raise RuntimeError(f"Time order violation for ticker={ticker}: dates are not strictly oldest -> newest")


def require_columns(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}; available columns={list(df.columns)}")


def normalize_reporting_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    normalized = df.copy()
    if "frequency_setting" not in normalized.columns and "frequency" in normalized.columns:
        normalized["frequency_setting"] = normalized["frequency"].astype(str)
    if "frequency" in normalized.columns and "frequency_setting" in normalized.columns:
        normalized = normalized.drop(columns=["frequency"])
    if "horizon" in normalized.columns:
        normalized["horizon"] = normalized["horizon"].astype(str)
        normalized["horizon"] = normalized["horizon"].str.replace("d$", "", regex=True)
        normalized["horizon"] = normalized["horizon"].astype(str)
    return normalized


def validate_lstm_summary_cells(summary_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    require_columns(summary_df, ["model", "frequency_setting", "horizon", "seed"], "LSTM summary")
    normalized = normalize_reporting_dataframe(summary_df, "LSTM summary").copy()
    if "model" in normalized.columns:
        bad_model = normalized[normalized["model"].astype(str).str.lower() != "lstm"]
        if not bad_model.empty:
            raise ValueError(f"LSTM summary contains unexpected model values: {sorted(bad_model['model'].astype(str).unique().tolist())}")
    normalized["seed"] = normalized["seed"].astype(int)
    normalized["horizon"] = normalized["horizon"].astype(str)
    if len(normalized) != 9:
        raise ValueError(f"LSTM summary must contain exactly 9 rows; found {len(normalized)}")
    if normalized.duplicated(subset=["model", "frequency_setting", "horizon", "seed"]).any():
        dupes = normalized[normalized.duplicated(subset=["model", "frequency_setting", "horizon", "seed"], keep=False)]
        raise ValueError(f"LSTM summary contains duplicate model/frequency/horizon/seed rows: {dupes.to_dict(orient='records')}")
    cells = {
        (str(row["frequency_setting"]), str(row["horizon"]), int(row["seed"]))
        for _, row in normalized.iterrows()
    }
    expected = {
        (freq, str(horizon), 42)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in [5, 10, 20]
    }
    missing = sorted(expected - cells)
    extra = sorted(cells - expected)
    if missing:
        raise ValueError(f"LSTM summary missing expected seed-42 cells: {missing}")
    if extra:
        raise ValueError(f"LSTM summary contains unexpected seed-42 cells: {extra}")
    return cells


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_main_v2_lstm_paths(output_dir: Path) -> dict[str, Path]:
    expected_directory = (ROOT / "dataset" / "audit" / "main_v2" / "lstm").resolve()
    lstm_dir = Path(output_dir).resolve()
    if lstm_dir != expected_directory:
        raise ValueError(
            "Formal MAIN V2 LSTM output directory must be "
            f"{expected_directory}; got {lstm_dir}"
        )
    if not lstm_dir.exists():
        raise FileNotFoundError(f"MAIN V2 LSTM output directory not found: {lstm_dir}")
    main_v2_root = lstm_dir.parent
    return {
        "lstm_dir": lstm_dir,
        "main_v2_root": main_v2_root,
        "ridge_path": main_v2_root / "ridge" / "main_v2_ridge_comparison.csv",
        "naive_path": main_v2_root / "naive" / "main_v2_naive_summary.csv",
        "lstm_summary_path": lstm_dir / "panel_lstm_summary_metrics.csv",
        "lstm_predictions_path": lstm_dir / "panel_lstm_test_predictions.csv",
        "lstm_rank_ic_path": lstm_dir / "panel_lstm_rank_ic_by_date.csv",
        "lstm_history_path": lstm_dir / "panel_lstm_training_history.csv",
    }


def validate_main_v2_predictions(pred_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    require_columns(pred_df, ["ticker", "anchor_date", "horizon", "frequency_setting", "y_true", "y_pred", "seed"], "LSTM predictions")
    if "model" in pred_df.columns:
        if not pred_df["model"].astype(str).str.lower().eq("lstm").all():
            raise ValueError("LSTM predictions include unexpected model values outside LSTM.")
    pred_df = pred_df.copy()
    pred_df["horizon"] = pred_df["horizon"].astype(str).str.replace("d$", "", regex=True)
    pred_df["seed"] = pred_df["seed"].astype(int)
    key_cols = ["frequency_setting", "horizon", "seed", "ticker", "anchor_date"]
    if pred_df.duplicated(subset=key_cols).any():
        dupes = pred_df[pred_df.duplicated(subset=key_cols, keep=False)].drop_duplicates(subset=key_cols)
        raise ValueError(f"LSTM predictions contain duplicate keys for {key_cols}: {dupes.head().to_dict(orient='records')}")
    expected = {
        (freq, str(horizon), 42)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in [5, 10, 20]
    }
    observed = {
        (str(row["frequency_setting"]), str(row["horizon"]), int(row["seed"]))
        for _, row in pred_df.iterrows()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        raise ValueError(f"Prediction evidence missing required cells: {missing}")
    if extra:
        raise ValueError(f"Prediction evidence contains unexpected cells: {extra}")
    per_cell = pred_df.groupby(["frequency_setting", "horizon", "seed"]).size()
    if len(per_cell) != 9:
        raise ValueError(f"Prediction evidence has {len(per_cell)} cells instead of 9 expected cells.")
    if not per_cell.eq(6647).all():
        bad = per_cell[per_cell != 6647]
        raise ValueError(f"Prediction evidence row counts are not exactly 6647 per cell: {bad.to_dict()}")
    if not np.isfinite(pred_df["y_true"]).all() or not np.isfinite(pred_df["y_pred"]).all():
        raise ValueError("Prediction evidence contains non-finite y_true or y_pred values.")
    return expected


def validate_main_v2_rank_ic(rank_ic_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    required = ["frequency_setting", "horizon", "anchor_date", "rank_ic"]
    if not all(col in rank_ic_df.columns for col in required):
        missing = [col for col in required if col not in rank_ic_df.columns]
        raise ValueError(f"Rank-IC evidence missing required columns: {missing}; available columns={list(rank_ic_df.columns)}")
    rank_ic_df = rank_ic_df.copy()
    if "seed" in rank_ic_df.columns:
        rank_ic_df["seed"] = rank_ic_df["seed"].astype(int)
        if not rank_ic_df["seed"].eq(42).all():
            raise ValueError("Rank-IC evidence contains non-nominal seed values; expected all seed=42.")
    else:
        rank_ic_df["seed"] = 42
    rank_ic_df["horizon"] = rank_ic_df["horizon"].astype(str).str.replace("d$", "", regex=True)
    key_cols = ["frequency_setting", "horizon", "seed", "anchor_date"]
    if rank_ic_df.duplicated(subset=key_cols).any():
        dupes = rank_ic_df[rank_ic_df.duplicated(subset=key_cols, keep=False)].drop_duplicates(subset=key_cols)
        raise ValueError(f"Rank-IC evidence contains duplicate keys for {key_cols}: {dupes.head().to_dict(orient='records')}")
    expected = {
        (freq, str(horizon), 42)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in [5, 10, 20]
    }
    observed = {
        (str(row["frequency_setting"]), str(row["horizon"]), int(row["seed"]))
        for _, row in rank_ic_df.iterrows()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        raise ValueError(f"Rank-IC evidence missing required cells: {missing}")
    if extra:
        raise ValueError(f"Rank-IC evidence contains unexpected cells: {extra}")
    per_cell = rank_ic_df.groupby(["frequency_setting", "horizon", "seed"]).size()
    if len(per_cell) != 9:
        raise ValueError(f"Rank-IC evidence has {len(per_cell)} cells instead of 9 expected cells.")
    if not per_cell.eq(391).all():
        bad = per_cell[per_cell != 391]
        raise ValueError(f"Rank-IC evidence row counts are not exactly 391 per cell: {bad.to_dict()}")
    return expected


def validate_main_v2_training_history(history_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    required = ["frequency_setting", "horizon", "epoch", "train_loss", "val_loss"]
    missing = [col for col in required if col not in history_df.columns]
    if missing:
        raise ValueError(f"Training history missing required columns: {missing}; available columns={list(history_df.columns)}")
    history_df = history_df.copy()
    if "model" not in history_df.columns:
        history_df["model"] = "lstm"
    if "seed" in history_df.columns:
        history_df["seed"] = history_df["seed"].astype(int)
        if not history_df["seed"].eq(42).all():
            raise ValueError("Training-history evidence contains non-nominal seed values; expected all seed=42.")
    else:
        history_df["seed"] = 42
    history_df["horizon"] = history_df["horizon"].astype(str).str.replace("d$", "", regex=True)
    history_df["train_loss"] = pd.to_numeric(history_df["train_loss"], errors="coerce")
    history_df["val_loss"] = pd.to_numeric(history_df["val_loss"], errors="coerce")
    if not np.isfinite(history_df["train_loss"]).all() or not np.isfinite(history_df["val_loss"]).all():
        raise ValueError("Training history contains non-finite train_loss or val_loss values.")
    key_cols = ["model", "frequency_setting", "horizon", "seed", "epoch"]
    if history_df.duplicated(subset=key_cols).any():
        dupes = history_df[history_df.duplicated(subset=key_cols, keep=False)].drop_duplicates(subset=key_cols)
        raise ValueError(f"Training history contains duplicate keys for {key_cols}: {dupes.head().to_dict(orient='records')}")
    expected = {
        (freq, str(horizon), 42)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in [5, 10, 20]
    }
    observed = {
        (str(row["frequency_setting"]), str(row["horizon"]), int(row["seed"]))
        for _, row in history_df.iterrows()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        raise ValueError(f"Training history missing expected cells: {missing}")
    if extra:
        raise ValueError(f"Training history contains unexpected cells: {extra}")
    return expected


def validate_main_v2_cell_consistency(summary_cells: set[tuple[str, str, int]], pred_cells: set[tuple[str, str, int]], rank_ic_cells: set[tuple[str, str, int]], history_cells: set[tuple[str, str, int]]) -> None:
    expected = {
        (freq, str(horizon), 42)
        for freq in ["daily_only", "weekly_only", "daily_weekly"]
        for horizon in [5, 10, 20]
    }
    mismatch = {
        "summary": sorted(expected - summary_cells),
        "predictions": sorted(expected - pred_cells),
        "rank_ic": sorted(expected - rank_ic_cells),
        "history": sorted(expected - history_cells),
    }
    if any(values for values in mismatch.values()):
        raise ValueError(f"Cross-artifact cell consistency failed: {mismatch}")


def write_main_v2_lstm_run_manifest(output_dir: Path, summary_df: pd.DataFrame, pred_df: pd.DataFrame, rank_ic_df: pd.DataFrame, history_df: pd.DataFrame) -> Path:
    manifest_path = Path(output_dir).resolve() / "main_v2_lstm_run_manifest.csv"
    script_path = ROOT / "scripts" / "python" / "panel_baseline_lstm.py"
    manifest = {
        "run_family": "MAIN_V2",
        "model": "LSTM",
        "temporal_status": "POST_TEMPORAL_FIX",
        "dataset_contract_version": 2,
        "dataset_audit_status": "PASS",
        "nominal_seed": 42,
        "formal_tickers": 17,
        "common_anchor_dates": 2563,
        "total_samples": 43571,
        "train_samples": 30634,
        "validation_samples": 6290,
        "test_samples": 6647,
        "completed_cells": 9,
        "predictions_per_cell": 6647,
        "total_prediction_rows": int(len(pred_df)),
        "rank_ic_dates_per_cell": 391,
        "total_rank_ic_rows": int(len(rank_ic_df)),
        "training_script": "scripts/python/panel_baseline_lstm.py",
        "training_script_sha256": "NOT_CAPTURED",
        "training_script_sha256_status": "not captured before post-training finalization edits",
        "finalization_script": "scripts/python/panel_baseline_lstm.py",
        "finalization_script_sha256": "<computed-at-runtime>",
        "training_completed_before_finalization_edits": True,
        "post_training_edits_scope": "output resume, reporting, schema validation, finalization, and provenance safeguards",
        "training_logic_intentionally_changed_after_completed_run": False,
        "dataset_manifest_sha256": "fc6943db7b05aa47f18c39c61e4b3350014d55014816e9884a161be4e59fab0c",
        "lstm_output_dir": str(Path(output_dir).resolve()),
        "ridge_reference": "dataset/audit/main_v2/ridge/main_v2_ridge_comparison.csv",
        "naive_reference": "dataset/audit/main_v2/naive/main_v2_naive_summary.csv",
        "historical_results_used_for_training": False,
        "historical_results_used_for_model_selection": False,
        "historical_results_used_for_formal_comparison": False,
        "finalized_from_existing_training_artifacts": True,
    }
    if script_path.exists():
        manifest["finalization_script_sha256"] = compute_sha256(script_path)
    pd.DataFrame([manifest]).to_csv(manifest_path, index=False)
    return manifest_path


def merge_existing_rows(path: Path, new_df: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    if not all(col in new_df.columns for col in key_cols):
        raise ValueError(f"merge_existing_rows requires key columns {key_cols} in the new dataframe; available columns are {list(new_df.columns)}")
    if path.exists():
        existing = pd.read_csv(path)
        if existing.empty:
            combined = new_df.copy()
        else:
            if not all(col in existing.columns for col in key_cols):
                raise ValueError(f"merge_existing_rows cannot merge {path} because it is missing required key columns {key_cols}; available columns are {list(existing.columns)}")
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def save_prediction_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for i in range(len(rr["ticker"])):
            rows.append({
                "ticker": rr["ticker"][i],
                "anchor_date": rr["anchor_date"][i],
                "horizon": rr["horizon"],
                "frequency_setting": rr["frequency_setting"],
                "y_true": rr["y_true"][i],
                "y_pred": rr["y_pred"][i],
                "seed": rr["seed"],
                "best_epoch": rr["best_epoch"],
            })
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_lstm_test_predictions.csv"
    if append_mode:
        return merge_existing_rows(
            path,
            df,
            ["frequency_setting", "horizon", "seed", "ticker", "anchor_date"],
            ["frequency_setting", "horizon", "anchor_date", "ticker"],
        )
    df.to_csv(path, index=False)
    return df


def save_summary_rows(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        rows.append({
            "model": rr["model"],
            "frequency_setting": rr["frequency_setting"],
            "horizon": rr["horizon"],
            "seed": rr["seed"],
            "best_epoch": rr["best_epoch"],
            "epochs_trained": rr.get("epochs_trained", rr["best_epoch"]),
            "runtime_seconds": rr.get("runtime_seconds", np.nan),
            "val_loss": rr["val_loss"],
            "test_mse": rr["test_mse"],
            "test_mae": rr["test_mae"],
            "test_corr": rr["test_corr"],
            "ticker_avg_mse": rr["ticker_avg_mse"],
            "ticker_avg_mae": rr["ticker_avg_mae"],
            "ticker_avg_corr": rr["ticker_avg_corr"],
            "mean_rank_ic": rr["mean_rank_ic"],
            "median_rank_ic": rr["median_rank_ic"],
            "rank_ic_std": rr["rank_ic_std"],
            "positive_ic_ratio": rr["positive_ic_ratio"],
            "rank_icir": rr["rank_icir"],
            "direction_accuracy": rr["direction_accuracy"],
            "pred_std": rr["pred_std"],
            "true_std": rr["true_std"],
            "pred_std_true_std_ratio": rr["pred_std_true_std_ratio"],
            "no_extreme_scale_pathology": rr.get("no_extreme_scale_pathology", np.nan),
        })
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_lstm_summary_metrics.csv"
    if append_mode:
        return merge_existing_rows(
            path,
            df,
            ["frequency_setting", "horizon", "seed"],
            ["frequency_setting", "horizon", "seed"],
        )
    df.to_csv(path, index=False)
    return df


def save_training_history(results: list[dict], append_mode: bool = False) -> pd.DataFrame:
    rows = []
    for rr in results:
        for epoch_row in rr["history"]:
            rows.append({
                "frequency_setting": epoch_row["frequency_setting"],
                "horizon": epoch_row["horizon"],
                "epoch": epoch_row["epoch"],
                "train_loss": epoch_row["train_loss"],
                "val_loss": epoch_row["val_loss"],
                "elapsed_seconds": epoch_row.get("elapsed_seconds", np.nan),
            })
    df = pd.DataFrame(rows)
    path = AUDIT_DIR / "panel_lstm_training_history.csv"
    if append_mode:
        return merge_existing_rows(
            path,
            df,
            ["frequency_setting", "horizon", "epoch"],
            ["frequency_setting", "horizon", "epoch"],
        )
    df.to_csv(path, index=False)
    return df


def append_rank_ic_by_date(res: dict) -> pd.DataFrame:
    path = AUDIT_DIR / "panel_lstm_rank_ic_by_date.csv"
    df = res["rank_ic_by_date"].copy()
    df["frequency_setting"] = res["frequency_setting"]
    df["horizon"] = res["horizon"]
    df["model"] = "lstm"
    columns = ["model", "frequency_setting", "horizon", "anchor_date", "n_tickers", "rank_ic"]
    out = df.reindex(columns=columns)
    return merge_existing_rows(
        path,
        out,
        ["frequency_setting", "horizon", "anchor_date"],
        ["frequency_setting", "horizon", "anchor_date"],
    )


def append_experiment_outputs(res: dict) -> None:
    save_summary_rows([res], append_mode=True)
    save_prediction_rows([res], append_mode=True)
    save_training_history([res], append_mode=True)
    append_rank_ic_by_date(res)


def canonicalize_main_v2_ridge(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    require_columns(normalized, ["frequency", "horizon", "test_mse", "test_mae", "corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"], "Ridge MAIN V2 comparison")
    normalized["frequency_setting"] = normalized["frequency"].astype(str)
    normalized["horizon"] = normalized["horizon"].astype(str).str.replace("d$", "", regex=True)
    normalized = normalized.rename(columns={
        "test_mse": "pooled_mse",
        "test_mae": "pooled_mae",
        "corr": "pooled_corr",
    })
    normalized = normalized[[
        "frequency_setting",
        "horizon",
        "pooled_mse",
        "pooled_mae",
        "pooled_corr",
        "direction_accuracy",
        "mean_rank_ic",
        "pred_std_true_std_ratio",
    ]].copy()
    expected_configs = {(freq, horizon) for freq in ["daily_only", "weekly_only", "daily_weekly"] for horizon in ["5", "10", "20"]}
    actual_configs = {(row["frequency_setting"], str(row["horizon"])) for _, row in normalized.iterrows()}
    if len(normalized) != 9:
        raise ValueError(f"Expected exactly 9 Ridge rows after canonicalization, found {len(normalized)}")
    if normalized.duplicated(subset=["frequency_setting", "horizon"]).any():
        dupes = normalized[normalized.duplicated(subset=["frequency_setting", "horizon"], keep=False)]
        raise ValueError(f"Ridge canonicalized data contains duplicate frequency_setting/horizon rows: {dupes.to_dict(orient='records')}")
    if actual_configs != expected_configs:
        raise ValueError(f"Ridge configs do not match expected 9 frequency x horizon settings: {sorted(actual_configs)}")
    return normalized


def canonicalize_main_v2_naive(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    require_columns(normalized, ["model", "horizon", "pooled_mse"], "Naive MAIN V2 summary")
    normalized["horizon"] = normalized["horizon"].astype(str).str.replace("d$", "", regex=True)
    formal = normalized[normalized["model"].isin(["zero", "train_mean"])].copy()
    expected = {(model, horizon) for model in ["zero", "train_mean"] for horizon in ["5", "10", "20"]}
    actual = {(row["model"], str(row["horizon"])) for _, row in formal.iterrows()}
    if len(formal) != 6:
        raise ValueError(f"Naive MAIN V2 summary must contain exactly 6 required formal rows, found {len(formal)}")
    if formal.duplicated(subset=["model", "horizon"]).any():
        dupes = formal[formal.duplicated(subset=["model", "horizon"], keep=False)]
        raise ValueError(f"Naive MAIN V2 summary contains duplicate model/horizon rows: {dupes.to_dict(orient='records')}")
    if actual != expected:
        raise ValueError(f"Naive MAIN V2 summary does not match the required formal zero/train_mean x 5/10/20 cells: {sorted(actual)}")
    return formal


def load_ridge_comparison_data(audit_dir: Path = AUDIT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_v2_ridge_path = ROOT / "dataset" / "audit" / "main_v2" / "ridge" / "main_v2_ridge_comparison.csv"
    main_v2_naive_path = ROOT / "dataset" / "audit" / "main_v2" / "naive" / "main_v2_naive_summary.csv"

    if not main_v2_ridge_path.exists() or not main_v2_naive_path.exists():
        raise FileNotFoundError("Accepted MAIN V2 Ridge/naive reference files not found. Freeze the Ridge benchmark first.")

    ridge_df = canonicalize_main_v2_ridge(pd.read_csv(main_v2_ridge_path))
    naive_df = canonicalize_main_v2_naive(pd.read_csv(main_v2_naive_path))
    return ridge_df, naive_df


def create_lstm_vs_ridge_summary(summary_df: pd.DataFrame, ridge_df: pd.DataFrame, naive_df: pd.DataFrame) -> pd.DataFrame:
    """Comparison builder for already-canonical MAIN V2 inputs.

    The original raw CSVs are canonicalized exactly once at the file-read boundary;
    downstream comparison code must consume canonical Ridge/Naive data only.
    """
    summary_df = normalize_reporting_dataframe(summary_df, "LSTM summary")
    require_columns(summary_df, ["frequency_setting", "horizon", "test_mse", "mean_rank_ic"], "LSTM summary")
    require_columns(ridge_df, ["frequency_setting", "horizon", "pooled_mse", "pooled_mae", "pooled_corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"], "Ridge MAIN V2 comparison")
    require_columns(naive_df, ["model", "horizon", "pooled_mse"], "Naive MAIN V2 summary")

    rows = []
    for _, row in summary_df.iterrows():
        freq = str(row["frequency_setting"])
        horizon = str(row["horizon"])
        ridge_match = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon)]
        if len(ridge_match) != 1:
            raise ValueError(f"Expected exactly one canonical Ridge match for frequency={freq}, horizon={horizon}; found {len(ridge_match)}")
        ridge_row = ridge_match.iloc[0]
        ridge_mse = ridge_row["pooled_mse"]
        ridge_mae = ridge_row["pooled_mae"]
        ridge_corr = ridge_row["pooled_corr"]
        ridge_rank_ic = ridge_row["mean_rank_ic"]
        ridge_da = ridge_row["direction_accuracy"]
        ridge_ratio = ridge_row["pred_std_true_std_ratio"]

        naive_h = naive_df[(naive_df["horizon"] == horizon)]
        if len(naive_h) != 2:
            raise ValueError(f"Expected exactly two naive rows for horizon={horizon}; found {len(naive_h)}")
        best_naive_mse = float(naive_h["pooled_mse"].min())

        rows.append({
            "frequency_setting": freq,
            "horizon": horizon,
            "lstm_mse": row["test_mse"],
            "ridge_mse": ridge_mse,
            "naive_best_mse": best_naive_mse,
            "mse_vs_ridge_delta": row["test_mse"] - ridge_mse,
            "mse_vs_naive_delta": row["test_mse"] - best_naive_mse,
            "lstm_mae": row["test_mae"],
            "ridge_mae": ridge_mae,
            "lstm_corr": row["test_corr"],
            "ridge_corr": ridge_corr,
            "lstm_rank_ic": row["mean_rank_ic"],
            "ridge_rank_ic": ridge_rank_ic,
            "lstm_da": row["direction_accuracy"],
            "ridge_da": ridge_da,
            "lstm_pred_std_true_std_ratio": row["pred_std_true_std_ratio"],
            "ridge_pred_std_true_std_ratio": ridge_ratio,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(AUDIT_DIR / "panel_lstm_vs_ridge_summary.csv", index=False)
    return comparison_df


def build_main_v2_lstm_comparison(summary_df: pd.DataFrame, ridge_df: pd.DataFrame, naive_df: pd.DataFrame) -> pd.DataFrame:
    """Comparison builder for already-canonical MAIN V2 inputs.

    The canonical Ridge/Naive CSVs are validated and normalized once at the start
    of the formal finalize-only pipeline. Do not re-canonicalize downstream.
    """
    summary_norm = normalize_reporting_dataframe(summary_df, "LSTM summary")
    require_columns(summary_norm, ["frequency_setting", "horizon", "seed", "best_epoch", "val_loss", "test_mse", "test_mae", "test_corr", "direction_accuracy", "mean_rank_ic", "rank_ic_std", "pred_std", "true_std", "pred_std_true_std_ratio"], "LSTM summary")
    require_columns(ridge_df, ["frequency_setting", "horizon", "pooled_mse", "pooled_mae", "pooled_corr", "direction_accuracy", "mean_rank_ic", "pred_std_true_std_ratio"], "Ridge MAIN V2 comparison")
    require_columns(naive_df, ["model", "horizon", "pooled_mse"], "Naive MAIN V2 summary")

    rows = []
    for _, row in summary_norm.iterrows():
        freq = str(row["frequency_setting"])
        horizon = str(row["horizon"])
        ridge_match = ridge_df[(ridge_df["frequency_setting"] == freq) & (ridge_df["horizon"] == horizon)]
        if len(ridge_match) != 1:
            raise ValueError(f"Expected exactly one canonical Ridge match for frequency={freq}, horizon={horizon}; found {len(ridge_match)}")
        ridge_row = ridge_match.iloc[0]
        ridge_mse = float(ridge_row["pooled_mse"])
        ridge_rank_ic = float(ridge_row["mean_rank_ic"])

        zero_match = naive_df[(naive_df["model"] == "zero") & (naive_df["horizon"] == horizon)]
        train_mean_match = naive_df[(naive_df["model"] == "train_mean") & (naive_df["horizon"] == horizon)]
        if len(zero_match) != 1:
            raise ValueError(f"Expected exactly one zero baseline for horizon={horizon}; found {len(zero_match)}")
        if len(train_mean_match) != 1:
            raise ValueError(f"Expected exactly one train_mean baseline for horizon={horizon}; found {len(train_mean_match)}")

        rows.append({
            "horizon": int(horizon),
            "frequency_setting": freq,
            "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]),
            "val_loss": float(row["val_loss"]),
            "test_mse": float(row["test_mse"]),
            "test_mae": float(row["test_mae"]),
            "test_corr": float(row["test_corr"]),
            "direction_accuracy": float(row["direction_accuracy"]),
            "mean_rank_ic": float(row["mean_rank_ic"]),
            "rank_ic_std": float(row["rank_ic_std"] if pd.notna(row["rank_ic_std"]) else np.nan),
            "pred_std": float(row["pred_std"]),
            "true_std": float(row["true_std"]),
            "pred_std_true_std_ratio": float(row["pred_std_true_std_ratio"]),
            "zero_mse": float(zero_match.iloc[0]["pooled_mse"]),
            "train_mean_mse": float(train_mean_match.iloc[0]["pooled_mse"]),
            "ridge_mse_same_frequency": ridge_mse,
            "ridge_rank_ic_same_frequency": ridge_rank_ic,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(["horizon", "frequency_setting"], kind="mergesort").reset_index(drop=True)
    comparison_df.to_csv(AUDIT_DIR / "main_v2_lstm_comparison.csv", index=False)
    return comparison_df


def write_final_summary(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    summary_df = normalize_reporting_dataframe(summary_df, "LSTM summary")
    best_mse_by_horizon = {}
    best_rank_ic_by_horizon = {}
    for horizon in ["5", "10", "20"]:
        df_h = summary_df[summary_df["horizon"] == horizon]
        best_mse_by_horizon[horizon] = df_h.loc[df_h["test_mse"].idxmin(), "frequency_setting"]
        best_rank_ic_by_horizon[horizon] = df_h.loc[df_h["mean_rank_ic"].idxmax(), "frequency_setting"]

    lines = []
    lines.append("MAIN V2")
    lines.append("POST-TEMPORAL-FIX")
    lines.append("Dataset Contract V2")
    lines.append("LSTM")
    lines.append("nominal seed = 42")
    lines.append("completed cells = 9 / 9")
    lines.append("")
    lines.append("PRE-TEMPORAL-FIX results were not used for formal comparison or model selection.")
    lines.append("Finalization script hash recorded in main_v2_lstm_run_manifest.csv; exact training-time script hash was not captured before post-training finalization edits.")
    lines.append("")
    lines.append("Best MSE by horizon:")
    for horizon, setting in best_mse_by_horizon.items():
        lines.append(f"  {horizon}d: {setting}")
    lines.append("")
    lines.append("Best Rank IC by horizon:")
    for horizon, setting in best_rank_ic_by_horizon.items():
        lines.append(f"  {horizon}d: {setting}")
    lines.append("")

    mse_wins_ridge = int((comparison_df["mse_vs_ridge_delta"] < 0).sum())
    mse_wins_naive = int((comparison_df["mse_vs_naive_delta"] < 0).sum())
    rank_ic_wins_ridge = int((comparison_df["lstm_rank_ic"] > comparison_df["ridge_rank_ic"]).sum())

    lines.append(f"MSE wins vs Ridge: {mse_wins_ridge}/9")
    lines.append(f"MSE wins vs naive: {mse_wins_naive}/9")
    lines.append(f"Rank IC wins vs Ridge: {rank_ic_wins_ridge}/9")
    lines.append("")

    for horizon in ["5", "10", "20"]:
        h_df = comparison_df[comparison_df["horizon"] == horizon].copy()
        dw = h_df[h_df["frequency_setting"] == "daily_weekly"].iloc[0]
        d_only = h_df[h_df["frequency_setting"] == "daily_only"].iloc[0]
        w_only = h_df[h_df["frequency_setting"] == "weekly_only"].iloc[0]
        daily_weekly_rank_ic_gt_daily_only = bool(dw["lstm_rank_ic"] > d_only["lstm_rank_ic"])
        daily_weekly_rank_ic_gt_weekly_only = bool(dw["lstm_rank_ic"] > w_only["lstm_rank_ic"])
        lines.append(f"{horizon}d: Daily+Weekly Rank IC > Daily-only = {daily_weekly_rank_ic_gt_daily_only}")
        lines.append(f"{horizon}d: Daily+Weekly Rank IC > Weekly-only = {daily_weekly_rank_ic_gt_weekly_only}")
    lines.append("")

    for horizon in ["5", "10", "20"]:
        h_df = comparison_df[comparison_df["horizon"] == horizon].copy()
        best_by_mse = h_df.loc[h_df["lstm_mse"].idxmin(), "frequency_setting"]
        best_by_rank_ic = h_df.loc[h_df["lstm_rank_ic"].idxmax(), "frequency_setting"]
        lines.append(f"{horizon}d: best MSE = {best_by_mse}; best Rank IC = {best_by_rank_ic}")
    lines.append("")

    avg_ratio = comparison_df["lstm_pred_std_true_std_ratio"].mean()
    lines.append(f"Mean LSTM PredStd/TrueStd ratio: {avg_ratio:.4f}")
    lines.append("")
    lines.append("Prediction dispersion behavior:")
    lines.append("  Compare the ratio against the Ridge frozen baseline to determine whether the LSTM is under-dispersed or unstable.")
    lines.append("  Also check the training history for overfitting / collapse / non-convergence.")
    lines.append("")
    lines.append("Output file list:")
    lines.append("  dataset/audit/main_v2/lstm/main_v2_lstm_run_manifest.csv")
    lines.append("  dataset/audit/main_v2/lstm/main_v2_lstm_comparison.csv")
    lines.append("  dataset/audit/main_v2/lstm/panel_lstm_vs_ridge_summary.csv")
    lines.append("  dataset/audit/main_v2/lstm/panel_lstm_final_summary.txt")

    (AUDIT_DIR / "panel_lstm_final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal MAIN V2 LSTM baseline on the accepted 17-stock panel.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_AUDIT_DIR / "main_v2" / "lstm"),
        help="Directory for LSTM outputs. Defaults to dataset/audit/main_v2/lstm.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        choices=["daily_only", "weekly_only", "daily_weekly", "all"],
        default="all",
        help="Run a single frequency or all frequencies. Default: all.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 10, 20],
        default=None,
        help="Run a single horizon: 5, 10, or 20. Default: all horizons.",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Equivalent to --frequency all with all horizons.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Read the existing 9-cell LSTM artifacts and regenerate final summary/comparison outputs without retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    expected_output_dir = (ROOT / "dataset" / "audit" / "main_v2" / "lstm").resolve()
    requested_output_dir = Path(args.output_dir).resolve()
    if requested_output_dir != expected_output_dir:
        raise ValueError(
            "Formal MAIN V2 outputs must be written to "
            f"{expected_output_dir}; got {requested_output_dir}"
        )

    global AUDIT_DIR
    AUDIT_DIR = requested_output_dir
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        paths = resolve_main_v2_lstm_paths(AUDIT_DIR)
        for required in [
            paths["ridge_path"],
            paths["naive_path"],
            paths["lstm_summary_path"],
            paths["lstm_predictions_path"],
            paths["lstm_rank_ic_path"],
            paths["lstm_history_path"],
        ]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required MAIN V2 file: {required}")

        summary_df = pd.read_csv(paths["lstm_summary_path"])
        summary_df = normalize_reporting_dataframe(summary_df, "LSTM summary")
        require_columns(summary_df, ["model", "frequency_setting", "horizon", "seed", "test_mse", "mean_rank_ic"], "LSTM summary")
        summary_cells = validate_lstm_summary_cells(summary_df)

        pred_df = pd.read_csv(paths["lstm_predictions_path"])
        pred_cells = validate_main_v2_predictions(pred_df)

        rank_ic_df = pd.read_csv(paths["lstm_rank_ic_path"])
        rank_ic_cells = validate_main_v2_rank_ic(rank_ic_df)

        history_df = pd.read_csv(paths["lstm_history_path"])
        history_cells = validate_main_v2_training_history(history_df)

        validate_main_v2_cell_consistency(summary_cells, pred_cells, rank_ic_cells, history_cells)

        ridge_df = canonicalize_main_v2_ridge(pd.read_csv(paths["ridge_path"]))
        naive_df = canonicalize_main_v2_naive(pd.read_csv(paths["naive_path"]))

        comparison_df = create_lstm_vs_ridge_summary(summary_df, ridge_df, naive_df)
        build_main_v2_lstm_comparison(summary_df, ridge_df, naive_df)
        write_main_v2_lstm_run_manifest(AUDIT_DIR, summary_df, pred_df, rank_ic_df, history_df)
        write_final_summary(summary_df, comparison_df)
        print("FINALIZE-ONLY complete: regenerated MAIN V2 LSTM comparison and summary artifacts without retraining.")
        return

    set_seed(SEED)

    selected_frequencies = FREQUENCY_SETTINGS if (args.run_all or args.frequency == "all") else [args.frequency]
    selected_horizons = list(HORIZONS) if args.horizon is None else [args.horizon]
    selected_pairs = [(freq, horizon) for freq in selected_frequencies for horizon in selected_horizons]
    total_experiments = len(selected_pairs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    script_start = time.perf_counter()

    print("============================================================")
    print("LSTM PANEL BASELINE RUNSETUP")
    print(f"device = {device.type.upper()}")
    print(f"output_dir = {AUDIT_DIR}")
    print(f"total_experiments = {total_experiments}")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"max_epochs = {MAX_EPOCHS}")
    print(f"early_stopping_patience = {EARLY_STOPPING_PATIENCE}")
    print(f"selected_frequencies = {selected_frequencies}")
    print(f"selected_horizons = {selected_horizons}")
    print("============================================================")

    ds = load_frozen_panel()
    assert_frozen_panel_checks(ds)
    stats = fit_train_only_normalizer(ds)

    results = []
    all_history = []
    experiment_runtimes = []

    for exp_counter, (frequency, horizon) in enumerate(selected_pairs, start=1):
        print(f"\n[Experiment {exp_counter}/{total_experiments}]")
        print(f"frequency_setting = {frequency}")
        print(f"horizon = {horizon}d")
        res = train_and_eval_frequency(ds, frequency, int(horizon), stats)
        results.append(res)
        all_history.extend(res["history"])
        experiment_runtimes.append(float(res["runtime_seconds"]))
        append_experiment_outputs(res)

        completed = len(experiment_runtimes)
        total_elapsed = time.perf_counter() - script_start
        mean_experiment_time = float(np.mean(experiment_runtimes)) if experiment_runtimes else 0.0
        remaining_experiments = max(total_experiments - completed, 0)
        estimated_remaining = mean_experiment_time * remaining_experiments

        print(f"Experiment runtime: {format_elapsed(res['runtime_seconds'])}")
        print(f"best_epoch = {res['best_epoch']}")
        print(f"best_validation_loss = {res['val_loss']:.6f}")
        print(f"test_mse = {res['test_mse']:.6f}")
        print(f"mean_rank_ic = {res['mean_rank_ic']:.6f}")
        print(f"PredStd/TrueStd = {res['pred_std_true_std_ratio']:.4f}")
        print(f"no_extreme_scale_pathology = {res['no_extreme_scale_pathology']}")
        print(f"\nProgress: {completed}/{total_experiments} experiments completed")
        print(f"Total elapsed: {format_elapsed(total_elapsed)}")
        print(f"Estimated remaining: ~{format_elapsed(estimated_remaining)} (approximate; early stopping epochs differ)")

    print("\n===========================================================")
    print("LSTM PANEL BASELINE TRAINING COMPLETE")
    print("Evidence was persisted cell-by-cell via append_experiment_outputs(res).")
    print("To generate final derived reports, run finalize-only in a separate stage:")
    print("  python scripts/python/panel_baseline_lstm.py --output-dir dataset/audit/main_v2/lstm --finalize-only")
    print(f"Total runtime: {format_elapsed(time.perf_counter() - script_start)}")
    print(f"Completed experiments: {len(results)}/{total_experiments}")
    print("===========================================================")
    return


if __name__ == "__main__":
    main()
