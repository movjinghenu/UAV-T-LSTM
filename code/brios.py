from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# A.py-style metrics (authoritative)
# -----------------------------
def curvature_energy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 3:
        return float("nan")
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.sum(d2 ** 2))


def total_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return float("nan")
    d1 = np.diff(x)
    return float(np.sum(np.abs(d1)))


def count_peaks_valleys(x: np.ndarray, eps: float = 1e-3) -> Tuple[int, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 3:
        return 0, 0

    dx = np.diff(x)
    sign = np.sign(dx)
    sign[np.abs(dx) < eps] = 0

    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    for i in range(len(sign) - 2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i + 1]

    flips = sign[1:] * sign[:-1] < 0
    flip_idx = np.where(flips)[0]

    peaks = 0
    valleys = 0
    for i in flip_idx:
        if sign[i] > 0 and sign[i + 1] < 0:
            peaks += 1
        if sign[i] < 0 and sign[i + 1] > 0:
            valleys += 1
    return int(peaks), int(valleys)


def summarize(name: str, series: np.ndarray, eps: float = 1e-3) -> Dict:
    ce = curvature_energy(series)
    tv = total_variation(series)
    p, v = count_peaks_valleys(series, eps=eps)
    return {
        "Method": name,
        "Length": int(len(series)),
        "CE": ce,
        "TV": tv,
        "N_peaks": int(p),
        "N_valleys": int(v),
        "N_turning_points": int(p + v),
    }


def fill_nan_linear(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)  # robust for (1,L) etc.
    s = pd.Series(arr)
    return s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float64)


def last_valid_index(x: np.ndarray) -> int:
    x = np.asarray(x)
    idx = np.where(~np.isnan(x))[0]
    return int(idx[-1]) if len(idx) > 0 else -1


# -----------------------------
# Auxiliary smoothing (BRIOS-like)
# -----------------------------
def moving_average_3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x.copy()
    xp = np.pad(x, (1, 1), mode="edge")
    return (xp[:-2] + xp[1:-1] + xp[2:]) / 3.0


def wide_savgol(x: np.ndarray, half_width: int = 6, polyorder: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x.copy()
    win = 2 * int(half_width) + 1
    win = max(win, polyorder + 2)
    if win % 2 == 0:
        win += 1
    if win > len(x):
        win = len(x) if len(x) % 2 == 1 else max(1, len(x) - 1)
    if win < polyorder + 2:
        return x.copy()
    return savgol_filter(x, window_length=win, polyorder=polyorder, mode="interp")


# -----------------------------
# Dataset
# -----------------------------
class ExcelSequenceDataset(Dataset):
    """
    Reads input Excel and builds a 1..L daily grid.

    Required columns: doy, s_data, u_data
    Optional baseline columns for metrics: spline, net, T-RNN, line
    """
    def __init__(self, xlsx_path: str, seq_len: int = 0):
        df = pd.read_excel(xlsx_path)

        for col in ["doy", "s_data", "u_data"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}. Found: {list(df.columns)}")

        df = df.copy()
        df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
        df["s_data"] = pd.to_numeric(df["s_data"], errors="coerce")
        df["u_data"] = pd.to_numeric(df["u_data"], errors="coerce")

        # ---------- Key fix for 25-year shorter effective length ----------
        # If seq_len==0, infer L as "last day with any valid data" rather than max(doy)
        if seq_len and int(seq_len) > 0:
            L = int(seq_len)
        else:
            candidate_cols = [c for c in ["u_data", "s_data", "spline", "net", "T-RNN", "line"] if c in df.columns]
            any_valid = np.zeros(len(df), dtype=bool)
            for c in candidate_cols:
                any_valid |= pd.to_numeric(df[c], errors="coerce").notna().to_numpy()

            valid_doy = df.loc[any_valid, "doy"]
            if valid_doy.notna().any():
                L = int(np.nanmax(valid_doy.to_numpy(dtype=np.float64)))
            else:
                # fallback
                L = int(np.nanmax(df["doy"].to_numpy(dtype=np.float64))) if df["doy"].notna().any() else int(len(df))

        if L <= 0:
            raise ValueError("Invalid inferred sequence length")

        df = df.sort_values("doy")
        df = df[(df["doy"] >= 1) & (df["doy"] <= L)].copy()
        df = df.set_index("doy").reindex(range(1, L + 1)).reset_index().rename(columns={"index": "doy"})

        self.df = df
        self.L = L

        self.s_mask = (~df["s_data"].isna()).to_numpy(dtype=np.float32)

        u = df["u_data"].to_numpy(dtype=np.float64)
        if np.isnan(u).any():
            u = fill_nan_linear(u)
        self.u_filled = u.astype(np.float32)

        s = df["s_data"].to_numpy(dtype=np.float64)
        self.s_raw = s.astype(np.float32)

        # dt since last satellite observation
        dt = np.zeros(L, dtype=np.float32)
        last = None
        for t in range(L):
            if self.s_mask[t] > 0.5:
                last = t
                dt[t] = 0.0
            else:
                dt[t] = float(t - last) if last is not None else float(t + 1)
        self.dt = dt

        doy = df["doy"].to_numpy(dtype=np.float64)
        self.time_sin = np.sin(2.0 * np.pi * doy / 365.0).astype(np.float32)

        u_ma3 = moving_average_3(self.u_filled)
        u_sg = wide_savgol(u_ma3, half_width=6, polyorder=2)

        self.u_ma3 = u_ma3.astype(np.float32)
        self.u_sg = u_sg.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        return {
            "s_raw": torch.from_numpy(self.s_raw).float(),
            "s_mask": torch.from_numpy(self.s_mask).float(),
            "u_raw": torch.from_numpy(self.u_filled).float(),
            "u_ma3": torch.from_numpy(self.u_ma3).float(),
            "u_sg": torch.from_numpy(self.u_sg).float(),
            "dt": torch.from_numpy(self.dt).float(),
            "time_sin": torch.from_numpy(self.time_sin).float(),
        }


# -----------------------------
# BRIOS-like recurrent cell (decay-aware)
# -----------------------------
class DecayGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, dt_t: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(-torch.clamp(self.gamma, 1e-4, 10.0) * torch.clamp(dt_t, 0.0, 1e6))
        h_bar = h_prev * decay
        return self.gru_cell(x_t, h_bar)


class BRIOSImputer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.f_cell = DecayGRUCell(input_dim, hidden_dim)
        self.b_cell = DecayGRUCell(input_dim, hidden_dim)
        self.f_out = nn.Linear(hidden_dim, 1)
        self.b_out = nn.Linear(hidden_dim, 1)

    def forward_one_direction(self, x: torch.Tensor, dt: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        B, L, _ = x.shape
        h = torch.zeros(B, self.f_cell.hidden_dim, device=x.device, dtype=x.dtype)
        preds = []
        idxs = range(L - 1, -1, -1) if reverse else range(L)
        cell = self.b_cell if reverse else self.f_cell
        head = self.b_out if reverse else self.f_out

        for t in idxs:
            h = cell(x[:, t, :], h, dt[:, t, :])
            preds.append(head(h))
        if reverse:
            preds = preds[::-1]
        return torch.cat(preds, dim=1)  # (B,L)

    def forward(self, x: torch.Tensor, dt: torch.Tensor):
        y_f = self.forward_one_direction(x, dt, reverse=False)
        y_b = self.forward_one_direction(x, dt, reverse=True)
        y = 0.5 * (y_f + y_b)
        return y, y_f, y_b


# -----------------------------
# Training / reconstruction
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    hidden_dim: int = 64
    weight_consistency: float = 0.1
    weight_smooth_l2: float = 0.0
    device: str = "cpu"
    seed: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_brios(model: BRIOSImputer, batch: Dict[str, torch.Tensor], cfg: TrainConfig) -> Dict[str, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # squeeze batch dim: (1,L)->(L,)
    s_raw = batch["s_raw"].to(cfg.device).squeeze(0)
    s_mask = batch["s_mask"].to(cfg.device).squeeze(0)
    u_raw = batch["u_raw"].to(cfg.device).squeeze(0)
    u_ma3 = batch["u_ma3"].to(cfg.device).squeeze(0)
    u_sg = batch["u_sg"].to(cfg.device).squeeze(0)
    dt = batch["dt"].to(cfg.device).squeeze(0)
    time_sin = batch["time_sin"].to(cfg.device).squeeze(0)

    L = int(s_raw.shape[0])

    s_filled = s_raw.clone()
    if torch.isnan(s_filled).any():
        s_np = s_filled.detach().cpu().numpy().astype(np.float64).reshape(-1)
        s_np = fill_nan_linear(s_np)
        s_filled = torch.from_numpy(s_np.astype(np.float32)).to(cfg.device)

    x = torch.stack([u_sg, u_ma3, u_raw, time_sin, s_filled, s_mask], dim=-1).unsqueeze(0)
    dt_in = dt.view(1, L, 1)

    obs_idx = s_mask > 0.5
    if int(obs_idx.sum().item()) < 3:
        raise ValueError("Too few observed satellite points to train (need >=3).")

    best = {"loss": float("inf")}
    for ep in range(cfg.epochs):
        opt.zero_grad()
        y, y_f, y_b = model(x, dt_in)
        y = y.squeeze(0)
        y_f = y_f.squeeze(0)
        y_b = y_b.squeeze(0)

        loss_fit = torch.mean((y[obs_idx] - s_filled[obs_idx]) ** 2)
        loss_cons = torch.mean((y_f - y_b) ** 2)

        if cfg.weight_smooth_l2 > 0:
            dy = y[1:] - y[:-1]
            loss_smooth = torch.mean(dy ** 2)
        else:
            loss_smooth = torch.tensor(0.0, device=cfg.device)

        loss = loss_fit + cfg.weight_consistency * loss_cons + cfg.weight_smooth_l2 * loss_smooth
        loss.backward()
        opt.step()

        if loss.item() < best["loss"]:
            best = {
                "loss": float(loss.item()),
                "loss_fit": float(loss_fit.item()),
                "loss_cons": float(loss_cons.item()),
                "loss_smooth": float(loss_smooth.item()),
                "epoch": int(ep + 1),
            }

    return best


@torch.no_grad()
def reconstruct_brios(model: BRIOSImputer, batch: Dict[str, torch.Tensor], cfg: TrainConfig) -> np.ndarray:
    model.eval()

    s_raw = batch["s_raw"].to(cfg.device).squeeze(0)
    s_mask = batch["s_mask"].to(cfg.device).squeeze(0)
    u_raw = batch["u_raw"].to(cfg.device).squeeze(0)
    u_ma3 = batch["u_ma3"].to(cfg.device).squeeze(0)
    u_sg = batch["u_sg"].to(cfg.device).squeeze(0)
    dt = batch["dt"].to(cfg.device).squeeze(0)
    time_sin = batch["time_sin"].to(cfg.device).squeeze(0)

    L = int(s_raw.shape[0])

    s_filled = s_raw.clone()
    if torch.isnan(s_filled).any():
        s_np = s_filled.detach().cpu().numpy().astype(np.float64).reshape(-1)
        s_np = fill_nan_linear(s_np)
        s_filled = torch.from_numpy(s_np.astype(np.float32)).to(cfg.device)

    x = torch.stack([u_sg, u_ma3, u_raw, time_sin, s_filled, s_mask], dim=-1).unsqueeze(0)
    dt_in = dt.view(1, L, 1)

    y, _, _ = model(x, dt_in)
    y = y.squeeze(0).detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.clip(y, -1.0, 1.0)


def light_smooth_for_reporting(y: np.ndarray, method: str = "ma", window: int = 5, alpha: float = 0.2) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if method == "none":
        return y.copy()
    if method == "ema":
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
        return out
    if window < 3:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(yp, kernel, mode="valid")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=r"C:\Users\movjing\Desktop\25data.xlsx",
                        help="Input Excel (same format as 25data.xlsx)")
    parser.add_argument("--output", type=str, default=r"C:\Users\movjing\Desktop\25databrios.xlsx",
                        help="Output Excel path")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--weight-cons", type=float, default=0.1, help="Bidirectional consistency weight")
    parser.add_argument("--weight-smooth", type=float, default=0.0, help="Optional smoothness (L2 diff) weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda/cpu")

    parser.add_argument("--smooth-method", type=str, default="ma", choices=["none", "ma", "ema"],
                        help="Post smoothing for reported curve")
    parser.add_argument("--smooth-window", type=int, default=5, help="MA window (odd)")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="EMA alpha")

    # you can still force L explicitly; otherwise it uses "last valid day" inference
    parser.add_argument("--seq-len", type=int, default=0,
                        help="Sequence length (0 = infer as last day with any valid data)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    set_seed(args.seed)

    ds = ExcelSequenceDataset(args.input, seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden,
        weight_consistency=args.weight_cons,
        weight_smooth_l2=args.weight_smooth,
        device=args.device,
        seed=args.seed,
    )

    model = BRIOSImputer(input_dim=6, hidden_dim=cfg.hidden_dim).to(cfg.device)

    best = train_brios(model, batch, cfg)

    pred_raw = reconstruct_brios(model, batch, cfg)
    pred = light_smooth_for_reporting(pred_raw, method=args.smooth_method,
                                      window=args.smooth_window, alpha=args.smooth_alpha)
    pred = np.clip(pred, -1.0, 1.0)

    df = ds.df.copy()
    df["uav_ma3"] = ds.u_ma3.astype(np.float64)
    df["uav_sg"] = ds.u_sg.astype(np.float64)
    df["s_mask"] = ds.s_mask.astype(np.float64)
    df["dt"] = ds.dt.astype(np.float64)
    df["brios_pred_raw"] = pred_raw.astype(np.float64)
    df["brios_pred"] = pred.astype(np.float64)

    eps_tp = 1e-3

    end_idx = len(pred) - 1
    if "spline" in df.columns:
        end_idx = min(end_idx, last_valid_index(pd.to_numeric(df["spline"], errors="coerce").to_numpy()))
    if "net" in df.columns:
        end_idx = min(end_idx, last_valid_index(pd.to_numeric(df["net"], errors="coerce").to_numpy()))

    if end_idx < 2:
        raise ValueError("Effective length too short for CE (need length>=3).")

    series_dict: Dict[str, np.ndarray] = {}

    def add_series(name: str, arr: np.ndarray):
        x = np.asarray(arr, dtype=np.float64).reshape(-1)[: end_idx + 1]
        if np.isnan(x).any():
            x = fill_nan_linear(x)
        series_dict[name] = x

    for col, mname in [("spline", "Spline"), ("net", "Net"), ("T-RNN", "T-RNN"), ("line", "Line")]:
        if col in df.columns:
            add_series(mname, pd.to_numeric(df[col], errors="coerce").to_numpy())

    add_series("BRIOS_raw", pred_raw)
    add_series("BRIOS_smooth", pred)

    metrics_table = pd.DataFrame([summarize(k, series_dict[k], eps=eps_tp) for k in series_dict.keys()])

    meta_df = pd.DataFrame({
        "key": [
            "inferred_L", "eval_end_idx", "eval_length", "eps_peaks_valleys",
            "smooth_method", "smooth_window", "smooth_alpha",
            "epochs", "lr", "hidden", "weight_consistency", "weight_smooth_l2",
            "best_epoch", "best_loss", "best_loss_fit", "best_loss_cons", "best_loss_smooth"
        ],
        "value": [
            int(ds.L), int(end_idx), int(end_idx + 1), float(eps_tp),
            args.smooth_method, int(args.smooth_window), float(args.smooth_alpha),
            int(cfg.epochs), float(cfg.lr), int(cfg.hidden_dim), float(cfg.weight_consistency), float(cfg.weight_smooth_l2),
            int(best.get("epoch", -1)), float(best.get("loss", np.nan)),
            float(best.get("loss_fit", np.nan)), float(best.get("loss_cons", np.nan)), float(best.get("loss_smooth", np.nan)),
        ]
    })

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        keep_cols = ["doy", "s_data", "u_data", "s_mask", "dt", "uav_ma3", "uav_sg", "brios_pred_raw", "brios_pred"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df[keep_cols].to_excel(writer, index=False, sheet_name="reconstruction")
        metrics_table.to_excel(writer, index=False, sheet_name="metrics")
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    print("\n=== Metrics table (A.py style) ===")
    print(metrics_table.to_string(index=False))
    print("\nMeta:")
    print(meta_df.to_string(index=False))
    print(f"\nSaved:\n  {args.output}")


if __name__ == "__main__":
    main()
