
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIGAN baseline runner (single-file) for the user's 2024 UAV–satellite NDVI dataset.

Inputs:
  - 25data.xlsx with columns: doy, s_data (satellite NDVI, sparse), u_data (UAV NDVI, sparse),
    and optionally other columns (net/line/spline/T-RNN) for convenience comparison.

Outputs:
  - CSV: pigan_25data_outputs.csv (doy, s_data, u_data, pigan_pred)
  - NPY: pigan_pred.npy (length 360)
  - Prints structural metrics: CE / TV / Turning Points

Notes:
  - This script implements a minimal UAV-guided instantiation consistent with the PIGAN-style TSFGAN codebase:
    * Satellite NDVI is the target (s2_len=360).
    * UAV NDVI is used as auxiliary guidance input (s1_len=360).
    * ROI-level time feature (sin DOY) is provided as an additional channel.
    * Missing satellite points are masked; a small subset of observed satellite points are randomly "faked" as missing
      for self-supervised training (imputation GAN training).
  - Default runs on CPU. You may reduce epochs for quick checks.

Run:
  python pigan_25data_baseline.py --xlsx /path/to/25data.xlsx --epochs 50 --fake-num 10
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Structural metrics (paper-style)
# ----------------------------
def curvature_energy(x: np.ndarray) -> float:
    """
    Curvature Energy (CE): sum of squared second differences.
    Matched to A.py evaluation script.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return float("nan")
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.sum(d2 ** 2))



def total_variation(x: np.ndarray) -> float:
    """
    Total Variation (TV): sum absolute first differences.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return float("nan")
    d1 = np.diff(x)
    return float(np.sum(np.abs(d1)))

def last_valid_index(x: np.ndarray) -> int:
    x = np.asarray(x)
    idx = np.where(~np.isnan(x))[0]
    return int(idx[-1]) if len(idx) > 0 else -1


def fill_nan_linear(x: np.ndarray) -> np.ndarray:
    # Linear interpolation for metric computation only (matched to A.py).
    s = pd.Series(x)
    return s.interpolate(method="linear", limit_direction="both").to_numpy()


def count_peaks_valleys(x: np.ndarray, eps: float = 1e-3):
    """Count peaks/valleys using sign-change on first differences with epsilon pruning (matched to A.py)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return 0, 0
    d = np.diff(x)
    d[np.abs(d) < eps] = 0.0
    s = np.sign(d)

    # Fill zero signs to avoid spurious peaks/valleys
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]

    peaks = int(np.sum((s[:-1] > 0) & (s[1:] < 0)))
    valleys = int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
    return peaks, valleys


def turning_points(x: np.ndarray, eps: float = 1e-3) -> int:
    p, v = count_peaks_valleys(x, eps=eps)
    return int(p + v)

def summarize(name: str, series: np.ndarray, eps: float = 1e-3) -> dict:
    """A.py-style row: CE/TV/peaks/valleys/turning points."""
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





def smooth_moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple centered moving-average smoothing."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def smooth_exponential(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential moving average smoothing."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y



# ----------------------------
# Losses (from provided PIGAN codebase style)
# ----------------------------
class GANLoss(nn.Module):
    """
    MSE GAN loss (LSGAN style), consistent with the provided utils.py implementation.
    Accepts either:
      - bool target (real/fake), or
      - a tensor target mask/probability map (as used in the provided TSFGAN code).
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        device = prediction.device
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction).to(device)

    def forward(self, prediction: torch.Tensor, target) -> torch.Tensor:
        # bool -> constant label; tensor -> regression target
        if isinstance(target, (bool, np.bool_)):
            tgt = self.get_target_tensor(prediction, bool(target))
        else:
            tgt = target.to(prediction.device)
            if tgt.shape != prediction.shape:
                tgt = tgt.expand_as(prediction)
        return self.loss(prediction, tgt)


class TVLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        return diff.sum()


class IMaskLoss2(nn.Module):
    """
    imask_loss_2 equivalent (lam1=5, lam2=0.002) in the provided code.
    """
    def __init__(self, lam1: float = 5.0, lam2: float = 0.002):
        super().__init__()
        self.l1 = nn.MSELoss()
        self.tv = TVLoss()
        self.lam1 = lam1
        self.lam2 = lam2

    def forward(
        self,
        predict: torch.Tensor,
        predict_ori: torch.Tensor,
        target: torch.Tensor,
        mask_fake: torch.Tensor,
        mask_real: torch.Tensor
    ) -> torch.Tensor:
        # mask_fake/mask_real are 1 for "missing", 0 for "observed" in the codebase convention
        # but the loss uses (1 - mask_fake) and mask_real as weights
        loss = (
            self.lam1 * self.l1(
                predict * (1.0 - mask_fake.unsqueeze(1)),
                target * (1.0 - mask_fake.unsqueeze(1)),
            ) / torch.mean(1.0 - mask_fake.unsqueeze(1))
            + self.l1(
                predict_ori * mask_real.unsqueeze(1),
                target * mask_real.unsqueeze(1),
            ) / torch.mean(mask_real.unsqueeze(1))
            + self.lam2 * self.tv(predict)
        )
        return loss


def huber_loss(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    error = y_pred - y_true
    squared = 0.5 * error ** 2
    linear = delta * (torch.abs(error) - 0.5 * delta)
    is_small = torch.abs(error) <= delta
    loss = torch.where(is_small, squared, linear)
    return torch.mean(loss)


# ----------------------------
# Minimal discriminator (dilated)
# ----------------------------
class AllOnesConv(nn.Module):
    """
    Fixed conv1d with all-ones kernel (normalized), used to build probability mask.
    Consistent with provided Dis.py behavior (gap_day=5, kernel_size=360/5).
    """
    def __init__(self, s2_len: int = 360, gap_day: int = 5):
        super().__init__()
        self.gap_day = gap_day
        self.kernel_size = int(s2_len / gap_day)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=1, dilation=gap_day, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.kernel_size)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ModelDDilate(nn.Module):
    def __init__(self, s2_len: int = 360, input_channels: int = 1, gap_day: int = 5):
        super().__init__()
        self.feat = 64
        self.gap_day = gap_day
        self.kernel_size = int(s2_len / gap_day)
        self.dilate_mov = nn.Conv1d(input_channels, self.feat, kernel_size=self.kernel_size, dilation=self.gap_day)
        self.mlp = nn.Sequential(
            nn.Linear(self.feat, self.feat * 4),
            nn.ReLU(True),
            nn.Linear(self.feat * 4, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dilate_mov(x)
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        return self.sig(x)


# ----------------------------
# Generator model (adapted from provided Gen.py)
# ----------------------------
class BlockModel(nn.Module):
    def __init__(self, channels: int, input_len: int, out_len: int, individual: bool, use_all: bool):
        super().__init__()
        self.channels = channels
        self.in_channels = 2
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual
        self.use_all = use_all

        if self.use_all:
            self.ac = nn.Sequential(
                nn.Conv1d(self.channels, self.in_channels, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.in_channels, self.channels, 5, 1, 2),
            )
        if self.individual:
            self.linear_channel = nn.ModuleList([nn.Linear(self.input_len, self.out_len) for _ in range(self.channels)])
        else:
            self.linear_channel = nn.Linear(self.input_len, self.out_len)

        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_all:
            x = self.ac(x)
        if self.individual:
            out = torch.zeros((x.size(0), x.size(1), self.out_len), dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, i, :] = self.linear_channel[i](x[:, i, :])
        else:
            out = self.linear_channel(x)
        return out


class GeneratorModel(nn.Module):
    """
    Same core U-shaped temporal model as provided Gen.py (ROI-level),
    with two branches: auxiliary (s1) and target+time+mask (s2 input).
    """
    def __init__(self, s1_len: int, s2_len: int, s1_channels: int, s2_channels: int, act: bool = True):
        super().__init__()
        self.s1_len = s1_len
        self.s2_len = s2_len
        self.s1_channels = s1_channels
        self.s2_channels = s2_channels
        self.mid_channels = 4
        self.individual = True
        self.act = act

        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        down_in_s1 = [int(np.ceil(self.s1_len / f)) for f in filters]
        down_in_s2 = [int(np.ceil(self.s2_len / f)) for f in filters]
        down_out = [int(np.ceil(self.s2_len / f)) for f in filters]

        self.s1b = nn.Sequential(nn.Conv1d(self.s1_channels, self.mid_channels, 1, 1), nn.ReLU(inplace=True))
        self.s2b = nn.Sequential(nn.Conv1d(self.s2_channels, self.mid_channels, 1, 1), nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.down1_s1 = BlockModel(self.mid_channels, down_in_s1[0], down_out[0], self.individual, True)
        self.down2_s1 = BlockModel(self.mid_channels, down_in_s1[1], down_out[1], self.individual, True)
        self.down3_s1 = BlockModel(self.mid_channels, down_in_s1[2], down_out[2], self.individual, True)
        self.down4_s1 = BlockModel(self.mid_channels, down_in_s1[3], down_out[3], self.individual, True)

        self.down1_s2 = BlockModel(self.mid_channels, down_in_s2[0], down_out[0], self.individual, True)
        self.down2_s2 = BlockModel(self.mid_channels, down_in_s2[1], down_out[1], self.individual, True)
        self.down3_s2 = BlockModel(self.mid_channels, down_in_s2[2], down_out[2], self.individual, True)
        self.down4_s2 = BlockModel(self.mid_channels, down_in_s2[3], down_out[3], self.individual, True)

        self.up3 = BlockModel(self.mid_channels * 2, down_out[2] + down_out[3], down_out[2], self.individual, True)
        self.up2 = BlockModel(self.mid_channels * 2, down_out[1] + down_out[2], down_out[1], self.individual, True)
        self.up1 = BlockModel(self.mid_channels * 2, down_out[0] + down_out[1], down_out[0], self.individual, True)

        self.out = nn.Conv1d(self.mid_channels * 2, 1, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s1: [B, C1, L1], s2: [B, C2, L2]; L2 == s2_len
        x1_1 = self.s1b(s1)
        e1_1 = self.down1_s1(x1_1)

        x2_1 = self.maxpool1(x1_1)
        e2_1 = self.down2_s1(x2_1)

        x3_1 = self.maxpool2(x2_1)
        e3_1 = self.down3_s1(x3_1)

        x4_1 = self.maxpool3(x3_1)
        e4_1 = self.down4_s1(x4_1)

        x1_2 = self.s2b(s2)
        e1_2 = self.down1_s2(x1_2)

        x2_2 = self.maxpool1(x1_2)
        e2_2 = self.down2_s2(x2_2)

        x3_2 = self.maxpool2(x2_2)
        e3_2 = self.down3_s2(x3_2)

        x4_2 = self.maxpool3(x3_2)
        e4_2 = self.down4_s2(x4_2)

        e4 = torch.cat((e4_1, e4_2), dim=1)
        e3 = torch.cat((e3_1, e3_2), dim=1)
        e2 = torch.cat((e2_1, e2_2), dim=1)
        e1 = torch.cat((e1_1, e1_2), dim=1)

        d4 = torch.cat((e3, e4), dim=2)
        d4 = self.up3(d4)

        d3 = torch.cat((e2, d4), dim=2)
        d3 = self.up2(d3)

        d2 = torch.cat((e1, d3), dim=2)
        out = self.up1(d2)

        out = self.out(out)
        if self.act:
            out = self.tanh(out)

        ori = out  # generator raw output in [-1, 1] then masked fusion happens below

        # s2 input convention: channel 0 is NDVI_in; last channel is "mask" (1 observed, 0 missing)
        ndvi_in = s2[:, 0, :].unsqueeze(1)
        obs_mask = s2[:, -1, :].unsqueeze(1)
        fused = ndvi_in * obs_mask + ori * (1.0 - obs_mask)
        return fused, ori


# ----------------------------
# TSFGAN wrapper (PIGAN-style)
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.5, 0.999)
    fake_num: int = 10
    seed: int = 42
    out_dir: str = "./models/PIGAN_25data"


class TSFGAN(nn.Module):
    def __init__(self, s1_len: int, s2_len: int, s1_channels: int, ind_channels: int, device: torch.device):
        super().__init__()
        self.s1_len = s1_len
        self.s2_len = s2_len
        self.device = device

        # generator input: ind_channels + 1 (mask)
        self.netG = GeneratorModel(
            s1_len=s1_len,
            s2_len=s2_len,
            s1_channels=s1_channels,
            s2_channels=ind_channels + 1,
            act=True,
        )
        self.netD = ModelDDilate(s2_len=s2_len, input_channels=1)

        self.criterion_train = IMaskLoss2()
        self.criterion_gan = GANLoss()
        self.l2 = nn.MSELoss()
        self.pro_conv = AllOnesConv(s2_len=s2_len)
        self.cl_loss = huber_loss

        self.optimizer_G: Optional[torch.optim.Optimizer] = None
        self.optimizer_D: Optional[torch.optim.Optimizer] = None

        self.to(self.device)

    def init_optim(self, lr: float, betas: Tuple[float, float]) -> None:
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=betas)

    def set_input(self, batch: Dict[str, torch.Tensor]) -> None:
        # Convention:
        #   s*_mask_real/fake are 1 for "missing", 0 for "observed"
        self.real_s1 = batch["s1"].to(self.device)          # [B, C1, L1]
        self.real_s1_syn = batch["s1_fake"].to(self.device)

        self.s1_mask = (1.0 - batch["s1_mask"]).to(self.device)
        self.s1_mask_fake = (1.0 - batch["s1_mask_fake"]).to(self.device)
        self.s1_mask_real = (1.0 - batch["s1_mask_real"]).to(self.device)

        self.ind = batch["ind"].to(self.device)             # [B, Cind, L2]
        self.ind_syn = batch["ind_fake"].to(self.device)

        self.ndvi = batch["NDVI"].to(self.device)           # [B, 1, L2]
        self.ndvi_syn = batch["NDVI_fake"].to(self.device)

        self.s2_mask = (1.0 - batch["s2_mask"]).to(self.device)          # 1 observed, 0 masked
        self.s2_mask_fake = (1.0 - batch["s2_mask_fake"]).to(self.device)
        self.s2_mask_real = (1.0 - batch["s2_mask_real"]).to(self.device)

    def forward_G(self, use_syn: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        ind = self.ind_syn if use_syn else self.ind
        mask = self.s2_mask if use_syn else self.s2_mask_real
        s2_in = torch.cat([ind, mask.unsqueeze(1)], dim=1)
        return self.netG(self.real_s1, s2_in)

    def optimize_parameters(self) -> Dict[str, float]:
        assert self.optimizer_G is not None and self.optimizer_D is not None

        # G forward with synthetic missingness
        self.fake_B, self.fake_B_ori = self.forward_G(use_syn=True)
        self.pro_mask = self.pro_conv(self.s2_mask.unsqueeze(1))

        # ---- Update D ----
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        pred_fake = self.netD(self.fake_B.detach())
        loss_D = self.criterion_gan(pred_fake, self.pro_mask.detach())
        loss_D.backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)

        # ---- Update G ----
        self.optimizer_G.zero_grad()

        # supervised + TV loss on (fake-masked positions) plus observed loss on real observed positions
        loss_G_L2 = self.criterion_train(self.fake_B, self.fake_B_ori, self.ndvi, self.s2_mask_fake, self.s2_mask_real)

        # consistency loss: compare ori outputs between synthetic and full observed mask
        full_B, full_B_ori = self.forward_G(use_syn=False)
        consist = self.cl_loss(self.fake_B_ori, full_B_ori.detach())

        # GAN loss for generator
        mean_value = self.pro_mask.mean()
        pro_gt = torch.where(self.pro_mask < mean_value, mean_value, self.pro_mask)
        pred = self.netD(self.fake_B)
        loss_G_GAN = self.criterion_gan(pred, pro_gt.detach())

        loss_G = 0.01 * loss_G_GAN + loss_G_L2 + consist
        loss_G.backward()
        self.optimizer_G.step()

        return {
            "loss_D": float(loss_D.detach().cpu().item()),
            "loss_G": float(loss_G.detach().cpu().item()),
            "loss_G_L2": float(loss_G_L2.detach().cpu().item()),
            "loss_G_GAN": float(loss_G_GAN.detach().cpu().item()),
            "loss_consist": float(consist.detach().cpu().item()),
        }

    @torch.no_grad()
    def reconstruct(self) -> np.ndarray:
        """
        Final reconstruction using full observed satellite mask (no synthetic masking).
        Returns a 1D numpy array of length s2_len (fused output).
        """
        self.eval()
        fused, _ori = self.forward_G(use_syn=False)
        pred = fused[0, 0, :].detach().cpu().numpy().astype(np.float64)
        return pred

    def set_requires_grad(self, net: nn.Module, requires_grad: bool) -> None:
        for p in net.parameters():
            p.requires_grad = requires_grad


# ----------------------------
# Single-sample dataset from 25data.xlsx
# ----------------------------
class DatasetXLSX(Dataset):
    def __init__(self, xlsx_path: str, s2_len: int = 0, fake_num: int = 10):
        self.xlsx_path = xlsx_path
        self.s2_len = s2_len
        self.fake_num = fake_num

        df = pd.read_excel(xlsx_path)
        required = {"doy", "s_data", "u_data"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"25data.xlsx is missing required columns: {sorted(missing)}")

        # infer sequence length if not provided:
        # - Prefer max(doy) if available; otherwise use row count.
        if s2_len is None or int(s2_len) <= 0:
            if df["doy"].notna().any():
                inferred = int(np.nanmax(df["doy"].to_numpy(dtype=np.float64)))
            else:
                inferred = int(len(df))
            s2_len = inferred
        self.s2_len = int(s2_len)

        # enforce DOY 1..self.s2_len, and reindex to a complete 1..L grid
        df = df.sort_values("doy")
        df = df[(df["doy"] >= 1) & (df["doy"] <= self.s2_len)].copy()
        df = df.set_index("doy").reindex(range(1, self.s2_len + 1)).reset_index().rename(columns={"index": "doy"})

        self.df = df

        # satellite / UAV series
        self.s_data = df["s_data"].to_numpy(dtype=np.float64)
        self.u_data = df["u_data"].to_numpy(dtype=np.float64)

        # observation masks (True = observed)
        self.s_obs = np.isfinite(self.s_data)
        self.u_obs = np.isfinite(self.u_data)

        # fill NaNs with 0 for network input, consistent with original codebase convention
        self.s_in = np.where(self.s_obs, self.s_data, 0.0)
        self.u_in = np.where(self.u_obs, self.u_data, 0.0)

        # time feature (sin DOY)
        doy = df["doy"].to_numpy(dtype=np.float64)
        days_in_year = 365.0  # use 365 for 2025; acceptable for partial-year series
        self.t_sin = np.sin(2.0 * np.pi * doy / days_in_year).astype(np.float64)

    def __len__(self) -> int:
        # single ROI sample
        return 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # s2 missing flag like QA60: 1 for missing, 0 for observed
        s2_mask_real = (~self.s_obs).astype(np.float32)

        # fake mask: choose fake_num points among observed satellite points
        obs_idx = np.where(self.s_obs)[0]
        if obs_idx.size == 0:
            raise ValueError("No observed satellite points found in s_data; cannot train imputation baseline.")
        n = min(self.fake_num, obs_idx.size)
        fake_idx = np.random.choice(obs_idx, size=n, replace=False)
        s2_mask_fake = np.zeros(self.s2_len, dtype=np.float32)
        s2_mask_fake[fake_idx] = 1.0

        s2_mask = np.clip(s2_mask_real + s2_mask_fake, 0.0, 1.0).astype(np.float32)

        # ind channels: [NDVI_in, time_sin]
        ind = np.stack([self.s_in.astype(np.float32), self.t_sin.astype(np.float32)], axis=0)  # [2, L]
        ind_fake = ind.copy()
        # zero-out where masked (real missing + fake missing)
        ind_fake[:, s2_mask.astype(bool)] = 0.0

        ndvi = self.s_in.astype(np.float32).reshape(1, self.s2_len)
        ndvi_fake = ind_fake[0:1, :]

        # s1: UAV guidance as a 1-channel series (same length as s2 here)
        s1 = self.u_in.astype(np.float32).reshape(1, self.s2_len)
        s1_mask_real = (~self.u_obs).astype(np.float32)
        s1_mask_fake = np.zeros_like(s1_mask_real, dtype=np.float32)
        s1_mask = np.clip(s1_mask_real + s1_mask_fake, 0.0, 1.0).astype(np.float32)
        s1_fake = s1.copy()
        # zero-out UAV missing in the input
        s1[:, s1_mask_real.astype(bool)] = 0.0
        s1_fake[:, s1_mask_real.astype(bool)] = 0.0

        # Provide placeholders for keys expected by TSFGAN
        batch: Dict[str, Any] = {
            "cropland": torch.tensor([0], dtype=torch.int64),
            "s1": torch.tensor(s1, dtype=torch.float32),
            "s1_fake": torch.tensor(s1_fake, dtype=torch.float32),
            "s1_mask_real": torch.tensor(s1_mask_real, dtype=torch.float32),
            "s1_mask_fake": torch.tensor(s1_mask_fake, dtype=torch.float32),
            "s1_mask": torch.tensor(s1_mask, dtype=torch.float32),

            "ind": torch.tensor(ind, dtype=torch.float32),
            "ind_fake": torch.tensor(ind_fake, dtype=torch.float32),

            "NDVI": torch.tensor(ndvi, dtype=torch.float32),
            "NDVI_fake": torch.tensor(ndvi_fake, dtype=torch.float32),

            "s2_mask_real": torch.tensor(s2_mask_real, dtype=torch.float32),
            "s2_mask_fake": torch.tensor(s2_mask_fake, dtype=torch.float32),
            "s2_mask": torch.tensor(s2_mask, dtype=torch.float32),

            # not used by our TSFGAN wrapper; kept for completeness
            "s2": torch.zeros((1, self.s2_len), dtype=torch.float32),
            "s2_fake": torch.zeros((1, self.s2_len), dtype=torch.float32),
        }
        return batch


# ----------------------------
# Train / Run
# ----------------------------
def train_model(model: TSFGAN, loader: DataLoader, cfg: TrainConfig) -> None:
    model.train()
    model.init_optim(cfg.lr, cfg.betas)

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for batch in loader:
            model.set_input(batch)
            loss_dict = model.optimize_parameters()
            losses.append(loss_dict)

        # epoch logging
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            mean_loss = {k: float(np.mean([d[k] for d in losses])) for k in losses[0].keys()}
            print(f"[Epoch {epoch:03d}/{cfg.epochs}] "
                  f"G={mean_loss['loss_G']:.6f} "
                  f"(L2={mean_loss['loss_G_L2']:.6f}, GAN={mean_loss['loss_G_GAN']:.6f}, C={mean_loss['loss_consist']:.6f}) "
                  f"D={mean_loss['loss_D']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default=r"C:\Users\movjing\Desktop\25data.xlsx", help="Path to 25data.xlsx")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length (0 = infer from xlsx; e.g., 230 for 2025)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep 1 for single ROI)")
    parser.add_argument("--fake-num", type=int, default=10, help="Number of observed satellite points to fake-mask")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smooth-method", type=str, default="ma", choices=["none","ma","ema"], help="Post-smoothing method for pigan prediction")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window (odd) when smooth-method=ma")
    parser.add_argument("--smooth-alpha", type=float, default=0.2, help="EMA alpha when smooth-method=ema")
    parser.add_argument("--out-dir", type=str, default=r"C:\Users\movjing\Desktop", help="Output directory")
    parser.add_argument("--out-xlsx", type=str, default=r"C:\Users\movjing\Desktop\25datagan.xlsx", help="Output Excel path (.xlsx)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataset
    ds = DatasetXLSX(args.xlsx, s2_len=(args.seq_len if args.seq_len else 0), fake_num=args.fake_num)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    # model lengths (infer from xlsx): s1_len == s2_len == L
    L = ds.s2_len
    model = TSFGAN(s1_len=L, s2_len=L, s1_channels=1, ind_channels=2, device=device)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, fake_num=args.fake_num, seed=args.seed)

    # train
    train_model(model, loader, cfg)

    # reconstruct (full observed mask)
    eval_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(eval_loader))
    model.set_input(batch)
    pred = model.reconstruct()
    pred = np.clip(pred, -1.0, 1.0)  # model uses tanh; keep consistent

    # optional post-smoothing (does NOT affect training; only affects reported curve)
    if args.smooth_method == "ma":
        pred_smooth = smooth_moving_average(pred, window=args.smooth_window)
    elif args.smooth_method == "ema":
        pred_smooth = smooth_exponential(pred, alpha=args.smooth_alpha)
    else:
        pred_smooth = pred.copy()
    pred_smooth = np.clip(pred_smooth, -1.0, 1.0)

    # structural metrics (raw and smoothed) using A.py evaluation protocol:
    # - Align length by truncating to a common valid tail (prefer 'spline'/'net' columns if present)
    # - If truncated baseline series still contain NaN, linearly interpolate for metric computation only
    eps_tp = 1e-3

    end_idx = len(pred) - 1
    if "spline" in ds.df.columns and "net" in ds.df.columns:
        spline_ref = pd.to_numeric(ds.df["spline"], errors="coerce").astype(np.float64).to_numpy()
        net_ref = pd.to_numeric(ds.df["net"], errors="coerce").astype(np.float64).to_numpy()
        end_idx = min(end_idx, last_valid_index(spline_ref), last_valid_index(net_ref))
    elif "spline" in ds.df.columns:
        spline_ref = pd.to_numeric(ds.df["spline"], errors="coerce").astype(np.float64).to_numpy()
        end_idx = min(end_idx, last_valid_index(spline_ref))
    elif "net" in ds.df.columns:
        net_ref = pd.to_numeric(ds.df["net"], errors="coerce").astype(np.float64).to_numpy()
        end_idx = min(end_idx, last_valid_index(net_ref))

    if end_idx < 2:
        raise ValueError("有效数据长度太短，无法计算 CE（至少需要长度>=3）。")

    pred_raw_eval = np.asarray(pred, dtype=np.float64)[: end_idx + 1]
    pred_smooth_eval = np.asarray(pred_smooth, dtype=np.float64)[: end_idx + 1]

    if np.isnan(pred_raw_eval).any():
        pred_raw_eval = fill_nan_linear(pred_raw_eval)
    if np.isnan(pred_smooth_eval).any():
        pred_smooth_eval = fill_nan_linear(pred_smooth_eval)

    ce = curvature_energy(pred_raw_eval)
    tv = total_variation(pred_raw_eval)
    tp = turning_points(pred_raw_eval, eps=eps_tp)

    ce_s = curvature_energy(pred_smooth_eval)
    tv_s = total_variation(pred_smooth_eval)
    tp_s = turning_points(pred_smooth_eval, eps=eps_tp)
    # save outputs (Excel requested)
    out_xlsx = args.out_xlsx
    out_npy_raw = os.path.join(args.out_dir, "pigan_pred_raw.npy")
    out_npy_smooth = os.path.join(args.out_dir, "pigan_pred.npy")

    df = ds.df.copy()
    df["pigan_pred_raw"] = pred
    df["pigan_pred"] = pred_smooth  # smoothed curve for reporting

    # Build A.py-style metrics table (one row per method)
    series_dict = {}
    # Optional baselines from the input sheet (if present)
    if "spline" in df.columns:
        x = pd.to_numeric(df["spline"], errors="coerce").astype(np.float64).to_numpy()[: end_idx + 1]
        if np.isnan(x).any():
            x = fill_nan_linear(x)
        series_dict["Spline"] = x
    if "net" in df.columns:
        x = pd.to_numeric(df["net"], errors="coerce").astype(np.float64).to_numpy()[: end_idx + 1]
        if np.isnan(x).any():
            x = fill_nan_linear(x)
        series_dict["Net"] = x
    if "T-RNN" in df.columns:
        x = pd.to_numeric(df["T-RNN"], errors="coerce").astype(np.float64).to_numpy()[: end_idx + 1]
        if np.isnan(x).any():
            x = fill_nan_linear(x)
        series_dict["T-RNN"] = x
    if "line" in df.columns:
        x = pd.to_numeric(df["line"], errors="coerce").astype(np.float64).to_numpy()[: end_idx + 1]
        if np.isnan(x).any():
            x = fill_nan_linear(x)
        series_dict["Line"] = x
    
    # PIGAN raw / smoothed
    series_dict["PIGAN_raw"] = pred_raw_eval
    series_dict["PIGAN_smooth"] = pred_smooth_eval
    
    rows = [summarize(name, series_dict[name], eps=eps_tp) for name in series_dict.keys()]
    metrics_table = pd.DataFrame(rows)
    
    meta_df = pd.DataFrame({
        "key": ["eval_end_idx", "eval_length", "eps_peaks_valleys", "smooth_method", "smooth_window", "smooth_alpha"],
        "value": [int(end_idx), int(end_idx + 1), float(eps_tp), args.smooth_method, int(args.smooth_window), float(args.smooth_alpha)],
    })

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df[["doy", "s_data", "u_data", "pigan_pred_raw", "pigan_pred"]].to_excel(writer, index=False, sheet_name="pred")
        metrics_table.to_excel(writer, index=False, sheet_name="metrics")
        meta_df.to_excel(writer, index=False, sheet_name="meta")

    np.save(out_npy_raw, pred)
    np.save(out_npy_smooth, pred_smooth)

    # print structural metrics
    print("\n=== PIGAN (UAV-guided) structural metrics on 2025 dataset (A.py metrics; truncated alignment) ===")
    print(f"Eval length: {end_idx+1} (end_idx={end_idx}), peaks/valleys eps={eps_tp}")
    print("\n=== Metrics table (A.py style) ===")
    print(metrics_table.to_string(index=False))
    print("Raw prediction:")
    print(f"  CE  = {ce:.6f}")
    print(f"  TV  = {tv:.6f}")
    print(f"  TPs = {tp:d}")
    print("Smoothed prediction (reported curve):")
    print(f"  CE  = {ce_s:.6f}")
    print(f"  TV  = {tv_s:.6f}")
    print(f"  TPs = {tp_s:d}")

    # Optional: if the sheet contains other trajectories, compute their metrics for quick sanity checks
    for col in ["net", "spline", "T-RNN", "line"]:
        if col in df.columns:
            arr = df[col].to_numpy(dtype=np.float64)
            if np.isfinite(arr).all():
                print(f"\n--- Metrics for column '{col}' (from 25data.xlsx) ---")
                print(f"CE  = {curvature_energy(arr):.6f}")
                print(f"TV  = {total_variation(arr):.6f}")
                print(f"TPs = {turning_points(arr):d}")

    print(f"\nSaved:\n  {out_xlsx}\n  {out_npy_smooth}\n  {out_npy_raw}")


if __name__ == "__main__":
    main()