import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# 0. 配置
# =========================
EXCEL_PATH = r'C:/Users/movjing/Desktop/latex/results/Fig1map/result25.xlsx'
SAVE_NAME = 'T-RNN_result_2024_trend_residual.png'

EPOCHS = 1200
LR = 0.001
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

HIDDEN_DIM = 4           # 若仍有轻微尖峰：8
HUBER_BETA = 0.02         # 0.01~0.05

# Trend 频率阶数（越大越能表示更复杂的长期波段，但仍很平滑）
FOURIER_K = 3             # 2~5 都可，建议先用 3

# 关键正则项（抑振核心）
LAMBDA_RES_L2 = 5.0       # 残差幅度惩罚（越大越不允许尖峰）
LAMBDA_RES_D1 = 2.0       # 残差一阶平滑（抑制锯齿）
LAMBDA_PRED_D2 = 10     # 输出二阶弱平滑（防局部曲率尖峰，别太大）

CLIP_NDVI = True
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# 1. 读数据
# =========================
df = pd.read_excel(EXCEL_PATH, index_col=0)

s = pd.to_numeric(df['s_data'], errors='coerce').astype(np.float32).to_numpy()
u = pd.to_numeric(df['u_data'], errors='coerce').astype(np.float32).to_numpy()

s_mask = ~np.isnan(s)
u_mask = ~np.isnan(u)
T = len(s)

print(f"T={T}, obs_count={(s_mask | u_mask).sum()}")

# =========================
# 2. 构造 delta_t（缺失段递增）并做 log1p
# =========================
delta_t = []
last_obs = -1
for i in range(T):
    has_obs = bool(s_mask[i] or u_mask[i])
    if has_obs:
        dt = 0.0 if last_obs < 0 else float(i - last_obs)
        last_obs = i
    else:
        dt = 1.0 if last_obs < 0 else float(i - last_obs)
    delta_t.append(dt)
delta_t = np.log1p(np.array(delta_t, dtype=np.float32)).astype(np.float32)

# =========================
# 3. 输入： [s_in, u_in, dt, s_mask, u_mask]
# =========================
s_in = np.nan_to_num(s, nan=0.0).astype(np.float32)
u_in = np.nan_to_num(u, nan=0.0).astype(np.float32)

X_np = np.stack([s_in, u_in, delta_t, s_mask.astype(np.float32), u_mask.astype(np.float32)], axis=1)
X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)  # (1,T,5)

# =========================
# 4. 监督目标：仅观测点监督（UAV优先）
# =========================
y_np = np.full((T,), np.nan, dtype=np.float32)
y_np[u_mask] = u[u_mask]
y_np[~u_mask & s_mask] = s[~u_mask & s_mask]

y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)          # (1,T,1)
mask = torch.tensor(~np.isnan(y_np), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
y_filled = torch.nan_to_num(y, nan=0.0)

# =========================
# 5. 损失：masked Huber
# =========================
huber = nn.SmoothL1Loss(reduction='none', beta=HUBER_BETA)

def masked_huber(pred, target_filled, mask, eps=1e-8):
    loss = huber(pred, target_filled)
    return (loss * mask).sum() / (mask.sum() + eps)

def smooth_d1(x):
    d1 = x[:, 1:, :] - x[:, :-1, :]
    return (d1 ** 2).mean()

def smooth_d2(x):
    d1 = x[:, 1:, :] - x[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return (d2 ** 2).mean()

# =========================
# 6. Trend 模块：低频 Fourier 基函数（长期波段）
#    Trend(t) = a0 + a1*t + Σ [bk*sin(2πkt) + ck*cos(2πkt)]
# =========================
class TrendFourier(nn.Module):
    def __init__(self, T, K=3):
        super().__init__()
        self.T = T
        self.K = K
        # 参数：a0, a1, 以及每阶的 sin/cos 系数
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(K))  # sin
        self.c = nn.Parameter(torch.zeros(K))  # cos

        # 预先构造 t（0~1）
        t = torch.linspace(0, 1, steps=T).view(1, T, 1)  # (1,T,1)
        self.register_buffer("t", t)

    def forward(self):
        t = self.t  # (1,T,1)
        trend = self.a0 + self.a1 * t.squeeze(-1)  # (1,T)
        for k in range(1, self.K + 1):
            trend = trend + self.b[k-1] * torch.sin(2 * np.pi * k * t.squeeze(-1)) \
                           + self.c[k-1] * torch.cos(2 * np.pi * k * t.squeeze(-1))
        return trend.unsqueeze(-1)  # (1,T,1)

# =========================
# 7. Residual 模块：TRNN 只输出残差 residual(t)
# =========================
class ResidualTRNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decay = nn.Parameter(torch.ones(hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        outputs = []
        for tt in range(T):
            dt = x[:, tt, 2:3]
            gamma = torch.exp(-dt * torch.abs(self.decay))
            h = h * gamma
            h = torch.tanh(self.Wx(x[:, tt, :]) + self.Wh(h))
            # 残差建议限制幅度，避免尖峰（tanh 后再缩放）
            r = 0.3 * torch.tanh(self.out(h))  # residual in roughly [-0.3,0.3]
            outputs.append(r)
        return torch.stack(outputs, dim=1)  # (B,T,1)

trend_model = TrendFourier(T=T, K=FOURIER_K)
res_model = ResidualTRNN(input_dim=5, hidden_dim=HIDDEN_DIM)

params = list(trend_model.parameters()) + list(res_model.parameters())
optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

# =========================
# 8. 训练
# =========================
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    trend = trend_model()              # (1,T,1)
    residual = res_model(X)            # (1,T,1)
    pred = trend + residual            # (1,T,1)

    # 观测点损失（拟合真实观测）
    loss_obs = masked_huber(pred, y_filled, mask)

    # 核心抑振：强约束 residual（幅度 + 平滑）
    loss_res_l2 = (residual ** 2).mean()
    loss_res_d1 = smooth_d1(residual)

    # 输出弱二阶平滑，防止局部曲率尖峰（别太强）
    loss_pred_d2 = 2*smooth_d2(pred)

    loss = (loss_obs
            + LAMBDA_RES_L2 * loss_res_l2
            + LAMBDA_RES_D1 * loss_res_d1
            + LAMBDA_PRED_D2 * loss_pred_d2)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | loss={loss.item():.6f} | obs={loss_obs.item():.6f} "
              f"| res_l2={loss_res_l2.item():.6f} | res_d1={loss_res_d1.item():.6f} | pred_d2={loss_pred_d2.item():.6f}")

# =========================
# 9. 推理与绘图
# =========================
trend_model.eval()
res_model.eval()
with torch.no_grad():
    trend = trend_model()
    residual = res_model(X)
    pred = (trend + residual).squeeze().cpu().numpy()

if CLIP_NDVI:
    pred = np.clip(pred, -1.0, 1.0)

spline = pd.to_numeric(df['spline'], errors='coerce').astype(np.float32).to_numpy()
net = pd.to_numeric(df['net'], errors='coerce').astype(np.float32).to_numpy()

plt.figure(figsize=(12, 4))
plt.plot(spline, label='Spline', linestyle='--')
plt.plot(net, label='UAV-T-LSTM (Ours)', linewidth=2)
plt.plot(pred, label='T-RNN (Trend+Residual)', linewidth=2)
plt.scatter(np.where(u_mask)[0], u[u_mask], s=18, c='k', label='UAV obs')

plt.title('T-RNN vs Proposed Method (2024)')
plt.xlabel('Time index')
plt.ylabel('NDVI')
plt.legend()
plt.tight_layout()

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, SAVE_NAME)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图片已保存到桌面：{save_path}")

import numpy as np
import pandas as pd

# =========================
# A) 计算有效截尾长度：以 spline / net 的有效末尾为准
# =========================
def last_valid_index(x):
    x = np.asarray(x)
    idx = np.where(~np.isnan(x))[0]
    return int(idx[-1]) if len(idx) > 0 else -1

end_spline = last_valid_index(spline)
end_net = last_valid_index(net)

end_idx = min(end_spline, end_net)
if end_idx < 2:
    raise ValueError("有效数据长度太短，无法计算 CE（至少需要长度>=3）。")

# 截尾：三者对齐长度
spline_cut = np.asarray(spline)[:end_idx+1]
net_cut = np.asarray(net)[:end_idx+1]
pred_cut = np.asarray(pred)[:end_idx+1]

print(f"截尾对齐长度：{len(pred_cut)} (end_idx={end_idx})")
print(f"spline_cut NaN count: {np.isnan(spline_cut).sum()}, net_cut NaN count: {np.isnan(net_cut).sum()}")

# =========================
# B) 若截尾后 spline/net 仍有 NaN：为保证 CE/TV 可计算，做线性填补
#    （仅用于结构指标计算，论文中可注明）
# =========================
def fill_nan_linear(x):
    s = pd.Series(x)
    return s.interpolate(method='linear', limit_direction='both').to_numpy()

spline_eval = fill_nan_linear(spline_cut) if np.isnan(spline_cut).any() else spline_cut
net_eval = fill_nan_linear(net_cut) if np.isnan(net_cut).any() else net_cut
pred_eval = pred_cut  # T-RNN 通常无 NaN

# =========================
# C) 指标：CE / TV / 峰谷数量
# =========================
def curvature_energy(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x[2:] - 2*x[1:-1] + x[:-2])**2))

def total_variation(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(np.abs(np.diff(x))))

def count_peaks_valleys(x, eps=1e-3):
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x)
    d[np.abs(d) < eps] = 0.0
    s = np.sign(d)

    # 填补 0 符号，避免虚假峰谷
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    for i in range(len(s)-2, -1, -1):
        if s[i] == 0:
            s[i] = s[i+1]

    peaks = int(np.sum((s[:-1] > 0) & (s[1:] < 0)))
    valleys = int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
    return peaks, valleys

def summarize(name, series, eps=1e-3):
    ce = curvature_energy(series)
    tv = total_variation(series)
    p, v = count_peaks_valleys(series, eps=eps)
    return {
        "Method": name,
        "Length": len(series),
        "CE": ce,
        "TV": tv,
        "N_peaks": p,
        "N_valleys": v,
        "N_turning_points": p + v
    }

rows = [
    summarize("Spline", spline_eval, eps=1e-3),
    summarize("T-RNN", pred_eval, eps=1e-3),
    summarize("UAV-T-LSTM (Ours)", net_eval, eps=1e-3),
]

metrics_df = pd.DataFrame(rows)
print("\n===== Metrics on aligned (truncated) series =====")
print(metrics_df)

# 可选：导出到桌面
import os
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
metrics_path = os.path.join(desktop, "metrics_summary_aligned.xlsx")
metrics_df.to_excel(metrics_path, index=False)
print(f"\n指标汇总已导出到：{metrics_path}")

