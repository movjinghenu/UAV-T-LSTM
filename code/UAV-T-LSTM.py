import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import copy
import os
from scipy.signal import savgol_filter  # 添加SG平滑滤波

# 确保输出目录存在
os.makedirs('C:/Users/movjing/Desktop', exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===================== 数据预处理 =====================
def load_data(file_path):
    df = pd.read_excel(file_path)
    doy = df.iloc[:, 0].values.astype(float)
    data = df.iloc[:, 1].values.astype(float)
    return doy, data


# 加载三个数据集
u_doy, u_data = load_data('C:/Users/movjing/Desktop/u24_data.xlsx')
s_doy, s_data = load_data('C:/Users/movjing/Desktop/s23_data.xlsx')
us_df = pd.read_excel('C:/Users/movjing/Desktop/us23_data.xlsx')
us_doy = us_df.iloc[:, 0].values.astype(float)
us_u = us_df.iloc[:, 1].values.astype(float)
us_s = us_df.iloc[:, 2].values.astype(float)

# 归一化数据值到[0,1]范围
scaler_u = MinMaxScaler(feature_range=(0, 1)).fit(u_data.reshape(-1, 1))
scaler_s = MinMaxScaler(feature_range=(0, 1)).fit(s_data.reshape(-1, 1))

u_data_norm = scaler_u.transform(u_data.reshape(-1, 1)).flatten()
s_data_norm = scaler_s.transform(s_data.reshape(-1, 1)).flatten()
us_u_norm = scaler_u.transform(us_u.reshape(-1, 1)).flatten()
us_s_norm = scaler_s.transform(us_s.reshape(-1, 1)).flatten()

# 创建无人机特征增强的卫星数据集
enhanced_s_data = []
enhanced_s_deltas = []

for i, s_d in enumerate(s_doy):
    # 找到时间上最接近的无人机数据点
    time_diffs = np.abs(u_doy - s_d)
    closest_idx = np.argmin(time_diffs)

    # 创建增强特征: [卫星数据, 无人机数据, 时间差]
    u_val = u_data_norm[closest_idx]
    time_diff = time_diffs[closest_idx]
    enhanced_s_data.append([s_data_norm[i], u_val, time_diff])

    # 计算时间间隔
    if i == 0:
        enhanced_s_deltas.append(np.mean(np.diff(s_doy)))
    else:
        enhanced_s_deltas.append(s_doy[i] - s_doy[i - 1])

enhanced_s_data = np.array(enhanced_s_data)
enhanced_s_deltas = np.array(enhanced_s_deltas)

# 归一化增强特征
s_scaler = MinMaxScaler(feature_range=(0, 1)).fit(enhanced_s_data)
enhanced_s_data_norm = s_scaler.transform(enhanced_s_data)


# 计算时间间隔并归一化
def calc_normalized_deltas(doy):
    deltas = np.zeros_like(doy)
    deltas[1:] = doy[1:] - doy[:-1]
    deltas[0] = np.mean(deltas[1:])  # 第一个点用平均值替代

    # 归一化时间间隔
    delta_scaler = MinMaxScaler(feature_range=(0.01, 1))
    normalized_deltas = delta_scaler.fit_transform(deltas.reshape(-1, 1)).flatten()
    return normalized_deltas, delta_scaler


u_deltas, u_delta_scaler = calc_normalized_deltas(u_doy)
s_deltas, s_delta_scaler = calc_normalized_deltas(s_doy)
us_deltas, us_delta_scaler = calc_normalized_deltas(us_doy)

# 为UAV分支创建增强特征
enhanced_u_data = []
for i, u_d in enumerate(u_doy):
    # 创建增强特征: [无人机数据, 0, 0] - 使用0作为占位符
    enhanced_u_data.append([u_data_norm[i], 0, 0])

enhanced_u_data = np.array(enhanced_u_data)
enhanced_u_data_norm = s_scaler.transform(enhanced_u_data)  # 使用相同的scaler


# ===================== T-LSTM单元 =====================
class UAVEnhancedTLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 更稳健的初始化
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # 初始化权重
        for name, param in self.lstm_cell.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        # 时间衰减参数
        self.Wd = nn.Parameter(torch.randn(hidden_size))
        nn.init.normal_(self.Wd, mean=0.0, std=0.01)

        # 无人机特征影响参数
        self.UavImpact = nn.Parameter(torch.randn(hidden_size))
        nn.init.normal_(self.UavImpact, mean=0.5, std=0.1)

        # 添加小的epsilon防止数值不稳定
        self.epsilon = 1e-7

    def forward(self, x, prev_state, delta_t, uav_feature=None):
        h_prev, c_prev = prev_state

        # 时间衰减处理
        decay = torch.exp(-delta_t * torch.clamp(self.Wd, min=self.epsilon, max=10))
        c_hat = c_prev * decay

        # LSTM计算
        h_new, c_new = self.lstm_cell(x, (h_prev, c_hat))

        # 如果提供了无人机特征，增强其影响
        if uav_feature is not None:
            # 直接添加无人机特征（增强影响）
            h_new = h_new + 1.5 * uav_feature

        # 梯度裁剪防止爆炸
        h_new = torch.clamp(h_new, min=-10, max=10)
        c_new = torch.clamp(c_new, min=-10, max=10)

        return h_new, c_new


# ===================== 波动放大损失 =====================
class AmplifiedFluctuationLoss(nn.Module):
    def __init__(self, amplification_factor=1.5):
        super().__init__()
        self.amplification_factor = amplification_factor

    def forward(self, pred, target):
        # 计算原始数据的波动（一阶差分）
        target_diff = torch.diff(target, dim=0)

        # 计算预测数据的波动
        pred_diff = torch.diff(pred, dim=0)

        # 鼓励预测波动大于原始波动
        amplification_loss = torch.mean(
            torch.relu(self.amplification_factor * torch.abs(target_diff) - torch.abs(pred_diff)))

        return amplification_loss


# ===================== 无人机增强模型 =====================
class UAVEnhancedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # UAV特征提取器
        self.uav_feature_extractor = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 共享的T-LSTM单元
        self.tlstm_cell = UAVEnhancedTLSTMCell(input_size, hidden_size)

        # UAV分支输出层
        self.uav_fc = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.uav_fc.weight)
        nn.init.constant_(self.uav_fc.bias, 0)

        # US分支输出层
        self.us_fc = nn.Linear(hidden_size * 2, 2)
        nn.init.xavier_uniform_(self.us_fc.weight)
        nn.init.constant_(self.us_fc.bias, 0)

        # 卫星分支重构层 - 简化为前馈网络以保留波动
        self.satellite_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 添加层归一化
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size * 2)

    def forward(self, mode, x, deltas, uav_features=None, prev_state=None):
        seq_len = x.size(0)
        outputs = []
        states = []

        # 初始化状态
        if prev_state is None:
            h = torch.zeros(1, self.hidden_size).to(device)
            c = torch.zeros(1, self.hidden_size).to(device)
        else:
            h, c = prev_state

        # 提取无人机特征（如果提供）
        uav_feats = None
        if uav_features is not None:
            uav_feats = self.uav_feature_extractor(uav_features.unsqueeze(-1))

        # T-LSTM时序处理
        for t in range(seq_len):
            delta_t = deltas[t].view(1, 1)

            # 获取当前时间步的输入
            x_t = x[t].unsqueeze(0)

            # 获取当前时间步的无人机特征（如果可用）
            uav_feat = uav_feats[t] if uav_feats is not None else None

            # 增强无人机影响的LSTM计算
            h, c = self.tlstm_cell(x_t, (h, c), delta_t, uav_feat)

            # 层归一化
            h = self.ln1(h)
            outputs.append(h)
            states.append((h.detach(), c.detach()))

        outputs = torch.cat(outputs, dim=0)

        # 分支处理
        if mode == 'uav':
            # 使用Sigmoid约束输出范围
            return torch.sigmoid(self.uav_fc(outputs)), states[-1]

        elif mode == 'us':
            # 使用共享特征和卫星特征
            u_feat = outputs
            s_feat = outputs
            combined = torch.cat([u_feat, s_feat], dim=1)
            combined = self.ln2(combined)
            # 使用Sigmoid约束输出范围
            return torch.sigmoid(self.us_fc(combined)), states[-1]

        elif mode == 'satellite':
            # 使用简单的前馈网络重构（保留波动）
            decoded = self.satellite_decoder(outputs)
            # 使用Sigmoid约束输出范围
            return torch.sigmoid(decoded), states[-1]


# ===================== 训练配置 =====================
HIDDEN_SIZE = 256
NUM_LAYERS = 4
LEARNING_RATE = 0.0001
UAV_IMPACT_FACTOR = 2.0
EPOCHS = 2000
GRAD_CLIP = 1.0
AMPLIFICATION_FACTOR = 1  # 波动放大系数

model = UAVEnhancedModel(input_size=3,  # 固定输入大小为3个特征
                         hidden_size=HIDDEN_SIZE,
                         num_layers=NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# 使用MSELoss和波动放大损失
mse_criterion = nn.MSELoss()
fluctuation_criterion = AmplifiedFluctuationLoss(AMPLIFICATION_FACTOR)


# 检查NaN的函数
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False


# ===================== 增强无人机影响的训练循环 =====================
def train_uav_branch(data, deltas, epochs):
    state = None
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 准备数据 - 使用增强的无人机特征
        tensor_data = torch.FloatTensor(data).to(device)
        tensor_deltas = torch.FloatTensor(deltas).to(device)

        # 模型前向
        output, state = model('uav', tensor_data, tensor_deltas, prev_state=state)

        # 计算目标 - 只取无人机数据部分
        target = torch.FloatTensor(data[:, 0]).view(-1, 1).to(device)

        # 计算损失
        mse_loss = mse_criterion(output, target)
        fluctuation_loss = fluctuation_criterion(output, target)
        loss = mse_loss + fluctuation_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if epoch % 500 == 0:
            print(
                f'UAV Branch - Epoch {epoch}, Loss: {loss.item():.6f} (MSE: {mse_loss.item():.6f}, Fluctuation: {fluctuation_loss.item():.6f})')

    print('UAV Branch training completed')
    return state


def train_us_branch(u_data, s_data, deltas, epochs, prev_state=None):
    state = prev_state
    # 创建增强特征
    enhanced_us_data = []
    for i in range(len(s_data)):
        # 创建增强特征: [卫星数据, 无人机数据, 0] - 使用0作为时间差占位符
        enhanced_us_data.append([s_data[i], u_data[i], 0])
    enhanced_us_data = np.array(enhanced_us_data)

    # 归一化增强特征
    enhanced_us_data_norm = s_scaler.transform(enhanced_us_data)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 准备数据
        tensor_data = torch.FloatTensor(enhanced_us_data_norm).to(device)
        tensor_deltas = torch.FloatTensor(deltas).to(device)
        tensor_uav = torch.FloatTensor(u_data).view(-1, 1).to(device)

        # 模型前向
        output, state = model('us', tensor_data, tensor_deltas, uav_features=tensor_uav, prev_state=state)

        # 计算目标
        target = torch.FloatTensor(np.column_stack((u_data, s_data))).to(device)

        # 计算损失
        mse_loss = mse_criterion(output, target)
        fluctuation_loss = fluctuation_criterion(output, target)
        loss = mse_loss + fluctuation_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if epoch % 500 == 0:
            print(
                f'US Branch - Epoch {epoch}, Loss: {loss.item():.6f} (MSE: {mse_loss.item():.6f}, Fluctuation: {fluctuation_loss.item():.6f})')

    print('US Branch training completed')
    return state


def train_satellite_branch(data, deltas, uav_features, epochs, prev_state=None):
    state = prev_state
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 准备数据
        tensor_data = torch.FloatTensor(data).to(device)
        tensor_deltas = torch.FloatTensor(deltas).to(device)
        tensor_uav = torch.FloatTensor(uav_features).to(device)

        # 模型前向
        output, state = model('satellite', tensor_data, tensor_deltas, uav_features=tensor_uav, prev_state=state)

        # 计算目标
        target = torch.FloatTensor(data[:, 0]).view(-1, 1).to(device)  # 目标是卫星数据

        # 计算损失
        mse_loss = mse_criterion(output, target)
        fluctuation_loss = fluctuation_criterion(output, target)
        loss = mse_loss + fluctuation_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if epoch % 500 == 0:
            print(
                f'Satellite Branch - Epoch {epoch}, Loss: {loss.item():.6f} (MSE: {mse_loss.item():.6f}, Fluctuation: {fluctuation_loss.item():.6f})')

    print('Satellite Branch training completed')
    return state


# ===================== 增强无人机影响的训练流程 =====================
print("=" * 50)
print("Training UAV Branch with Enhanced Impact...")
uav_state = train_uav_branch(enhanced_u_data_norm, u_deltas, int(EPOCHS * UAV_IMPACT_FACTOR))

print("=" * 50)
print("Training US Branch with UAV Features...")
us_state = train_us_branch(us_u_norm, us_s_norm, us_deltas, EPOCHS, prev_state=uav_state)

print("=" * 50)
print("Training Satellite Branch with UAV Features...")
# 准备无人机特征数据
uav_features_for_s = []
for s_d in s_doy:
    time_diffs = np.abs(u_doy - s_d)
    closest_idx = np.argmin(time_diffs)
    uav_features_for_s.append(u_data_norm[closest_idx])
uav_features_for_s = np.array(uav_features_for_s)

sat_state = train_satellite_branch(
    enhanced_s_data_norm,
    s_deltas,
    uav_features_for_s,
    EPOCHS,
    prev_state=us_state
)


# ===================== 卫星数据重构与全年插值 =====================
def reconstruct_full_year():
    model.eval()
    with torch.no_grad():
        # 创建全年DOY范围（1-365）
        full_doy = np.arange(1, 366)

        # 准备插值输入特征
        interpolated_features = []
        for d in full_doy:
            # 找到最近的卫星数据点
            s_idx = np.argmin(np.abs(s_doy - d))

            # 找到最近的无人机数据点
            u_idx = np.argmin(np.abs(u_doy - d))

            # 创建特征向量 [卫星数据, 无人机数据, 时间差]
            feature = [
                s_data_norm[s_idx],
                u_data_norm[u_idx],
                np.abs(u_doy[u_idx] - d)
            ]
            interpolated_features.append(feature)

        # 归一化特征
        interpolated_features = np.array(interpolated_features)
        interpolated_features_norm = s_scaler.transform(interpolated_features)

        # 计算时间间隔（假设均匀分布）
        interpolated_deltas = np.full(len(full_doy), np.mean(s_deltas))

        # 准备无人机特征
        interpolated_uav_features = []
        for d in full_doy:
            u_idx = np.argmin(np.abs(u_doy - d))
            interpolated_uav_features.append(u_data_norm[u_idx])
        interpolated_uav_features = np.array(interpolated_uav_features)

        # 使用模型重构全年数据
        tensor_data = torch.FloatTensor(interpolated_features_norm).to(device)
        tensor_deltas = torch.FloatTensor(interpolated_deltas).to(device)
        tensor_uav = torch.FloatTensor(interpolated_uav_features).to(device)

        reconstructed_norm, _ = model('satellite', tensor_data, tensor_deltas, uav_features=tensor_uav)
        reconstructed_norm = reconstructed_norm.cpu().numpy().flatten()

        # 确保值在[0,1]范围内
        reconstructed_norm = np.clip(reconstructed_norm, 0, 1)

        # 反归一化
        reconstructed = scaler_s.inverse_transform(reconstructed_norm.reshape(-1, 1)).flatten()

        # 应用SG平滑滤波
        window_length = min(31, len(reconstructed) // 10 + 1)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(5, window_length)
        smoothed_reconstructed = savgol_filter(reconstructed,
                                               window_length=window_length,
                                               polyorder=2)
        return full_doy, reconstructed, smoothed_reconstructed


# 执行全年插值
full_doy, full_reconstructed, smoothed_reconstructed = reconstruct_full_year()


# ===================== 结果可视化 =====================
plt.figure(figsize=(18, 8))

# 主图：全年插值结果对比
ax1 = plt.subplot(1, 1, 1)
# 原始卫星数据
ax1.scatter(s_doy, s_data, c='blue', s=70, label='Original Satellite Data', alpha=0.7)

# 全年插值结果
ax1.plot(full_doy, full_reconstructed, 'r-', linewidth=1.5, label='Full-Year Interpolation', alpha=0.7)

# SG平滑结果
ax1.plot(full_doy, smoothed_reconstructed, 'g-', linewidth=2.5, label='SG Smoothed Reconstruction')

# 无人机数据
ax1.scatter(u_doy, u_data, c='green', marker='x', s=80, label='UAV Data', alpha=0.8)

# 同步点数据
ax1.scatter(us_doy, us_s, c='purple', s=120, marker='*', label='US Sync Points', alpha=0.9)

ax1.set_title('UAV-Enhanced Satellite Data Reconstruction with SG Smoothing', fontsize=16)
ax1.set_xlabel('Day of Year (DOY)', fontsize=14)
ax1.set_ylabel('Data Value', fontsize=14)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('C:/Users/movjing/Desktop/uav_enhanced_full_year.png')
plt.show()

# 保存重构结果
result_df = pd.DataFrame({
    'DOY': full_doy,
    'Reconstructed': full_reconstructed,
    'SG_Smoothed': smoothed_reconstructed  # 添加平滑后的数据列
})
result_df.to_csv('C:/Users/movjing/Desktop/full_year_reconstruction.csv', index=False)

print("Reconstruction results saved to desktop/results directory")
print("Program completed successfully")
