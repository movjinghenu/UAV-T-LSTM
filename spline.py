import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取Excel文件
file_path = 'C:/Users/movjing/Desktop/chazhi.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 确保第二列是数值类型，并将空字符串等转换为NaN
df['s_data'] = pd.to_numeric(df['s_data'], errors='coerce')

# 提取非空数据的索引和值
known_indices = df['doy'][df['s_data'].notna()].values
known_values = df['s_data'][df['s_data'].notna()].values

# 创建3次和5次样条插值函数
spline3 = InterpolatedUnivariateSpline(known_indices, known_values, k=3)
spline5 = InterpolatedUnivariateSpline(known_indices, known_values, k=5)

# 对所有行进行插值
all_indices = df['doy'].values
interpolated_values_3 = spline3(all_indices)
interpolated_values_5 = spline5(all_indices)

# 创建更密集的点用于绘制平滑曲线
dense_indices = np.linspace(min(all_indices), max(all_indices), 500)
dense_values_3 = spline3(dense_indices)
dense_values_5 = spline5(dense_indices)

# 绘制图形
plt.figure(figsize=(12, 8))

# 绘制原始数据点
plt.scatter(known_indices, known_values, color='red', s=50, zorder=5, label='原始数据点')

# 绘制3次样条插值曲线
plt.plot(dense_indices, dense_values_3, 'b-', linewidth=2, label='3次样条插值')

# 绘制5次样条插值曲线
plt.plot(dense_indices, dense_values_5, 'g--', linewidth=2, label='5次样条插值')

# 标记插值点
plt.scatter(all_indices, interpolated_values_3, color='blue', s=20, alpha=0.6, marker='o', label='3次插值点')
plt.scatter(all_indices, interpolated_values_5, color='green', s=20, alpha=0.6, marker='^', label='5次插值点')

# 添加图例和标题
plt.legend(loc='best')
plt.title('3次与5次多项式样条插值结果比较')
plt.xlabel('s_doy')
plt.ylabel('s_data')
plt.grid(True, alpha=0.3)

# 保存图像
#plt.savefig('样条插值比较图.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建包含原始数据和插值结果的DataFrame
result_df = df.copy()
result_df['3次样条插值'] = interpolated_values_3
result_df['5次样条插值'] = interpolated_values_5

# 保存结果到Excel文件
output_path = 'C:/Users/movjing/Desktop/样条插值结果比较.xlsx'
result_df.to_excel(output_path, index=False)

print("插值完成，结果已保存到:", output_path)
print("图像已保存为: 样条插值比较图.png")