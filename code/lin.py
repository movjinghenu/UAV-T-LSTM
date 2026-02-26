import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('C:/Users/movjing/Desktop/chazhi.xlsx', sheet_name='Sheet1')

# 将空字符串转换为NaN
df['s_data'] = df['s_data'].replace('', np.nan)

# 使用线性插值填充缺失值
df['s_data'] = df['s_data'].interpolate(method='linear')

# 保存结果到新文件
df.to_excel('C:/Users/movjing/Desktop/线性填充_完成.xlsx', index=False)

print("线性填充完成，结果已保存到'线性填充_完成.xlsx'")
