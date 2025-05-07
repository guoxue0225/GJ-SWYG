import pandas as pd
import joblib
import numpy as np

# 加载已保存的RF模型
rf_loaded = joblib.load('E:\WYY_JB\贡菊-分类\RF-660_SG_model.joblib')  # 替换为你的模型路径

# 读取Excel新样本（假设格式与训练数据一致）
new_data = pd.read_excel("E:\WYY_JB\贡菊-分类\GJ-SG-55-yz.xlsx", header=0)  # header=0表示第1行是标题
new_samples = new_data.iloc[:, 0:].values  # 提取特征列（注意索引范围）

# ---- 关键：与训练数据相同的预处理 ----
#from 回归.光谱预处理 import D2  # 若训练时用了预处理（如SNV）
#new_samples = D2(new_samples)  # 取消注释并应用相同预处理

# 预测新样本
new_labels = rf_loaded.predict(new_samples)

# 输出结果
print("预测标签:", new_labels)


# 若有真实标签，计算准确率
if 'Label' in new_data.columns:
    true_labels = new_data['Label'].values.astype(int)
    accuracy = np.mean(true_labels == new_labels)
    print(f"外部验证准确率: {accuracy:.4f}")
