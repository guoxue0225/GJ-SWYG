import joblib  # 导入joblib模块
import numpy as np  # 导入numpy模块
import pandas as pd  # 导入pandas模块
from sklearn.metrics import r2_score, mean_squared_error
# 加载模型
plsr_loaded = joblib.load(r"E:\三维荧光-回归\Luteoloside\RFR_luteo_SG.pkl")

# 加载新的光谱数据
new_samples_x = np.loadtxt(r"E:\三维荧光-回归\CA\SG-GPSJ-55.csv", delimiter=',')

# 使用加载的模型进行预测
new_predictions = plsr_loaded.predict(new_samples_x)
# 输出预测结果
print(new_predictions)

# 假设你已经有了新的样本真实含量值
new_samples_y = np.loadtxt(r"E:\三维荧光-回归\Luteoloside\Luteo-55-zsz.csv", delimiter=',')

# 计算R²
r2_test_new = r2_score(new_samples_y, new_predictions)
print(f"外部验证集 R²: {r2_test_new:.4f}")

# 计算RMSE
rmse_test_new = np.sqrt(mean_squared_error(new_samples_y, new_predictions))
print(f"外部验证集 RMSE: {rmse_test_new:.4f}")

# 计算RPD (Relative Percentage Deviation)
SD_new = np.std(new_samples_y)
rpd_new = SD_new / rmse_test_new
print(f"外部验证集 RPD: {rpd_new:.2f}")

# 创建一个 DataFrame 来保存实际值和预测值
results_df = pd.DataFrame({
    '真实值': new_samples_y,
    '预测值': new_predictions.flatten()  # 使用 flatten() 将二维数组转为一维数组
})

# 保存到 结果Excel 文件
output_path = r"E:\三维荧光-回归\Luteoloside\RFR_Luteo-yz-result_SG.xlsx"
results_df.to_excel(output_path, index=False, engine='openpyxl')

# 绘制实际值与预测值的散点图
import matplotlib.pyplot as plt

plt.scatter(new_samples_y, new_predictions, color='blue', alpha=0.7)
plt.plot([min(new_samples_y), max(new_samples_y)], [min(new_samples_y), max(new_samples_y)], color='black', linestyle='--')  # 45度参考线
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('外部验证：实际值 vs 预测值')
plt.show()
