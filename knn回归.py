# 顾佳盛
# I am Iron Man
# 开发时间：2022/12/6 13:12
from skimage.metrics import mean_squared_error

from 回归.光谱预处理 import *
import numpy as np
from sklearn.metrics import  r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

path=r"E:\三维荧光-回归\Luteoloside\SG-luteo-660.csv"#选择文件路径
data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)  # 读取数据
data_x = data[:, 1:]  # 反射率数据
data_y = data[:, 0]  # 分类标签

# 预处理 （MSC D1 D2 SNV SG）
#data_x = SNV(data_x)  ##改这里的函数名就可以得到不同的预处理
#data_x = np.array(data_x)

# 划分数据集
table_random_state =7  # random_state就是为了保证程序每次运行都分割一样的训练集和测试集,用来复现结果
test_ratio = 0.3  # 划分测试集比例
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        shuffle=True,
                                                        random_state=table_random_state)  # ,stratify=data_y
#保存划分好的训练集数据、 测试集数据
y_train_2 = y_train.reshape(-1, 1)#转成列向量
y_test_2 = y_test.reshape(-1, 1)
combined_train = np.hstack((y_train_2, X_train))#合并
combined_test = np.hstack((y_test_2, X_test))#合并
#np.savetxt(r"G:\数据\test-result\knn_train_dataset.csv", combined_train, fmt='%.5f', delimiter=',')#保存划分好的训练集数据
#np.savetxt(r"G:\数据\test-result\knn_test_dataset.csv", combined_test, fmt='%.5f', delimiter=',')#保存划分好的测试集数据


#交叉验证选择合适的k值‌
k_scores = []
for k in range(1, 5):
    model = KNeighborsRegressor(n_neighbors=k)

    kf = KFold(n_splits=5, shuffle=True, random_state=table_random_state)
    loo = LeaveOneOut()
    scores = np.sqrt(-cross_val_score(model, X_train, y_train,
                                      scoring="neg_mean_squared_error", cv=kf))  # cv= kf ,cv=loo留一
    k_scores.append(scores.mean())
print('最佳k值得分:', '%.4f' % min(k_scores))
index = np.argmin(k_scores)
print('最佳k值:', range(1, 21)[index])
plt.plot(range(1, 5), k_scores)
plt.axvline(range(1,5)[index] , color='k', linestyle='--', linewidth=1)
plt.xticks( np.arange(0,10,2) )
plt.xlabel('Number of k')#最佳k值
plt.ylabel('k_scores')#
plt.title('k_selection')
plt.tight_layout()
plt.show()

#knn建模 K-近邻算法 该算法的基本思想是对于每个输入的新数据点，找到其在样本数据集中最近的K个数据点，根据这K个邻居的类别来预测新数据点的类别。
knn = KNeighborsRegressor(n_neighbors=2, weights='uniform') #n_neighbors:指定邻居的数量，即k的个数，默认值为5。较小的k值会增加模型的复杂度，可能导致过拟合；较大的k值则会减小模型的复杂度，可能导致欠拟合。通常通过交叉验证选择合适的k值‌。
knn.fit(X_train, y_train)            #放入训练数据进行训练
#预测
pred_train = knn.predict(X_train)
pred_test = knn.predict(X_test)

#模型预测结果
r2_test = r2_score(y_test, pred_test)#测试集
r2_train = r2_score(y_train, pred_train)#预测集
print('训练集R\u00b2:','%.4f' % r2_train)#保留后4位小数
print('预测集R\u00b2:','%.4f' % r2_test)
print('RMSE train: %.4f\nRMSE predict: %.4f' % (#RMSE 是通过取 MSE 的平方根来计算的。RMSE 也称为均方根偏差。
    # 它测量误差的平均幅度，并关注与实际值的偏差。RMSE 值为零表示模型具有完美拟合。RMSE 越低，模型及其预测就越好。
        np.sqrt(mean_squared_error(y_train, pred_train)),
        np.sqrt(mean_squared_error(y_test, pred_test))))
rmsep=np.sqrt(mean_squared_error(y_test, pred_test))
SD=np.std(y_test)
rpd=SD/rmsep
print('相对百分比偏差(RPD):','%.2f' %rpd)

#保存模型预测的训练集数据、 测试集数据
pred_train2 = pred_train.reshape(-1, 1)#转成列向量
pred_test2 = pred_test.reshape(-1, 1)
combined_train2 = np.hstack((pred_train2, X_train))#合并
combined_test2 = np.hstack((pred_test2, X_test))#合并
#np.savetxt(r"G:\数据\test-result\knn_train_dataset_pred.csv", combined_train2, fmt='%.5f', delimiter=',')#保存模型预测好的训练集数据
#np.savetxt(r"G:\数据\test-result\knn_test_dataset_pred.csv", combined_test2, fmt='%.5f', delimiter=',')#保存模型预测好的测试集数据


import joblib
from sklearn.neighbors import KNeighborsRegressor

# 假设knn模型已经训练好了
knn = KNeighborsRegressor(n_neighbors=2, weights='uniform')
knn.fit(X_train, y_train)

# 保存模型
#joblib.dump(knn, 'E:\三维荧光-回归\Luteoloside\knn_luteo_SG.pkl')
