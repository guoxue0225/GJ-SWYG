#import SPXY
from 回归.光谱预处理 import *
import numpy as np
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error,r2_score

path="E:\三维荧光-回归\Luteoloside\SG-luteo-660.csv"
data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)#读取数据
data_x = data[:, 1:]#特征数据
data_y = data[:, 0]#成分含量浓度

#path =r"C:\Users\86191\Desktop\贡菊-HIS\GJ-FM-4.xlsx"
#data = pd.read_excel(path)
#data_x = data.iloc[1:, 0:396].values #第0列到395列
#data_y = data.iloc[1:, 396].values  # 第396列到399列

#预处理（MSC D2 D1 SG SNV）
#data_x=SNV(data_x)##改这里的函数名就可以使用不同的预处理
#data_x=np.array(data_x)
test_ratio = 0.3#测试集占比
table_random_state =7#random_state用来复现结果
#分为训练集 、验证集 、 测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(data_x, data_y, test_size=test_ratio,random_state=table_random_state)#,stratify=data_y
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25,random_state=table_random_state)#划分训练集和验证集

#保存划分好的训练集数据、验证集、 测试集数据
y_train_2 = y_train.reshape(-1, 1)#转成列向量
y_test_2 = y_test.reshape(-1, 1)
y_val_2 = y_val.reshape(-1, 1)
combined_train = np.hstack((y_train_2, X_train))#合并
combined_test = np.hstack((y_test_2, X_test))#合并
combined_val = np.hstack((y_val_2, X_val))
#np.savetxt(r"E:\三维荧光-回归\图\CA-train-zsz.csv", combined_train, fmt='%.5f', delimiter=',')#保存划分好的训练集数据
#np.savetxt(r"E:\三维荧光-回归\图\CA-train-zsz-1.csv", combined_val, fmt='%.5f', delimiter=',')#保存划分好的验证集数据
#np.savetxt(r"E:\三维荧光-回归\图\CA-test-zsz.csv", combined_test, fmt='%.5f', delimiter=',')#保存划分好的测试集数据

#最优参数网格搜索
param_grid ={"n_estimators":[50,100,150,200] #树的数量过多，训练时间会增加(20-200)
              ,"max_depth":  range(1, 5), #过高可能过拟合（1-10）
             'max_features': range(1, 11)}#1-10,sqrt
kf = KFold(n_splits=5, shuffle=True, random_state=table_random_state)
model = ms.GridSearchCV(RandomForestRegressor(random_state=table_random_state), param_grid, cv=kf, n_jobs=-1,
                      return_train_score=True,
                     refit=True) # 当refit=True估计器是分类器时才可用
model.fit(X_trainval, y_trainval)
print('最优分数:%.4lf' % model.best_score_)
print('最优参数:', model.best_params_)

#建模
#修改参数
RF = RandomForestRegressor(n_estimators=150, max_depth=9, max_features="sqrt",
                           random_state=table_random_state, oob_score=True)
RF.fit(X_trainval, y_trainval)
pred_test= RF.predict(X_test)
pred_train=RF.predict(X_trainval)
r2_test = r2_score(y_test, pred_test)
r2_train = r2_score(y_trainval, pred_train)
print('训练集R\u00b2:','%.4f' % r2_train)#保留后4位小数
print('预测集R\u00b2:','%.4f' % r2_test)
print('RMSE train: %.4f\nRMSE test: %.4f' % (#RMSE 是通过取 MSE 的平方根来计算的。RMSE 也称为均方根偏差。
    # 它测量误差的平均幅度，并关注与实际值的偏差。RMSE 值为零表示模型具有完美拟合。RMSE 越低，模型及其预测就越好。
        np.sqrt(mean_squared_error(y_trainval, pred_train)),
        np.sqrt(mean_squared_error(y_test, pred_test))))
rmsep = np.sqrt(mean_squared_error(y_test, pred_test))
SD = np.std(y_test)
rpd = SD/rmsep
print('相对百分比偏差(RPD):','%.2f' %rpd)

#保存模型预测的训练集（训练集+验证集）数据、 测试集数据
pred_train2 = pred_train.reshape(-1, 1)#转成列向量
pred_test2 = pred_test.reshape(-1, 1)
combined_train2 = np.hstack((pred_train2, X_trainval))#合并
combined_test2 = np.hstack((pred_test2, X_test))#合并
#np.savetxt(r"E:\三维荧光-回归\图\CA-train_ycz.csv", combined_train2, fmt='%.5f', delimiter=',')#保存模型预测好的训练集数据
#np.savetxt(r"E:\三维荧光-回归\图\CA-test_ycz.csv", combined_test2, fmt='%.5f', delimiter=',')#保存模型预测好的测试集数据


#使用留一交叉验证或者十折交叉验证评估建立模型的泛化能力
loo = LeaveOneOut()#留一交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=table_random_state) #5折交叉验证
rmsecv = np.sqrt(-cross_val_score(RF, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))#cv=loo或cv=kf，#改cv参数更换交叉验证方法
print("交叉验证迭代次数:", len(rmsecv))
print("交叉验证最小RMSECV: {:.4f}".format(rmsecv.mean()))#交叉验证的均方根误差 (RMSECV) 越小越好


#画拟合曲线
plt.rcParams['font.sans-serif'] = ['Times New Roman']# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(pred_test,y_test,alpha=0.6)
w=np.linspace(min(pred_test),max(pred_test),100)
plt.plot(w,w,'gray',lw=3)
plt.xlabel('实测值',fontdict={'family': 'Simsun', 'size': 15})
plt.ylabel('预\n测\n值',fontdict={'family': 'Simsun', 'size': 15}, rotation=0, labelpad=15)
plt.title('Random Forest Prediction')
plt.legend(['Prediction set'],loc='best',fontsize='large', frameon=False)#'Training set',
plt.show()


import joblib
import os

# 指定保存路径（确保文件夹存在）
#save_path ="E:\三维荧光-回归\Luteoloside\RFR_luteo_SG.pkl"

# 保存训练好的模型到指定路径
#joblib.dump(RF, save_path)