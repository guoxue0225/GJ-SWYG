# 开发时间：2023/4/19 15:08
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split,LeaveOneOut
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Times New Roman']#Microsoft YaHei
plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
plt.rcParams['font.size'] = 14

path=r"E:\三维荧光-回归\Luteoloside\SG-luteo-660.csv"
data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)#读取数据
data_x = data[:, 1:]#特征数据，所有的行，从第2列-最后列，
data_y = data[:,0]#成分含量浓度，所有的行，第1列,读取含量数据
#data_y=np.random.uniform(50,100,size=660)
#print(np.shape(data_x))

#path=r"E:\三维荧光-回归\Luteoloside\SG-luteo-660.csv"
#data = pd.read_excel(path)  #xlsx
#data_x = data.iloc[1:, 1:].values#光谱反射率数据
#data_y = data.iloc[1:, 0].values#成分含量
#data_raw = data_x


#预处理 (MSC、SNV、SG、D1、D2、wave)
#data_MSC = MSC(data_x) #MSC预处理
#data_D1 = D1(data_x) #D1预处理
#data_D2 = D2(data_x) #D2预处理
#data_SNV = SNV(data_x) #SNV预处理
#data_SG = SG(data_x) #SG预处理
#data_wave = wave(data_x) #wave小波变换预处理
#保存数据
# np.savetxt('F:\pythonProject1\整理/raw.csv',data_raw, fmt='%.5f', delimiter=',', encoding='utf-8')
# np.savetxt('F:\pythonProject1\整理/MSC.csv',data_MSC, fmt='%.5f', delimiter=',', encoding='utf-8')
# np.savetxt('F:\pythonProject1\整理/D1.csv',data_D1, fmt='%.5f', delimiter=',')
# np.savetxt('F:\pythonProject1\整理/D2.csv',data_D2, fmt='%.5f', delimiter=',')
# np.savetxt('F:\pythonProject1\整理/SNV.csv',data_SNV, fmt='%.5f', delimiter=',')
# np.savetxt('F:\pythonProject1\整理/SG.csv',data_SG, fmt='%.5f', delimiter=',')
# np.savetxt('F:\pythonProject1\整理/wave.csv',data_wave, fmt='%.5f', delimiter=',')

#划分数据集
test_ratio = 0.3#测试集占比
table_random_state =7#random_state就是为了保证程序每次运行都分割一样的训练集和测试集,用来复现结果
#将数据集分割为训练集和测试集(随机划分)
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,random_state=table_random_state)#改第一个参数更换为不同预处理处理后的数据，例如 data_MSC
#X_train, X_test, y_train, y_test = SPXY.spxy(data_x, data_y, test_size=test_ratio,)#SPXY数据集划分方法可以同时兼顾光谱信息和含量信息，保证划分样本间的差异性和代表性，提高模型稳定性。

#保存划分好的训练集数据、 测试集数据
y_train_2 = y_train.reshape(-1, 1)#转成列向量
y_test_2 = y_test.reshape(-1, 1)
combined_train = np.hstack((y_train_2, X_train))#合并
combined_test = np.hstack((y_test_2, X_test))#合并
#np.savetxt(r"G:\数据\test-result\plsr_train_dataset.csv", combined_train, fmt='%.5f', delimiter=',')#保存划分好的训练集数据
#np.savetxt(r"G:\数据\test-result\plsr_test_dataset.csv", combined_test, fmt='%.5f', delimiter=',')#保存划分好的测试集数据

#采用留一交叉验证方法。当交叉验证的均方根误差 (RMSECV) 达到最小值时，选择了PLSR模型的最佳潜变量 (LVs)。
scores_rmse = []
for k in range(1,12):
    model = PLSRegression(n_components=k)
    kf = KFold(n_splits=5, shuffle=True, random_state=table_random_state)#5折交叉验证
    loo = LeaveOneOut()#留一交叉验证
    rmsecv = np.sqrt(-cross_val_score(model, X_train, y_train,
                                      scoring="neg_mean_squared_error", cv=kf))#cv= kf ,cv=loo
    scores_rmse.append(np.mean(rmsecv))#求每个主成分的rmsecv的均值
print('最小RMSECV:','%.4f' %min(scores_rmse))
index = np.argmin(scores_rmse)
print('最佳潜在变量（LVs）数量:', range(1,12)[index])

plt.plot(range(1, 12), scores_rmse)
plt.axvline(range(1,12)[index] , color='k', linestyle='--', linewidth=1)
plt.xticks( np.arange(0,15,2) )
plt.xlabel('Number of Components')#最佳潜在变量（LVs）数量
plt.ylabel('RMSECV')#RMSECV
plt.title('Leave-one-out Cross-validation Error')
plt.tight_layout()
plt.show()

#建模
plsr=PLSRegression(n_components=9)
plsr.fit(X_train,y_train)
#预测
pred_train=plsr.predict(X_train)
pred_test=plsr.predict(X_test)

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
#np.savetxt(r"G:\数据\test-result\plsr_train_dataset_pred.csv", combined_train2, fmt='%.5f', delimiter=',')#保存模型预测好的训练集数据
#np.savetxt(r"G:\数据\test-result\plsr_test_dataset_pred.csv", combined_test2, fmt='%.5f', delimiter=',')#保存模型预测好的测试集数据


#画拟合曲线
plt.rcParams['xtick.direction'] = 'in'#刻度朝内
plt.rcParams['ytick.direction'] = 'in'
fig,ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.tick_params(which='major', width=0.75, length=5)#设置刻度
scatter1 = plt.scatter(x=y_train, y=pred_train, s=50, marker='o',c='#FFBCDE' , alpha=0.8, label='Training set', linewidths=0.3, edgecolor='#17223b')
scatter2 = plt.scatter(x=y_test, y=pred_test,s=50, marker='o',c='#00C8F4' ,alpha=0.8, label='Prediction set', linewidths=0.3, edgecolor='#17223b')
scatter3 = plt.scatter(x=y_test,y=pred_test,marker='o', c='green',alpha=0)
ax.plot([data_y.min(),data_y.max()],[data_y.min(),data_y.max()],'--',c='black',alpha=0.3)
plt.xlabel('实测值',fontdict={'family': 'Simsun', 'size': 15})
plt.ylabel('预\n测\n值',fontdict={'family': 'Simsun', 'size': 15}, rotation=0, labelpad=18)
l1 =plt.legend(['R\u00b2 = {:.4f}'.format(r2_train),'R\u00b2 = {:.4f}'.format(r2_test),
'RPD = {:.2f}'.format(rpd)],loc='lower right',fontsize=15,frameon=False, handletextpad=0.1)#R$^2$ , 'RPD = {:.2f}'.format(rpd)
plt.legend(loc='upper left',fontsize=15, frameon=False, labelspacing=0.5, handletextpad=0.1)#size 15画图,labelcolor='#00C8F4'
plt.gca().add_artist(l1)
plt.show()


# 保存模型
#joblib.dump(plsr, 'E:\三维荧光-回归\Luteoloside\plsr_luteo_SG.pkl')
