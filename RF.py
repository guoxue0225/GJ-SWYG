from 回归.光谱预处理 import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV


path =r"E:\WYY_JB\贡菊-分类\GJ-YS-660.csv"#选择文件路径
data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)#读取数据
data_x = data[:, 1:]#反射率数据
data_x=np.array(data_x)
data_y = data[:,0]  # 分类标签

#预处理 （MSC D1 D2 SNV SG）
#data_x=SG(data_x)##改这里的函数名就可以得到不同的预处理

#划分数据集
table_random_state =7#random_state就是为了保证程序每次运行都分割一样的训练集和测试集,用来复现结果
test_ratio = 0.3#划分测试集比例
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                            shuffle=True,random_state=table_random_state)#,stratify=data_y
#保存划分好的训练集数据、 测试集数据
y_train_2 = y_train.reshape(-1, 1)#转成列向量
y_test_2 = y_test.reshape(-1, 1)
combined_train = np.hstack((y_train_2, X_train))#合并
combined_test = np.hstack((y_test_2, X_test))#合并
#np.savetxt(r"E:\WYY_JB\结果\PLSDA_YS_660_T.csv", combined_train, fmt='%.5f', delimiter=',')#保存划分好的训练集数据
#np.savetxt(r"E:\WYY_JB\结果\PLSDA_YS_660_C.csv", combined_test, fmt='%.5f', delimiter=',')#保存划分好的测试集数据


#网格搜索,使用5折交叉选择最优的超参数
param_grid ={"n_estimators": [50,100,150,200,250,300],
                  "max_depth":range(5,10),
            }  #'max_features': range(1, 11)或者sqrt、log2
kf=KFold(n_splits=5, shuffle=True, random_state=table_random_state)
model = GridSearchCV(RandomForestClassifier(random_state=table_random_state), param_grid, cv=kf, n_jobs=-1,
                      return_train_score=True,
                     refit=True)
model.fit(X_train, y_train)
print('最优分数=交叉验证的平均准确率:%.4lf' % model.best_score_)
print('最优参数:',model.best_params_)



#建立随机森林模型
RF = RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", random_state=table_random_state)
RF.fit(X_train, y_train)
pred_test = RF.predict(X_test)
pred_train = RF.predict(X_train)
acc_test = accuracy_score(y_test, pred_test)
acc_train = accuracy_score(y_train, pred_train)
print('训练集准确率:', '%.4f' % acc_train)
print('测试集准确率:', '%.4f' % acc_test)
#print('混淆矩阵：\n',confusion_matrix(y_test,pred_test))

#保存模型预测的训练集数据、 测试集数据
pred_train2 = pred_train.reshape(-1, 1)#转成列向量
pred_test2 = pred_test.reshape(-1, 1)
combined_train2 = np.hstack((pred_train2, X_train))#合并
combined_test2 = np.hstack((pred_test2, X_test))#合并
#np.savetxt(r"E:\WYY_JB\结果\PLSDA_YS_660_T_result.csv", combined_train2, fmt='%.5f', delimiter=',')#保存模型预测好的训练集数据
#np.savetxt(r"E:\WYY_JB\结果\PLSDA_YS_660_C_result.csv", combined_test2, fmt='%.5f', delimiter=',')#保存模型预测好的测试集数据




#使用留一交叉验证或者5折交叉,是对训练集（70%的数据）进行的交叉验证，用于评估训练好的模型的泛化能力。
loo = LeaveOneOut()#留一交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=table_random_state) #十折交叉验证
scores = cross_val_score(RF, X_train, y_train, scoring="accuracy",  n_jobs=-1, cv=kf)#cv=loo或cv=kf，#改cv参数更换交叉验证方法
print("交叉验证迭代次数:", len(scores))
print("交叉验证 Mean accuracy: {:.4f}".format(scores.mean()))


#结果可视化（混淆矩阵）
y__pred = np.array(pred_test,dtype=int)
y_pred = list(y__pred)#再转为列表
y__true = np.array(y_test,dtype=int)
y_true = list(y__true)
classes = list(set(y_true))#
classes.sort()# 排序，准确对上分类结果
confusion = confusion_matrix(y_true,y_pred)# 对比，得到混淆矩阵
print(confusion)
f = plt.figure(figsize=(12,8),dpi=100)
plt.imshow(confusion,interpolation='nearest',  cmap=plt.cm.Wistia,alpha=0.4)# ,norm=norm根据最下面的图按自己需求更改颜色  ,  Wistia    GnBu
ax = plt.subplot()
indices = range(len(confusion))
plt.rcParams['font.sans-serif'] = ['Times New Roman']#Times New Roman  SimHei
plt.rcParams['axes.unicode_minus'] = False# 解决中文显示问题
plt.rcParams['font.size'] = 10
labels_name=['1', '2','3','4','5','6','7','8','9','10','11']#分类标签
plt.xticks(indices, labels_name,size=5,font='Simsun',rotation=30)#
plt.yticks(indices, labels_name,size=5,font='Simsun')
# 热度显示仪
cb=plt.colorbar(format='%.0f',)
plt.ylabel('真\n实\n值', fontdict={'family': 'Simsun', 'size':10}, rotation=0, labelpad=15)#, labelpad=15
plt.xlabel('预测值', fontdict={'family': 'Simsun', 'size':10})
plt.title('Confusion  Matrix', fontdict={'family': 'Times New Roman', 'size': 15})
# 显示数据，直观些
for first_index in range(len(confusion)):#0~10#第几行
    for second_index in range(len(confusion[first_index])):#第几列
        plt.text(second_index, first_index, confusion[first_index][second_index],
                 horizontalalignment='center', verticalalignment='center', fontsize=18,
                 )#保持中心  fontsize=20
ax.tick_params(which='major', width=1.00, length=4)
plt.show()

import joblib

# 保存模型到文件
#joblib.dump(RF, 'E:\WYY_JB\贡菊-分类/RF_YS_model.joblib')  # 你可以选择任何你想要的文件名和路径
