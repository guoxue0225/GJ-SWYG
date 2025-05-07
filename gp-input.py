# 顾佳盛Gu
# I am Iron Man
# 开发时间：2022/10/20 15:02
from 回归.光谱预处理 import *
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.direction'] = 'in'  # 刻度朝内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # SimHei黑体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15
# 载入数据
path ='E:\WYY_JB\回归\羟基-y-山椒素.csv' #改成自己的路径

data = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
print("数据导入")
data_x = data[:, 1:]  # 所有行，从第2列-最后列
data_y = data[:, 0]  # 所有的行，第1列
x_length = len(data_x)  # 获取波段数
print(data.shape)  # 数据形状
data_raw = np.array(data_x) #原始数据

# 绘制预处理前的光谱波长图片
plt.figure(1, dpi=100, figsize=(12, 6))  # figsize=(8,6) 12,6
ax = plt.subplot()
plt.plot(data_y, data_raw)  #
plt.xlabel("波长/nm", family='SimSun', fontweight='medium', fontsize='15')
plt.ylabel("反射率", family='SimSun', fontweight='medium', fontsize='15')
plt.title("The spectrum of the raw ", fontweight='semibold', fontsize='18')
plt.savefig(r"E:\WYY_JB\raw.png",dpi=800) #保存图片


##MSC数据预处理、可视化
# 默认读入列向量，须要转置为行向量才适于显示
data_MSC = MSC(np.transpose(data_x))#MSC预处理
data_MSC=np.transpose(data_MSC)
# 绘制   预处理后图片
plt.figure(2, dpi=100, figsize=(12, 6))
plt.plot(data_y, data_MSC)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')  ##记得改名字MSC               semibold黑体
plt.title("The spectrum of the MSC ", fontweight='semibold', fontsize='18')  # fontsize设置字体大小,默认12,fontweight设置字体粗细
plt.savefig(r"E:\WYY_JB\MSC.png",dpi=500)


##D1数据预处理、可视化
data_y1= data_y[:-1]#不要最后一个
data_D1 =D1(np.transpose(data_x))#D1预处理
data_D1=np.transpose(data_D1)
# 绘制   预处理后图片
plt.figure(3, dpi=100, figsize=(12, 6))
plt.plot(data_y1, data_D1)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')  ##记得改名字MSC               semibold黑体
plt.title("The spectrum of the D1 ", fontweight='semibold', fontsize='18')  # fontsize设置字体大小,默认12,fontweight设置字体粗细
plt.savefig(r"E:\WYY_JB\D1.png",dpi=500)

##D2数据预处理、可视化
data_y2=data_y[:-2]#不要最后两个
data_D2 =D2(np.transpose(data_x))#D2预处理
data_D2=np.transpose(data_D2)
# 绘制   预处理后图片
plt.figure(4, dpi=100, figsize=(12, 6))
plt.plot(data_y2, data_D2)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')  ##记得改名字MSC               semibold黑体
plt.title("The spectrum of the D2 ", fontweight='semibold', fontsize='18')  # fontsize设置字体大小,默认12,fontweight设置字体粗细
plt.savefig(r"E:\WYY_JB\D2.png",dpi=500)

##SNV数据预处理、可视化
data_SNV =SNV(np.transpose(data_x))#SG预处理
data_SNV=np.transpose(data_SNV)
# 绘制   预处理后图片
plt.figure(5, dpi=100, figsize=(12, 6))
plt.plot(data_y, data_SNV)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')  ##记得改名字MSC               semibold黑体
plt.title("The spectrum of the SNV ", fontweight='semibold', fontsize='18')  # fontsize设置字体大小,默认12,fontweight设置字体粗细
plt.savefig(r"E:\WYY_JB\SNV.png",dpi=500)

##SG数据预处理、可视化
data_SG =SG(np.transpose(data_x))#SG预处理
data_SG=np.transpose(data_SG)
# 绘制   预处理后图片
plt.figure(6, dpi=100, figsize=(12, 6))
plt.plot(data_y, data_SG)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')  ##记得改名字MSC               semibold黑体
plt.title("The spectrum of the SG ", fontweight='semibold', fontsize='18')  # fontsize设置字体大小,默认12,fontweight设置字体粗细
plt.savefig(r"E:\WYY_JB\SG.png",dpi=500)


##wave数据预处理、可视化
data_wave = wave(np.transpose(data_x))     # Wave 数据预处理
data_wave = np.transpose(data_wave)
# 绘制预处理后波段图像
plt.figure(dpi=100, figsize=(12, 6))
plt.plot(data_y, data_wave)
plt.xlabel("波长/nm", fontweight='medium', fontsize='15', family='Simsun')
plt.ylabel("Reflectance", fontweight='medium', fontsize='15')
plt.title("The spectrum of the Wave", fontweight='semibold', fontsize='18')
plt.savefig(r"E:\WYY_JB\wave.png", dpi=500)
plt.show()