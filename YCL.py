# I am Iron Man
# 开发时间：2022/10/21 9:53
import numpy as np
from copy import deepcopy
import pandas as pd
import pywt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression# 使用sklearn库可以很方便的实现各种基本的机器学习算法

# 多元散射校正MSC 每个光谱减去基线平移量和偏移量，从而修正光谱数据的基线平移和偏移现象
def MSC(Data):#默认读入列向量，须要转置为行向量才适于显示
    #计算平均光谱
    n,p = Data.shape#张量大小
    msc = np.ones((n,p))#全为1

    for j in range(n):#  求光谱均值
        mean = np.mean(Data,axis=0)#
    #线性拟合
    for i in range(n):
        y = Data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1,1),y.reshape(-1,1))
        k = l.coef_#逻辑回归 coef_代表的是模型参数#该coef_包含每个目标的预测系数。
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc#横向求均值

# Savitzky-Golay平滑   #要转置
def SG(data, w=21, p=3):#w:平滑窗口大小  p:多项式次数
    return signal.savgol_filter(data, w, p)
#window_length(w)即窗口长度；取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
#polyorder为多项式拟合的阶数。它越小，则平滑效果越明显；越大，则更贴近原始曲线。

# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])#diff()求导
    return Di

# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di

# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    #print(m, n)  #
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  data_snv  #沿着垂直方向求均值

# 均值中心化
def CT(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data

# 最大最小值归一化    X不用转置  最小最大归一化将数据缩放到[0,1]
def MMS(data):
    return MinMaxScaler().fit_transform(data)

# 标准化            X不用转置 计算每个特征的均值和标准差来进行数据标准化，标准化后的数据会被转换为零均值和单位方差的正态分布
def SS(data):
    return StandardScaler().fit_transform(data)

#小波变换Wave
def wave(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
       CWT中小波参数 (小波基础和分解尺度) 的选择至关重要，直接决定了后续模型的优劣
       经过试验盘算和分析，本研究选择了Daubechies族的db4小波基，分解规模定为64。
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))

    return tmp

# 趋势校正(DT)
def DT(data):
    """
          :param data: raw spectrum data, shape (n_samples, n_features)
          :return: data after DT :(n_samples, n_features)
       """
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out
# 移动平均平滑           #要转置
def MA(a, WSZ=21):
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return a