import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# 需求：
# 先对数据做线性回归，
# 计算并定义残差+常数项，s
# 通过基于人工鱼群算法改进的LSTM神经网络对定义的这项进行非线性拟合，再进行预测
data = pd.read_csv(r"data\raw_data.csv",header=None)
