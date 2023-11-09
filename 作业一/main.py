import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# 创建图
plt.figure()

#--------任务(a)-------
# 导入生成的随机数据
data = pd.read_excel('data.xlsx', index_col = 0)
# print(data)
x1 = data['x']
y1 = data['y']

#画图
# plt.subplot(2, 2, 1)
plt.scatter(x1,y1,color = 'c',marker = 'p')
# plt.legend() #图例
plt.title("随机数据散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)
plt.show()

#--------任务(b)-------
# 构造np.array类型的数据矩阵A
A = np.array(data)
# print(A)

# 对每一个属性的样本求均值
MEAN = np.mean(A, axis=0)  # 沿轴0调用mean函数
# print(MEAN)

# 去中心化
X = np.subtract(A, MEAN)
# print(X)
# print(X.T)  # 矩阵的转置

# 计算协方差矩阵
COV = np.cov(X.T)
# print(COV)

# 计算特征值和特征向量
W, V = np.linalg.eig(COV)
# print(W)  # 特征值
# print(V)  # 特征向量

# 计算主成分贡献率以及累计贡献率
sum_lambda = np.sum(W)  # 特征值的和
# print(sum_lambda)
f = np.divide(W, sum_lambda)  # 每个特征值的贡献率（特征值 / 总和）
# print(f)
# 要求保留两个维度 此处不计算前几个贡献率的和>0.9
# 前两大特征值对应的特征向量为：
e1 = V.T[0]
# print(e1)
e2 = V.T[1]
# print(e2)

# 计算主成分值（已去中心化）X是去中心化后的结果
# print(X.shape)
z1 = np.dot(X, e1)
# print(z1)
z2 = np.dot(X, e2)
# print(z2)

# 输出降维后的结果（已去中心化）
RES = np.array([z2, z1])
# print(RES)
# print(RES.T)
RES = RES.T # 转制一遍之后是最终结果
# print(RES)

# 画图
RES_df = pd.DataFrame(RES)
# RES_df.to_excel('my_RES.xlsx')
RES_df.columns = ['x', 'y']
# print(RES_df)
x2 = RES_df['x']
y2 = RES_df['y']

# plt.subplot(2, 2, 2)
plt.scatter(x2,y2,color = 'c',marker = 'p')
# plt.legend() #图例
plt.title("PCA变换散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)
plt.show()

#--------任务(c)-------
new_W = W ** (-1 / 2)
D = np.diag(new_W)

white_V = np.dot(V, D)
e1 = white_V.T[0]
# print(e1)
e2 = white_V.T[1]
# print(e2)

# 计算主成分值（已去中心化）X是去中心化后的结果
# print(X.shape)
z1 = np.dot(X, e1)
# print(z1.shape)
z2 = np.dot(X, e2)
# print(z2.shape)

# 输出降维后的结果（已去中心化）
RES_white = np.array([z1, z2])
# print(RES)
# print(RES.T)
RES_white = RES_white.T # 转制一遍之后是最终结果
# print(RES)

# 画图
RES_df_white = pd.DataFrame(RES_white)
RES_df_white.to_excel('my_white_RES.xlsx')
RES_df_white.columns = ['x', 'y']
# print(RES_df)
x3 = RES_df_white['x']
y3 = RES_df_white['y']

# plt.subplot(2, 2, 3)
plt.scatter(x3,y3,color = 'c',marker = 'p')
# plt.legend() #图例
plt.title("白化变换散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)
plt.show()