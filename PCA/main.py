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
x1 = data['x']
y1 = data['y']

# 画图
plt.scatter(x1,y1,color = 'c',marker = 'p', label = '原始数据')
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)

#--------任务(b)-------
# 构造np.array类型的数据矩阵A
A = np.array(data)

# 对每一个属性的样本求均值
MEAN = np.mean(A, axis=0)  # 沿轴0调用mean函数

# 去中心化
X = np.subtract(A, MEAN)

# 计算协方差矩阵
COV = np.cov(X.T)

# 计算特征值和特征向量 W:特征值 V:特征向量
W, V = np.linalg.eig(COV)
# 这里求出的W并非按照大小进行排序后的结果 此处进行优化 以保证与api求得结果相似
# 对特征值按照大小降序排序 此处返回值是特征值对应的下标
sorted_index = np.argsort(-W) # 此处将参数设定为[-][参数名称]以表明是降序
tW = W[sorted_index[::1]] # 按sorted_index中的顺序依次取W中元素 存储在tW中
W = tW
tV = V[:, sorted_index[::1]] # 按sorted_index中的顺序依次取V中元素 存储在tV中
V = tV

# 计算主成分贡献率以及累计贡献率
sum_lambda = np.sum(W)  # 特征值的和
f = np.divide(W, sum_lambda)  # 每个特征值的贡献率（特征值 / 总和）
# 要求保留两个维度 此处不计算前几个贡献率的和>0.9
# 前两大特征值对应的特征向量为：
e1 = V.T[0]
e2 = V.T[1]

# 计算主成分值（已去中心化）X是去中心化后的结果
z1 = np.dot(X, e1)
z2 = np.dot(X, e2)

# 输出降维后的结果（已去中心化）
RES = np.array([z1, z2])
RES = RES.T # 转制一遍之后是最终结果

# 画图
RES_df = pd.DataFrame(RES)
# RES_df.to_excel('my_RES.xlsx')
RES_df.columns = ['x', 'y']
x2 = RES_df['x'] * (-1)
y2 = RES_df['y']

# 画图
plt.scatter(x2,y2,color = 'r',marker = 'p', label = 'PCA变换')
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)

#--------任务(c)-------
# 创建特征值构成的对角矩阵D 求D的-1/2次方
new_W = W ** (-1 / 2)
D = np.diag(new_W)

# V、D相乘 作为白化处理中前面要乘的矩阵
white_V = np.dot(V, D)
e1 = white_V.T[0]
e2 = white_V.T[1]

# 计算主成分值（已去中心化）X是去中心化后的结果
z1 = np.dot(X, e1)
z2 = np.dot(X, e2)

# 输出降维后的结果（已去中心化）
RES_white = np.array([z1, z2])
RES_white = RES_white.T # 转制一遍之后是最终结果

# 画图
RES_df_white = pd.DataFrame(RES_white)
# RES_df_white.to_excel('my_white_RES.xlsx')
RES_df_white.columns = ['x', 'y']
x3 = RES_df_white['x']
y3 = RES_df_white['y']

# 画图
plt.scatter(x3,y3,color = 'g',marker = 'p', label = '白化变换')
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)

# 最终展示
plt.legend() # 图例
plt.title('手搓结果')
plt.show()