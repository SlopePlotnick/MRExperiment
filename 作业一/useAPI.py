import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

data = pd.read_excel('data.xlsx', index_col = 0)
x = data['x']
y = data['y']
# print(data)

A = np.array(data)
pca = PCA(n_components = 2)
pca.fit(A)
RES = pca.transform(A)
plt.scatter(x,y,color = 'c',marker = 'p', label = '原始数据')
# plt.legend() #图例
# plt.title("PCA变换散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)
# print(RES)

RES_df = pd.DataFrame(RES)
# RES_df.to_excel('api_RES.xlsx')
RES_df.columns = ['x', 'y']
x1 = RES_df['x']
y1 = RES_df['y']

# plt.subplot(2, 2, 2)
plt.scatter(x1,y1,color = 'r',marker = 'p', label = 'PCA变换')
# plt.legend() #图例
# plt.title("PCA变换散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)

pca = PCA(n_components = 2, whiten = True)
pca.fit(A)
RES = pca.transform(A)

RES_df = pd.DataFrame(RES)
# RES_df.to_excel('api_white_RES.xlsx')
RES_df.columns = ['x', 'y']
x2 = RES_df['x']
y2 = RES_df['y']

# plt.subplot(2, 2, 2)
plt.scatter(x2,y2,color = 'g',marker = 'p', label = '白化变换')
# plt.legend() #图例
# plt.title("白化变换散点图",fontsize = 14)
plt.xlabel("x",fontsize = 12)
plt.ylabel("y",fontsize = 12)
plt.legend()
plt.title('api计算结果')
plt.show()