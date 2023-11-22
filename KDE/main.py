import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import math

plt.figure()

# X: 符合对数正态分布的数据
xdata = pd.read_excel('data.xlsx')
xdata = np.array(xdata['x'])
xdata = np.sort(xdata)
X = xdata[:, np.newaxis]

# 创建一个[1,52]范围内包含1200个数据的等差数列X_plot
X_plot = np.linspace(1, 52, 1200)[:, np.newaxis]
# 计算mu = 2 sigma = 0.5的对数正态分布的真实概率密度
sigma = 0.5
mu = 2
true_dens = st.lognorm.pdf(X_plot[:, 0], s = sigma, scale = math.exp(mu))
plt.plot(X_plot[:, 0], true_dens, 'b-', label='true')  # 绘制真实数据

# # 求出最佳带宽 求解出的最佳带宽为:1.9630406500402715
# bandwidths = 10 ** np.linspace(-1, 1, 100)
# grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
# grid.fit(X)
# best_KDEbandwidth = grid.best_params_['bandwidth']

# 依次选取带宽
bands = [1.9630406500402715, 0.2, 5]
colors = ["r", "g", "darkorange"]
for i in range(len(bands)):
    # 用X数据训练模型
    kde = KernelDensity(kernel='gaussian', bandwidth=bands[i]).fit(X)
    # 在X_plot数据上测试
    log_dens = kde.score_samples(X_plot)
    # 画图
    lab = 'bandwidth=' + str(bands[i])
    if bands[i] == 1.9630406500402715:
        lab = lab + '(best)'
    plt.plot(X_plot[:, 0], np.exp(log_dens), color=colors[i], label=lab)  # kde数据

plt.legend()
plt.show()
