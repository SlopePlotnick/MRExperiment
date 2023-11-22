import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.neighbors import KernelDensity
import math
import sklearn as skl

# ----------------------------------------------------------------------
# Plot a 1D density example
# ---------------------------------------------------------------------------
'''
用随机种子生成100个数据，其中30个是符合高斯分布（0,1）的数据，70个是符合高斯分布(5,1)的数据，
（0,1）表示以x轴上的0为中心点，宽度为1的高斯分布。
（5,1）表示以x轴上5为中心店，宽度为1的高斯分布
'''
# ---------------------------------------------------------------------------
N = 100
np.random.seed(1)
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]
print(X)
# ---------------------------------------------------------------------------

# 创建一个[-5,10]范围内包含1000个数据的等差数列
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
# 使用简单的高斯模型norm得到两个高斯分布的概率密度作为真实值（我不觉得这是最佳的办法）
true_dens = 0.3 * st.norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * st.norm(5, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
# 填充出用简单高斯模型得出的密度真实值
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
# 使用不同的内核进行拟合，我也不推荐这样做，我们首先应该是观察数据的分布，然后选择模型，而不是
# 一个个尝试，应该做的是调整我们的带宽。
kernels = ["gaussian", "tophat", "epanechnikov"]
# 划线的粗细
lw = 2

for color, kernel in zip(colors, kernels):
    # 用X数据进行训练模型
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    # 在X_plot数据上测试
    log_dens = kde.score_samples(X_plot)
    # 画图
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=color,
        lw=lw,
        linestyle="-",
        label="kernel = '{0}'".format(kernel),
    )

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc="upper left")
# 用'+'代表真实的数据并且画出，用于观察数据分布集中情况
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()