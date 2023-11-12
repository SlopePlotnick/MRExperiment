import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成随机数据
rd = np.random.normal(size = (2000, 2)) @ np.array([[2, 1], [1, 2]])
data = pd.DataFrame([])
data['x'] = rd[:, 0]
data['y'] = rd[:, 1]
data.to_excel('data.xlsx')