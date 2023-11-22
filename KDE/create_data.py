import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
import math

sigma = 0.5
mu = 2

xdata = st.lognorm.rvs(s = sigma, scale = math.exp(mu), size = 1000)

data = pd.DataFrame([])
data['x'] = xdata

data.to_excel('data.xlsx')