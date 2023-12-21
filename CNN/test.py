import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 16, 81, 256, 625]
y3 = [1, 8, 27, 64, 125]

# 创建第一张图并绘制多条线
fig1, ax1 = plt.subplots()
ax1.plot(x, y1, label='Line 1')
ax1.plot(x, y2, label='Line 2')
ax1.set_title('Lines in Figure 1')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()

# 创建第二张图并绘制多条线
fig2, ax2 = plt.subplots()
ax2.plot(x, y2, label='Line 2')
ax2.plot(x, y3, label='Line 3')
ax2.set_title('Lines in Figure 2')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.legend()

# 显示图像
plt.show()
