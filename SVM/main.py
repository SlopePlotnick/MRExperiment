from libsvm.svm import *
from libsvm.svmutil import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# #-----任务1-----
# train_label_1,train_pixel_1 = svm_read_problem('svmguide1.txt')
# predict_label_1,predict_pixel_1 = svm_read_problem('svmguide1_test.txt')
# m1 = svm_train(train_label_1, train_pixel_1)
# print("#1 result:")
# p_label_1, p_acc_1, p_val_1 = svm_predict(predict_label_1, predict_pixel_1, m1);
# print(p_acc_1)
#
# #-----任务2-----
# train_label_2,train_pixel_2 = svm_read_problem('scaledata.txt')
# predict_label_2,predict_pixel_2 = svm_read_problem('scaledata_test.txt')
# m2 = svm_train(train_label_2, train_pixel_2)
# print("#2 result:")
# p_label_2, p_acc_2, p_val_2 = svm_predict(predict_label_2, predict_pixel_2, m2);
# print(p_acc_2)
#
# #-----任务3-----
# train_label_3,train_pixel_3 = svm_read_problem('svmguide1.txt')
# predict_label_3,predict_pixel_3 = svm_read_problem('svmguide1_test.txt')
# m3 = svm_train(train_label_3, train_pixel_3, '-t 0')
# print("#3 result:")
# p_label_3, p_acc_3, p_val_3 = svm_predict(predict_label_3, predict_pixel_3, m3);
# print(p_acc_3)
#
# #-----任务4-----
# train_label_4,train_pixel_4 = svm_read_problem('svmguide1.txt')
# predict_label_4,predict_pixel_4 = svm_read_problem('svmguide1_test.txt')
# m4 = svm_train(train_label_4, train_pixel_4, '-c 1000 -t 2')
# print("#4 result:")
# p_label_4, p_acc_4, p_val_4 = svm_predict(predict_label_4, predict_pixel_4, m4);
# print(p_acc_4)
#
# #-----任务5-----
# train_label_5,train_pixel_5 = svm_read_problem('scaledata.txt')
# predict_label_5,predict_pixel_5 = svm_read_problem('scaledata_test.txt')
# m5 = svm_train(train_label_5, train_pixel_5, '-c 8 -g 2 -t 2')
# print("#5 result:")
# p_label_5, p_acc_5, p_val_5 = svm_predict(predict_label_5, predict_pixel_5, m5);
# print(p_acc_5)

#-----任务6-----
train_label_6,train_pixel_6 = svm_read_problem('w1a.txt')
predict_label_6,predict_pixel_6 = svm_read_problem('w1a.t')
m = svm_train(train_label_6, train_pixel_6)
print("#6 result:")
p_label_6, p_acc_6, p_val_6 = svm_predict(predict_label_6, predict_pixel_6, m)
print(p_acc_6)
all = 0
right = 0
for i in range(len(predict_label_6)):
    if predict_label_6[i] == 1:
        all = all + 1
        if p_label_6[i] == 1:
            right = right + 1
print('TPR:')
print(right / all)

xdata = np.array([]) # w1取值
ydata = np.array([]) # 真阳率
acc = np.array([]) # 样本总数
for i in range(20, 51, 5):
    option = '-w1 ' + str(i)
    xdata = np.append(xdata, i)
    m6 = svm_train(train_label_6, train_pixel_6, option)
    p_label, p_acc, p_val = svm_predict(predict_label_6, predict_pixel_6, m6);
    right = 0
    all = 0
    for i in range(len(predict_label_6)):
        if predict_label_6[i] == 1:
            all = all + 1
            if p_label[i] == 1:
                right = right + 1
    ydata = np.append(ydata, (right / all) * 100)
    acc = np.append(acc, p_acc[0])
data = pd.DataFrame([])
data['w1取值'] = xdata
data['真阳率%'] = ydata
data['准确率%'] = acc
data.to_excel('运行结果.xlsx')

plt.figure()
plt.plot(xdata, ydata, '-b')
plt.xlabel('w1取值')
plt.ylabel('真阳率%')
plt.show()

