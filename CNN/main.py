import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# 超参数定义
n_epochs = 20 # 训练参数的轮数
batch_size_train = 100 # 训练集的bacthsize
batch_size_test = 100 # 测试集的batchsize
learning_rate = 0.001 # 学习率 控制每次更新只采用负梯度的一小部分
# momentum = 0.5 # 优化器超参数？
# log_interval = 10 # ？
random_seed = 0 # 为任何使用随机数产生的东西设置随机种子
torch.manual_seed(random_seed)

mnist = torchvision.datasets.MNIST('./data/', train=True, download=False,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                           ]))
# 选取前10000条数据进行训练
train_set = Subset(mnist, range(10000))

# MNIST数据集的dataloader 数据将自动下载至目录的data文件夹
# 其中的0.1307和0.3081是MNIST数据集的全局平均值和标准偏差 用来作标准化
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# 网络构建
# 在此列出作业要求
# 卷积层 nn.Conv2d(nIn, nOut, kernel_size = (x, y), stride = 步长, padding = 填补)
# 最大汇合 nn.MaxPool2d(kernel_size=(2, 2), stride=2)
# ReLu层 nn.ReLu(inplace = True)
# 分类层 Softmax函数 nn.Softmax(dim = 1) dim = 1表示按行计算，dim = 0表示按列计算
class BASE(nn.Module):
    def __init__(self):
        super(BASE, self).__init__()
        self.model = torch.nn.Sequential(
            # 第一层
            torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1, padding=0),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding = 0),

            # 第二层
            torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=1, padding=0),
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding = 0),

            # 第三层
            torch.nn.Conv2d(in_channels=50, out_channels=500, kernel_size=(4, 4), stride=1, padding=0),
            torch.nn.ReLU(inplace = True),

            # 第四层
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=500, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output

# 网络初始化
base = BASE()
# 构建迭代器与损失函数
lossF = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = optim.Adam(base.parameters(), learning_rate) # Adam迭代器 学习率为之前设定的0.001

# 存储训练过程
history = {'Test Loss' : [], 'Test Accuracy' : []}
for epoch in range(1, n_epochs + 1):
    # 构建tqdm进度条
    processBar = tqdm(train_loader, unit = 'step')
    # 打开网络的训练模式
    base.train(True)

    # 开始对训练集的dataloader进行迭代
    for step, (train_imgs, labels) in enumerate(processBar):
        # --------------------------------训练代码--------------------------------
        # 清空模型梯度
        base.zero_grad()
        # 前向推理
        outputs = base(train_imgs)

        # 计算本轮损失
        loss = lossF(outputs, labels)
        # 计算本轮准确率
        predictions = torch.argmax(outputs, dim = 1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]

        # 反向传播求出模型参数的梯度
        loss.backward()
        # 使用迭代器更新模型权重
        optimizer.step()

        # 将本step结果进行可视化处理
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, n_epochs, loss.item(), accuracy.item()))

        #--------------------------------测试代码--------------------------------
        # 最后一次训练完成后
        if step == len(processBar) - 1:
            correct = 0 # 正确测试数量
            total_loss = 0 # 总损失

            # 关闭模型训练状态
            base.train(False)
            # 对测试集的dataloader迭代
            for test_imgs, labels in test_loader:
                outputs = base(test_imgs)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim = 1)

                # 存储测试结果
                total_loss += loss
                correct += torch.sum(predictions == labels)

            # 计算平均准确率
            test_accuracy = correct / (batch_size_test * len(test_loader))
            # 计算平均损失
            test_loss = total_loss / len(test_loader)
            # 存储历史数据
            history['Test Loss'].append(test_loss.item())
            history['Test Accuracy'].append(test_accuracy.item())

            # 展示本轮测试结果
            processBar.set_description("Test Loss: %.4f, Test Acc: %.4f" % (test_loss.item(), test_accuracy.item()))

    processBar.close()

#对测试Loss进行可视化
plt.plot(history['Test Loss'],label = 'Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#对测试准确率进行可视化
plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()