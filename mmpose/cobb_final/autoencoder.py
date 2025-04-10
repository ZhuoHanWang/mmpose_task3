import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 使用 Sigmoid 确保输出在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded





if __name__ == '__main__':
    for i in range(17):
        # 1. 加载数据
        data = pd.read_csv('tan_theta.csv', usecols=[i], header=None)  # 替换为你的 CSV 文件路径

        # 2. 数据预处理
        features = data.values  # 获取特征值
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)  # 标准化数据

        # 转换为 PyTorch 张量
        features_tensor = torch.FloatTensor(features_scaled)

        # 划分训练集和测试集
        # train_data, test_data = train_test_split(features_tensor, test_size=0.2, random_state=42)

        train_data = features_tensor
        # 3. 构建自动编码器模型



        # 初始化模型、损失函数和优化器
        autoencoder = Autoencoder()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        # 4. 训练模型
        num_epochs = 300
        for epoch in range(num_epochs):
            autoencoder.train()
            optimizer.zero_grad()

            # 前向传播
            outputs = autoencoder(train_data)
            loss = criterion(outputs, train_data)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save(autoencoder.state_dict(), './autoencoder/' + str(i) + '.pth')

        # 5. 预测重构误差
        # autoencoder.eval()
        # with torch.no_grad():
        #     reconstructed = autoencoder(test_data)
        #     reconstruction_error = torch.mean((reconstructed - test_data) ** 2, dim=1).numpy()
        #
        # # 6. 设置阈值并进行异常检测
        # threshold = np.percentile(reconstruction_error, 95)  # 设定阈值为重构误差的95百分位数
        # anomalies = reconstruction_error > threshold
        #
        #
        #
        # 可视化重构误差
        # plt.figure(figsize=(10, 6))
        # plt.hist(reconstruction_error, bins=50)
        # plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
        # plt.title('Reconstruction Error Histogram')
        # plt.xlabel('Reconstruction Error')
        # plt.ylabel('Frequency')
        # plt.show()