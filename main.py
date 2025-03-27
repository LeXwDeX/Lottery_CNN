import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import openpyxl

# ---------------------------
# 参数设置
# ---------------------------
EPOCHS = 100                # 训练轮数
BATCH_SIZE = 10             # 批大小
PATIENCE = 20               # EarlyStopping 的耐心值

NUM_HIDDEN_LAYERS_RED = 3   # 红球模型中间隐藏层的数量
NUM_HIDDEN_LAYERS_BLUE = 3  # 蓝球模型中间隐藏层的数量

# 红球模型参数
INPUT_DIM = 33              # 输入维度（经过one-hot编码后每个红球）
RED_FIRST_UNITS = 128       # 第一隐藏层神经元数
RED_HIDDEN_UNITS = 64       # 每个中间隐藏层神经元数
RED_OUTPUT_DIM = 33         # 输出层神经元数

# 蓝球模型参数
BLUE_INPUT_DIM = 16         # 蓝球输入维度
BLUE_FIRST_UNITS = 128      # 第一层神经元数
BLUE_HIDDEN_UNITS = 128     # 中间隐藏层神经元数
BLUE_OUTPUT_DIM = 16        # 输出层

# 优化器设置（两模型统一采用相同学习率）
LEARNING_RATE = 0.001

# ---------------------------
# 红球数据预处理
# ---------------------------
# 读取数据
data = pd.read_excel('双色球.xlsx')
input_features = data[['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']]

# One-Hot编码：将每个红球数字（1~33）转换为33维的one-hot向量
def one_hot_encode(balls):
    # 确保球号在1-33范围内
    balls = np.clip(balls, 1, 33)
    # 创建正确维度的one-hot数组
    one_hot = np.zeros((balls.size, 33))
    for i, ball in enumerate(balls.flatten()):
        one_hot[i, int(ball)-1] = 1
    one_hot = one_hot.reshape(-1, 6, 33)
    return one_hot.sum(axis=1)  # 形状：(样本数, 33)

input_features_one_hot = one_hot_encode(input_features.values)

# 转换为PyTorch张量
red_X = torch.FloatTensor(input_features_one_hot)
red_dataset = TensorDataset(red_X, red_X)
red_loader = DataLoader(red_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 构建红球模型（自编码器结构）
# ---------------------------
class RedModel(nn.Module):
    def __init__(self):
        super(RedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, RED_FIRST_UNITS),
            nn.ReLU(),
            nn.Linear(RED_FIRST_UNITS, RED_HIDDEN_UNITS),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(RED_HIDDEN_UNITS, RED_HIDDEN_UNITS),
                nn.ReLU()
            ) for _ in range(NUM_HIDDEN_LAYERS_RED-1)],
            nn.Linear(RED_HIDDEN_UNITS, RED_OUTPUT_DIM),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

red_model = RedModel()
red_criterion = nn.BCELoss()
red_optimizer = optim.Adam(red_model.parameters(), lr=LEARNING_RATE)

# 红球模型训练
best_red_loss = float('inf')
red_patience_counter = 0

for epoch in range(EPOCHS):
    red_model.train()
    epoch_loss = 0
    for batch_X, _ in tqdm(red_loader, desc=f'Red Epoch {epoch+1}'):
        red_optimizer.zero_grad()
        outputs = red_model(batch_X)
        loss = red_criterion(outputs, batch_X)
        loss.backward()
        red_optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(red_loader)
    
    # EarlyStopping
    if avg_loss < best_red_loss:
        best_red_loss = avg_loss
        red_patience_counter = 0
    else:
        red_patience_counter += 1
        if red_patience_counter >= PATIENCE:
            print(f'Red Early stopping at epoch {epoch+1}')
            break

# 红球预测
red_model.eval()
with torch.no_grad():
    red_predictions = red_model(red_X[-1].unsqueeze(0))
    predicted_indices = torch.topk(red_predictions[0], 6).indices
    predicted_red_balls = predicted_indices.numpy() + 1  # 转换为1-33编号

# ---------------------------
# 蓝球数据预处理
# ---------------------------
blue_balls = data['BlueBall'].values
# 将蓝球数字(1-16)转换为16维one-hot向量
blue_one_hot = np.zeros((blue_balls.size, 16))
for i, ball in enumerate(blue_balls):
    blue_one_hot[i, int(ball)-1] = 1

# 转换为PyTorch张量
blue_X = torch.FloatTensor(blue_one_hot)
blue_dataset = TensorDataset(blue_X, blue_X)
blue_loader = DataLoader(blue_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 构建蓝球模型（自编码器结构）
# ---------------------------
class BlueModel(nn.Module):
    def __init__(self):
        super(BlueModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(BLUE_INPUT_DIM, BLUE_FIRST_UNITS),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(BLUE_HIDDEN_UNITS, BLUE_HIDDEN_UNITS),
                nn.ReLU()
            ) for _ in range(NUM_HIDDEN_LAYERS_BLUE)],
            nn.Linear(BLUE_HIDDEN_UNITS, BLUE_OUTPUT_DIM),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

blue_model = BlueModel()
blue_criterion = nn.MSELoss()
blue_optimizer = optim.Adam(blue_model.parameters(), lr=LEARNING_RATE)

# 蓝球模型训练
best_blue_loss = float('inf')
blue_patience_counter = 0

for epoch in range(EPOCHS):
    blue_model.train()
    epoch_loss = 0
    for batch_X, _ in tqdm(blue_loader, desc=f'Blue Epoch {epoch+1}'):
        blue_optimizer.zero_grad()
        outputs = blue_model(batch_X)
        loss = blue_criterion(outputs, batch_X)
        loss.backward()
        blue_optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(blue_loader)
    
    # EarlyStopping
    if avg_loss < best_blue_loss:
        best_blue_loss = avg_loss
        blue_patience_counter = 0
    else:
        blue_patience_counter += 1
        if blue_patience_counter >= PATIENCE:
            print(f'Blue Early stopping at epoch {epoch+1}')
            break

    # 蓝球预测
    blue_model.eval()
    with torch.no_grad():
        blue_prediction = blue_model(blue_X[-1].unsqueeze(0))
        predicted_index = torch.argmax(blue_prediction[0]).item()
        predicted_blue_ball = predicted_index + 1  # 转换为1-16编号

# ---------------------------
# 输出预测结果
# ---------------------------
print('Predicted Red Balls:', predicted_red_balls)
print('Predicted Blue Ball:', predicted_blue_ball)
