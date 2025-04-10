import openpyxl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------------------
# 参数设置
# ---------------------------
EPOCHS = 500                # 训练轮数
BATCH_SIZE = 100            # 批大小
PATIENCE = 20               # EarlyStopping 的耐心值

NUM_HIDDEN_LAYERS_RED = 5   # 红球模型中间隐藏层的数量
NUM_HIDDEN_LAYERS_BLUE = 5  # 蓝球模型中间隐藏层的数量

# 红球模型参数(大乐透红球范围1-35)
INPUT_DIM = 35              # 输入维度
RED_FIRST_UNITS = 512       # 第一隐藏层神经元数
RED_HIDDEN_UNITS = 512      # 每个中间隐藏层神经元数
RED_OUTPUT_DIM = 35         # 输出层神经元数

# 蓝球模型参数(大乐透蓝球范围1-12)
BLUE_INPUT_DIM = 12         # 蓝球输入维度
BLUE_FIRST_UNITS = 512      # 第一层神经元数
BLUE_HIDDEN_UNITS = 512     # 中间隐藏层神经元数
BLUE_OUTPUT_DIM = 12        # 输出层

# 优化器设置
LEARNING_RATE = 0.001

# ---------------------------
# 红球数据预处理
# ---------------------------
# 读取数据
data = pd.read_excel('大乐透.xlsx')
input_features = data[['红球1', '红球2', '红球3', '红球4', '红球5']]

# One-Hot编码：将每个红球数字(1~35)转换为35维的one-hot向量
def one_hot_encode(balls):
    balls = np.clip(balls, 1, 35)
    one_hot = np.zeros((balls.size, 35))
    for i, ball in enumerate(balls.flatten()):
        one_hot[i, int(ball)-1] = 1
    one_hot = one_hot.reshape(-1, 5, 35)
    return one_hot.sum(axis=1)  # 形状：(样本数, 35)

input_features_one_hot = one_hot_encode(input_features.values)

# 转换为PyTorch张量
red_X = torch.FloatTensor(input_features_one_hot).to(device)
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

red_model = RedModel().to(device)
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
    predicted_indices = torch.topk(red_predictions[0], 5).indices.cpu()
    predicted_red_balls = predicted_indices.numpy() + 1  # 转换为1-35编号

# ---------------------------
# 蓝球数据预处理
# ---------------------------
blue_balls = data[['蓝球1', '蓝球2']].values
# 将蓝球数字(1-12)转换为12维one-hot向量
blue_one_hot = np.zeros((blue_balls.size, 24))  # 2个蓝球×12维
for i, (ball1, ball2) in enumerate(blue_balls):
    blue_one_hot[i, int(ball1)-1] = 1
    blue_one_hot[i, 12 + int(ball2)-1] = 1

# 转换为PyTorch张量
blue_X = torch.FloatTensor(blue_one_hot).to(device)
blue_dataset = TensorDataset(blue_X, blue_X)
blue_loader = DataLoader(blue_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 构建蓝球模型（自编码器结构）
# ---------------------------
class BlueModel(nn.Module):
    def __init__(self):
        super(BlueModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(BLUE_INPUT_DIM*2, BLUE_FIRST_UNITS),  # 2个蓝球×12维=24
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(BLUE_HIDDEN_UNITS, BLUE_HIDDEN_UNITS),
                nn.ReLU()
            ) for _ in range(NUM_HIDDEN_LAYERS_BLUE)],
            nn.Linear(BLUE_HIDDEN_UNITS, BLUE_OUTPUT_DIM*2),  # 输出2个蓝球
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

blue_model = BlueModel().to(device)
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
    # 预测第一个蓝球
    blue1_pred = torch.argmax(blue_prediction[0, :12]).item() + 1
    
    # 预测第二个蓝球(确保不等于第一个)
    blue2_probs = blue_prediction[0, 12:].clone()
    blue2_probs[blue1_pred-1] = -1  # 将第一个蓝球的概率设为-1
    blue2_pred = torch.argmax(blue2_probs).item() + 1

# ---------------------------
# 输出预测结果
# ---------------------------
print('Predicted Red Balls:', predicted_red_balls)
print('Predicted Blue Balls:', [blue1_pred, blue2_pred])
