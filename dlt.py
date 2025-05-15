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
EPOCHS = 100                # 训练轮数
BATCH_SIZE = 1              # 批大小
PATIENCE = 10               # EarlyStopping 的耐心值

# 极简网络结构
NUM_HIDDEN_LAYERS_RED = 1   # 红球模型中间隐藏层的数量
NUM_HIDDEN_LAYERS_BLUE = 1  # 蓝球模型中间隐藏层的数量

# 红球模型参数(大乐透红球范围1-35)
INPUT_DIM = 35              # 输入维度
RED_FIRST_UNITS = 64        # 第一隐藏层神经元数
RED_HIDDEN_UNITS = 64       # 每个中间隐藏层神经元数
RED_OUTPUT_DIM = 35         # 输出层神经元数

# 蓝球模型参数(大乐透蓝球范围1-12)
BLUE_INPUT_DIM = 12         # 蓝球输入维度
BLUE_FIRST_UNITS = 32       # 第一层神经元数
BLUE_HIDDEN_UNITS = 32      # 中间隐藏层神经元数
BLUE_OUTPUT_DIM = 12        # 输出层

# 优化器设置
LEARNING_RATE = 0.0001

# ---------------------------
# 红球数据预处理
# ---------------------------
# 读取数据
data = pd.read_excel('大乐透.xlsx')
# 过滤掉红球或蓝球任意一列为"-"的数据行
valid_mask = (data[['红球1', '红球2', '红球3', '红球4', '红球5', '蓝球1', '蓝球2']] != "-").all(axis=1)
data = data[valid_mask].reset_index(drop=True)
input_features = data[['红球1', '红球2', '红球3', '红球4', '红球5']]

# One-Hot编码：将每个红球数字(1~35)转换为35维的one-hot向量
def one_hot_encode(balls):
    balls = np.clip(balls, 1, 35)
    one_hot = np.zeros((balls.size, 35))
    for i, ball in enumerate(balls.flatten()):
        one_hot[i, int(ball)-1] = 1
    one_hot = one_hot.reshape(-1, 5, 35)
    return one_hot.sum(axis=1)  # 形状：(样本数, 35)

# 滑动窗口特征：用前window_size期预测下一期，体现数字运动的先后规律
red_features = data[['红球1', '红球2', '红球3', '红球4', '红球5']].values
window_size = 10  # 可调整，表示用最近10期预测下一期
if len(red_features) <= window_size:
    raise ValueError("有效数据不足，无法进行红球滑动窗口预测！")

red_X_list = []
red_y_list = []
for i in range(len(red_features) - window_size):
    x = one_hot_encode(red_features[i:i+window_size]).flatten()
    y = one_hot_encode(red_features[i+window_size:i+window_size+1]).flatten()  # 保证y为(35,)
    red_X_list.append(x)
    red_y_list.append(y)
red_X_np = np.stack(red_X_list)
red_y_np = np.stack(red_y_list)

# 转换为PyTorch张量
red_X = torch.FloatTensor(red_X_np).to(device)
red_y = torch.FloatTensor(red_y_np).to(device)
red_dataset = TensorDataset(red_X, red_y)
red_loader = DataLoader(red_dataset, batch_size=1, shuffle=True)

# ---------------------------
# 构建红球模型（自编码器结构）
# ---------------------------
class RedModel(nn.Module):
    def __init__(self, input_dim):
        super(RedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, RED_FIRST_UNITS),
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

red_model = RedModel(red_X.shape[1]).to(device)
red_criterion = nn.BCELoss()
red_optimizer = optim.Adam(red_model.parameters(), lr=LEARNING_RATE)

# 红球模型训练
best_red_loss = float('inf')
red_patience_counter = 0

for epoch in range(EPOCHS):
    red_model.train()
    epoch_loss = 0
    for batch_X, batch_y in tqdm(red_loader, desc=f'Red Epoch {epoch+1}'):
        red_optimizer.zero_grad()
        outputs = red_model(batch_X)
        loss = red_criterion(outputs, batch_y)
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
# 用最新window_size期预测下一期
red_model.eval()
with torch.no_grad():
    latest_x = one_hot_encode(red_features[-window_size:]).flatten().reshape(1, -1)
    latest_x_tensor = torch.FloatTensor(latest_x).to(device)
    red_predictions = red_model(latest_x_tensor)
    predicted_indices = torch.topk(red_predictions[0], 5).indices.cpu()
    predicted_red_balls = predicted_indices.numpy() + 1  # 转换为1-35编号
    print("数字运动预测红球:", predicted_red_balls)

# ---------------------------
# 蓝球数据预处理
# ---------------------------
blue_balls = data[['蓝球1', '蓝球2']].values
if len(blue_balls) < 2:
    raise ValueError("有效数据不足，无法进行蓝球预测！")

# 滑动窗口特征：用前window_size_blue期预测下一期，体现数字运动的先后规律
def blue_one_hot_encode(balls):
    # balls: (样本数, 2)
    one_hot = np.zeros((balls.shape[0], 24))
    for i, (ball1, ball2) in enumerate(balls):
        one_hot[i, int(ball1)-1] = 1
        one_hot[i, 12 + int(ball2)-1] = 1
    return one_hot

window_size_blue = 10  # 可调整，表示用最近10期预测下一期
if len(blue_balls) <= window_size_blue:
    raise ValueError("有效数据不足，无法进行蓝球滑动窗口预测！")

blue_X_list = []
blue_y_list = []
for i in range(len(blue_balls) - window_size_blue):
    x = blue_one_hot_encode(blue_balls[i:i+window_size_blue]).flatten()
    y = blue_one_hot_encode(blue_balls[i+window_size_blue:i+window_size_blue+1]).flatten()  # 保证y为(24,)
    blue_X_list.append(x)
    blue_y_list.append(y)
blue_X_np = np.stack(blue_X_list)
blue_y_np = np.stack(blue_y_list)

# 转换为PyTorch张量
blue_X = torch.FloatTensor(blue_X_np).to(device)
blue_y = torch.FloatTensor(blue_y_np).to(device)
blue_dataset = TensorDataset(blue_X, blue_y)
blue_loader = DataLoader(blue_dataset, batch_size=1, shuffle=True)

# ---------------------------
# 构建蓝球模型（自编码器结构）
# ---------------------------
class BlueModel(nn.Module):
    def __init__(self, input_dim):
        super(BlueModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, BLUE_FIRST_UNITS),
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

blue_model = BlueModel(blue_X.shape[1]).to(device)
blue_criterion = nn.MSELoss()
blue_optimizer = optim.Adam(blue_model.parameters(), lr=LEARNING_RATE)

# 蓝球模型训练
best_blue_loss = float('inf')
blue_patience_counter = 0

for epoch in range(EPOCHS):
    blue_model.train()
    epoch_loss = 0
    for batch_X, batch_y in tqdm(blue_loader, desc=f'Blue Epoch {epoch+1}'):
        blue_optimizer.zero_grad()
        outputs = blue_model(batch_X)
        loss = blue_criterion(outputs, batch_y)
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
# 用最新window_size_blue期预测下一期
blue_model.eval()
with torch.no_grad():
    latest_x_blue = blue_one_hot_encode(blue_balls[-window_size_blue:]).flatten().reshape(1, -1)
    latest_x_blue_tensor = torch.FloatTensor(latest_x_blue).to(device)
    blue_prediction = blue_model(latest_x_blue_tensor)
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

# 结果覆盖写入到TXT文件
with open('dlt_predictions.txt', 'w') as f:
    f.write('Predicted Red Balls: ' + ', '.join(map(str, predicted_red_balls)) + '\n')
    f.write('Predicted Blue Balls: ' + str(blue1_pred) + ', ' + str(blue2_pred) + '\n')
