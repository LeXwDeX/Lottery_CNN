import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
import openpyxl

# ---------------------------
# 参数设置
# ---------------------------
EPOCHS = 1000               # 训练轮数
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
BLUE_INPUT_DIM = 1          # 蓝球输入维度
BLUE_FIRST_UNITS = 128      # 第一层神经元数
BLUE_HIDDEN_UNITS = 128     # 中间隐藏层神经元数
BLUE_OUTPUT_DIM = 1         # 输出层

# 优化器设置（两模型统一采用相同学习率）
LEARNING_RATE = 0.001
red_optimizer = Adam(learning_rate=LEARNING_RATE)
blue_optimizer = Adam(learning_rate=LEARNING_RATE)

# ---------------------------
# 红球数据预处理
# ---------------------------
# 读取数据
data = pd.read_excel('双色球.xlsx')
input_features = data[['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']]

# One-Hot编码：将每个红球数字（1~33）转换为33维的one-hot向量，
# 然后对每行（6个球）求和，得到的向量中每个位置的值代表该数字出现的次数（0~6，一般为0或1）
label_binarizer = LabelBinarizer()
label_binarizer.fit(range(1, 34))
one_hot_encoded = label_binarizer.transform(input_features.values.flatten())
one_hot_encoded = one_hot_encoded.reshape(-1, 6, 33)
input_features_one_hot = one_hot_encoded.sum(axis=1)  # 形状：(样本数, 33)

# ---------------------------
# 构建红球模型（自编码器结构）
# ---------------------------
red_model = Sequential()
red_model.add(Input(shape=(INPUT_DIM,)))
red_model.add(Dense(RED_FIRST_UNITS, activation='relu'))

for _ in range(NUM_HIDDEN_LAYERS_RED):
    red_model.add(Dense(RED_HIDDEN_UNITS, activation='relu'))

red_model.add(Dense(RED_OUTPUT_DIM, activation='sigmoid'))

red_model.compile(optimizer=red_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 红球模型训练：目标是重构输入（自编码器）
red_model.fit(input_features_one_hot,  
              input_features_one_hot,  
              epochs=EPOCHS,  
              batch_size=BATCH_SIZE,  
              verbose=0,  
              callbacks=[
                  TqdmCallback(verbose=1),
                  EarlyStopping(monitor='loss', patience=PATIENCE)
              ])

# 红球预测：预测最后一条数据，并选出概率最高的6个数字作为预测结果
red_predictions = red_model.predict(input_features_one_hot[-1].reshape(1, -1))
predicted_indices = np.argsort(red_predictions[0])[::-1][:6]
predicted_red_balls = np.array(range(1, 34))[predicted_indices]

# ---------------------------
# 蓝球数据预处理
# ---------------------------
blue_balls = data['BlueBall'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
blue_balls_scaled = scaler.fit_transform(blue_balls)

# ---------------------------
# 构建蓝球模型（自编码器结构）
# ---------------------------
blue_model = Sequential()
blue_model.add(Input(shape=(BLUE_INPUT_DIM,)))
blue_model.add(Dense(BLUE_FIRST_UNITS, activation='relu'))

for _ in range(NUM_HIDDEN_LAYERS_BLUE):
    blue_model.add(Dense(BLUE_HIDDEN_UNITS, activation='relu'))

blue_model.add(Dense(BLUE_OUTPUT_DIM, activation='sigmoid'))

blue_model.compile(optimizer=blue_optimizer, loss='mean_squared_error')

# 蓝球模型训练：目标同样是重构输入（归一化后的蓝球数字）
blue_model.fit(blue_balls_scaled,  
               blue_balls_scaled,  
               epochs=EPOCHS,  
               batch_size=BATCH_SIZE,
               verbose=0,
               callbacks=[
                   TqdmCallback(verbose=1),
                   EarlyStopping(monitor='loss', patience=PATIENCE)
               ])

# 蓝球预测：预测最后一条数据，再通过逆归一化转换为原始数值
blue_prediction = blue_model.predict(blue_balls_scaled[-1].reshape(1, -1))
predicted_blue_ball = scaler.inverse_transform(blue_prediction).astype(int)

# ---------------------------
# 输出预测结果
# ---------------------------
print('Predicted Red Balls:', predicted_red_balls)
print('Predicted Blue Ball:', predicted_blue_ball.flatten()[0])
