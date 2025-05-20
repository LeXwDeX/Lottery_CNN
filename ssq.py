import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
warnings.filterwarnings("ignore")

# ---------------------------
# 读取数据
# ---------------------------
console.rule("[bold green]双色球智能预测系统")
console.print(Panel("数据加载中...", style="cyan"))
data = pd.read_excel('双色球.xlsx')
# 适配Excel表头为A~G或0~6的情况，自动重命名为标准列名
if list(data.columns)[:7] == list('ABCDEFG'):
    data.columns = ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6', 'BlueBall']
elif list(data.columns)[:7] == list(range(7)):
    data.columns = ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6', 'BlueBall']
# 过滤掉红球或蓝球任意一列为"-"或空的数据行
cols = ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6', 'BlueBall']
valid_mask = (data[cols].replace("-", np.nan).notnull()).all(axis=1)
data = data[valid_mask].reset_index(drop=True)
data[cols] = data[cols].astype(int)
console.print(f"[green]数据加载成功，共 {len(data)} 期有效数据。[/green]")

# ---------------------------
# 特征工程
# ---------------------------
def get_features(df, idx, window=10):
    # 统计特征：频率、遗漏、和值、奇偶、区间、连号
    reds = df.loc[idx-window:idx-1, ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']].values
    if len(reds) < window:
        reds = df.loc[:idx-1, ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']].values
    flat = reds.flatten()
    # 1. 频率
    freq = np.zeros(33)
    for n in flat:
        freq[n-1] += 1
    freq = freq / flat.size
    # 2. 当前遗漏
    last_seen = np.zeros(33)
    for i in range(33):
        for j in range(idx-1, idx-window-1, -1):
            if j < 0: break
            if (i+1) in df.loc[j, ['Ball1', 'Ball2', 'Ball3', 'Ball4', 'Ball5', 'Ball6']].values:
                last_seen[i] = idx-1-j
                break
        else:
            last_seen[i] = window
    last_seen = last_seen / window
    # 3. 和值
    sumv = flat.sum() / (6*33)
    # 4. 奇偶比
    odd = np.sum(flat % 2)
    even = flat.size - odd
    odd_even = np.array([odd/flat.size, even/flat.size])
    # 5. 区间分布（1-11,12-22,23-33）
    seg = [0,0,0]
    for n in flat:
        if n <= 11:
            seg[0] += 1
        elif n <= 22:
            seg[1] += 1
        else:
            seg[2] += 1
    seg = np.array(seg)/flat.size
    # 6. 连号数
    sorted_balls = np.sort(flat)
    lianhao = np.sum(np.diff(sorted_balls)==1) / (flat.size-1)
    return np.concatenate([freq, last_seen, [sumv], odd_even, seg, [lianhao]])

def get_blue_features(df, idx, window=10):
    blues = df.loc[idx-window:idx-1, ['BlueBall']].values
    if len(blues) < window:
        blues = df.loc[:idx-1, ['BlueBall']].values
    flat = blues.flatten()
    # 1. 频率
    freq = np.zeros(16)
    for n in flat:
        freq[n-1] += 1
    freq = freq / flat.size
    # 2. 当前遗漏
    last_seen = np.zeros(16)
    for i in range(16):
        for j in range(idx-1, idx-window-1, -1):
            if j < 0: break
            if (i+1) == df.loc[j, 'BlueBall']:
                last_seen[i] = idx-1-j
                break
        else:
            last_seen[i] = window
    last_seen = last_seen / window
    # 3. 和值
    sumv = flat.sum() / (1*16*window)
    # 4. 奇偶
    odd = np.sum(flat % 2)
    even = flat.size - odd
    odd_even = np.array([odd/flat.size, even/flat.size])
    return np.concatenate([freq, last_seen, [sumv], odd_even])

# 构造红球训练集
console.print(Panel("特征工程中...", style="cyan"))
window = 10
red_X = []
red_y = [[] for _ in range(6)]
for i in range(window, len(data)-1):
    feat = get_features(data, i, window)
    red_X.append(feat)
    for k in range(6):
        red_y[k].append(data.iloc[i, k])  # Ball1~Ball6
red_X = np.array(red_X)
red_y = [np.array(y) for y in red_y]

# 构造蓝球训练集
blue_X = []
blue_y = []
for i in range(window, len(data)-1):
    feat = get_blue_features(data, i, window)
    blue_X.append(feat)
    blue_y.append(data.loc[i, 'BlueBall'])
blue_X = np.array(blue_X)
blue_y = np.array(blue_y)
console.print("[green]特征工程完成。[/green]")

# ---------------------------
# 随机森林建模
# ---------------------------
console.print(Panel("红球随机森林建模...", style="cyan"))
red_models = []
for k in range(6):
    y = red_y[k]
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(red_X, y)
    red_models.append(clf)
    console.print(f"[yellow]红球{k+1} 随机森林训练完成[/yellow]")
console.print("[green]红球随机森林全部完成。[/green]")

console.print(Panel("蓝球随机森林建模...", style="cyan"))
blue_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
blue_model.fit(blue_X, blue_y)
console.print(f"[yellow]蓝球 随机森林训练完成[/yellow]")
console.print("[green]蓝球随机森林全部完成。[/green]")

# ---------------------------
# 预测下一期
# ---------------------------
console.print(Panel("预测下一期...", style="cyan"))
latest_red_feat = get_features(data, len(data), window).reshape(1, -1)
latest_blue_feat = get_blue_features(data, len(data), window).reshape(1, -1)

predicted_red_balls = []
for k, model in enumerate(red_models):
    # 预测概率分布，取概率最大的球号
    proba = model.predict_proba(latest_red_feat)[0]
    pred = np.argmax(proba) + 1
    predicted_red_balls.append(pred)
predicted_red_balls = sorted(set(predicted_red_balls))
while len(predicted_red_balls) < 6:
    for i in range(1, 34):
        if i not in predicted_red_balls:
            predicted_red_balls.append(i)
        if len(predicted_red_balls) == 6:
            break
predicted_red_balls = sorted(predicted_red_balls)

blue_proba = blue_model.predict_proba(latest_blue_feat)[0]
predicted_blue_ball = np.argmax(blue_proba) + 1
predicted_blue_ball = max(1, min(16, predicted_blue_ball))

# ---------------------------
# 输出预测结果
# ---------------------------
table = Table(title="双色球随机森林预测结果", show_header=True, header_style="bold magenta")
table.add_column("红球", style="red")
table.add_column("蓝球", style="blue")
red_str = " ".join([str(x) for x in predicted_red_balls])
blue_str = str(predicted_blue_ball)
table.add_row(red_str, blue_str)
console.print(table)
console.print(Panel("[bold green]预测结果已写入 ssq_predictions.txt[/bold green]"))

with open('ssq_predictions.txt', 'w') as f:
    f.write('Predicted Red Balls: ' + ', '.join(map(str, predicted_red_balls)) + '\n')
    f.write('Predicted Blue Ball: ' + str(predicted_blue_ball) + '\n')
