import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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
data = pd.read_excel("双色球.xlsx")

# 兼容 A~G 或 0~6 列头
cols0 = list(data.columns)[:7]
if cols0 == list("ABCDEFG"):
    data.columns = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]
elif cols0 == list(range(7)):
    data.columns = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]

# 丢弃缺失或“-”行
cols = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]
mask = data[cols].replace("-", np.nan).notnull().all(axis=1)
data = data[mask].reset_index(drop=True)
data[cols] = data[cols].astype(int)

console.print(f"[green]数据加载成功，共 {len(data)} 期有效数据。[/green]")

# ---------------------------
# 特征工程
# ---------------------------
def get_features(df, idx, window=10):
    reds = df.loc[max(0, idx-window): idx-1, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values
    flat = reds.flatten()

    # 1. 频率
    freq = np.bincount(flat-1, minlength=33) / flat.size

    # 2. 当前遗漏（归一化）
    last_seen = np.zeros(33, dtype=float)
    for num in range(1,34):
        gap = 0
        for j in range(idx-1, max(-1, idx-window-1), -1):
            gap += 1
            if num in df.loc[j, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values:
                break
        last_seen[num-1] = min(gap, window)
    last_seen /= window

    # 3. 和值（归一化）
    sumv = flat.sum() / (6 * 33 * window)

    # 4. 奇偶比
    odd = np.sum(flat % 2)
    even = flat.size - odd
    odd_even = np.array([odd/flat.size, even/flat.size])

    # 5. 区间分布
    seg = [np.sum(flat<=11), np.sum((flat>11)&(flat<=22)), np.sum(flat>22)]
    seg = np.array(seg) / flat.size

    # 6. 连号率
    sorted_b = np.sort(flat)
    lianhao = np.sum(np.diff(sorted_b)==1) / (sorted_b.size-1)

    # 7. 二阶组合特征
    pair_cnt = int(33*(33-1)/2)
    pair_freq = np.zeros(pair_cnt, dtype=float)
    idx_pf = 0
    for a, b in combinations(range(1,34), 2):
        pair_freq[idx_pf] = np.sum(np.all(np.isin(reds, [a,b]), axis=1)) / window
        idx_pf += 1

    return np.concatenate([freq, last_seen, [sumv], odd_even, seg, [lianhao], pair_freq])

def get_blue_features(df, idx, window=10):
    blues = df.loc[max(0, idx-window): idx-1, "BlueBall"].values

    # 1. 频率
    freq = np.bincount(blues-1, minlength=16) / blues.size

    # 2. 当前遗漏
    last = np.zeros(16, dtype=float)
    for num in range(1,17):
        gap = 0
        for j in range(idx-1, max(-1, idx-window-1), -1):
            gap += 1
            if num == df.loc[j, "BlueBall"]:
                break
        last[num-1] = min(gap, window)
    last /= window

    # 3. 和值
    sumv = blues.sum() / (16 * window)

    # 4. 奇偶比
    odd = np.sum(blues % 2)
    even = blues.size - odd
    odd_even = np.array([odd/blues.size, even/blues.size])

    return np.concatenate([freq, last, [sumv], odd_even])

# 构造训练集
window = 10
X_red, y_red, X_blue, y_blue = [], [], [], []
for i in range(window, len(data)-1):
    X_red.append(get_features(data, i, window))
    y_red.append(data.loc[i, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values)
    X_blue.append(get_blue_features(data, i, window))
    y_blue.append(data.loc[i, "BlueBall"])

X_red = np.array(X_red)
Y_red = np.vstack(y_red)  # shape (n_samples,6)
X_blue = np.array(X_blue)
y_blue = np.array(y_blue)

console.print(Panel("模型训练中...", style="cyan"))
multi_red = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
multi_red.fit(X_red, Y_red)
console.print("[green]红球多输出模型训练完成。[/green]")

blue_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
blue_clf.fit(X_blue, y_blue)
console.print("[green]蓝球模型训练完成。[/green]")

# ---------------------------
# 预测下一期
# ---------------------------
console.print(Panel("预测下一期...", style="cyan"))
red_feat = get_features(data, len(data), window).reshape(1,-1)
blue_feat = get_blue_features(data, len(data), window).reshape(1,-1)

probas = multi_red.predict_proba(red_feat)
mean_p = np.mean([p[0] for p in probas], axis=0)
mean_p /= mean_p.sum()
pred_red = sorted((np.argsort(mean_p)[-6:] + 1).tolist())

blue_p = blue_clf.predict_proba(blue_feat)[0]
if len(blue_p) < 16:
    blue_p = np.pad(blue_p, (0, 16-len(blue_p)))
blue_p /= blue_p.sum()
pred_blue = int(np.argmax(blue_p) + 1)

# 输出结果
table = Table(title="双色球随机森林预测结果", show_header=True, header_style="bold magenta")
table.add_column("红球", style="red")
table.add_column("蓝球", style="blue")
table.add_row(" ".join(map(str, pred_red)), str(pred_blue))
console.print(table)

with open("ssq_predictions.txt","w") as f:
    f.write("Predicted Red Balls: " + ", ".join(map(str, pred_red)) + "\n")
    f.write("Predicted Blue Ball: " + str(pred_blue) + "\n")
console.print(Panel("[bold green]预测结果已写入 ssq_predictions.txt[/bold green]"))
