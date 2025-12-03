import pandas as pd
import numpy as np
import optuna
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
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
cols0 = list(data.columns)[:7]
if cols0 == list("ABCDEFG"):
    data.columns = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]
elif cols0 == list(range(7)):
    data.columns = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]
cols = ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6","BlueBall"]
mask = data[cols].replace("-", np.nan).notnull().all(axis=1)
data = data[mask].reset_index(drop=True)
data[cols] = data[cols].astype(int)
console.print(f"[green]数据加载成功，共 {len(data)} 期有效数据。[/green]")

# ---------------------------
# 特征工程
# ---------------------------
def get_features(df, idx, window=10):
    reds = df.loc[max(0, idx-window):idx-1, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values
    flat = reds.flatten()
    freq = np.bincount(flat-1, minlength=33) / flat.size

    last_seen = np.zeros(33, float)
    for num in range(1,34):
        gap = 0
        for j in range(idx-1, max(-1, idx-window-1), -1):
            gap += 1
            if num in df.loc[j, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values:
                break
        last_seen[num-1] = min(gap, window)
    last_seen /= window

    sumv = flat.sum() / (6 * 33 * window)
    odd = np.sum(flat % 2)
    even = flat.size - odd
    odd_even = np.array([odd/flat.size, even/flat.size])
    seg = np.array([np.sum(flat<=11), np.sum((flat>11)&(flat<=22)), np.sum(flat>22)]) / flat.size
    sorted_b = np.sort(flat)
    lianhao = np.sum(np.diff(sorted_b)==1) / (sorted_b.size-1)

    pair_cnt = 33*(33-1)//2
    pair_freq = np.zeros(pair_cnt, float)
    idx_pf = 0
    for a, b in combinations(range(1,34), 2):
        pair_freq[idx_pf] = np.sum(np.all(np.isin(reds, [a,b]), axis=1)) / window
        idx_pf += 1

    return np.concatenate([freq, last_seen, [sumv], odd_even, seg, [lianhao], pair_freq])

def get_blue_features(df, idx, window=10):
    blues = df.loc[max(0, idx-window):idx-1, "BlueBall"].values
    freq = np.bincount(blues-1, minlength=16) / blues.size

    last = np.zeros(16, float)
    for num in range(1,17):
        gap = 0
        for j in range(idx-1, max(-1, idx-window-1), -1):
            gap += 1
            if num == df.loc[j, "BlueBall"]:
                break
        last[num-1] = min(gap, window)
    last /= window

    sumv = blues.sum() / (16 * window)
    odd = np.sum(blues % 2)
    even = blues.size - odd
    odd_even = np.array([odd/blues.size, even/blues.size])
    return np.concatenate([freq, last, [sumv], odd_even])

# ---------------------------
# 构造训练集
# ---------------------------
window = 10
X_red, Y_red, X_blue, y_blue = [], [], [], []
for i in range(window, len(data)-1):
    X_red.append(get_features(data, i, window))
    Y_red.append(data.loc[i, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values)
    X_blue.append(get_blue_features(data, i, window))
    y_blue.append(data.loc[i, "BlueBall"])
X_red = np.array(X_red)
Y_red = np.vstack(Y_red)
X_blue = np.array(X_blue)
y_blue = np.array(y_blue)

# ---------------------------
# 超参优化与回测评估
# ---------------------------
console.print(Panel("开始超参优化与回测评估...", style="cyan"))
def objective(trial):
    n_est = trial.suggest_int("n_estimators", 50, 200)
    max_d = trial.suggest_int("max_depth", 5, 20)
    tss = TimeSeriesSplit(n_splits=5)
    losses = []
    for tr_idx, val_idx in tss.split(X_red):
        X_tr, X_va = X_red[tr_idx], X_red[val_idx]
        Y_tr, Y_va = Y_red[tr_idx], Y_red[val_idx]
        clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42))
        clf.fit(X_tr, Y_tr)
        probas = clf.predict_proba(X_va)
        for k, probas_k in enumerate(probas):
            # 处理多输出每个模型的概率矩阵
            y_true = Y_va[:, k] - 1
            p = probas_k  # shape (n_val_samples, n_classes)
            if p.shape[1] < 33:
                p = np.pad(p, ((0, 0), (0, 33 - p.shape[1])))
            losses.append(log_loss(y_true, p, labels=list(range(33))))
    return np.mean(losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best = study.best_params
console.print(Panel(f"[green]最佳超参: {best}, log_loss: {study.best_value:.4f}[/green]"))

# 使用最佳参数重训练
console.print(Panel("训练最终模型...", style="cyan"))
multi_red = MultiOutputClassifier(RandomForestClassifier(**best, random_state=42))
multi_red.fit(X_red, Y_red)
blue_clf = RandomForestClassifier(**best, random_state=42)
blue_clf.fit(X_blue, y_blue)
console.print("[green]模型训练完成。[/green]")

# ---------------------------
# 回测评估
# ---------------------------
console.print(Panel("开始回测评估...", style="cyan"))
hits_red = hits_blue = 0
tests = len(data) - window - 1
for i in range(window, len(data)-1):
    pr_feat = get_features(data, i, window).reshape(1,-1)
    probas_r = multi_red.predict_proba(pr_feat)
    avg_p = np.mean([p[0] for p in probas_r], axis=0)
    top6 = set((np.argsort(avg_p)[-6:]+1).tolist())
    actual_r = set(data.loc[i, ["Ball1","Ball2","Ball3","Ball4","Ball5","Ball6"]].values)
    if top6 & actual_r: hits_red += 1
    pb_feat = get_blue_features(data, i, window).reshape(1,-1)
    pb = blue_clf.predict_proba(pb_feat)[0]
    if int(np.argmax(pb)+1) == data.loc[i, "BlueBall"]: hits_blue += 1
console.print(f"红球 Top-6 回测命中率: {hits_red/tests:.2%}")
console.print(f"蓝球 回测命中率: {hits_blue/tests:.2%}")

# ---------------------------
# 预测下一期并输出
# ---------------------------
console.print(Panel("预测下一期...", style="cyan"))
red_feat = get_features(data, len(data), window).reshape(1,-1)
blue_feat = get_blue_features(data, len(data), window).reshape(1,-1)
probas = multi_red.predict_proba(red_feat)
mean_p = np.mean([p[0] for p in probas], axis=0)
mean_p /= mean_p.sum()
pred_red = sorted((np.argsort(mean_p)[-6:]+1).tolist())
blue_p = blue_clf.predict_proba(blue_feat)[0]
pred_blue = int(np.argmax(blue_p)+1)

table = Table(title="双色球预测结果", show_header=True, header_style="bold magenta")
table.add_column("红球", style="red")
table.add_column("蓝球", style="blue")
table.add_row(" ".join(map(str, pred_red)), str(pred_blue))
console.print(table)

with open("ssq_predictions.txt", "w") as f:
    f.write("Predicted Red Balls: "+", ".join(map(str, pred_red))+"\n")
    f.write("Predicted Blue Ball: "+str(pred_blue)+"\n")
console.print(Panel("[bold green]预测结果已写入 ssq_predictions.txt[/bold green]"))
