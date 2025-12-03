import argparse
import datetime as dt
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

console = Console()

RED_COLS = ['红球1', '红球2', '红球3', '红球4', '红球5']
BLUE_COLS = ['蓝球1', '蓝球2']


@dataclass
class Ticket:
    strategy: str
    reds: List[int]
    blues: List[int]
    note: str
    score: float


@dataclass
class Strategy:
    name: str
    description: str
    generator: Callable[['StrategyContext', np.random.Generator], Ticket]


@dataclass
class StrategyContext:
    prob_red: np.ndarray
    prob_blue: np.ndarray
    miss_red: Dict[int, int]
    miss_blue: Dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="大乐透号码相关性建模与预测工具",
    )
    parser.add_argument("--data", default="大乐透.xlsx", help="历史数据Excel文件路径")
    parser.add_argument("--window", type=int, default=12, help="构建特征时的历史窗口长度")
    parser.add_argument("--n-estimators", type=int, default=200, help="随机森林树数量")
    parser.add_argument("--max-depth", type=int, default=10, help="随机森林最大深度")
    parser.add_argument("--tickets", type=int, default=4, help="输出推荐注数")
    parser.add_argument("--eval", type=int, default=25, help="回测最近多少期（0表示跳过）")
    parser.add_argument("--seed", type=int, default=2024, help="随机数种子，保证可重复性")
    return parser.parse_args()


def load_dlt_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    header = list(df.columns)[:7]
    if header == list('ABCDEFG') or header == list(range(7)):
        df.columns = RED_COLS + BLUE_COLS
    df = df[[*RED_COLS, *BLUE_COLS]]
    mask = df.replace("-", np.nan).notnull().all(axis=1)
    df = df[mask].reset_index(drop=True)
    df = df.astype(int)
    return df


def compute_red_features(df: pd.DataFrame, idx: int, window: int) -> np.ndarray:
    subset = df.loc[max(0, idx - window):idx - 1, RED_COLS]
    if subset.empty:
        subset = df.loc[:idx - 1, RED_COLS]
    reds = subset.values.flatten()
    freq = np.zeros(35)
    for n in reds:
        freq[n - 1] += 1
    freq /= max(len(reds), 1)
    last_seen = np.full(35, window, dtype=float)
    for num in range(1, 36):
        for j in range(idx - 1, idx - window - 1, -1):
            if j < 0:
                break
            if num in df.loc[j, RED_COLS].values:
                last_seen[num - 1] = idx - 1 - j
                break
    last_seen /= max(window, 1)
    sumv = reds.sum() / (5 * 35) if len(reds) else 0
    odd = np.sum(reds % 2)
    total = len(reds) or 1
    odd_even = np.array([odd / total, (total - odd) / total])
    seg = np.zeros(3)
    for n in reds:
        if n <= 12:
            seg[0] += 1
        elif n <= 24:
            seg[1] += 1
        else:
            seg[2] += 1
    seg = seg / total
    sorted_balls = np.sort(reds)
    lianhao = np.sum(np.diff(sorted_balls) == 1) / (max(total - 1, 1))
    return np.concatenate([freq, last_seen, [sumv], odd_even, seg, [lianhao]])


def compute_blue_features(df: pd.DataFrame, idx: int, window: int) -> np.ndarray:
    subset = df.loc[max(0, idx - window):idx - 1, BLUE_COLS]
    if subset.empty:
        subset = df.loc[:idx - 1, BLUE_COLS]
    blues = subset.values.flatten()
    freq = np.zeros(12)
    for n in blues:
        freq[n - 1] += 1
    freq /= max(len(blues), 1)
    last_seen = np.full(12, window, dtype=float)
    for num in range(1, 13):
        for j in range(idx - 1, idx - window - 1, -1):
            if j < 0:
                break
            if num in df.loc[j, BLUE_COLS].values:
                last_seen[num - 1] = idx - 1 - j
                break
    last_seen /= max(window, 1)
    sumv = blues.sum() / (2 * 12 * max(window, 1)) if len(blues) else 0
    odd = np.sum(blues % 2)
    total = len(blues) or 1
    odd_even = np.array([odd / total, (total - odd) / total])
    return np.concatenate([freq, last_seen, [sumv], odd_even])


def build_training_sets(df: pd.DataFrame, window: int):
    red_X: List[np.ndarray] = []
    red_y = [[] for _ in range(5)]
    blue_X: List[np.ndarray] = []
    blue_y = [[] for _ in range(2)]
    for i in range(window, len(df)):
        feat_red = compute_red_features(df, i, window)
        red_X.append(feat_red)
        for k in range(5):
            red_y[k].append(int(df.loc[i, RED_COLS[k]]))
        feat_blue = compute_blue_features(df, i, window)
        blue_X.append(feat_blue)
        for k in range(2):
            blue_y[k].append(int(df.loc[i, BLUE_COLS[k]]))
    red_X_arr = np.array(red_X) if red_X else np.empty((0, 35 + 35 + 1 + 2 + 3 + 1))
    blue_X_arr = np.array(blue_X) if blue_X else np.empty((0, 12 + 12 + 1 + 2))
    red_y_arr = [np.array(y) for y in red_y]
    blue_y_arr = [np.array(y) for y in blue_y]
    return red_X_arr, red_y_arr, blue_X_arr, blue_y_arr


def train_random_forests(
    red_X: np.ndarray,
    red_y: List[np.ndarray],
    blue_X: np.ndarray,
    blue_y: List[np.ndarray],
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
):
    def train_group(X: np.ndarray, y_targets: List[np.ndarray], label: str):
        models = []
        for idx, y in enumerate(y_targets):
            if X.size == 0:
                break
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state + idx,
            )
            clf.fit(X, y)
            models.append(clf)
            console.log(f"{label}{idx + 1} 模型训练完成（样本数 {len(y)}）")
        return models

    red_models = train_group(red_X, red_y, "红球")
    blue_models = train_group(blue_X, blue_y, "蓝球")
    return red_models, blue_models


def aggregate_probabilities(models: List[RandomForestClassifier], latest_feat: np.ndarray, upper: int) -> np.ndarray:
    if not models:
        return np.ones(upper) / upper
    proba = np.zeros(upper, dtype=float)
    for model in models:
        classes = model.classes_.astype(int)
        probs = model.predict_proba(latest_feat)[0]
        temp = np.zeros(upper, dtype=float)
        temp[classes - 1] = probs
        proba += temp
    proba = proba / len(models)
    total = proba.sum()
    if total <= 0:
        return np.ones(upper) / upper
    return proba / total


def compute_miss_counts(df: pd.DataFrame, columns: List[str], max_number: int) -> Dict[int, int]:
    last_seen = {n: None for n in range(1, max_number + 1)}
    for idx in range(len(df) - 1, -1, -1):
        row_numbers = set(int(x) for x in df.loc[idx, columns].values)
        for number in row_numbers:
            if last_seen[number] is None:
                last_seen[number] = len(df) - 1 - idx
        if all(value is not None for value in last_seen.values()):
            break
    filled = {num: (val if val is not None else len(df)) for num, val in last_seen.items()}
    return filled


def blend_with_miss(prob: np.ndarray, miss: Dict[int, int], weight: Optional[float] = None) -> np.ndarray:
    if not miss:
        return prob
    miss_array = np.array([miss.get(i + 1, 0) for i in range(len(prob))], dtype=float)
    mean_gap = miss_array.mean()
    if mean_gap <= 0:
        return prob
    if weight is None:
        normalized = miss_array / (mean_gap + 1e-9)
        dispersion = float(np.std(normalized))
        recent_ratio = float(np.count_nonzero(miss_array == 0) / max(len(miss_array), 1))
        far_ratio = float(np.count_nonzero(miss_array >= mean_gap * 2.5) / max(len(miss_array), 1))
        weight = 0.25 + 0.25 * recent_ratio + 0.2 * dispersion + 0.15 * far_ratio
        weight = float(np.clip(weight, 0.2, 0.65))
    if weight <= 0:
        return prob
    adjusted = prob * (1.0 + weight * (miss_array / mean_gap))
    total = adjusted.sum()
    if total <= 0:
        return np.ones_like(prob) / len(prob)
    return adjusted / total


def avoid_recent_repeat(selection: List[int], miss: Dict[int, int], priority: np.ndarray) -> List[int]:
    last_draw = {num for num, gap in miss.items() if gap == 0}
    target_size = len(selection)
    if len(last_draw) != target_size or set(selection) != last_draw:
        return selection
    order = np.argsort(-priority)
    selection = selection.copy()
    for idx in order:
        candidate = int(idx) + 1
        if candidate in last_draw or candidate in selection:
            continue
        replace_idx = min(range(target_size), key=lambda i: priority[selection[i] - 1])
        selection[replace_idx] = candidate
        selection.sort()
        break
    return selection


def top_numbers(prob: np.ndarray, count: int) -> List[int]:
    order = np.argsort(-prob)
    picks = sorted(int(idx) + 1 for idx in order[:count])
    return picks


def weighted_numbers(prob: np.ndarray, count: int, rng: np.random.Generator) -> List[int]:
    available = list(range(1, len(prob) + 1))
    weights = prob.copy()
    selection: List[int] = []
    for _ in range(count):
        picked = pick_one(available, weights, rng)
        selection.append(picked)
        idx = available.index(picked)
        available.pop(idx)
        weights = np.delete(weights, idx)
    return sorted(selection)


def pick_one(candidates: List[int], weights: np.ndarray, rng: np.random.Generator) -> int:
    if len(candidates) == 1:
        return candidates[0]
    if weights.sum() <= 0:
        return int(rng.choice(candidates))
    probs = weights / weights.sum()
    return int(rng.choice(candidates, p=probs))


def balanced_numbers(prob: np.ndarray, rng: np.random.Generator) -> List[int]:
    segments = [
        (1, 12, 2),
        (13, 24, 2),
        (25, 35, 1),
    ]
    selected: List[int] = []
    available = set(range(1, len(prob) + 1))
    for start, end, quota in segments:
        segment = [n for n in range(start, end + 1) if n in available]
        if not segment:
            continue
        seg_probs = np.array([prob[n - 1] for n in segment], dtype=float)
        merged = []
        targets = min(quota, len(segment))
        for _ in range(targets):
            pick = pick_one(segment, seg_probs, rng)
            merged.append(pick)
            idx = segment.index(pick)
            segment.pop(idx)
            seg_probs = np.delete(seg_probs, idx)
        selected.extend(merged)
        available.difference_update(merged)
    if len(selected) < 5:
        remain = sorted(list(available))
        remain_probs = np.array([prob[n - 1] for n in remain], dtype=float)
        while len(selected) < 5 and remain:
            pick = pick_one(remain, remain_probs, rng)
            selected.append(pick)
            idx = remain.index(pick)
            remain.pop(idx)
            remain_probs = np.delete(remain_probs, idx)
    selected = sorted(selected)
    selected = adjust_parity(selected, prob, rng)
    return selected


def adjust_parity(selected: List[int], prob: np.ndarray, rng: np.random.Generator) -> List[int]:
    target_range = range(1, len(prob) + 1)
    odd_count = sum(n % 2 for n in selected)
    even_count = len(selected) - odd_count
    if 2 <= odd_count <= 3 and 2 <= even_count <= 3:
        return selected
    not_selected = [n for n in target_range if n not in selected]
    if odd_count < 2:
        odd_candidates = [n for n in not_selected if n % 2 == 1]
        if odd_candidates:
            replace_idx = min(
                (i for i, n in enumerate(selected) if n % 2 == 0),
                key=lambda i: prob[selected[i] - 1],
                default=None,
            )
            if replace_idx is not None:
                add = pick_one(odd_candidates, np.array([prob[n - 1] for n in odd_candidates]), rng)
                selected[replace_idx] = add
    elif even_count < 2:
        even_candidates = [n for n in not_selected if n % 2 == 0]
        if even_candidates:
            replace_idx = min(
                (i for i, n in enumerate(selected) if n % 2 == 1),
                key=lambda i: prob[selected[i] - 1],
                default=None,
            )
            if replace_idx is not None:
                add = pick_one(even_candidates, np.array([prob[n - 1] for n in even_candidates]), rng)
                selected[replace_idx] = add
    return sorted(selected)


def hot_cold_numbers(prob: np.ndarray, miss: Dict[int, int], hot_count: int, total: int) -> List[int]:
    prob_rank = sorted(range(1, len(prob) + 1), key=lambda n: (-prob[n - 1], n))
    miss_rank = sorted(range(1, len(prob) + 1), key=lambda n: (-miss[n], -prob[n - 1], n))
    selected: List[int] = []
    for num in prob_rank:
        if num not in selected:
            selected.append(num)
        if len(selected) >= hot_count:
            break
    for num in miss_rank:
        if num not in selected:
            selected.append(num)
        if len(selected) >= total:
            break
    if len(selected) < total:
        for num in prob_rank:
            if num not in selected:
                selected.append(num)
            if len(selected) >= total:
                break
    return sorted(selected[:total])


def geometric_score(prob_red: np.ndarray, prob_blue: np.ndarray, reds: List[int], blues: List[int]) -> float:
    values = [prob_red[n - 1] for n in reds] + [prob_blue[n - 1] for n in blues]
    values = [max(v, 1e-9) for v in values]
    product = np.prod(values)
    return float(product ** (1 / len(values)))


def create_ticket(strategy: str, note: str, reds: List[int], blues: List[int], context: StrategyContext) -> Ticket:
    score = geometric_score(context.prob_red, context.prob_blue, reds, blues)
    return Ticket(strategy=strategy, reds=sorted(reds), blues=sorted(blues), note=note, score=score)


def build_strategies() -> List[Strategy]:
    def weighted_strategy(context: StrategyContext, rng: np.random.Generator) -> Ticket:
        reds = weighted_numbers(context.prob_red, 5, rng)
        blues = weighted_numbers(context.prob_blue, 2, rng)
        return create_ticket("权重多样", "按概率权重采样，兼顾多样性", reds, blues, context)

    def top_strategy(context: StrategyContext, rng: np.random.Generator) -> Ticket:
        adjusted_red = blend_with_miss(context.prob_red, context.miss_red)
        adjusted_blue = blend_with_miss(context.prob_blue, context.miss_blue)
        reds = top_numbers(adjusted_red, 5)
        reds = avoid_recent_repeat(reds, context.miss_red, adjusted_red)
        blues = top_numbers(adjusted_blue, 2)
        blues = avoid_recent_repeat(blues, context.miss_blue, adjusted_blue)
        return create_ticket("概率优选", "概率权重叠加遗漏的稳健组合", reds, blues, context)

    def balanced_strategy(context: StrategyContext, rng: np.random.Generator) -> Ticket:
        reds = balanced_numbers(context.prob_red, rng)
        blues = [
            top_numbers(context.prob_blue, 1)[0],
            hot_cold_numbers(context.prob_blue, context.miss_blue, 1, 2)[-1],
        ]
        blues = sorted(set(blues))
        while len(blues) < 2:
            pool = [n for n in range(1, 13) if n not in blues]
            if not pool:
                pool = list(range(1, 13))
            blues.append(int(rng.choice(pool)))
            blues = sorted(set(blues))
        return create_ticket("区间均衡", "兼顾三大区间与奇偶的均衡组合", reds, blues[:2], context)

    def hotcold_strategy(context: StrategyContext, rng: np.random.Generator) -> Ticket:
        reds = hot_cold_numbers(context.prob_red, context.miss_red, hot_count=3, total=5)
        blues = hot_cold_numbers(context.prob_blue, context.miss_blue, hot_count=1, total=2)
        return create_ticket("冷热混搭", "热门号码配合遗漏较久号码", reds, blues, context)

    return [
        Strategy("概率优选", "概率最高且适度规避刚开出号码", top_strategy),
        Strategy("权重多样", "基于概率的随机探索组合", weighted_strategy),
        Strategy("区间均衡", "兼顾区间覆盖和奇偶配比", balanced_strategy),
        Strategy("冷热混搭", "热门+冷门结合，冲击意外惊喜", hotcold_strategy),
    ]


def generate_tickets(
    context: StrategyContext,
    strategies: List[Strategy],
    tickets_requested: int,
    seed: int,
) -> List[Ticket]:
    tickets: List[Ticket] = []
    for idx, strat in enumerate(strategies):
        rng = np.random.default_rng(seed + idx * 13)
        tickets.append(strat.generator(context, rng))
    extra_needed = max(0, tickets_requested - len(tickets))
    if extra_needed:
        weighted_strat = next((s for s in strategies if s.name == "权重多样"), strategies[0])
        for k in range(extra_needed):
            rng = np.random.default_rng(seed + 101 + k * 17)
            ticket = weighted_strat.generator(context, rng)
            ticket.strategy += f"+{k + 1}"
            ticket.note = "多次随机采样增强覆盖"
            tickets.append(ticket)
    tickets.sort(key=lambda t: t.score, reverse=True)
    return tickets[:tickets_requested]


def backtest(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    model_args: Dict[str, int],
    strategies: List[Strategy],
    seed: int,
) -> List[Dict[str, float]]:
    if horizon <= 0 or len(df) <= window + 1:
        return []
    start_idx = max(window + 1, len(df) - horizon)
    records: Dict[str, List[Dict[str, int]]] = {s.name: [] for s in strategies}
    for target_idx in range(start_idx, len(df)):
        history = df.iloc[:target_idx]
        if len(history) <= window:
            continue
        red_X, red_y, blue_X, blue_y = build_training_sets(history, window)
        if red_X.size == 0 or blue_X.size == 0:
            continue
        red_models, blue_models = train_random_forests(
            red_X,
            red_y,
            blue_X,
            blue_y,
            model_args["n_estimators"],
            model_args["max_depth"],
            model_args["random_state"] + target_idx,
        )
        latest_red_feat = compute_red_features(history, len(history), window).reshape(1, -1)
        latest_blue_feat = compute_blue_features(history, len(history), window).reshape(1, -1)
        prob_red = aggregate_probabilities(red_models, latest_red_feat, 35)
        prob_blue = aggregate_probabilities(blue_models, latest_blue_feat, 12)
        miss_red = compute_miss_counts(history, RED_COLS, 35)
        miss_blue = compute_miss_counts(history, BLUE_COLS, 12)
        context = StrategyContext(prob_red=prob_red, prob_blue=prob_blue, miss_red=miss_red, miss_blue=miss_blue)
        actual_red = set(int(x) for x in df.iloc[target_idx][RED_COLS])
        actual_blue = set(int(x) for x in df.iloc[target_idx][BLUE_COLS])
        for idx, strat in enumerate(strategies):
            rng = np.random.default_rng(seed + target_idx * 37 + idx * 19)
            ticket = strat.generator(context, rng)
            red_hits = len(set(ticket.reds) & actual_red)
            blue_hits = len(set(ticket.blues) & actual_blue)
            records[strat.name].append(
                {
                    "red_hits": red_hits,
                    "blue_hits": blue_hits,
                    "total_hits": red_hits + blue_hits,
                }
            )
    summary: List[Dict[str, float]] = []
    for strat in strategies:
        outcomes = records[strat.name]
        if not outcomes:
            continue
        red_hits = np.array([o["red_hits"] for o in outcomes])
        blue_hits = np.array([o["blue_hits"] for o in outcomes])
        total_hits = np.array([o["total_hits"] for o in outcomes])
        summary.append(
            {
                "strategy": strat.name,
                "samples": len(outcomes),
                "avg_red_hits": float(red_hits.mean()),
                "avg_blue_hits": float(blue_hits.mean()),
                "avg_total_hits": float(total_hits.mean()),
                "max_total_hits": int(total_hits.max()),
                "hit_rate_3plus": float((total_hits >= 3).mean()),
            }
        )
    return summary


def render_probability_table(
    prob: np.ndarray,
    miss: Dict[int, int],
    title: str,
    top_n: int,
    color: str,
) -> Table:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("排名", justify="right")
    table.add_column("号码", style=color, justify="center")
    table.add_column("概率%", justify="right")
    table.add_column("遗漏期数", justify="right")
    order = np.argsort(-prob)[:top_n]
    for rank, idx in enumerate(order, start=1):
        number = idx + 1
        probability = prob[idx] * 100
        omission = miss.get(number, 0)
        table.add_row(str(rank), str(number), f"{probability:.2f}", str(omission))
    return table


def render_ticket_table(tickets: List[Ticket]) -> Table:
    table = Table(title="推荐投注组合", show_header=True, header_style="bold magenta")
    table.add_column("策略")
    table.add_column("红球", style="red")
    table.add_column("蓝球", style="blue")
    table.add_column("策略说明", overflow="fold")
    table.add_column("组合强度(几何均值%)", justify="right")
    for ticket in tickets:
        red_display = " ".join(f"{n:02d}" for n in ticket.reds)
        blue_display = " ".join(f"{n:02d}" for n in ticket.blues)
        table.add_row(
            ticket.strategy,
            red_display,
            blue_display,
            ticket.note,
            f"{ticket.score * 100:.2f}",
        )
    return table


def render_backtest_table(results: List[Dict[str, float]]) -> Optional[Table]:
    if not results:
        return None
    table = Table(title="策略回测表现（最近区间）", show_header=True, header_style="bold magenta")
    table.add_column("策略")
    table.add_column("样本数", justify="right")
    table.add_column("平均红球命中", justify="right")
    table.add_column("平均蓝球命中", justify="right")
    table.add_column("平均总命中", justify="right")
    table.add_column("最高总命中", justify="right")
    table.add_column("命中≥3概率", justify="right")
    for row in results:
        table.add_row(
            row["strategy"],
            str(row["samples"]),
            f"{row['avg_red_hits']:.2f}",
            f"{row['avg_blue_hits']:.2f}",
            f"{row['avg_total_hits']:.2f}",
            str(row["max_total_hits"]),
            f"{row['hit_rate_3plus'] * 100:.1f}%",
        )
    return table


def save_predictions(tickets: List[Ticket], path: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"==== {timestamp} ====",
    ]
    for ticket in tickets:
        lines.append(
            f"{ticket.strategy}: 红球 {', '.join(f'{n:02d}' for n in ticket.reds)} | "
            f"蓝球 {', '.join(f'{n:02d}' for n in ticket.blues)} | "
            f"Score {ticket.score * 100:.2f}% | {ticket.note}"
        )
    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    console.rule("[bold green]大乐透智能关联预测")
    console.print(Panel("开始加载历史数据", style="cyan"))
    df = load_dlt_data(args.data)
    console.print(f"[green]成功加载 {len(df)} 期历史数据[/green]")
    if len(df) <= args.window:
        console.print(f"[red]数据量不足，至少需要 {args.window + 1} 期数据以训练模型。[/red]")
        return
    console.print(Panel("构建训练集与训练模型", style="cyan"))
    red_X, red_y, blue_X, blue_y = build_training_sets(df, args.window)
    if red_X.size == 0 or blue_X.size == 0:
        console.print("[red]训练数据构建失败，请检查窗口参数或数据文件。[/red]")
        return
    red_models, blue_models = train_random_forests(
        red_X,
        red_y,
        blue_X,
        blue_y,
        args.n_estimators,
        args.max_depth,
        args.seed,
    )
    latest_red_feat = compute_red_features(df, len(df), args.window).reshape(1, -1)
    latest_blue_feat = compute_blue_features(df, len(df), args.window).reshape(1, -1)
    prob_red = aggregate_probabilities(red_models, latest_red_feat, 35)
    prob_blue = aggregate_probabilities(blue_models, latest_blue_feat, 12)
    miss_red = compute_miss_counts(df, RED_COLS, 35)
    miss_blue = compute_miss_counts(df, BLUE_COLS, 12)
    context = StrategyContext(prob_red=prob_red, prob_blue=prob_blue, miss_red=miss_red, miss_blue=miss_blue)
    strategies = build_strategies()
    tickets = generate_tickets(context, strategies, args.tickets, args.seed)
    console.print(Panel("核心数据特征", style="cyan"))
    console.print(render_probability_table(prob_red, miss_red, "红球概率排行", top_n=12, color="red"))
    console.print(render_probability_table(prob_blue, miss_blue, "蓝球概率排行", top_n=6, color="blue"))
    console.print(Panel("生成推荐投注组合", style="cyan"))
    console.print(render_ticket_table(tickets))
    if args.eval > 0:
        console.print(Panel(f"执行最近 {args.eval} 期的滑窗回测", style="cyan"))
        backtest_results = backtest(
            df,
            args.window,
            args.eval,
            {"n_estimators": args.n_estimators, "max_depth": args.max_depth, "random_state": args.seed},
            strategies,
            args.seed,
        )
        table = render_backtest_table(backtest_results)
        if table:
            console.print(table)
        else:
            console.print("[yellow]历史期数不足，暂未生成回测结果。[/yellow]")
    save_predictions(tickets, "dlt_predictions.txt")
    console.print(Panel("[bold green]预测结果已追加至 dlt_predictions.txt[/bold green]"))


if __name__ == "__main__":
    main()
