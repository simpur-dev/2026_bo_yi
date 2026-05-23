"""
诊断 selfplay 数据多样性的脚本（不动训练管道，只读现有 jsonl）。

度量：
  1. 每个 iter 内独立 (board, player) hash 占比 —— 多样性
  2. winner 分布 —— 验证白先手优势 (理论值: 0.925 白)
  3. policy entropy 平均与分布 —— 探索深度
  4. iter 之间 board 重叠率 —— 看数据是否同质化
  5. 每个 board 在 dataset 中出现次数分布 —— 极值即 collapse 严重度

输出：纯文本表格，不写文件。
"""

import json
import os
import sys
import hashlib
import math
from collections import Counter, defaultdict


def board_hash(board_list):
    """对 7*11*11 板状态做稳定 hash。floats 在 jsonl 中是 -1/0/1/0.25 等几个固定值。"""
    # 拼成短字符串再 hash，性能优于直接 str()
    s = ",".join(f"{v:.4g}" for v in board_list)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def policy_entropy(policy, legal_mask):
    """归一化熵 in [0, 1]: entropy / log(num_legal)。"""
    total_legal = sum(legal_mask)
    if total_legal <= 1:
        return 0.0
    s = 0.0
    for p, m in zip(policy, legal_mask):
        if m and p > 1e-9:
            s -= p * math.log(p)
    return s / math.log(total_legal)


def load_iter(iter_dir, max_lines=None):
    """读取 iter 的所有 jsonl 文件，返回样本列表。"""
    samples = []
    if not os.path.isdir(iter_dir):
        return samples
    files = sorted(f for f in os.listdir(iter_dir) if f.endswith(".jsonl"))
    for f in files:
        with open(os.path.join(iter_dir, f), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if max_lines and len(samples) >= max_lines:
                    return samples
    return samples


def analyze_iter(samples, label):
    if not samples:
        print(f"{label:>18}: <empty>")
        return None

    n = len(samples)
    boards = [board_hash(s["board"]) for s in samples]
    unique_boards = len(set(boards))
    diversity = unique_boards / n

    boards_with_player = [(boards[i], samples[i].get("player", 0)) for i in range(n)]
    unique_bp = len(set(boards_with_player))
    diversity_bp = unique_bp / n

    winners = Counter(s.get("winner", 0) for s in samples)
    players = Counter(s.get("player", 0) for s in samples)

    entropies = [policy_entropy(s["policy"], s["legal_mask"]) for s in samples]
    avg_ent = sum(entropies) / n

    confs = [max(s["policy"]) for s in samples]
    avg_conf = sum(confs) / n

    # board 出现次数分布
    bc = Counter(boards)
    max_repeat = max(bc.values())
    top5_repeat = sorted(bc.values(), reverse=True)[:5]

    print(f"{label:>18}: N={n:5d}  unique_board={unique_boards:5d} ({diversity*100:5.1f}%)  "
          f"unique_(b,player)={unique_bp:5d} ({diversity_bp*100:5.1f}%)  "
          f"avgEnt={avg_ent:.3f}  avgMaxP={avg_conf:.3f}  "
          f"W:B={winners.get(1,0)}/{winners.get(-1,0)}/{winners.get(0,0)}  "
          f"player+/-={players.get(1,0)}/{players.get(-1,0)}  "
          f"top5_repeat={top5_repeat}")
    return {
        "n": n,
        "unique_board": unique_boards,
        "diversity": diversity,
        "diversity_bp": diversity_bp,
        "winners": winners,
        "players": players,
        "avg_ent": avg_ent,
        "avg_conf": avg_conf,
        "boards": boards,
    }


def cross_iter_overlap(stats_a, stats_b, label_a, label_b):
    set_a = set(stats_a["boards"])
    set_b = set(stats_b["boards"])
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = inter / union if union else 0
    overlap_a = inter / len(set_a) if set_a else 0
    overlap_b = inter / len(set_b) if set_b else 0
    print(f"{label_a} ∩ {label_b}: inter={inter} jaccard={jaccard:.3f} "
          f"overlap_in_{label_a}={overlap_a:.3f} overlap_in_{label_b}={overlap_b:.3f}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "../data/backward_run1"
    base = os.path.abspath(base)
    print(f"Analyzing: {base}\n")

    # 收集所有 iter_*_st0 目录（按编号排序）
    iters = []
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if not os.path.isdir(full) or not d.startswith("iter_"):
            continue
        try:
            num = int(d.split("_")[1])
        except (IndexError, ValueError):
            continue
        # 跳过空目录
        if not any(f.endswith(".jsonl") for f in os.listdir(full)):
            continue
        iters.append((num, d, full))
    iters.sort()

    # 由于 backward_training 把数据放到 combined/，原 iter_*_stN 可能空
    # 检查 combined 目录中按文件名前缀分组
    combined_dir = os.path.join(base, "combined")
    all_stats = {}

    if os.path.isdir(combined_dir):
        print("=" * 80)
        print("Analyzing combined/ (grouped by iter prefix in filename)")
        print("=" * 80)
        files = sorted(f for f in os.listdir(combined_dir) if f.endswith(".jsonl"))
        # group by iter prefix (e.g., "iter_0001_st25__...")
        groups = defaultdict(list)
        for f in files:
            prefix = f.split("__")[0]
            groups[prefix].append(f)

        for prefix in sorted(groups.keys()):
            samples = []
            for f in groups[prefix]:
                with open(os.path.join(combined_dir, f), "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            samples.append(json.loads(line))
            stats = analyze_iter(samples, prefix)
            if stats:
                all_stats[prefix] = stats

    # 跨 iter 重叠：相邻几对
    print()
    print("=" * 80)
    print("Cross-iter overlap (adjacent + first vs last)")
    print("=" * 80)
    keys = sorted(all_stats.keys())
    for i in range(len(keys) - 1):
        cross_iter_overlap(all_stats[keys[i]], all_stats[keys[i+1]], keys[i], keys[i+1])
    if len(keys) >= 2:
        print()
        cross_iter_overlap(all_stats[keys[0]], all_stats[keys[-1]], keys[0], keys[-1])

    # 全局合并的多样性
    print()
    print("=" * 80)
    print("Global summary across all loaded iters")
    print("=" * 80)
    all_boards = []
    for s in all_stats.values():
        all_boards.extend(s["boards"])
    total = len(all_boards)
    unique = len(set(all_boards))
    print(f"Total samples: {total}  unique boards: {unique}  ({unique/total*100:.1f}%)")

    bc = Counter(all_boards)
    repeats = sorted(bc.values(), reverse=True)
    print(f"Top10 board repeat counts: {repeats[:10]}")
    print(f"Samples appearing ≥2 times: {sum(1 for v in repeats if v >= 2)}  "
          f"(占总独立 board {sum(1 for v in repeats if v >= 2)/unique*100:.1f}%)")


if __name__ == "__main__":
    main()
