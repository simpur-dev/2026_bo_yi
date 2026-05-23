"""
X1: 复刻 dataset.py 的 file split 逻辑，看 val 集来自哪些 iter，与 train 集
    的 winner/player/policy entropy 分布是否一致。
X3: 读 iter17 metrics_history，统计 "best" epoch 与 epoch 1 的差距。
"""

import json
import math
import os
import random
import sys
from collections import Counter


VAL_SPLIT = 0.1
SPLIT_SEED = 2026


def split_files(data_dir):
    filenames = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(filenames)
    val_count = max(1, min(int(len(filenames) * VAL_SPLIT), len(filenames) - 1))
    val_files = sorted(filenames[:val_count])
    train_files = sorted(filenames[val_count:])
    return train_files, val_files


def policy_entropy(policy, legal_mask):
    total = sum(legal_mask)
    if total <= 1:
        return 0.0
    s = 0.0
    for p, m in zip(policy, legal_mask):
        if m and p > 1e-9:
            s -= p * math.log(p)
    return s / math.log(total)


def collect_stats(data_dir, files):
    samples = []
    for f in files:
        with open(os.path.join(data_dir, f), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    if not samples:
        return None
    n = len(samples)
    winners = Counter(s.get("winner", 0) for s in samples)
    players = Counter(s.get("player", 0) for s in samples)
    ents = [policy_entropy(s["policy"], s["legal_mask"]) for s in samples]
    confs = [max(s["policy"]) for s in samples]
    # 按 file iter prefix 分组
    iter_dist = Counter(f.split("__")[0] for f in files)
    return {
        "n": n,
        "winner_w": winners.get(1, 0) / n,
        "winner_b": winners.get(-1, 0) / n,
        "player_w": players.get(1, 0) / n,
        "player_b": players.get(-1, 0) / n,
        "avg_ent": sum(ents) / n,
        "avg_conf": sum(confs) / n,
        "iter_dist": iter_dist,
    }


def main():
    data_dir = "../data/backward_run1/combined"
    data_dir = os.path.abspath(data_dir)
    print(f"data_dir: {data_dir}\n")
    print("=" * 80)
    print("X1: train/val split distribution shift analysis")
    print("=" * 80)

    train_files, val_files = split_files(data_dir)
    print(f"train_files ({len(train_files)}):")
    for f in train_files:
        print(f"  {f}")
    print(f"\nval_files ({len(val_files)}):")
    for f in val_files:
        print(f"  {f}")

    print("\n--- Train ---")
    ts = collect_stats(data_dir, train_files)
    print(f"  N={ts['n']:5d}  W_winrate={ts['winner_w']:.3f}  B_winrate={ts['winner_b']:.3f}  "
          f"player+={ts['player_w']:.3f}  player-={ts['player_b']:.3f}  "
          f"avgEnt={ts['avg_ent']:.3f}  avgMaxP={ts['avg_conf']:.3f}")
    print(f"  iters in train: {dict(ts['iter_dist'])}")

    print("\n--- Val ---")
    vs = collect_stats(data_dir, val_files)
    print(f"  N={vs['n']:5d}  W_winrate={vs['winner_w']:.3f}  B_winrate={vs['winner_b']:.3f}  "
          f"player+={vs['player_w']:.3f}  player-={vs['player_b']:.3f}  "
          f"avgEnt={vs['avg_ent']:.3f}  avgMaxP={vs['avg_conf']:.3f}")
    print(f"  iters in val: {dict(vs['iter_dist'])}")

    # Shift 指标
    def kl_bernoulli(p, q):
        eps = 1e-9
        p = min(max(p, eps), 1-eps)
        q = min(max(q, eps), 1-eps)
        return p * math.log(p/q) + (1-p) * math.log((1-p)/(1-q))

    print("\n--- Shift metrics (Bernoulli KL) ---")
    print(f"  winner_w shift: train={ts['winner_w']:.3f} val={vs['winner_w']:.3f} KL={kl_bernoulli(ts['winner_w'], vs['winner_w']):.4f}")
    print(f"  player_w shift: train={ts['player_w']:.3f} val={vs['player_w']:.3f} KL={kl_bernoulli(ts['player_w'], vs['player_w']):.4f}")

    print("\n" + "=" * 80)
    print("X3: iter17 metrics_history analysis")
    print("=" * 80)

    report_path = "../data/models/backward_st0_iter17_report.json"
    report_path = os.path.abspath(report_path)
    if not os.path.exists(report_path):
        print(f"NOT FOUND: {report_path}")
        return
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    mh = report.get("metrics_history", [])
    if not mh:
        print("empty metrics_history")
        return
    print(f"{'epoch':>5} {'train_loss':>10} {'train_pl':>9} {'val_loss':>9} {'val_pl':>9} {'val_top1':>9} {'val_mae':>9} {'kl_div':>9}")
    for r in mh:
        ep = r.get("epoch", "?")
        tl = r.get("train_loss", float("nan"))
        tp = r.get("train_policy_loss", float("nan"))
        vl = r.get("val_loss", "")
        vp = r.get("val_policy_loss", "")
        vt = r.get("val_policy_acc_top1", "")
        vm = r.get("val_value_mae", "")
        kl = r.get("kl_divergence", "")
        def f(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"{ep:>5} {tl:>10.4f} {tp:>9.4f} {f(vl):>9} {f(vp):>9} {f(vt):>9} {f(vm):>9} {f(kl):>9}")

    # 找 best val_policy_loss epoch
    val_records = [r for r in mh if "val_policy_loss" in r]
    if val_records:
        best_pl = min(val_records, key=lambda r: r["val_policy_loss"])
        worst_pl = max(val_records, key=lambda r: r["val_policy_loss"])
        ep1 = val_records[0]
        print(f"\nBest val_policy_loss: {best_pl['val_policy_loss']:.4f} @ epoch {best_pl['epoch']}  Top1={best_pl.get('val_policy_acc_top1')}")
        print(f"Worst val_policy_loss: {worst_pl['val_policy_loss']:.4f} @ epoch {worst_pl['epoch']}")
        print(f"Epoch 1 val_policy_loss: {ep1['val_policy_loss']:.4f}  Top1={ep1.get('val_policy_acc_top1')}")


if __name__ == "__main__":
    main()
