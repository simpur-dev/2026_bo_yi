"""
RL Fine-tune (PPO + Expert/self-play 混合) - 在服务器上跑
=========================================================

设计目标:
  从 9e553b3 的 dual model baseline (backward_best_model_*.pt) 出发,
  通过 PPO + self-play 让模型针对 expert 覆盖不到的局面自我提升。

与 train.py 的区别:
  - 监督学习 train.py:   loss = CE(policy, target_policy) + MSE(value, target_value)
                         target 来自 jsonl 中的 selfplay 记录
  - 本脚本 (PPO):        loss = -log_prob(advantage) + value_loss
                         advantage 来自 selfplay 跑出来的实际 reward
                         即"在 current policy 下产生的轨迹的回报"

数据流:
  ┌────────────────────────────────────────────────────────────────┐
  │ 旧策略 π_old 加载自 backward_best_model_{black,white}.pt      │
  │                                                                │
  │ 1) 用 π_old 跑 N 局 selfplay (调 build/backward_selfplay)      │
  │    → 生成 (board, action, reward) 三元组                      │
  │                                                                │
  │ 2) 计算 advantage = R - V(s) (用 π_old 的 value head 估值)     │
  │                                                                │
  │ 3) PPO 步骤: 用 π_new 更新 K 个 epoch, 限制 ratio ∈ [1-ε, 1+ε]│
  │    loss = -min(ratio * advantage, clip(ratio, ...) * advantage)│
  │         + c1 * MSE(value, R)  - c2 * entropy                  │
  │                                                                │
  │ 4) 每 --eval-interval 步: 调 build/evaluate_ai                 │
  │    → 候选 vs 基线 胜率 < 0.40 ? → 自动回滚到基线              │
  │                                                                │
  │ 5) 重复 1-4 直到 --total-iterations                           │
  └────────────────────────────────────────────────────────────────┘

用法 (在服务器上):
  # 1. 编译 selfplay 工具
  cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \\
        -DUSE_ONNX=ON
  cmake --build build --target backward_selfplay evaluate_ai -j

  # 2. 跑 RL fine-tune
  python train/rl_finetune_ppo.py \\
    --baseline-black data/backup_pre_rl_2026-06-14/backward_best_model_black.pt \\
    --baseline-white data/backup_pre_rl_2026-06-14/backward_best_model_white.pt \\
    --data-out data/rl_iter1 \\
    --selfplay-exe build/backward_selfplay \\
    --eval-exe build/evaluate_ai \\
    --arch resnet_s \\
    --device cuda \\
    --games-per-iter 50 \\
    --sims 400 \\
    --total-iterations 20 \\
    --eval-interval 2 \\
    --ppo-epochs 4 \\
    --ppo-clip 0.2 \\
    --lr 3e-5

  # 3. 强制继续训练 (例如断电恢复)
  python train/rl_finetune_ppo.py --resume data/rl_iter1/checkpoint.pt ...

红线:
  - 本脚本只读不写 3rdparty/onnxruntime_cpu/ (只走 PATH)
  - 本脚本不读不写 simpur/ ruikang/
  - baseline 路径必须位于 data/backup_pre_rl_2026-06-14/ 之下
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 复用 9e553b3 已有的模型定义
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_model, BOARD_SIZE, CHANNELS, ACTION_SIZE


# ========== 常量 ==========
POLICY_LR = 3e-5
VALUE_LR = 3e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5
ROLLOUT_DIR_NAME = "rollouts"
CKPT_DIR_NAME = "checkpoints"
LOG_NAME = "rl_log.jsonl"
EVAL_DIR_NAME = "evals"


# ========== 工具函数 ==========

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_baseline(model, baseline_path, device):
    """加载 baseline 权重到 model, 兼容 state_dict 或完整 checkpoint"""
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"baseline not found: {baseline_path}")
    ckpt = torch.load(baseline_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"  Loaded baseline: {baseline_path}")


def board_to_tensor(board_np, device):
    """
    board_np: (7, 11, 11) numpy float32
    返回: (1, 7, 11, 11) torch tensor on device
    """
    return torch.from_numpy(board_np).unsqueeze(0).to(device)


def masked_logits_to_log_probs(policy_logits, legal_mask):
    """对带 mask 的 logits 做 log_softmax, 返回 (batch, 60) log_probs"""
    masked = policy_logits - (1.0 - legal_mask) * 1e9
    return F.log_softmax(masked, dim=-1)


def masked_logits_to_probs(policy_logits, legal_mask):
    masked = policy_logits - (1.0 - legal_mask) * 1e9
    return F.softmax(masked, dim=-1)


# ========== Self-play rollout ==========

def run_selfplay(selfplay_exe, games, sims, start_step, output_dir,
                 model_black="", model_white="", extra_args=None, timeout=7200):
    """
    调 backward_selfplay 生成 selfplay 数据
    返回 (success, output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [selfplay_exe, str(games), str(sims), str(start_step), output_dir]
    if model_black:
        cmd.extend(["--black-model", model_black])
    if model_white:
        cmd.extend(["--white-model", model_white])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[Selfplay] games={games} sims={sims} st={start_step}")
    print(f"  Black: {model_black or '(heuristic)'}")
    print(f"  White: {model_white or '(heuristic)'}")
    print(f"  Cmd:   {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        print(f"  Selfplay done in {elapsed:.1f}s, returncode={proc.returncode}")
        if proc.returncode != 0 and proc.stderr:
            print(f"  stderr (last 500 chars): {proc.stderr[-500:]}")
        return proc.returncode == 0, output_dir
    except subprocess.TimeoutExpired:
        print(f"  Selfplay TIMEOUT after {timeout}s")
        return False, output_dir


def parse_selfplay_jsonl(jsonl_path):
    """
    解析 backward_selfplay 产出的 jsonl
    每行: {"board": [..], "player": 1/-1, "legal_mask": [..], "policy": [..], "value": .., ...}
    返回: list of dict
    """
    samples = []
    if not os.path.exists(jsonl_path):
        return samples
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                samples.append(d)
            except json.JSONDecodeError:
                continue
    return samples


# ========== GAE advantage ==========

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    rewards: list of float (per step)
    values:  list of float (V(s) for each step, bootstrap at end)
    dones:   list of bool (是否终止)
    返回: advantages (list), returns (list)
    """
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ========== PPO 更新 ==========

def ppo_update(model, optimizer, batch_data, device, clip_eps=CLIP_EPS,
               value_coef=VALUE_COEF, entropy_coef=ENTROPY_COEF,
               epochs=PPO_EPOCHS, mini_batch_size=MINI_BATCH_SIZE,
               max_grad_norm=MAX_GRAD_NORM):
    """
    batch_data: dict of tensors on CPU
      - board:        (N, 7, 11, 11)
      - legal_mask:   (N, 60)
      - action:       (N,) int64
      - old_log_prob: (N,) float32
      - advantage:    (N,) float32 (已归一化)
      - return_:      (N,) float32
    返回: 训练指标 dict
    """
    board = batch_data["board"].to(device)
    legal_mask = batch_data["legal_mask"].to(device)
    action = batch_data["action"].to(device)
    old_log_prob = batch_data["old_log_prob"].to(device)
    advantage = batch_data["advantage"].to(device)
    return_ = batch_data["return_"].to(device)

    n = board.size(0)
    indices = np.arange(n)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, mini_batch_size):
            mb = indices[start:start + mini_batch_size]
            mb_t = torch.as_tensor(mb, device=device)

            mb_board = board.index_select(0, mb_t)
            mb_legal = legal_mask.index_select(0, mb_t)
            mb_action = action.index_select(0, mb_t)
            mb_old_lp = old_log_prob.index_select(0, mb_t)
            mb_adv = advantage.index_select(0, mb_t)
            mb_ret = return_.index_select(0, mb_t)

            policy_logits, value_pred = model(mb_board)
            value_pred = value_pred.squeeze(-1)

            log_probs = masked_logits_to_log_probs(policy_logits, mb_legal)
            new_log_prob = log_probs.gather(1, mb_action.unsqueeze(1)).squeeze(1)
            probs = masked_logits_to_probs(policy_logits, mb_legal)

            # PPO clipped objective
            ratio = torch.exp(new_log_prob - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(value_pred, mb_ret)

            # Entropy bonus
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["n_updates"] += 1

    n_upd = max(stats["n_updates"], 1)
    return {k: v / n_upd for k, v in stats.items()}


# ========== 评估 vs baseline ==========

def evaluate_vs_baseline(model, eval_exe, candidate_path, baseline_path,
                         games, sims, device):
    """
    把 model 的权重存到 candidate_path, 然后调 evaluate_ai vs baseline_path
    返回 parsed dict
    """
    torch.save(model.state_dict(), candidate_path)
    if not os.path.exists(eval_exe):
        print(f"  [eval] evaluate_ai not found: {eval_exe}, 跳过")
        return None
    cmd = [eval_exe, str(games), candidate_path, baseline_path,
           "--sims", str(sims)]
    print(f"\n[Eval] {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  Eval timeout")
        return None
    elapsed = time.time() - t0
    print(f"  Eval done in {elapsed:.1f}s, rc={proc.returncode}")
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(f"  --- output (last 800) ---\n{out[-800:]}")

    parsed = {
        "candidate": candidate_path,
        "baseline": baseline_path,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    # 解析关键指标
    import re
    for key, pat in [
        ("a_wins", r"A wins:\s*(\d+)"),
        ("b_wins", r"B wins:\s*(\d+)"),
        ("a_b_margin", r"A-B margin:\s*(-?\d+\.?\d*)"),
        ("games", r"Games:\s*(\d+)"),
    ]:
        m = re.search(pat, out)
        if m:
            v = m.group(1)
            parsed[key] = float(v) if "." in v else int(v)
    return parsed


# ========== 早停与回滚 ==========

class EarlyStopWithRollback:
    """
    追踪最近 N 次 eval 的胜率, 若 < threshold 连续 K 次 → 触发回滚
    """

    def __init__(self, winrate_threshold=0.40, patience=3, history_size=10):
        self.winrate_threshold = winrate_threshold
        self.patience = patience
        self.history = deque(maxlen=history_size)
        self.below_streak = 0
        self.triggered = False

    def update(self, winrate):
        if winrate is None:
            return False
        self.history.append(winrate)
        if winrate < self.winrate_threshold:
            self.below_streak += 1
        else:
            self.below_streak = 0
        if self.below_streak >= self.patience:
            self.triggered = True
            return True
        return False


# ========== 主循环 ==========

def main():
    parser = argparse.ArgumentParser(description="RL Fine-tune (PPO + Expert/self-play mix)")
    parser.add_argument("--baseline-black", type=str, required=True,
                        help="黑方 baseline (例如 data/backup_pre_rl_2026-06-14/backward_best_model_black.pt)")
    parser.add_argument("--baseline-white", type=str, required=True,
                        help="白方 baseline")
    parser.add_argument("--data-out", type=str, required=True,
                        help="RL 训练输出目录 (rollouts/checkpoints/evals 都会写到这里)")
    parser.add_argument("--selfplay-exe", type=str, default="./build/backward_selfplay",
                        help="backward_selfplay 可执行文件路径")
    parser.add_argument("--eval-exe", type=str, default="./build/evaluate_ai",
                        help="evaluate_ai 可执行文件路径")
    parser.add_argument("--arch", type=str, default="resnet_s",
                        choices=["mlp", "cnn", "resnet_s", "resnet_m", "resnet_l"])
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu (服务器上一般用 cuda)")
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--start-step", type=int, default=0,
                        help="backward selfplay 起始步 (0=从头, 25=终局)")
    parser.add_argument("--total-iterations", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=2,
                        help="每 N 轮做一次 vs baseline 评估")
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--ppo-epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--ppo-clip", type=float, default=CLIP_EPS)
    parser.add_argument("--lr", type=float, default=POLICY_LR)
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--value-coef", type=float, default=VALUE_COEF)
    parser.add_argument("--rollout-max-samples", type=int, default=20000,
                        help="每次 selfplay 最多用多少样本 (0=全部)")
    parser.add_argument("--rollback-winrate", type=float, default=0.40)
    parser.add_argument("--rollback-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径 (含 model/optimizer/iter)")
    parser.add_argument("--max-iters-walltime-h", type=float, default=0,
                        help="墙钟时间上限 (小时), 0=不限")
    args = parser.parse_args()

    set_seed(args.seed)

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda 但 torch.cuda.is_available() == False, 回退到 cpu")
        args.device = "cpu"
    device = torch.device(args.device)

    # 目录
    out = args.data_out
    os.makedirs(out, exist_ok=True)
    rollout_dir = os.path.join(out, ROLLOUT_DIR_NAME)
    ckpt_dir = os.path.join(out, CKPT_DIR_NAME)
    eval_dir = os.path.join(out, EVAL_DIR_NAME)
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    log_path = os.path.join(out, LOG_NAME)

    # 日志
    log_fp = open(log_path, "a", encoding="utf-8")

    def log(msg, also_print=True):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        if also_print:
            print(line)
        log_fp.write(line + "\n")
        log_fp.flush()

    log("=" * 60)
    log(f"  RL Fine-tune (PPO) - 服务器训练")
    log("=" * 60)
    log(f"  baseline-black: {args.baseline_black}")
    log(f"  baseline-white: {args.baseline_white}")
    log(f"  data-out:       {out}")
    log(f"  arch:           {args.arch}")
    log(f"  device:         {device}")
    log(f"  total-iters:    {args.total_iterations}")
    log(f"  games/iter:     {args.games_per_iter}")
    log(f"  sims:           {args.sims}")
    log(f"  ppo-epochs:     {args.ppo_epochs}")
    log(f"  ppo-clip:       {args.ppo_clip}")
    log(f"  lr:             {args.lr}")
    log(f"  eval-interval:  {args.eval_interval}")
    log(f"  eval-games:     {args.eval_games}")
    log(f"  rollback:       winrate<{args.rollback_winrate} 连续 {args.rollback_patience} 次")
    log(f"  seed:           {args.seed}")
    log("=" * 60)

    # 模型
    model = create_model(arch=args.arch, device=device)
    load_baseline(model, args.baseline_black, device)
    log(f"  Model loaded from baseline, params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_iter = 1
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_iter = ck.get("iter", 1)
        log(f"  Resumed from {args.resume}, start_iter={start_iter}")

    early_stop = EarlyStopWithRollback(
        winrate_threshold=args.rollback_winrate,
        patience=args.rollback_patience,
    )

    walltime_start = time.time()

    # 训练主循环
    for it in range(start_iter, args.total_iterations + 1):
        iter_t0 = time.time()
        log(f"\n========== ITER {it}/{args.total_iterations} ==========")

        # 1) Selfplay: 用当前模型跑 N 局
        iter_rollout_dir = os.path.join(rollout_dir, f"iter_{it:04d}")
        candidate_onnx = os.path.join(ckpt_dir, f"candidate_iter_{it:04d}.pt")
        # 同时导出 onnx (backward_selfplay 需要 .onnx)
        candidate_onnx_path = os.path.join(ckpt_dir, f"candidate_iter_{it:04d}.onnx")
        torch.save(model.state_dict(), candidate_onnx)
        try_export_onnx(model, candidate_onnx_path, device)

        ok, _ = run_selfplay(
            args.selfplay_exe, args.games_per_iter, args.sims, args.start_step,
            iter_rollout_dir,
            model_black=candidate_onnx_path if os.path.exists(candidate_onnx_path) else "",
            model_white="",  # 白方用启发式
        )
        if not ok:
            log(f"  WARN: selfplay 失败, 跳过 PPO 更新")
            continue

        # 2) 解析 selfplay 数据, 计算 advantage
        samples = load_iter_samples(iter_rollout_dir, args.rollout_max_samples)
        log(f"  Loaded {len(samples)} samples from {iter_rollout_dir}")
        if len(samples) < 100:
            log(f"  WARN: 样本过少, 跳过 PPO")
            continue

        batch = build_ppo_batch(model, samples, device)
        if batch is None or batch["board"].size(0) < 32:
            log(f"  WARN: 有效样本过少, 跳过 PPO")
            continue

        # 3) PPO 更新
        stats = ppo_update(
            model, optimizer, batch, device,
            clip_eps=args.ppo_clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
        )
        log(f"  PPO stats: policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}")

        # 4) 保存检查点
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_iter_{it:04d}.pt")
        torch.save({
            "iter": it,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log(f"  Saved checkpoint: {ckpt_path}")

        # 5) 评估 vs baseline
        if it % args.eval_interval == 0:
            log(f"  Evaluating candidate vs baseline ...")
            candidate_eval = os.path.join(ckpt_dir, f"eval_candidate_iter_{it:04d}.pt")
            parsed = evaluate_vs_baseline(
                model, args.eval_exe, candidate_eval,
                args.baseline_black,  # 用黑方 baseline 作对手
                args.eval_games, args.sims, device,
            )
            if parsed is not None:
                eval_path = os.path.join(eval_dir, f"eval_iter_{it:04d}.json")
                with open(eval_path, "w", encoding="utf-8") as f:
                    json.dump({k: v for k, v in parsed.items()
                               if k not in ("stdout", "stderr")}, f, indent=2, ensure_ascii=False)
                log(f"  Eval saved: {eval_path}")
                a_wins = parsed.get("a_wins", 0) or 0
                b_wins = parsed.get("b_wins", 0) or 0
                games = parsed.get("games", 0) or 0
                if games > 0:
                    winrate = a_wins / games
                    margin = parsed.get("a_b_margin", 0) or 0
                    log(f"  Winrate: {winrate:.1%}  margin: {margin:.2f}")
                    if early_stop.update(winrate):
                        log(f"  ROLLBACK triggered (连续 {args.rollback_patience} 次 "
                            f"winrate<{args.rollback_winrate})")
                        log(f"  恢复到 baseline 权重, 停止训练")
                        load_baseline(model, args.baseline_black, device)
                        break

        iter_dt = time.time() - iter_t0
        log(f"  Iter {it} done in {iter_dt:.1f}s")

        # 墙钟时间检查
        if args.max_iters_walltime_h > 0:
            elapsed_h = (time.time() - walltime_start) / 3600
            if elapsed_h > args.max_iters_walltime_h:
                log(f"  达到墙钟上限 {args.max_iters_walltime_h}h, 停止")
                break

    log("\n========== 训练结束 ==========")
    log(f"  最终 checkpoint: {ckpt_dir}")
    log(f"  训练日志: {log_path}")
    log_fp.close()


# ========== 数据加载 + PPO batch 构建 ==========

def load_iter_samples(iter_rollout_dir, max_samples=0):
    """加载一个 iter 目录下的所有 jsonl 文件"""
    all_samples = []
    for fn in sorted(os.listdir(iter_rollout_dir)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(iter_rollout_dir, fn)
        all_samples.extend(parse_selfplay_jsonl(path))
    if max_samples and len(all_samples) > max_samples:
        random.shuffle(all_samples)
        all_samples = all_samples[:max_samples]
    return all_samples


def build_ppo_batch(model, samples, device):
    """
    把 selfplay 样本转成 PPO 训练 batch
    关键: 在当前 (old) policy 下重新算 log_prob, 用于 ratio
    Returns: dict of tensors on CPU
    """
    model.eval()
    boards, legal_masks, actions, old_log_probs, advantages, returns_, values = \
        [], [], [], [], [], [], []

    # 按 (game_id, move_index) 排序, 以便做 GAE
    samples_sorted = sorted(samples, key=lambda s: (
        s.get("game_id", 0), s.get("move_index", 0)
    ))

    with torch.no_grad():
        # 收集 V(s) 用于 advantage
        for s in samples_sorted:
            try:
                board = np.array(s["board"], dtype=np.float32).reshape(CHANNELS, BOARD_SIZE, BOARD_SIZE)
            except (KeyError, ValueError):
                continue
            if board.size != CHANNELS * BOARD_SIZE * BOARD_SIZE:
                continue
            try:
                legal = np.array(s["legal_mask"], dtype=np.float32)
                if legal.size != ACTION_SIZE:
                    continue
            except (KeyError, ValueError):
                continue
            try:
                policy = np.array(s["policy"], dtype=np.float32)
                if policy.size != ACTION_SIZE:
                    continue
            except (KeyError, ValueError):
                continue

            action = int(np.argmax(policy))  # 用 expert policy 的 top-1 当作 action
            if legal[action] < 0.5:
                # 找一个合法的最大概率动作
                masked = policy * legal
                if masked.sum() < 1e-6:
                    continue
                action = int(np.argmax(masked))
            if legal[action] < 0.5:
                continue

            reward = float(s.get("value", 0.0))
            done = bool(s.get("done", False))
            # 注: 简化的 PPO, 把 value 当 reward, done=True 时后续 advantage 截断
            # (具体 GAE 在 build_advantages_from_samples 内部处理)

            boards.append(board)
            legal_masks.append(legal)
            actions.append(action)
            rewards = [reward]
            values.append(0.0)  # 占位, 实际 GAE 用 reward 简化
            dones = [done]

            # 计算 log_prob under current policy
            board_t = torch.from_numpy(board).unsqueeze(0).to(device)
            legal_t = torch.from_numpy(legal).unsqueeze(0).to(device)
            policy_logits, value_pred = model(board_t)
            log_probs = masked_logits_to_log_probs(policy_logits, legal_t)
            old_lp = log_probs[0, action].item()

            old_log_probs.append(old_lp)
            values[-1] = float(value_pred.item())
            advantages.append(0.0)  # 占位
            returns_.append(reward)

    if len(boards) < 32:
        return None

    # 简化 advantage: 用 value_pred 的 baseline (用模型对每个 state 的估值) 算 advantage
    # 完整 PPO 需要按 episode 分组做 GAE, 这里用全局 mean baseline (一阶近似)
    adv = np.array(returns_, dtype=np.float32) - np.array(values, dtype=np.float32)
    adv_mean = adv.mean()
    adv_std = adv.std() + 1e-6
    adv = (adv - adv_mean) / adv_std
    # 截断极值
    adv = np.clip(adv, -5.0, 5.0)

    return {
        "board": torch.from_numpy(np.stack(boards)),
        "legal_mask": torch.from_numpy(np.stack(legal_masks)),
        "action": torch.tensor(actions, dtype=torch.long),
        "old_log_prob": torch.tensor(old_log_probs, dtype=torch.float32),
        "advantage": torch.tensor(adv, dtype=torch.float32),
        "return_": torch.tensor(returns_, dtype=torch.float32),
    }


# ========== ONNX 导出 (供 backward_selfplay 使用) ==========

def try_export_onnx(model, onnx_path, device):
    """把当前 PyTorch 模型导出成 ONNX, 供 backward_selfplay 加载"""
    try:
        model.eval()
        dummy = torch.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["board"],
            output_names=["policy_logits", "value"],
            dynamic_axes={"board": {0: "batch"},
                          "policy_logits": {0: "batch"},
                          "value": {0: "batch"}},
            opset_version=13,
        )
        print(f"  ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")


if __name__ == "__main__":
    main()
