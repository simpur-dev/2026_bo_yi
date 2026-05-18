"""
AlphaZero Trainer v2

特性:
  - AdamW 优化器 (默认)
  - 完整 checkpoint (model + optimizer + scheduler + epoch + args + metrics)
  - KL 散度监控 (每 eval_interval 计算新旧策略的 KL)
  - Top-k 策略准确率 (top-1, top-3, top-5)
  - 训练报告 JSON (所有指标历史，可用于绘图)
  - 可复现性 (全局 seed)
  - 向下兼容所有 v1 参数

用法:
  python train.py --data_dir ../data/selfplay --epochs 100 --augment --val_split 0.1
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import create_model
from dataset import create_dataloaders


# ========== 工具函数 ==========

def safe_run_name(name):
    name = (name or "").strip()
    if not name:
        return ""
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def named_model_path(model_dir, run_name, filename):
    if run_name:
        return os.path.join(model_dir, f"{run_name}_{filename}")
    return os.path.join(model_dir, filename)


def set_seed(seed):
    """设置全局随机种子以确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== 损失函数 ==========

def masked_cross_entropy(policy_logits, target_policy, legal_mask, reduction="mean"):
    """
    计算带合法动作 mask 的交叉熵损失

    Args:
        policy_logits: (batch, 60) 网络输出的原始 logits
        target_policy: (batch, 60) 目标概率分布
        legal_mask: (batch, 60) 合法动作掩码
    """
    masked_logits = policy_logits - (1 - legal_mask) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    loss = -torch.sum(target_policy * log_probs, dim=1)
    if reduction == "none":
        return loss
    return loss.mean()


def sharpen_policy(target_policy, legal_mask, temperature):
    if temperature == 1.0:
        return target_policy

    if temperature <= 0.0:
        actions = target_policy.argmax(dim=1, keepdim=True)
        sharpened = torch.zeros_like(target_policy)
        sharpened.scatter_(1, actions, 1.0)
        return sharpened

    probs = target_policy * legal_mask
    powered = torch.pow(torch.clamp(probs, min=0.0), 1.0 / temperature) * legal_mask
    denom = powered.sum(dim=1, keepdim=True)
    fallback = legal_mask / legal_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(denom > 0.0, powered / denom.clamp_min(1e-12), fallback)


def policy_sample_weights(target_policy, legal_mask, mode):
    if mode == "none":
        return None

    probs = target_policy * legal_mask
    denom = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs = probs / denom

    if mode == "confidence":
        return probs.max(dim=1).values.clamp_min(1e-6)

    legal_count = legal_mask.sum(dim=1).clamp_min(2.0)
    entropy = -torch.sum(torch.where(probs > 0.0, probs * torch.log(probs.clamp_min(1e-12)), torch.zeros_like(probs)), dim=1)
    normalized_entropy = entropy / torch.log(legal_count)
    return (1.0 - normalized_entropy).clamp_min(1e-6)


# ========== KL 散度 ==========

def compute_kl_divergence(model, dataloader, device, old_policy_logits_list):
    """
    计算新策略相对旧策略的 KL 散度: KL(old || new)

    Args:
        model: 当前模型
        dataloader: 数据加载器
        old_policy_logits_list: 训练前保存的旧 logits 列表 (每个 batch 一个 tensor)

    Returns:
        平均 KL 散度
    """
    model.eval()
    total_kl = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (board, legal_mask, _, _) in enumerate(dataloader):
            if batch_idx >= len(old_policy_logits_list):
                break
            board = board.to(device)
            legal_mask = legal_mask.to(device)
            old_logits = old_policy_logits_list[batch_idx].to(device)

            new_logits, _ = model(board)

            # masked softmax
            old_masked = old_logits - (1 - legal_mask) * 1e9
            new_masked = new_logits - (1 - legal_mask) * 1e9

            old_probs = F.softmax(old_masked, dim=1)
            new_log_probs = F.log_softmax(new_masked, dim=1)
            old_log_probs = F.log_softmax(old_masked, dim=1)

            # KL(old || new) = sum(old * (log_old - log_new))
            kl = torch.sum(old_probs * (old_log_probs - new_log_probs), dim=1)
            total_kl += kl.sum().item()
            num_samples += board.size(0)

    return total_kl / max(num_samples, 1)


def snapshot_policy_logits(model, dataloader, device, max_batches=20):
    """保存当前模型在前 max_batches 个 batch 上的 policy logits"""
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch_idx, (board, _, _, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            board = board.to(device)
            policy_logits, _ = model(board)
            logits_list.append(policy_logits.cpu())
    return logits_list


# ========== 训练与评估 ==========

def train_epoch(model, dataloader, optimizer, device, policy_temperature, policy_weight_mode, grad_clip=0.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for board, legal_mask, target_policy, target_value in dataloader:
        board = board.to(device)
        legal_mask = legal_mask.to(device)
        target_policy = target_policy.to(device)
        target_value = target_value.to(device)
        policy_weights = policy_sample_weights(target_policy, legal_mask, policy_weight_mode)
        target_policy = sharpen_policy(target_policy, legal_mask, policy_temperature)

        optimizer.zero_grad()

        policy_logits, predicted_value = model(board)
        predicted_value = predicted_value.squeeze(-1)

        # 策略损失
        per_sample_policy_loss = masked_cross_entropy(policy_logits, target_policy, legal_mask, reduction="none")
        if policy_weights is None:
            policy_loss = per_sample_policy_loss.mean()
        else:
            policy_loss = torch.sum(per_sample_policy_loss * policy_weights) / torch.sum(policy_weights).clamp_min(1e-8)

        # 价值损失
        value_loss = F.mse_loss(predicted_value, target_value)

        # 总损失
        loss = policy_loss + value_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_policy = total_policy_loss / max(num_batches, 1)
    avg_value = total_value_loss / max(num_batches, 1)
    return avg_loss, avg_policy, avg_value


def evaluate(model, dataloader, device, policy_temperature):
    """
    评估模型，返回完整指标字典

    包含: loss, policy_loss, value_loss, policy_acc_top1/3/5, value_mae, value_sign_acc
    """
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    total_value_err = 0.0
    total_value_sign_acc = 0.0
    num_samples = 0
    num_batches = 0

    with torch.no_grad():
        for board, legal_mask, target_policy, target_value in dataloader:
            board = board.to(device)
            legal_mask = legal_mask.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            target_policy = sharpen_policy(target_policy, legal_mask, policy_temperature)

            policy_logits, predicted_value = model(board)
            predicted_value = predicted_value.squeeze(-1)

            policy_loss = masked_cross_entropy(policy_logits, target_policy, legal_mask)
            value_loss = F.mse_loss(predicted_value, target_value)
            loss = policy_loss + value_loss

            # Top-k 准确率
            masked_logits = policy_logits - (1 - legal_mask) * 1e9
            target_actions = target_policy.argmax(dim=1)

            # Top-1
            pred_top1 = masked_logits.argmax(dim=1)
            total_top1 += (pred_top1 == target_actions).float().sum().item()

            # Top-3
            _, pred_top3 = masked_logits.topk(3, dim=1)
            total_top3 += (pred_top3 == target_actions.unsqueeze(1)).any(dim=1).float().sum().item()

            # Top-5
            _, pred_top5 = masked_logits.topk(5, dim=1)
            total_top5 += (pred_top5 == target_actions.unsqueeze(1)).any(dim=1).float().sum().item()

            # 价值误差
            total_value_err += F.l1_loss(predicted_value, target_value, reduction='sum').item()
            pred_sign = torch.sign(predicted_value)
            target_sign = torch.sign(target_value)
            total_value_sign_acc += (pred_sign == target_sign).float().sum().item()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_samples += board.size(0)
            num_batches += 1

    n = max(num_samples, 1)
    b = max(num_batches, 1)
    return {
        "loss": total_loss / b,
        "policy_loss": total_policy_loss / b,
        "value_loss": total_value_loss / b,
        "policy_acc_top1": total_top1 / n,
        "policy_acc_top3": total_top3 / n,
        "policy_acc_top5": total_top5 / n,
        "value_mae": total_value_err / n,
        "value_sign_acc": total_value_sign_acc / n,
    }


# ========== Checkpoint ==========

def save_checkpoint(path, model, optimizer, scheduler, epoch, args, metrics_history, best_val_metric):
    """保存完整 checkpoint"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "metrics_history": metrics_history,
        "best_val_metric": best_val_metric,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """加载完整 checkpoint，返回 (epoch, args_dict, metrics_history, best_val_metric)"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("args", {}), ckpt.get("metrics_history", []), ckpt.get("best_val_metric", float("inf"))


# ========== 报告 ==========

def write_report(report_path, args, metrics_history, summary):
    """写出训练报告 JSON"""
    report = {
        "trainer_version": 2,
        "args": vars(args),
        "summary": summary,
        "metrics_history": metrics_history,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Trainer v2")
    parser.add_argument("--data_dir", type=str, default="../data/selfplay",
                        help="训练数据目录")
    parser.add_argument("--model_dir", type=str, default="../data/models",
                        help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--lr", type=float, default=6e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减 (AdamW)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                        help="优化器: adamw (推荐) 或 adam")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto/cpu/cuda")
    parser.add_argument("--arch", type=str, default="mlp",
                        choices=["mlp", "cnn", "resnet_s", "resnet_m", "resnet_l"],
                        help="网络架构: mlp / resnet_s / resnet_m / resnet_l (cnn=resnet_s)")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练: checkpoint 路径 (完整 checkpoint 或仅 state_dict)")
    parser.add_argument("--run_name", type=str, default="candidate",
                        help="本次训练输出前缀，默认保存为 candidate_*")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最多加载多少条样本，0 表示全部")
    parser.add_argument("--scheduler_step", type=int, default=30,
                        help="学习率衰减步长 (epochs)")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5,
                        help="学习率衰减因子")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="每隔多少 epoch 做完整评估")
    parser.add_argument("--policy_temperature", type=float, default=1.0,
                        help="训练目标 policy 温度")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--split_seed", type=int, default=2026,
                        help="训练/验证划分随机种子")
    parser.add_argument("--split_mode", type=str, default="file",
                        choices=["sample", "file"],
                        help="验证集划分方式")
    parser.add_argument("--min_policy_confidence", type=float, default=0.0,
                        help="只加载 max(policy) >= 该阈值的样本")
    parser.add_argument("--policy_weight_mode", type=str, default="none",
                        choices=["none", "confidence", "entropy"],
                        help="policy loss 样本权重")
    parser.add_argument("--augment", action="store_true",
                        help="启用 D4 对称增强 (8 倍)")
    parser.add_argument("--value_mode", type=str, default="margin",
                        choices=["margin", "wdl", "q", "q+z"],
                        help="价值目标: margin/wdl/q/q+z (反向训练用 q 或 q+z)")
    parser.add_argument("--q_weight", type=float, default=0.25,
                        help="Q 值权重 (value_mode=q+z 时: q_weight*Q + (1-q_weight)*z)")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪最大范数, 0 表示不裁剪")
    args = parser.parse_args()
    run_name = safe_run_name(args.run_name)

    # 可复现性
    set_seed(args.seed)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  AlphaZero Trainer v2")
    print("=" * 60)
    print(f"  Run name:       {run_name if run_name else '(official)'}")
    print(f"  Device:         {device}")
    print(f"  Architecture:   {args.arch}")
    print(f"  Optimizer:      {args.optimizer} (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Scheduler:      StepLR(step={args.scheduler_step}, gamma={args.scheduler_gamma})")
    print(f"  Augmentation:   {'ON' if args.augment else 'OFF'}")
    print(f"  Val split:      {args.val_split} ({args.split_mode})")
    print(f"  Policy temp:    {args.policy_temperature}")
    print(f"  Weight mode:    {args.policy_weight_mode}")
    print(f"  Grad clip:      {args.grad_clip}")
    print(f"  Seed:           {args.seed}")
    print("=" * 60)

    # 创建模型
    model = create_model(arch=args.arch, device=device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # 优化器
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # 恢复训练
    start_epoch = 0
    metrics_history = []
    best_val_metric = float("inf")

    if args.resume:
        if os.path.exists(args.resume):
            try:
                start_epoch, _, metrics_history, best_val_metric = load_checkpoint(
                    args.resume, model, optimizer, scheduler, device)
                print(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
            except KeyError:
                # 旧格式: 只有 state_dict
                model.load_state_dict(torch.load(args.resume, map_location=device))
                print(f"Resumed model weights from: {args.resume}")
        else:
            print(f"WARNING: resume path not found: {args.resume}")

    # 数据
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        val_split=args.val_split,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        min_policy_confidence=args.min_policy_confidence,
        augment=args.augment,
        value_mode=args.value_mode,
        q_weight=args.q_weight,
    )
    if train_loader is None:
        print("No training data found. Please generate data first.")
        return

    # 模型/报告保存目录
    os.makedirs(args.model_dir, exist_ok=True)

    # ========== 训练循环 ==========
    train_start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()

        # KL: 训练前快照 policy
        do_eval = (args.eval_interval > 0 and
                   (epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs))
        old_logits = None
        if do_eval and val_loader is not None:
            old_logits = snapshot_policy_logits(model, val_loader, device)

        # 训练
        avg_loss, avg_policy, avg_value = train_epoch(
            model, train_loader, optimizer, device,
            args.policy_temperature, args.policy_weight_mode, args.grad_clip)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = scheduler.get_last_lr()[0]

        # 构建指标记录
        record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "train_policy_loss": round(avg_policy, 5),
            "train_value_loss": round(avg_value, 5),
            "lr": lr,
            "epoch_time_s": round(epoch_time, 1),
        }

        # 完整评估
        if do_eval:
            train_metrics = evaluate(model, train_loader, device, args.policy_temperature)
            record["train_policy_acc_top1"] = round(train_metrics["policy_acc_top1"], 4)
            record["train_policy_acc_top3"] = round(train_metrics["policy_acc_top3"], 4)
            record["train_policy_acc_top5"] = round(train_metrics["policy_acc_top5"], 4)
            record["train_value_mae"] = round(train_metrics["value_mae"], 4)
            record["train_value_sign_acc"] = round(train_metrics["value_sign_acc"], 4)

            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device, args.policy_temperature)
                record["val_loss"] = round(val_metrics["loss"], 5)
                record["val_policy_loss"] = round(val_metrics["policy_loss"], 5)
                record["val_value_loss"] = round(val_metrics["value_loss"], 5)
                record["val_policy_acc_top1"] = round(val_metrics["policy_acc_top1"], 4)
                record["val_policy_acc_top3"] = round(val_metrics["policy_acc_top3"], 4)
                record["val_policy_acc_top5"] = round(val_metrics["policy_acc_top5"], 4)
                record["val_value_mae"] = round(val_metrics["value_mae"], 4)
                record["val_value_sign_acc"] = round(val_metrics["value_sign_acc"], 4)

                # KL 散度
                if old_logits is not None:
                    kl = compute_kl_divergence(model, val_loader, device, old_logits)
                    record["kl_divergence"] = round(kl, 6)

        metrics_history.append(record)

        # 输出
        msg = (f"Epoch {epoch:3d}/{args.epochs} | "
               f"Loss: {avg_loss:.4f} (P:{avg_policy:.4f} V:{avg_value:.4f}) | "
               f"LR: {lr:.6f} | {epoch_time:.1f}s")

        if do_eval:
            msg += f" | Top1: {record.get('train_policy_acc_top1', 0):.4f}"
            msg += f" Top3: {record.get('train_policy_acc_top3', 0):.4f}"
            msg += f" Top5: {record.get('train_policy_acc_top5', 0):.4f}"
            if val_loader is not None:
                msg += (f" | Val: {record.get('val_loss', 0):.4f}"
                        f" Top1: {record.get('val_policy_acc_top1', 0):.4f}"
                        f" Top3: {record.get('val_policy_acc_top3', 0):.4f}"
                        f" Top5: {record.get('val_policy_acc_top5', 0):.4f}"
                        f" MAE: {record.get('val_value_mae', 0):.4f}")
                if "kl_divergence" in record:
                    msg += f" KL: {record['kl_divergence']:.5f}"

        print(msg)

        # 保存最佳模型 (基于 val_loss, 回退到 train_loss)
        current_metric = record.get("val_loss", avg_loss)
        if current_metric < best_val_metric:
            best_val_metric = current_metric
            save_path = named_model_path(args.model_dir, run_name, "best_model.pt")
            torch.save(model.state_dict(), save_path)

        # 定期 checkpoint
        if epoch % args.eval_interval == 0:
            ckpt_path = named_model_path(args.model_dir, run_name, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, args, metrics_history, best_val_metric)

    # ========== 训练结束 ==========
    total_time = time.time() - train_start_time

    # 最终保存
    final_path = named_model_path(args.model_dir, run_name, "final_model.pt")
    torch.save(model.state_dict(), final_path)

    final_ckpt_path = named_model_path(args.model_dir, run_name, "final_checkpoint.pt")
    save_checkpoint(final_ckpt_path, model, optimizer, scheduler, args.epochs, args, metrics_history, best_val_metric)

    # 汇总
    summary = {
        "total_epochs": args.epochs,
        "total_time_s": round(total_time, 1),
        "best_val_metric": round(best_val_metric, 5),
        "final_train_loss": round(avg_loss, 5),
        "param_count": param_count,
    }

    # 提取最佳 val 指标
    val_records = [r for r in metrics_history if "val_policy_acc_top1" in r]
    if val_records:
        best_top1_record = max(val_records, key=lambda r: r["val_policy_acc_top1"])
        summary["best_val_policy_acc_top1"] = best_top1_record["val_policy_acc_top1"]
        summary["best_val_policy_acc_top1_epoch"] = best_top1_record["epoch"]
        best_top3_record = max(val_records, key=lambda r: r["val_policy_acc_top3"])
        summary["best_val_policy_acc_top3"] = best_top3_record["val_policy_acc_top3"]
        best_top5_record = max(val_records, key=lambda r: r["val_policy_acc_top5"])
        summary["best_val_policy_acc_top5"] = best_top5_record["val_policy_acc_top5"]

    # 写报告
    report_path = named_model_path(args.model_dir, run_name, "report.json")
    write_report(report_path, args, metrics_history, summary)

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Best val metric:{best_val_metric:.5f}")
    if val_records:
        print(f"  Best ValTop1:   {summary['best_val_policy_acc_top1']:.4f} (epoch {summary['best_val_policy_acc_top1_epoch']})")
        print(f"  Best ValTop3:   {summary['best_val_policy_acc_top3']:.4f}")
        print(f"  Best ValTop5:   {summary['best_val_policy_acc_top5']:.4f}")
    print(f"  Report:         {report_path}")
    print(f"  Final model:    {final_path}")
    print(f"  Checkpoint:     {final_ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
