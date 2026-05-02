"""
AlphaZero 训练入口

损失函数:
  loss = value_loss + policy_loss + l2_regularization
  value_loss = MSE(predicted_value, target_value)
  policy_loss = -sum(target_policy * log(predicted_policy))

用法:
  python train.py --data_dir ../data/selfplay --epochs 100 --batch_size 256
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import create_model
from dataset import create_dataloaders


def safe_run_name(name):
    name = (name or "").strip()
    if not name:
        return ""
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def named_model_path(model_dir, run_name, filename):
    if run_name:
        return os.path.join(model_dir, f"{run_name}_{filename}")
    return os.path.join(model_dir, filename)


def masked_cross_entropy(policy_logits, target_policy, legal_mask):
    """
    计算带合法动作 mask 的交叉熵损失

    Args:
        policy_logits: (batch, 60) 网络输出的原始 logits
        target_policy: (batch, 60) 目标概率分布
        legal_mask: (batch, 60) 合法动作掩码
    """
    # 对非法动作施加大负数
    masked_logits = policy_logits - (1 - legal_mask) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    loss = -torch.sum(target_policy * log_probs, dim=1)
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


def train_epoch(model, dataloader, optimizer, device, policy_temperature):
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
        target_policy = sharpen_policy(target_policy, legal_mask, policy_temperature)

        optimizer.zero_grad()

        policy_logits, predicted_value = model(board)
        predicted_value = predicted_value.squeeze(-1)

        # 策略损失
        policy_loss = masked_cross_entropy(policy_logits, target_policy, legal_mask)

        # 价值损失
        value_loss = F.mse_loss(predicted_value, target_value)

        # 总损失
        loss = policy_loss + value_loss

        loss.backward()
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
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_acc = 0.0
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

            # 策略准确率：预测的最大概率动作是否与目标一致
            masked_logits = policy_logits - (1 - legal_mask) * 1e9
            pred_actions = masked_logits.argmax(dim=1)
            target_actions = target_policy.argmax(dim=1)
            total_policy_acc += (pred_actions == target_actions).float().sum().item()

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

    avg_loss = total_loss / max(num_batches, 1)
    avg_policy_loss = total_policy_loss / max(num_batches, 1)
    avg_value_loss = total_value_loss / max(num_batches, 1)
    policy_acc = total_policy_acc / max(num_samples, 1)
    value_err = total_value_err / max(num_samples, 1)
    value_sign_acc = total_value_sign_acc / max(num_samples, 1)
    return avg_loss, avg_policy_loss, avg_value_loss, policy_acc, value_err, value_sign_acc


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument("--data_dir", type=str, default="../data/selfplay",
                        help="训练数据目录")
    parser.add_argument("--model_dir", type=str, default="../data/models",
                        help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 正则化")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto/cpu/cuda")
    parser.add_argument("--arch", type=str, default="mlp", choices=["cnn", "mlp"],
                        help="网络架构: cnn (ResNet) 或 mlp (轻量MLP)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的模型路径")
    parser.add_argument("--run_name", type=str, default="candidate", help="本次训练输出前缀，默认保存为 candidate_*")
    parser.add_argument("--max_samples", type=int, default=0, help="最多加载多少条样本，0 表示全部")
    parser.add_argument("--scheduler_step", type=int, default=30, help="学习率衰减步长")
    parser.add_argument("--eval_interval", type=int, default=10, help="每隔多少 epoch 输出一次训练集指标")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="训练目标 policy 温度，低于 1 会锐化分布，0 表示 one-hot")
    parser.add_argument("--val_split", type=float, default=0.0, help="验证集比例，0 表示不划分验证集")
    parser.add_argument("--split_seed", type=int, default=2026, help="训练/验证划分随机种子")
    parser.add_argument("--split_mode", type=str, default="sample", choices=["sample", "file"], help="验证集划分方式: sample 随机样本划分，file 按 JSONL 文件划分")
    parser.add_argument("--min_policy_confidence", type=float, default=0.0, help="只加载 max(policy) 大于等于该阈值的样本，0 表示不过滤")
    args = parser.parse_args()
    run_name = safe_run_name(args.run_name)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Run name: {run_name if run_name else '(official names)'}")
    print(f"Policy temperature: {args.policy_temperature}")
    print(f"Validation split: {args.val_split}")
    print(f"Split mode: {args.split_mode}")
    print(f"Min policy confidence: {args.min_policy_confidence}")

    # 创建模型
    model = create_model(arch=args.arch, device=device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.5)

    # 数据
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        val_split=args.val_split,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        min_policy_confidence=args.min_policy_confidence,
    )
    if train_loader is None:
        print("No training data found. Please generate data first.")
        return

    # 模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 训练循环
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_policy, avg_value = train_epoch(model, train_loader, optimizer, device, args.policy_temperature)
        scheduler.step()

        message = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"TrainLoss: {avg_loss:.4f} | "
                   f"TrainPolicy: {avg_policy:.4f} | "
                   f"TrainValue: {avg_value:.4f} | "
                   f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if args.eval_interval > 0 and (epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs):
            _, _, _, policy_acc, value_mae, value_sign_acc = evaluate(model, train_loader, device, args.policy_temperature)
            message += (f" | TrainPolicyAcc: {policy_acc:.4f} | "
                        f"TrainValueMAE: {value_mae:.4f} | "
                        f"TrainValueSignAcc: {value_sign_acc:.4f}")
            if val_loader is not None:
                val_loss, val_policy_loss, val_value_loss, val_policy_acc, val_value_mae, val_value_sign_acc = evaluate(
                    model, val_loader, device, args.policy_temperature
                )
                message += (f" | ValLoss: {val_loss:.4f} | "
                            f"ValPolicy: {val_policy_loss:.4f} | "
                            f"ValValue: {val_value_loss:.4f} | "
                            f"ValPolicyAcc: {val_policy_acc:.4f} | "
                            f"ValValueMAE: {val_value_mae:.4f} | "
                            f"ValValueSignAcc: {val_value_sign_acc:.4f}")

        print(message)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = named_model_path(args.model_dir, run_name, "best_model.pt")
            torch.save(model.state_dict(), save_path)

        # 定期保存检查点
        if epoch % 10 == 0:
            save_path = named_model_path(args.model_dir, run_name, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved checkpoint: {save_path}")

    # 最终保存
    final_path = named_model_path(args.model_dir, run_name, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
