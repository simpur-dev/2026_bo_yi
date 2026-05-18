"""
BoxesZero 反向训练编排脚本

实现论文核心算法:
  阶段1 (Value Network Reinforcement):
    从终局开始, 逐步向前推进 start_step
    每步: 自对弈(MCTS从st开始) → 训练(value_mode=q) → 检查晋级条件 → 前进β步
    
  阶段2 (Policy Network Reinforcement):
    start_step=0, 全程MCTS自对弈
    value_mode=q+z (混合Q值和最终胜负)

用法:
  python backward_training.py --arch resnet_s --games_per_iter 50 --simulations 400

需要先编译 backward_selfplay:
  cmake -B build_onnx -DUSE_ONNX=ON -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
  cmake --build build_onnx --target backward_selfplay
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil


def run_backward_selfplay(exe_path, games, sims, start_step, output_dir, model_path=""):
    """执行反向自对弈"""
    cmd = [exe_path, str(games), str(sims), str(start_step), output_dir]
    if model_path:
        cmd.append(model_path)
    
    print(f"\n[Selfplay] st={start_step}, games={games}, sims={sims}")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_training(data_dir, model_dir, arch, epochs, run_name, value_mode, q_weight,
                 resume=None, augment=True, lr=6e-4):
    """执行一轮训练"""
    cmd = [
        sys.executable, "train.py",
        "--data_dir", data_dir,
        "--model_dir", model_dir,
        "--arch", arch,
        "--epochs", str(epochs),
        "--batch_size", "256",
        "--lr", str(lr),
        "--run_name", run_name,
        "--value_mode", value_mode,
        "--q_weight", str(q_weight),
        "--split_mode", "sample",
        "--val_split", "0.1",
    ]
    if resume:
        cmd.extend(["--resume", resume])
    if augment:
        cmd.append("--augment")
    
    print(f"\n[Train] arch={arch}, epochs={epochs}, value_mode={value_mode}")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def export_onnx(model_path, output_path, arch):
    """导出 ONNX 模型"""
    cmd = [
        sys.executable, "export_onnx.py",
        "--model_path", model_path,
        "--output", output_path,
        "--arch", arch,
    ]
    print(f"\n[Export] {model_path} -> {output_path}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Export failed: {result.stderr}")
        return False
    return True


def count_samples(data_dir):
    """统计目录中的样本数"""
    count = 0
    if not os.path.exists(data_dir):
        return 0
    for f in os.listdir(data_dir):
        if f.endswith(".jsonl"):
            with open(os.path.join(data_dir, f), "r") as fh:
                count += sum(1 for _ in fh)
    return count


def analyze_q_values(data_dir):
    """分析自对弈数据的Q值分布，用于晋级条件判断"""
    q_values = []
    if not os.path.exists(data_dir):
        return 0.0, 0.0
    
    for f in os.listdir(data_dir):
        if f.endswith(".jsonl"):
            with open(os.path.join(data_dir, f), "r") as fh:
                for line in fh:
                    sample = json.loads(line.strip())
                    if "root_q" in sample:
                        q_values.append(abs(sample["root_q"]))
    
    if not q_values:
        return 0.0, 0.0
    
    # mqr: proportion where |Q| >= threshold
    mqth = 0.5
    mqr = sum(1 for q in q_values if q >= mqth) / len(q_values)
    avg_abs_q = sum(q_values) / len(q_values)
    
    return mqr, avg_abs_q


def main():
    parser = argparse.ArgumentParser(description="BoxesZero Backward Training")
    parser.add_argument("--arch", type=str, default="resnet_s",
                        choices=["resnet_s", "resnet_m", "resnet_l"],
                        help="CNN 架构")
    parser.add_argument("--games_per_iter", type=int, default=50,
                        help="每次迭代的自对弈局数")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS 模拟次数")
    parser.add_argument("--epochs_per_iter", type=int, default=30,
                        help="每次迭代的训练轮数")
    parser.add_argument("--max_iterations", type=int, default=100,
                        help="最大迭代次数")
    parser.add_argument("--start_st", type=int, default=50,
                        help="初始 start_step (从第几步开始MCTS)")
    parser.add_argument("--beta", type=int, default=5,
                        help="每次晋级前进的步数")
    parser.add_argument("--mit", type=int, default=5,
                        help="最大迭代次数阈值 (同一st停留几轮后强制晋级)")
    parser.add_argument("--mqr_threshold", type=float, default=0.6,
                        help="Q值晋级阈值")
    parser.add_argument("--q_weight", type=float, default=0.25,
                        help="阶段2 value target 中 Q 权重")
    parser.add_argument("--selfplay_exe", type=str, 
                        default="../build_onnx/backward_selfplay.exe",
                        help="backward_selfplay 可执行文件路径")
    parser.add_argument("--output_base", type=str, default="../data/backward",
                        help="数据输出基目录")
    parser.add_argument("--model_dir", type=str, default="../data/models",
                        help="模型保存目录")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="学习率")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoxesZero Backward Training")
    print("=" * 60)
    print(f"  Architecture:    {args.arch}")
    print(f"  Games/iter:      {args.games_per_iter}")
    print(f"  Simulations:     {args.simulations}")
    print(f"  Epochs/iter:     {args.epochs_per_iter}")
    print(f"  Start step:      {args.start_st}")
    print(f"  Beta:            {args.beta}")
    print(f"  Max iter (mit):  {args.mit}")
    print(f"  MQR threshold:   {args.mqr_threshold}")
    print()

    os.makedirs(args.output_base, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 状态跟踪
    current_st = args.start_st
    current_model = ""  # 空 = 使用 heuristic evaluator
    iterations_at_st = 0
    total_iterations = 0
    stage = 1  # 1=value reinforcement, 2=policy reinforcement

    # 状态文件
    state_file = os.path.join(args.output_base, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
            current_st = state.get("current_st", args.start_st)
            current_model = state.get("current_model", "")
            iterations_at_st = state.get("iterations_at_st", 0)
            total_iterations = state.get("total_iterations", 0)
            stage = state.get("stage", 1)
            print(f"  Resumed from: st={current_st}, iter={total_iterations}, stage={stage}")
            print()

    for iteration in range(args.max_iterations):
        total_iterations += 1
        iterations_at_st += 1

        print(f"\n{'='*60}")
        print(f"  Iteration {total_iterations} | Stage {stage} | st={current_st} | iter_at_st={iterations_at_st}")
        print(f"{'='*60}")

        # --- 1. 自对弈 ---
        iter_data_dir = os.path.join(args.output_base, f"iter_{total_iterations:04d}_st{current_st}")
        ok = run_backward_selfplay(
            args.selfplay_exe,
            args.games_per_iter,
            args.simulations,
            current_st,
            iter_data_dir,
            current_model,
        )
        if not ok:
            print("[ERROR] Selfplay failed!")
            break

        samples = count_samples(iter_data_dir)
        print(f"  Generated {samples} samples")

        # --- 2. 训练 ---
        # 合并本轮和之前数据
        combined_dir = os.path.join(args.output_base, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        # 复制本轮数据到 combined
        for f in os.listdir(iter_data_dir):
            if f.endswith(".jsonl"):
                dst_name = f"iter_{total_iterations:04d}__{f}"
                shutil.copy2(os.path.join(iter_data_dir, f), os.path.join(combined_dir, dst_name))

        run_name = f"backward_st{current_st}_iter{total_iterations}"
        value_mode = "q" if stage == 1 else "q+z"
        resume_path = current_model.replace(".onnx", ".pt") if current_model.endswith(".onnx") else current_model

        # 查找最近的 best_model.pt
        best_pt = os.path.join(args.model_dir, f"backward_best_model.pt")
        if os.path.exists(best_pt):
            resume_path = best_pt

        ok = run_training(
            combined_dir, args.model_dir, args.arch,
            args.epochs_per_iter, run_name, value_mode, args.q_weight,
            resume=resume_path if os.path.exists(resume_path) else None,
            augment=True,
            lr=args.lr,
        )
        if not ok:
            print("[ERROR] Training failed!")
            break

        # --- 3. 导出 ONNX ---
        pt_path = os.path.join(args.model_dir, f"{run_name}_best_model.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(args.model_dir, f"{run_name}_epoch{args.epochs_per_iter}_model.pt")
        
        onnx_path = os.path.join(args.model_dir, "backward_model.onnx")
        if os.path.exists(pt_path):
            # 同时保存为 backward_best_model.pt
            shutil.copy2(pt_path, best_pt)
            export_onnx(pt_path, onnx_path, args.arch)
            current_model = onnx_path
        else:
            print(f"  Warning: no model found at {pt_path}")

        # --- 4. 晋级条件判断 ---
        if stage == 1:
            mqr, avg_q = analyze_q_values(iter_data_dir)
            print(f"\n  [Advancement] mqr={mqr:.3f} (th={args.mqr_threshold}), avg|Q|={avg_q:.3f}, iter_at_st={iterations_at_st}/{args.mit}")

            should_advance = (mqr >= args.mqr_threshold) or (iterations_at_st >= args.mit)
            
            if should_advance:
                old_st = current_st
                current_st = max(0, current_st - args.beta)
                iterations_at_st = 0
                print(f"  >>> ADVANCE: st {old_st} -> {current_st}")

                if current_st == 0:
                    stage = 2
                    print(f"  >>> ENTERING STAGE 2 (Policy Network Reinforcement)")
        else:
            # Stage 2: 持续训练, 不再晋级
            print(f"  [Stage 2] Full self-play training")

        # --- 5. 保存状态 ---
        state = {
            "current_st": current_st,
            "current_model": current_model,
            "iterations_at_st": iterations_at_st,
            "total_iterations": total_iterations,
            "stage": stage,
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        print(f"\n  State saved: {state_file}")

    print("\n" + "=" * 60)
    print("  Backward Training Complete!")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Final st: {current_st}")
    print(f"  Final model: {current_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
