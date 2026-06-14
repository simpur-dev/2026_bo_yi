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
import random
import subprocess
import sys
import time
import shutil


def resolve_path(path, script_dir, default_value=None):
    """解析路径: 默认参数按脚本目录解析；用户传入的相对路径优先按当前工作目录解析"""
    if not path or os.path.isabs(path):
        return path
    if default_value is not None and path == default_value:
        return os.path.abspath(os.path.join(script_dir, path))
    cwd_path = os.path.abspath(path)
    script_path = os.path.abspath(os.path.join(script_dir, path))
    if os.path.exists(cwd_path):
        return cwd_path
    if os.path.exists(script_path):
        return script_path
    return cwd_path


def run_backward_selfplay(exe_path, games, sims, start_step, output_dir, model_path="", black_model="", white_model="", black_sims_mult=1.0, eval_black=False):
    """执行反向自对弈。支持单/双模型 + 单侧模型 (league) + 不对称 sims。

    模型配置规则 (C++ 侧一致性):
      black_model + white_model → 双模型各自独立
      black_model only           → 黑=模型, 白=启发式 (league: 模型执黑 vs 弱对手)
      white_model only           → 黑=启发式, 白=模型 (league: 模型执白 vs 弱对手)
      model_path only            → 单模型双方共用 (标准 selfplay)
      无模型                     → 纯启发式双方
    """
    cmd = [exe_path, str(games), str(sims), str(start_step), output_dir]
    if black_model:
        cmd.extend(["--black-model", black_model])
    if white_model:
        cmd.extend(["--white-model", white_model])
    if not black_model and not white_model and model_path:
        cmd.append(model_path)
    if black_sims_mult != 1.0:
        cmd.extend(["--black-sims-mult", str(black_sims_mult)])
    if eval_black:
        cmd.append("--eval-black")

    # 描述性日志
    parts = [f"st={start_step}", f"games={games}", f"sims={sims}"]
    if black_sims_mult != 1.0:
        parts.append(f"black {black_sims_mult:.1f}x")
    print(f"\n[Selfplay] {', '.join(parts)}")

    if black_model and white_model:
        print(f"  Dual: BLACK={black_model}, WHITE={white_model}")
    elif black_model:
        print(f"  League: BLACK={black_model}, WHITE=heuristic")
    elif white_model:
        print(f"  League: BLACK=heuristic, WHITE={white_model}")
    else:
        print(f"  Model: {model_path or 'heuristic'}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False,
                            timeout=max(60, games * sims / 50 + 120))
    return result.returncode == 0


def detect_free_gpus(min_free_mb=10000, max_util=10, max_gpus=4):
    """动态检测空闲 GPU: 利用率 < max_util% 且显存 > min_free_mb MB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip().replace(" %", "").replace(" MiB", "") for p in line.split(",")]
            if len(parts) >= 3:
                idx, util, mem = int(parts[0]), int(parts[1]), int(parts[2])
                if util <= max_util and mem >= min_free_mb:
                    gpus.append(idx)
        return gpus[:max_gpus]
    except Exception:
        return []


def run_backward_selfplay_multi_gpu(exe_path, total_games, sims, start_step, output_dir,
                                     gpus, model_path="", black_model="", white_model="",
                                     black_sims_mult=1.0, eval_black=False):
    """多 GPU 并行 selfplay: 每个 GPU 跑一部分对局, 数据合并到 output_dir"""
    games_per_gpu = total_games // len(gpus)
    remainder = total_games % len(gpus)
    procs = []
    timeout_s = max(60, (games_per_gpu + 1) * sims / 50 + 120)

    for i, gpu in enumerate(gpus):
        gpu_games = games_per_gpu + (1 if i < remainder else 0)
        if gpu_games <= 0:
            continue
        sub_dir = os.path.join(output_dir, f"_gpu{gpu}")
        os.makedirs(sub_dir, exist_ok=True)

        cmd = [exe_path, str(gpu_games), str(sims), str(start_step), sub_dir]
        if black_model:
            cmd.extend(["--black-model", black_model])
        if white_model:
            cmd.extend(["--white-model", white_model])
        if not black_model and not white_model and model_path:
            cmd.append(model_path)
        if black_sims_mult != 1.0:
            cmd.extend(["--black-sims-mult", str(black_sims_mult)])
        if eval_black:
            cmd.append("--eval-black")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"  [GPU {gpu}] {gpu_games} games")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT)
        procs.append((p, sub_dir, gpu_games, gpu))

    # 等待所有进程完成
    ok = True
    for p, sub_dir, n_games, gpu in procs:
        try:
            p.wait(timeout=timeout_s)
            if p.returncode != 0:
                print(f"  [GPU {gpu}] FAILED (exit {p.returncode})")
                ok = False
        except subprocess.TimeoutExpired:
            print(f"  [GPU {gpu}] TIMEOUT ({timeout_s}s) — killing")
            p.kill()
            p.wait()
            ok = False

    # 合并数据: 各 GPU 子目录的 jsonl 移到 output_dir
    if ok:
        for _, sub_dir, _, gpu in procs:
            for f in os.listdir(sub_dir):
                if f.endswith(".jsonl"):
                    src = os.path.join(sub_dir, f)
                    dst = os.path.join(output_dir, f"G{gpu}_{f}")
                    shutil.move(src, dst)
            try:
                os.rmdir(sub_dir)
            except OSError:
                pass

    return ok


def run_training(data_dir, model_dir, arch, epochs, run_name, value_mode, q_weight,
                 resume=None, augment=True, lr=6e-4, split_mode="file",
                 sample_weight_mode="none", sample_weight_clip=5.0,
                 player_channel_mode="original"):
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
        "--split_mode", split_mode,
        "--val_split", "0.1",
        "--sample_weight_mode", sample_weight_mode,
        "--sample_weight_clip", str(sample_weight_clip),
        "--player_channel_mode", player_channel_mode,
    ]
    if resume:
        cmd.extend(["--resume", resume])
    if augment:
        cmd.append("--augment")
    
    print(f"\n[Train] arch={arch}, epochs={epochs}, value_mode={value_mode}")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                            timeout=max(300, epochs * 30))
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
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  Export TIMEOUT after 120s")
        return False
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


def run_arena(evaluate_exe, candidate_model, current_model, games_per_side, simulations, temp_moves, promote_threshold, min_black_winrate, min_white_winrate, side_gate_mode, side_tolerance, min_score_margin=0.5, fixed_color=False):
    """运行候选模型 vs 当前模型的 arena，返回多项指标 + 是否晋级。

    返回 tuple: (promoted, winrate, black_wr, white_wr, baseline_black_wr, baseline_white_wr, score_margin)
      - winrate           候选 (A) 总胜率
      - black_wr/white_wr  A 执黑/执白 胜率 (仅供诊断，swap-color 下不应作为门槛)
      - baseline_*         B 执黑/执白 胜率 (同上)
      - score_margin       A 平均得分 - B 平均得分。颜色对称连续量，主要晋级门槛。
    当 fixed_color=True 时: candidate 永远执黑, current 永远执白, 单边测试。"""
    if games_per_side <= 0:
        return True, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    if not current_model or not os.path.exists(current_model):
        return True, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    if not os.path.exists(evaluate_exe):
        print(f"  [Arena] evaluate_ai not found: {evaluate_exe}")
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cmd = [
        evaluate_exe,
        str(games_per_side),
        candidate_model,
        current_model,
        "--sims", str(simulations),
        "--temp", str(temp_moves),
    ]
    if fixed_color:
        cmd.append("--fixed-color")
    timeout_s = max(600, games_per_side * 2 * simulations // 25 + 600)
    print(f"\n[Arena] candidate vs current, games={games_per_side}, threshold={promote_threshold:.2f}"
          + (" [fixed-color]" if fixed_color else ""))
    print(f"  Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  [Arena] TIMEOUT after {timeout_s}s — killing")
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    winrate = None
    a_as_black_wins = None
    a_as_white_wins = None
    b_as_black_wins = None
    b_as_white_wins = None
    score_margin = 0.0
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if "A winrate:" in line:
            try:
                winrate = float(line.split("A winrate:")[1].split("%")[0].strip()) / 100.0
            except (IndexError, ValueError):
                pass
        elif "A as BLACK wins:" in line:
            try:
                a_as_black_wins = int(line.split("A as BLACK wins:")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "A as WHITE wins:" in line:
            try:
                a_as_white_wins = int(line.split("A as WHITE wins:")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "B as BLACK wins:" in line:
            try:
                b_as_black_wins = int(line.split("B as BLACK wins:")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "B as WHITE wins:" in line:
            try:
                b_as_white_wins = int(line.split("B as WHITE wins:")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "A-B margin:" in line:
            try:
                score_margin = float(line.split("A-B margin:")[1].strip())
            except (IndexError, ValueError):
                pass
    if winrate is None:
        print(f"  [Arena] PARSE FAILED — raw output ({len(result.stdout)} chars stdout, {len(result.stderr)} chars stderr):")
        print(f"  stdout tail: {result.stdout[-500:] if result.stdout else '(empty)'}")
        print(f"  stderr tail: {result.stderr[-500:] if result.stderr else '(empty)'}")
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    black_winrate = (a_as_black_wins / games_per_side) if a_as_black_wins is not None else 0.0
    white_winrate = (a_as_white_wins / games_per_side) if a_as_white_wins is not None else 0.0
    baseline_black_winrate = (b_as_black_wins / games_per_side) if b_as_black_wins is not None else 0.0
    baseline_white_winrate = (b_as_white_wins / games_per_side) if b_as_white_wins is not None else 0.0

    if side_gate_mode == "margin":
        # 推荐默认: 颜色对称连续量, 受 5x5 结构性白优影响远小于单边胜率
        # 优点: 记忆 c5900f29 指出在 N=20/side 下黑方胜率方差 ±15pp，
        #       不可靠。score-margin 作为连续量方差下降级为 O(1/sqrt(N)) box
        side_ok = score_margin >= min_score_margin
        print(f"  [Arena Gate] margin: A-B={score_margin:.3f} >= {min_score_margin:.3f}? {side_ok}")
    elif side_gate_mode == "absolute":
        side_ok = black_winrate >= min_black_winrate and white_winrate >= min_white_winrate
        print(f"  [Arena Gate] absolute: black={black_winrate:.3f}/{min_black_winrate:.3f}, white={white_winrate:.3f}/{min_white_winrate:.3f}")
    else:  # "relative" (legacy)
        side_ok = (
            black_winrate + side_tolerance >= baseline_black_winrate and
            white_winrate + side_tolerance >= baseline_white_winrate
        )
        print(
            f"  [Arena Gate] relative: "
            f"A_black={black_winrate:.3f} vs B_black={baseline_black_winrate:.3f}, "
            f"A_white={white_winrate:.3f} vs B_white={baseline_white_winrate:.3f}, "
            f"tol={side_tolerance:.3f}"
        )

    promoted = winrate >= promote_threshold and side_ok
    return promoted, winrate, black_winrate, white_winrate, baseline_black_winrate, baseline_white_winrate, score_margin


def collect_iter_dirs(output_base, max_iteration=None):
    """收集包含样本的 iter_* 目录，按迭代编号排序"""
    result = []
    for d in os.listdir(output_base):
        full = os.path.join(output_base, d)
        if not d.startswith("iter_") or not os.path.isdir(full):
            continue
        try:
            num = int(d.split("_")[1])
        except (IndexError, ValueError):
            continue
        if max_iteration is not None and num > max_iteration:
            continue
        if count_samples(full) > 0:
            result.append((num, d))
    return [d for _, d in sorted(result)]


def rebuild_training_window(output_base, combined_dir, window_size, anchor_size, max_iteration=None):
    """构建训练集: 前 anchor_size 轮锚点数据 + 最近 window_size 轮 replay 数据"""
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    os.makedirs(combined_dir, exist_ok=True)

    all_iter_dirs = collect_iter_dirs(output_base, max_iteration=max_iteration)
    anchor_dirs = all_iter_dirs[:anchor_size] if anchor_size > 0 else []
    recent_dirs = all_iter_dirs[-window_size:] if window_size > 0 else all_iter_dirs

    selected_dirs = []
    seen = set()
    for d in anchor_dirs + recent_dirs:
        if d not in seen:
            selected_dirs.append(d)
            seen.add(d)

    for iter_dir_name in selected_dirs:
        iter_dir_path = os.path.join(output_base, iter_dir_name)
        for f in os.listdir(iter_dir_path):
            if f.endswith(".jsonl"):
                dst_name = f"{iter_dir_name}__{f}"
                shutil.copy2(os.path.join(iter_dir_path, f), os.path.join(combined_dir, dst_name))

    return selected_dirs, count_samples(combined_dir)


def rebuild_training_window_dual(output_base, black_dir, white_dir, window_size, anchor_size, max_iteration=None):
    """双模型训练集: 按 player 字段拆分为黑/白两个 dataset"""
    import json as _json

    for d in [black_dir, white_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    all_iter_dirs = collect_iter_dirs(output_base, max_iteration=max_iteration)
    anchor_dirs = all_iter_dirs[:anchor_size] if anchor_size > 0 else []
    recent_dirs = all_iter_dirs[-window_size:] if window_size > 0 else all_iter_dirs

    selected_dirs = []
    seen = set()
    for d in anchor_dirs + recent_dirs:
        if d not in seen:
            selected_dirs.append(d)
            seen.add(d)

    samples_black = 0
    samples_white = 0

    for iter_dir_name in selected_dirs:
        iter_dir_path = os.path.join(output_base, iter_dir_name)
        for f in os.listdir(iter_dir_path):
            if not f.endswith(".jsonl"):
                continue
            src = os.path.join(iter_dir_path, f)
            dst_black = os.path.join(black_dir, f"{iter_dir_name}__{f}")
            dst_white = os.path.join(white_dir, f"{iter_dir_name}__{f}")

            # 按 player 拆分: player=BLACK(1) → black dataset, player=WHITE(-1) → white dataset
            black_lines = []
            white_lines = []
            with open(src, "r") as fh:
                for line in fh:
                    sample = _json.loads(line.strip())
                    if sample.get("player") == 1:  # BLACK
                        black_lines.append(line)
                    else:  # WHITE or other
                        white_lines.append(line)

            if black_lines:
                with open(dst_black, "w") as fh:
                    fh.writelines(black_lines)
                samples_black += len(black_lines)
            if white_lines:
                with open(dst_white, "w") as fh:
                    fh.writelines(white_lines)
                samples_white += len(white_lines)

    return samples_black, samples_white


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
                        help="本次启动最多新增迭代次数")
    parser.add_argument("--target_total_iterations", type=int, default=0,
                        help="目标总迭代数: total_iterations 达到该值即停止 (0=关闭)")
    parser.add_argument("--start_st", type=int, default=50,
                        help="初始 start_step (从第几步开始MCTS)")
    parser.add_argument("--beta", type=int, default=5,
                        help="每次晋级前进的步数")
    parser.add_argument("--mit", type=int, default=5,
                        help="最大迭代次数阈值 (同一st停留几轮后强制晋级)")
    parser.add_argument("--mqr_threshold", type=float, default=0.6,
                        help="Q值晋级阈值")
    parser.add_argument("--q_weight", type=float, default=1.0,
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
    parser.add_argument("--window_size", type=int, default=8,
                        help="滑动窗口大小: 只用最近 N 轮数据训练 (0=全部)")
    parser.add_argument("--anchor_size", type=int, default=15,
                        help="锚点数据轮数: 始终保留最早 N 轮 curriculum 数据 (0=关闭)")
    parser.add_argument("--stage2_value_mode", type=str, default="q",
                        choices=["q+z", "q+margin", "margin", "wdl", "q"],
                        help="阶段2 value target。默认 q (纯 MCTS root Q)；避免 z 导致 value head 颜色偏差。")
    parser.add_argument("--split_mode", type=str, default="file",
                        choices=["file", "sample"],
                        help="训练/验证划分方式")
    parser.add_argument("--sample_weight_mode", type=str, default="none",
                        choices=["none", "player", "winner", "player_winner"],
                        help="[DEPRECATED] 训练样本权重平衡模式。实验表明加权会扭曲真实先手优势信号，不推荐开启。")
    parser.add_argument("--sample_weight_clip", type=float, default=5.0,
                        help="训练样本权重上限，0 表示不裁剪")
    parser.add_argument("--player_channel_mode", type=str, default="original",
                        choices=["original", "zero", "random_flip"],
                        help="[DEPRECATED] 训练时处理当前玩家通道的方式。random_flip 实验表明会严重牌化，保持 original。")
    parser.add_argument("--evaluate_exe", type=str,
                        default="../build_onnx/evaluate_ai.exe",
                        help="evaluate_ai 可执行文件路径")
    parser.add_argument("--arena_games_per_side", type=int, default=50,
                        help="候选模型晋级 arena 每边局数 (0=关闭)。二项分布 95%% CI: N=100 局 ±10pp")
    parser.add_argument("--arena_simulations", type=int, default=800,
                        help="arena MCTS 模拟次数")
    parser.add_argument("--arena_temp", type=int, default=4,
                        help="arena 前 N 步温度采样")
    parser.add_argument("--promote_threshold", type=float, default=0.60,
                        help="candidate 晋级所需胜率。N=100 局二项分布 95%% CI 上限 = 0.598。")
    parser.add_argument("--min_black_winrate", type=float, default=0.20,
                        help="[仅 absolute 模式] candidate 执黑最低胜率门槛。记忆 c5900f29 证明在 5x5 中不可靠。")
    parser.add_argument("--min_white_winrate", type=float, default=0.50,
                        help="[仅 absolute 模式] candidate 执白最低胜率门槛")
    parser.add_argument("--side_gate_mode", type=str, default="margin",
                        choices=["margin", "relative", "absolute"],
                        help="arena 门槛模式。\n"
                             "  margin   = (默认) A-B 平均得分差 >= --min_score_margin。颜色对称连续量，推荐。\n"
                             "  relative = 对比 baseline 同颜色胜率。[legacy]\n"
                             "  absolute = 固定黑白胜率门槛。[legacy, 记忆 c5900f29 证明不可靠]")
    parser.add_argument("--min_score_margin", type=float, default=0.5,
                        help="[margin 模式] candidate 比 baseline 平均多赢几个格才能晋级。\n"
                             "5x5 总格数 25，0.5 代表 +2%% 总格优势；\n"
                             "高于 1.0 过于严苛 (iter22 自身 vs heuristic 也只能赢 ~2.3 box)。")
    parser.add_argument("--side_tolerance", type=float, default=0.14,
                        help="[relative 模式] 分侧门槛容忍度。按 N=50/side 二项分布 95%% CI: 1.96*sqrt(0.25/50)=0.139。")
    parser.add_argument("--dual", action="store_true", default=True,
                        help="启用双模型模式: 黑/白各自独立网络, 隔离 color bias (默认开启)")
    parser.add_argument("--no-dual", action="store_false", dest="dual",
                        help="禁用双模型, 回退到单模型模式")
    parser.add_argument("--black_sims_multiplier", type=float, default=1.0,
                        help="黑方 sims 倍数: 黑方搜索更深以补偿先手劣势 (推荐 2.0)")
    parser.add_argument("--league", action="store_true", default=False,
                        help="启用 League selfplay: 混合自对弈、vs 启发式、vs 旧模型，注入黑方胜局数据")
    parser.add_argument("--league_selfplay_frac", type=float, default=0.4,
                        help="League 中标准自对弈占比 (默认 0.4)")
    parser.add_argument("--league_heuristic_frac", type=float, default=0.5,
                        help="League 中模型(黑) vs 启发式(白) 占比 (默认 0.5)")
    parser.add_argument("--league_old_model_frac", type=float, default=0.1,
                        help="League 中模型(黑) vs 旧模型(白) 占比 (默认 0.1)")
    parser.add_argument("--league_old_model_pool_size", type=int, default=5,
                        help="League 旧模型对手池大小 (默认 5)")
    parser.add_argument("--gpus", type=str, default="",
                        help="并行 selfplay 的 GPU 列表, 逗号分隔 (如 '3,5,6,7') 或 'auto' 自动检测")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="自对弈探索温度 (1.0=高探索, 0.0=纯贪心)")
    parser.add_argument("--temperature_moves", type=int, default=10,
                        help="温度采样步数: 前 N 步渐进降温, 之后趋近贪心 (0=全程贪心)")
    args = parser.parse_args()
    gpu_list = []
    auto_gpu = args.gpus.lower() == "auto"
    if args.gpus and not auto_gpu:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    def do_selfplay(games, **kwargs):
        """单/多 GPU selfplay 调度。auto 模式下每轮重新检测空闲 GPU"""
        nonlocal gpu_list
        if auto_gpu:
            gpu_list = detect_free_gpus()
            if gpu_list:
                print(f"  [Auto GPU] detected: {gpu_list}", flush=True)
            else:
                print(f"  [Auto GPU] no free GPUs, falling back to single GPU", flush=True)
        # 自动注入温度参数（除非调用方显式覆盖）
        if "temperature" not in kwargs:
            kwargs["temperature"] = args.temperature
        if "temperature_moves" not in kwargs:
            kwargs["temperature_moves"] = args.temperature_moves
        if gpu_list:
            return run_backward_selfplay_multi_gpu(
                args.selfplay_exe, games, args.simulations, current_st,
                iter_data_dir, gpu_list, **kwargs)
        else:
            return run_backward_selfplay(
                args.selfplay_exe, games, args.simulations, current_st,
                iter_data_dir, **kwargs)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.selfplay_exe = resolve_path(args.selfplay_exe, script_dir, parser.get_default("selfplay_exe"))
    args.evaluate_exe = resolve_path(args.evaluate_exe, script_dir, parser.get_default("evaluate_exe"))
    args.output_base = resolve_path(args.output_base, script_dir, parser.get_default("output_base"))
    args.model_dir = resolve_path(args.model_dir, script_dir, parser.get_default("model_dir"))

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
    current_model = ""       # 单模型模式
    current_model_black = "" # 双模型模式: 黑方模型
    current_model_white = "" # 双模型模式: 白方模型
    iterations_at_st = 0
    total_iterations = 0
    stage = 1

    # 双模型: 模型文件路径
    best_pt_black = os.path.join(args.model_dir, "backward_best_model_black.pt")
    best_pt_white = os.path.join(args.model_dir, "backward_best_model_white.pt")
    onnx_black = os.path.join(args.model_dir, "backward_model_black.onnx")
    onnx_white = os.path.join(args.model_dir, "backward_model_white.onnx")

    # 状态文件
    state_file = os.path.join(args.output_base, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
            current_st = state.get("current_st", args.start_st)
            if args.dual:
                current_model_black = resolve_path(state.get("current_model_black", ""), script_dir)
                current_model_white = resolve_path(state.get("current_model_white", ""), script_dir)
                current_model = ""  # dual mode: no single model
            else:
                current_model = resolve_path(state.get("current_model", ""), script_dir)
            iterations_at_st = state.get("iterations_at_st", 0)
            total_iterations = state.get("total_iterations", 0)
            stage = state.get("stage", 1)
            mode_str = "dual" if args.dual else "single"
            print(f"  Resumed from: st={current_st}, iter={total_iterations}, stage={stage}, mode={mode_str}")
            print()

    # 双模型初始化: 首次运行时从现有单模型复制权重
    if args.dual:
        if not os.path.exists(best_pt_black) or not os.path.exists(best_pt_white):
            src_pt = os.path.join(args.model_dir, "backward_best_model.pt")
            if os.path.exists(src_pt):
                print("  [Dual Init] Copying existing model to black/white...")
                shutil.copy2(src_pt, best_pt_black)
                shutil.copy2(src_pt, best_pt_white)
            # 初始化 ONNX
            for pt, onnx in [(best_pt_black, onnx_black), (best_pt_white, onnx_white)]:
                if os.path.exists(pt) and not os.path.exists(onnx):
                    export_onnx(pt, onnx, args.arch)
        if not current_model_black:
            current_model_black = onnx_black if os.path.exists(onnx_black) else ""
        if not current_model_white:
            current_model_white = onnx_white if os.path.exists(onnx_white) else ""
        print(f"  Dual models: BLACK={current_model_black or 'N/A'}, WHITE={current_model_white or 'N/A'}")
        print()

    # League selfplay: 旧模型对手池 (model_path, iteration)
    old_model_pool = []
    # 从 state 恢复旧模型池
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            saved_state = json.load(f)
        for entry in saved_state.get("old_model_pool", []):
            if os.path.exists(entry[0]):
                old_model_pool.append(tuple(entry))

    for iteration in range(args.max_iterations):
        if args.target_total_iterations > 0 and total_iterations >= args.target_total_iterations:
            print(f"  Target total_iterations reached: {total_iterations}/{args.target_total_iterations}")
            break

        total_iterations += 1
        iterations_at_st += 1

        print(f"\n{'='*60}")
        print(f"  Iteration {total_iterations} | Stage {stage} | st={current_st} | iter_at_st={iterations_at_st}"
              + (" | DUAL" if args.dual else ""))
        print(f"{'='*60}")

        # --- 1. 自对弈 ---
        iter_data_dir = os.path.join(args.output_base, f"iter_{total_iterations:04d}_st{current_st}")
        if os.path.exists(iter_data_dir):
            print(f"  Removing existing iter dir before rerun: {iter_data_dir}")
            shutil.rmtree(iter_data_dir)

        if args.dual:
            ok = do_selfplay(
                args.games_per_iter,
                black_model=current_model_black, white_model=current_model_white,
                black_sims_mult=args.black_sims_multiplier)
        elif args.league and stage == 2 and current_model:
            # === League selfplay: 混合多种对手，注入黑方胜局数据 ===
            # 70% 标准自对弈 + 20% 模型(黑) vs 启发式(白) + 10% 模型(黑) vs 旧模型(白)
            total_games = args.games_per_iter
            n_selfplay = max(0, int(total_games * args.league_selfplay_frac))
            n_vs_heuristic = max(0, int(total_games * args.league_heuristic_frac))
            n_vs_old = total_games - n_selfplay - n_vs_heuristic

            ok = True
            league_batch = 0

            def _flush_league_batch():
                """重命名 iter_data_dir 中的 jsonl 文件，防止下一 batch 覆盖"""
                nonlocal league_batch
                for f in sorted(os.listdir(iter_data_dir)):
                    if f.endswith(".jsonl"):
                        src = os.path.join(iter_data_dir, f)
                        dst = os.path.join(iter_data_dir, f"L{league_batch}_{f}")
                        os.rename(src, dst)
                league_batch += 1

            # 1) 标准自对弈: 模型 vs 模型
            if ok and n_selfplay > 0:
                ok = do_selfplay(n_selfplay, model_path=current_model)
                _flush_league_batch()

            # 2) 模型(黑) vs 启发式(白): 注入黑方胜局
            #    --eval-black: 黑方无噪声贪心搜索, 发挥真实实力 vs 确定性启发式
            if ok and n_vs_heuristic > 0:
                ok = do_selfplay(n_vs_heuristic, black_model=current_model,
                                 eval_black=True)
                _flush_league_batch()

            # 3) 模型(黑) vs 旧模型(白): 多样性对手
            if ok and n_vs_old > 0 and old_model_pool:
                old_model_path, old_iter = random.choice(old_model_pool)
                print(f"  League opponent: iter_{old_iter} model")
                ok = do_selfplay(n_vs_old, black_model=current_model,
                                 white_model=old_model_path)
                _flush_league_batch()
            elif n_vs_old > 0:
                # 旧模型池为空时，回退为额外自对弈
                print(f"  League: old model pool empty, falling back to selfplay")
                ok = do_selfplay(n_vs_old, model_path=current_model)
                _flush_league_batch()
        else:
            ok = do_selfplay(args.games_per_iter, model_path=current_model)
        if not ok:
            print("[ERROR] Selfplay failed!")
            break

        samples = count_samples(iter_data_dir)
        print(f"  Generated {samples} samples")

        # --- 2. 训练 (双模型模式下分颜色训练) ---
        if args.dual:
            # 按玩家颜色拆分样本
            black_dir = os.path.join(args.output_base, "combined_black")
            white_dir = os.path.join(args.output_base, "combined_white")
            samples_black, samples_white = rebuild_training_window_dual(
                args.output_base, black_dir, white_dir,
                args.window_size, args.anchor_size, max_iteration=total_iterations)
            print(f"  Training window: black={samples_black}, white={samples_white}")

            promoted_black = False
            promoted_white = False

            for color, combined_dir, resume_pt, best_pt, onnx_path in [
                ("black", black_dir, best_pt_black, best_pt_black, onnx_black),
                ("white", white_dir, best_pt_white, best_pt_white, onnx_white),
            ]:
                print(f"\n  [{color.upper()}] Starting training...")
                run_name = f"backward_st{current_st}_iter{total_iterations}_{color}"
                value_mode = "q" if stage == 1 else args.stage2_value_mode
                resume = resume_pt if os.path.exists(resume_pt) else None

                ok = run_training(
                    combined_dir, args.model_dir, args.arch,
                    args.epochs_per_iter, run_name, value_mode, args.q_weight,
                    resume=resume, augment=True, lr=args.lr,
                    split_mode=args.split_mode,
                    sample_weight_mode=args.sample_weight_mode,
                    sample_weight_clip=args.sample_weight_clip,
                    player_channel_mode=args.player_channel_mode,
                )
                if not ok:
                    print(f"[ERROR] Training {color} failed!")
                    break

                # 导出 ONNX
                pt_path = os.path.join(args.model_dir, f"{run_name}_best_model.pt")
                if not os.path.exists(pt_path):
                    pt_path = os.path.join(args.model_dir, f"{run_name}_final_model.pt")
                candidate_onnx = os.path.join(args.model_dir, f"{run_name}_candidate.onnx")
                if not os.path.exists(pt_path) or not export_onnx(pt_path, candidate_onnx, args.arch):
                    continue

                # Arena: 用 --fixed-color 测试单色
                # black arena: black_candidate(BLACK) vs white_baseline(WHITE)
                #   → A=black_candidate 胜率 = 黑候选的黑方表现
                # white arena: black_baseline(BLACK) vs white_candidate(WHITE)
                #   → A=black_baseline 胜率, B=white_candidate 胜率 = 1-A_胜率
                if color == "black":
                    promoted, winrate, black_wr, white_wr, b_bl, b_wh, score_margin = run_arena(
                        args.evaluate_exe, candidate_onnx, current_model_white,
                        args.arena_games_per_side, args.arena_simulations, args.arena_temp,
                        args.promote_threshold, args.min_black_winrate, args.min_white_winrate,
                        args.side_gate_mode, args.side_tolerance, args.min_score_margin,
                        fixed_color=True)
                else:
                    # white_candidate = B, winrate 是 A(black_baseline) 的胜率
                    a_promoted, a_winrate, a_black_wr, a_white_wr, a_b_bl, a_b_wh, a_margin = run_arena(
                        args.evaluate_exe, current_model_black, candidate_onnx,
                        args.arena_games_per_side, args.arena_simulations, args.arena_temp,
                        args.promote_threshold, args.min_black_winrate, args.min_white_winrate,
                        args.side_gate_mode, args.side_tolerance, args.min_score_margin,
                        fixed_color=True)
                    # 取 B(white_candidate) 的胜率与 margin (取反)
                    winrate = 1.0 - a_winrate
                    score_margin = -a_margin
                    promoted = winrate >= args.promote_threshold

                if promoted:
                    try:
                        if os.path.exists(pt_path):
                            shutil.copy2(pt_path, best_pt)
                        if os.path.exists(candidate_onnx):
                            shutil.copy2(candidate_onnx, onnx_path)
                        if color == "black":
                            current_model_black = onnx_path
                            promoted_black = True
                        else:
                            current_model_white = onnx_path
                            promoted_white = True
                        print(f"  [Promote] {color} candidate promoted (arena winrate={winrate:.3f}, margin={score_margin:+.3f})")
                    except Exception as e:
                        print(f"  [Promote] {color} copy failed: {e}")
                else:
                    print(f"  [Reject] {color} candidate rejected (arena winrate={winrate:.3f}, margin={score_margin:+.3f})")
        else:
            # --- 单模型模式 (原逻辑) ---
            combined_dir = os.path.join(args.output_base, "combined")
            selected_dirs, window_samples = rebuild_training_window(
                args.output_base, combined_dir, args.window_size, args.anchor_size,
                max_iteration=total_iterations)
            print(f"  Training window: {len(selected_dirs)} iters, {window_samples} samples")
            print(f"  Data iters: {', '.join(selected_dirs)}")

            run_name = f"backward_st{current_st}_iter{total_iterations}"
            value_mode = "q" if stage == 1 else args.stage2_value_mode
            resume_path = current_model.replace(".onnx", ".pt") if current_model.endswith(".onnx") else current_model

            best_pt = os.path.join(args.model_dir, f"backward_best_model.pt")
            if os.path.exists(best_pt):
                resume_path = best_pt

            ok = run_training(
                combined_dir, args.model_dir, args.arch,
                args.epochs_per_iter, run_name, value_mode, args.q_weight,
                resume=resume_path if os.path.exists(resume_path) else None,
                augment=True, lr=args.lr,
                split_mode=args.split_mode,
                sample_weight_mode=args.sample_weight_mode,
                sample_weight_clip=args.sample_weight_clip,
                player_channel_mode=args.player_channel_mode,
            )
            if not ok:
                print("[ERROR] Training failed!")
                break

            # --- 3. 导出 ONNX ---
            pt_path = os.path.join(args.model_dir, f"{run_name}_best_model.pt")
            if not os.path.exists(pt_path):
                pt_path = os.path.join(args.model_dir, f"{run_name}_final_model.pt")

            onnx_path = os.path.join(args.model_dir, "backward_model.onnx")
            candidate_onnx = os.path.join(args.model_dir, f"{run_name}_candidate.onnx")
            if os.path.exists(pt_path):
                if not export_onnx(pt_path, candidate_onnx, args.arch):
                    print("[ERROR] Candidate export failed!")
                    break

                promoted, winrate, black_wr, white_wr, baseline_black_wr, baseline_white_wr, score_margin = run_arena(
                    args.evaluate_exe, candidate_onnx, current_model,
                    args.arena_games_per_side, args.arena_simulations, args.arena_temp,
                    args.promote_threshold, args.min_black_winrate, args.min_white_winrate,
                    args.side_gate_mode, args.side_tolerance, args.min_score_margin)
                if promoted:
                    # 将旧模型保存到 league 对手池 (先保存再覆盖)
                    if args.league and current_model and os.path.exists(current_model):
                        league_dir = os.path.join(args.model_dir, "league")
                        os.makedirs(league_dir, exist_ok=True)
                        league_path = os.path.join(league_dir, f"iter_{total_iterations:04d}.onnx")
                        shutil.copy2(current_model, league_path)
                        # 同时复制 .data 文件 (ONNX external weights)
                        data_src = current_model + ".data"
                        if os.path.exists(data_src):
                            shutil.copy2(data_src, league_path + ".data")
                        old_model_pool.append((league_path, total_iterations))
                        if len(old_model_pool) > args.league_old_model_pool_size:
                            # 移除最旧的，清理磁盘
                            removed_path, _ = old_model_pool[0]
                            if os.path.exists(removed_path):
                                os.remove(removed_path)
                                if os.path.exists(removed_path + ".data"):
                                    os.remove(removed_path + ".data")
                            old_model_pool = old_model_pool[-args.league_old_model_pool_size:]
                    shutil.copy2(pt_path, best_pt)
                    shutil.copy2(candidate_onnx, onnx_path)
                    current_model = onnx_path
                    print(f"  [Promote] candidate promoted (winrate={winrate:.3f}, margin={score_margin:+.3f}, A_black={black_wr:.2f}/B_black={baseline_black_wr:.2f}, A_white={white_wr:.2f}/B_white={baseline_white_wr:.2f})")
                else:
                    print(f"  [Reject] candidate rejected (winrate={winrate:.3f}, margin={score_margin:+.3f}, A_black={black_wr:.2f}/B_black={baseline_black_wr:.2f}, A_white={white_wr:.2f}/B_white={baseline_white_wr:.2f})")
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
            print(f"  [Stage 2] Full self-play training")

        # --- 5. 保存状态 ---
        state = {
            "current_st": current_st,
            "iterations_at_st": iterations_at_st,
            "total_iterations": total_iterations,
            "stage": stage,
        }
        if args.dual:
            state["current_model_black"] = current_model_black
            state["current_model_white"] = current_model_white
        else:
            state["current_model"] = current_model
        if args.league:
            state["old_model_pool"] = old_model_pool
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        print(f"\n  State saved: {state_file}")

    print("\n" + "=" * 60)
    print("  Backward Training Complete!")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Final st: {current_st}")
    if args.dual:
        print(f"  Final black model: {current_model_black}")
        print(f"  Final white model: {current_model_white}")
    else:
        print(f"  Final model: {current_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
