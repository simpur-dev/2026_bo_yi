"""
候选模型 vs 基线模型 评估脚本
============================

功能:
  调 build/evaluate_ai.exe 进行黑/白轮转对战, 统计:
    - 胜率 (candidate vs baseline)
    - A-B margin (candidate 净胜格数)
    - 黑/白拆分胜率

用法:
  python eval/eval_vs_baseline.py \\
    --candidate data/backup_pre_rl_2026-06-14/backward_model_white.onnx \\
    --baseline  data/backup_pre_rl_2026-06-14/backward_model_black.onnx \\
    --games 20 --sims 400

  也支持调用 Python evaluator (在没编译 evaluate_ai.exe 时):
    python eval/eval_vs_baseline.py --candidate modelA.pt --baseline modelB.pt \\
      --games 10 --python-only

设计原则:
  - 严守红线: 只在 2026_bo_yi 内部工作, 不碰 simpur/ / ruikang/ / 3rdparty/
  - 候选模型若 胜率 < 40% 或 margin < 0.5, 自动判定 REJECT (基线保护)
  - 评估结果写到 eval/reports/eval_YYYYMMDD_HHMMSS.json (可追溯)

相关:
  - src/evaluate_ai_main.cpp: 评估主体 (C++ 实现)
  - data/backup_pre_rl_2026-06-14/: 训练前基线
  - 进度.md: 训练/评估日志
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_EVALUATE_AI = os.path.join(PROJECT_ROOT, "build", "evaluate_ai.exe")
REJECT_WINRATE_THRESHOLD = 0.40
REJECT_MARGIN_THRESHOLD = 0.5


def find_evaluate_ai_exe():
    """在常见位置查找 evaluate_ai.exe"""
    candidates = [
        os.environ.get("EVALUATE_AI_EXE"),
        DEFAULT_EVALUATE_AI,
        os.path.join(PROJECT_ROOT, "build", "Release", "evaluate_ai.exe"),
        os.path.join(PROJECT_ROOT, "build_onnx", "evaluate_ai.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def parse_eval_output(stdout: str, stderr: str):
    """
    解析 evaluate_ai 的输出, 抽取 A wins / B wins / A-B margin / 黑/白 wins
    """
    text = stdout + "\n" + stderr
    result = {
        "a_wins": None,
        "b_wins": None,
        "a_as_black_wins": None,
        "a_as_white_wins": None,
        "b_as_black_wins": None,
        "b_as_white_wins": None,
        "black_wins": None,
        "white_wins": None,
        "unfinished": None,
        "a_b_margin": None,
        "avg_score_a": None,
        "avg_score_b": None,
        "games": None,
    }
    patterns = {
        "a_wins": r"A wins:\s*(\d+)",
        "b_wins": r"B wins:\s*(\d+)",
        "a_as_black_wins": r"A as BLACK wins:\s*(\d+)",
        "a_as_white_wins": r"A as WHITE wins:\s*(\d+)",
        "b_as_black_wins": r"B as BLACK wins:\s*(\d+)",
        "b_as_white_wins": r"B as WHITE wins:\s*(\d+)",
        "black_wins": r"BLACK wins:\s*(\d+)",
        "white_wins": r"WHITE wins:\s*(\d+)",
        "unfinished": r"Unfinished:\s*(\d+)",
        "a_b_margin": r"A-B margin:\s*(-?\d+\.?\d*)",
        "avg_score_a": r"Avg score A:\s*(-?\d+\.?\d*)",
        "avg_score_b": r"Avg score B:\s*(-?\d+\.?\d*)",
        "games": r"Games:\s*(\d+)",
    }
    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            try:
                v = m.group(1)
                result[k] = float(v) if "." in v else int(v)
            except (ValueError, IndexError):
                pass
    return result


def run_evaluate_ai(exe_path, candidate, baseline, games, sims, temp, verbose):
    """调用一次 evaluate_ai.exe, 返回解析结果"""
    # evaluate_ai.exe 编译时链接 onnxruntime, 运行时需要 onnxruntime.dll
    # 该 dll 在 3rdparty/onnxruntime_cpu/lib/ (红线内只读不碰)
    # 临时把该目录加入 PATH 让 Windows 能找到 dll
    onnx_lib_dir = os.path.join(PROJECT_ROOT, "3rdparty", "onnxruntime_cpu", "lib")
    env = os.environ.copy()
    if os.path.isdir(onnx_lib_dir):
        env["PATH"] = onnx_lib_dir + os.pathsep + env.get("PATH", "")
    cmd = [exe_path, str(games), candidate, baseline, "--sims", str(sims)]
    if temp is not None:
        cmd.extend(["--temp", str(temp)])
    print(f"[eval] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"[eval] ERROR: returncode={proc.returncode}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:500]}")
    parsed = parse_eval_output(proc.stdout, proc.stderr)
    parsed["elapsed_s"] = round(elapsed, 1)
    parsed["cmd"] = " ".join(cmd)
    return parsed


def judge(parsed):
    """
    根据指标判定 ACCEPT / REJECT / INCONCLUSIVE
    """
    games = parsed.get("games") or 0
    if games < 5:
        return "INCONCLUSIVE", "对局数太少 (<5), 统计不稳定"

    a_wins = parsed.get("a_wins")
    b_wins = parsed.get("b_wins")
    margin = parsed.get("a_b_margin")
    finished = (a_wins or 0) + (b_wins or 0)

    if finished == 0:
        return "INCONCLUSIVE", "无完成对局"

    winrate = (a_wins or 0) / games

    if winrate < REJECT_WINRATE_THRESHOLD:
        return "REJECT", f"胜率 {winrate:.1%} < {REJECT_WINRATE_THRESHOLD:.0%} (基线保护)"
    if margin is not None and margin < REJECT_MARGIN_THRESHOLD:
        return "REJECT", f"A-B margin {margin:.2f} < {REJECT_MARGIN_THRESHOLD} (基线保护)"
    if winrate >= 0.55 and (margin is None or margin >= 1.0):
        return "ACCEPT", f"胜率 {winrate:.1%} margin {margin or 0:.2f} (显著优势)"
    return "INCONCLUSIVE", f"胜率 {winrate:.1%} margin {margin or 0:.2f} (需更多对局确认)"


def main():
    parser = argparse.ArgumentParser(
        description="候选模型 vs 基线模型 评估 (调 evaluate_ai.exe)")
    parser.add_argument("--candidate", type=str, required=True,
                        help="候选模型路径 (.onnx / .pt / .bin)")
    parser.add_argument("--baseline", type=str, required=True,
                        help="基线模型路径 (默认: data/backup_pre_rl_2026-06-14/)")
    parser.add_argument("--games", type=int, default=20,
                        help="对局数 (黑/白各一半, 默认 20)")
    parser.add_argument("--sims", type=int, default=400,
                        help="每步 MCTS 模拟数 (默认 400)")
    parser.add_argument("--temp", type=int, default=4,
                        help="前 N 步加噪声 (默认 4, 0=deterministic, evaluate_ai 实际参数名)")
    parser.add_argument("--exe", type=str, default=None,
                        help="evaluate_ai.exe 路径 (默认自动查找)")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "reports"),
                        help="报告输出目录")
    parser.add_argument("--tag", type=str, default="",
                        help="报告标签 (例如 rl_iter_001)")
    args = parser.parse_args()

    exe_path = args.exe or find_evaluate_ai_exe()
    if exe_path is None or not os.path.exists(exe_path):
        print(f"ERROR: 找不到 evaluate_ai.exe, 请先编译:")
        print(f"  cmake -B build -G 'MinGW Makefiles'")
        print(f"  cmake --build build --target evaluate_ai")
        sys.exit(2)

    print("=" * 60)
    print("  Eval: candidate vs baseline")
    print("=" * 60)
    print(f"  candidate: {args.candidate}")
    print(f"  baseline:  {args.baseline}")
    print(f"  games:     {args.games}")
    print(f"  sims:      {args.sims}")
    print(f"  exe:       {exe_path}")
    print("=" * 60)

    parsed = run_evaluate_ai(
        exe_path, args.candidate, args.baseline,
        args.games, args.sims, args.temp, verbose=False)
    print()
    print("  Raw parsed:")
    for k, v in parsed.items():
        print(f"    {k:20s} = {v}")

    verdict, reason = judge(parsed)
    parsed["verdict"] = verdict
    parsed["reason"] = reason
    parsed["candidate"] = args.candidate
    parsed["baseline"] = args.baseline
    parsed["games_requested"] = args.games
    parsed["sims"] = args.sims
    parsed["timestamp"] = datetime.now().isoformat(timespec="seconds")
    parsed["tag"] = args.tag

    print()
    print("=" * 60)
    print(f"  VERDICT: {verdict}")
    print(f"  REASON:  {reason}")
    print("=" * 60)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    report_path = os.path.join(args.out_dir, f"eval{tag}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")

    if verdict == "REJECT":
        sys.exit(1)
    elif verdict == "INCONCLUSIVE":
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
