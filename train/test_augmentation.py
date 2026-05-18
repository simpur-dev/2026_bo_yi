"""
D4 增强测试与可视化脚本

测试验收标准:
  1. 任意合法 action 经 8 种增强后仍是合法 action
  2. sum(policy) 增强前后保持为 1
  3. sum(legal_mask) 增强前后相同
  4. 增强 4 次 90° 后回到原 action
  5. 棋盘编码增强与动作增强一致
  6. 使用真实训练数据验证

用法:
  python test_augmentation.py                    # 运行所有测试
  python test_augmentation.py --data_dir DIR     # 用真实数据验证
  python test_augmentation.py --visualize        # 可视化增强前后对比
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmentation import (
    validate_augmentation,
    augment_sample,
    transform_board,
    transform_action_array,
    loc_to_action,
    action_to_loc,
    ACTION_PERMS,
    NUM_TRANSFORMS,
    ACTION_SIZE,
    BOARD_SIZE,
)


def test_basic_validation():
    """基础验证: 排列双射、循环、sum 不变"""
    print("=" * 60)
    print("TEST 1: Basic augmentation validation")
    print("=" * 60)
    ok = validate_augmentation()
    assert ok, "Basic validation failed!"
    print()
    return True


def test_synthetic_sample():
    """合成样本测试: 验证 board / policy / legal_mask 一致性"""
    print("=" * 60)
    print("TEST 2: Synthetic sample augmentation consistency")
    print("=" * 60)

    rng = np.random.RandomState(123)

    # 创建一个合成 board: channel 3 (可落子边) 和 legal_mask 必须一致
    board = np.zeros((7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # 随机标记一些边为 "已占据" (channel 2) 和 "可落子" (channel 3)
    legal_actions = []
    for a in range(ACTION_SIZE):
        i, j = action_to_loc(a)
        if rng.random() < 0.5:
            board[2, i, j] = 1.0  # 已占据
        else:
            board[3, i, j] = 1.0  # 可落子
            legal_actions.append(a)

    # 构造 legal_mask 和 policy
    legal_mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for a in legal_actions:
        legal_mask[a] = 1.0

    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    if len(legal_actions) > 0:
        probs = rng.dirichlet(np.ones(len(legal_actions)))
        for i, a in enumerate(legal_actions):
            policy[a] = probs[i]

    ok = True
    for t in range(NUM_TRANSFORMS):
        new_board, new_mask, new_policy = augment_sample(board, legal_mask, policy, t)

        # 检查 1: legal_mask 计数不变
        assert int(new_mask.sum()) == int(legal_mask.sum()), \
            f"T{t}: legal count changed {int(legal_mask.sum())} -> {int(new_mask.sum())}"

        # 检查 2: policy sum 不变
        assert np.isclose(policy.sum(), new_policy.sum(), atol=1e-6), \
            f"T{t}: policy sum changed"

        # 检查 3: policy 只在 legal 位置非零
        for a in range(ACTION_SIZE):
            if new_mask[a] == 0 and new_policy[a] > 1e-8:
                print(f"[FAIL] T{t}: policy[{a}] = {new_policy[a]} but legal_mask[{a}] = 0")
                ok = False

        # 检查 4: board channel 3 与 new_mask 一致
        for a in range(ACTION_SIZE):
            i, j = action_to_loc(a)
            board_val = new_board[3, i, j]
            mask_val = new_mask[a]
            if abs(board_val - mask_val) > 1e-6:
                print(f"[FAIL] T{t}: board[3,{i},{j}]={board_val} != new_mask[{a}]={mask_val}")
                ok = False

        # 检查 5: board channel 2 与 ~new_mask 一致 (occupied = not free)
        for a in range(ACTION_SIZE):
            i, j = action_to_loc(a)
            occ_val = new_board[2, i, j]
            expected_occ = 1.0 - new_mask[a]
            if abs(occ_val - expected_occ) > 1e-6:
                print(f"[FAIL] T{t}: board[2,{i},{j}]={occ_val} != expected {expected_occ}")
                ok = False

    if ok:
        print("[OK] Synthetic sample: board/policy/legal_mask all consistent under 8 transforms")
    else:
        print("[FAIL] Synthetic sample tests had failures")
    print()
    return ok


def test_real_data(data_dir):
    """使用真实训练数据验证增强不破坏样本"""
    print("=" * 60)
    print(f"TEST 3: Real data augmentation ({data_dir})")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"[SKIP] Data directory not found: {data_dir}")
        print()
        return True

    # 找一个 JSONL 文件
    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        print("[SKIP] No JSONL files found")
        print()
        return True

    filepath = os.path.join(data_dir, jsonl_files[0])
    samples_tested = 0
    ok = True

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num >= 50:  # 测前 50 条
                break
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            board = np.array(sample["board"], dtype=np.float32).reshape(7, BOARD_SIZE, BOARD_SIZE)
            legal_mask = np.array(sample["legal_mask"], dtype=np.float32)
            policy = np.array(sample["policy"], dtype=np.float32)

            for t in range(NUM_TRANSFORMS):
                _, new_mask, new_policy = augment_sample(board, legal_mask, policy, t)

                # sum 检查
                if int(new_mask.sum()) != int(legal_mask.sum()):
                    print(f"[FAIL] Sample {line_num} T{t}: legal count {int(legal_mask.sum())} -> {int(new_mask.sum())}")
                    ok = False

                if not np.isclose(policy.sum(), new_policy.sum(), atol=1e-5):
                    print(f"[FAIL] Sample {line_num} T{t}: policy sum {policy.sum():.6f} -> {new_policy.sum():.6f}")
                    ok = False

                # policy 只在 legal 位置非零
                illegal_policy = new_policy * (1.0 - new_mask)
                if illegal_policy.sum() > 1e-6:
                    print(f"[FAIL] Sample {line_num} T{t}: policy on illegal actions = {illegal_policy.sum():.6f}")
                    ok = False

            samples_tested += 1

    if ok:
        print(f"[OK] {samples_tested} real samples × 8 transforms all pass")
    else:
        print(f"[FAIL] Some real data tests failed")
    print()
    return ok


def visualize_augmentation(data_dir=None):
    """可视化增强前后对比"""
    print("=" * 60)
    print("VISUALIZATION: D4 augmentation examples")
    print("=" * 60)

    # 用一个简单的例子
    if data_dir and os.path.exists(data_dir):
        jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
        if jsonl_files:
            filepath = os.path.join(data_dir, jsonl_files[0])
            with open(filepath, "r", encoding="utf-8") as f:
                sample = json.loads(f.readline().strip())
            legal_mask = np.array(sample["legal_mask"], dtype=np.float32)
            policy = np.array(sample["policy"], dtype=np.float32)
            print(f"Using real sample from: {jsonl_files[0]}")
        else:
            legal_mask = None
    else:
        legal_mask = None

    if legal_mask is None:
        # 合成数据
        rng = np.random.RandomState(42)
        legal_mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a in range(ACTION_SIZE):
            if rng.random() < 0.6:
                legal_mask[a] = 1.0
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        legal_indices = np.where(legal_mask > 0)[0]
        probs = rng.dirichlet(np.ones(len(legal_indices)))
        for i, a in enumerate(legal_indices):
            policy[a] = probs[i]
        print("Using synthetic sample")

    # 显示原始 top-5 动作
    top5 = np.argsort(policy)[::-1][:5]
    print(f"\nOriginal top-5 actions:")
    print(f"  {'Action':<8} {'LOC':<12} {'Prob':<10} {'Type'}")
    for a in top5:
        i, j = action_to_loc(a)
        atype = "HENG" if a < 30 else "SHU"
        print(f"  {a:<8} ({i},{j}){'':<6} {policy[a]:.4f}     {atype}")

    print(f"\nLegal actions: {int(legal_mask.sum())}")
    print(f"Policy sum: {policy.sum():.6f}")

    # 显示每种变换后的 top-1
    transform_names = [
        "identity", "rot90_CW", "rot180", "rot270_CW",
        "flip_H", "rot90+flip", "rot180+flip(flip_V)", "rot270+flip"
    ]
    print(f"\nTop-1 action under each D4 transform:")
    print(f"  {'Transform':<22} {'Action':<8} {'LOC':<12} {'Prob':<10} {'Type'}")
    for t in range(NUM_TRANSFORMS):
        new_policy = transform_action_array(policy, t)
        best_a = int(np.argmax(new_policy))
        i, j = action_to_loc(best_a)
        atype = "HENG" if best_a < 30 else "SHU"
        print(f"  {transform_names[t]:<22} {best_a:<8} ({i},{j}){'':<6} {new_policy[best_a]:.4f}     {atype}")

    print()


def main():
    parser = argparse.ArgumentParser(description="D4 增强测试与可视化")
    parser.add_argument("--data_dir", type=str, default="../data/selfplay", help="训练数据目录")
    parser.add_argument("--visualize", action="store_true", help="可视化增强效果")
    args = parser.parse_args()

    all_pass = True
    all_pass &= test_basic_validation()
    all_pass &= test_synthetic_sample()
    all_pass &= test_real_data(args.data_dir)

    if args.visualize:
        visualize_augmentation(args.data_dir)

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
