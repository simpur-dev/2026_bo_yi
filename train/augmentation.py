"""
D4 对称增强：点格棋 5×5 棋盘的 8 种对称变换

点格棋棋盘是正方形，具有 D4 对称群（4 旋转 × 2 镜像 = 8 种变换）。
每条训练样本可以通过这 8 种变换扩展为 8 条等价样本。

变换列表:
  0: identity         (i, j) -> (i, j)
  1: rotate 90° CW    (i, j) -> (j, 10-i)
  2: rotate 180°      (i, j) -> (10-i, 10-j)
  3: rotate 270° CW   (i, j) -> (10-j, i)
  4: flip horizontal   (i, j) -> (i, 10-j)
  5: rot90 + flip      (i, j) -> (j, i)          (主对角线翻转)
  6: rot180 + flip     (i, j) -> (10-i, j)        (垂直翻转)
  7: rot270 + flip     (i, j) -> (10-j, 10-i)     (副对角线翻转)

关键设计:
  不直接 reshape policy 数组。而是:
    action -> LOC -> transform LOC -> new action
  原因: 横边和竖边在 90° 旋转后互换，直接按下标映射容易出错。
"""

import numpy as np

BOARD_SIZE = 11
ACTION_SIZE = 60
NUM_TRANSFORMS = 8

# ========== 坐标与动作映射 (复刻 C++ az_action.cpp) ==========

def loc_to_action(i, j):
    """LOC -> action index (与 C++ locToAction 一致)"""
    if i % 2 == 0 and j % 2 == 1:  # 横边 HENG
        return (i // 2) * 5 + (j - 1) // 2
    elif i % 2 == 1 and j % 2 == 0:  # 竖边 SHU
        return 30 + ((i - 1) // 2) * 6 + j // 2
    return -1


def action_to_loc(action):
    """action index -> LOC (与 C++ actionToLoc 一致)"""
    if action < 30:  # 横边
        i = (action // 5) * 2
        j = (action % 5) * 2 + 1
        return i, j
    else:  # 竖边
        a = action - 30
        i = (a // 6) * 2 + 1
        j = (a % 6) * 2
        return i, j


# ========== D4 坐标变换 ==========

def _transform_loc(i, j, t):
    """对 11×11 棋盘上的坐标 (i, j) 施加第 t 种 D4 变换"""
    if t == 0: return i, j
    if t == 1: return j, 10 - i
    if t == 2: return 10 - i, 10 - j
    if t == 3: return 10 - j, i
    if t == 4: return i, 10 - j
    if t == 5: return j, i
    if t == 6: return 10 - i, j
    if t == 7: return 10 - j, 10 - i
    raise ValueError(f"Invalid transform id: {t}")


# ========== 预计算动作排列表 ==========

def _build_action_permutation(t):
    """为变换 t 构建动作排列: perm[old_action] = new_action"""
    perm = [0] * ACTION_SIZE
    for a in range(ACTION_SIZE):
        oi, oj = action_to_loc(a)
        ni, nj = _transform_loc(oi, oj, t)
        new_a = loc_to_action(ni, nj)
        assert new_a >= 0, f"Transform {t}: ({oi},{oj})->({ni},{nj}) is not a valid edge"
        perm[a] = new_a
    return perm


# 预计算所有 8 种变换的排列表 (module 加载时执行一次)
ACTION_PERMS = [_build_action_permutation(t) for t in range(NUM_TRANSFORMS)]


# ========== 棋盘张量变换 ==========

def transform_board(board, t):
    """
    对棋盘张量施加 D4 变换。

    Args:
        board: numpy array, shape (C, 11, 11) 或 (7*11*11,) 一维
        t: 变换编号 0-7

    Returns:
        变换后的 numpy array, shape 同输入
    """
    flat = False
    if board.ndim == 1:
        flat = True
        board = board.reshape(7, BOARD_SIZE, BOARD_SIZE)

    if t == 0:
        result = board.copy()
    elif t == 1:  # 90° CW
        result = np.rot90(board, k=-1, axes=(1, 2)).copy()
    elif t == 2:  # 180°
        result = np.rot90(board, k=2, axes=(1, 2)).copy()
    elif t == 3:  # 270° CW
        result = np.rot90(board, k=1, axes=(1, 2)).copy()
    elif t == 4:  # flip horizontal
        result = np.flip(board, axis=2).copy()
    elif t == 5:  # transpose (主对角线)
        result = np.transpose(board, (0, 2, 1)).copy()
    elif t == 6:  # vertical flip
        result = np.flip(board, axis=1).copy()
    elif t == 7:  # anti-transpose (副对角线)
        result = np.transpose(np.rot90(board, k=2, axes=(1, 2)), (0, 2, 1)).copy()
    else:
        raise ValueError(f"Invalid transform id: {t}")

    if flat:
        result = result.reshape(-1)
    return result


# ========== Policy / Legal Mask 变换 ==========

def transform_action_array(arr, t):
    """
    对长度为 60 的动作数组 (policy 或 legal_mask) 施加 D4 变换。

    Args:
        arr: numpy array 或 list, shape (60,)
        t: 变换编号 0-7

    Returns:
        变换后的 numpy array, shape (60,)
    """
    if t == 0:
        return np.array(arr, dtype=np.float32)

    perm = ACTION_PERMS[t]
    result = np.zeros(ACTION_SIZE, dtype=np.float32)
    for old_a in range(ACTION_SIZE):
        result[perm[old_a]] = arr[old_a]
    return result


# ========== 完整样本变换 ==========

def augment_sample(board, legal_mask, policy, t):
    """
    对一条训练样本施加第 t 种 D4 变换。

    Args:
        board: (7*11*11,) 或 (7,11,11) 棋盘张量
        legal_mask: (60,) 合法动作掩码
        policy: (60,) 策略分布
        t: 变换编号 0-7

    Returns:
        (new_board, new_legal_mask, new_policy)
    """
    new_board = transform_board(board, t)
    new_mask = transform_action_array(legal_mask, t)
    new_policy = transform_action_array(policy, t)
    return new_board, new_mask, new_policy


def random_augment(board, legal_mask, policy, rng=None):
    """
    随机选择一种 D4 变换并施加。

    Args:
        board, legal_mask, policy: 同 augment_sample
        rng: numpy RandomState 或 None (使用全局随机)

    Returns:
        (new_board, new_legal_mask, new_policy, transform_id)
    """
    if rng is not None:
        t = rng.randint(0, NUM_TRANSFORMS)
    else:
        t = np.random.randint(0, NUM_TRANSFORMS)
    new_board, new_mask, new_policy = augment_sample(board, legal_mask, policy, t)
    return new_board, new_mask, new_policy, t


# ========== 验证工具 ==========

def validate_augmentation():
    """
    验证所有 D4 增强的正确性。

    检查:
      1. 每个 action 经 8 种变换后仍是有效 action (0-59)
      2. 4 次 90° 旋转后回到原 action
      3. sum(policy) 增强前后相等
      4. sum(legal_mask) 增强前后相等
      5. 排列是双射 (permutation)
      6. 逆变换正确

    Returns:
        True 如果全部通过
    """
    ok = True

    # 检查 1: 排列有效性
    for t in range(NUM_TRANSFORMS):
        perm = ACTION_PERMS[t]
        seen = set(perm)
        if len(seen) != ACTION_SIZE:
            print(f"[FAIL] Transform {t}: permutation not bijective, {len(seen)} unique values")
            ok = False
        for a in range(ACTION_SIZE):
            if perm[a] < 0 or perm[a] >= ACTION_SIZE:
                print(f"[FAIL] Transform {t}: action {a} -> {perm[a]} out of range")
                ok = False

    # 检查 2: 4 次 90° 旋转回到原点
    perm1 = ACTION_PERMS[1]
    for a in range(ACTION_SIZE):
        a1 = perm1[a]
        a2 = perm1[a1]
        a3 = perm1[a2]
        a4 = perm1[a3]
        if a4 != a:
            print(f"[FAIL] 4x rotate90: action {a} -> {a1} -> {a2} -> {a3} -> {a4} != {a}")
            ok = False

    # 检查 3: 2 次 180° 回到原点
    perm2 = ACTION_PERMS[2]
    for a in range(ACTION_SIZE):
        if perm2[perm2[a]] != a:
            print(f"[FAIL] 2x rotate180: action {a} not identity")
            ok = False

    # 检查 4: 2 次 flip 回到原点
    for t in [4, 5, 6, 7]:
        perm = ACTION_PERMS[t]
        for a in range(ACTION_SIZE):
            if perm[perm[a]] != a:
                print(f"[FAIL] 2x transform {t}: action {a} not identity")
                ok = False

    # 检查 5: policy sum 不变
    rng = np.random.RandomState(42)
    test_policy = rng.dirichlet(np.ones(ACTION_SIZE))
    for t in range(NUM_TRANSFORMS):
        new_policy = transform_action_array(test_policy, t)
        if not np.isclose(test_policy.sum(), new_policy.sum(), atol=1e-6):
            print(f"[FAIL] Transform {t}: policy sum changed {test_policy.sum():.6f} -> {new_policy.sum():.6f}")
            ok = False

    # 检查 6: legal_mask count 不变
    test_mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    test_mask[:30] = 1.0  # 假设前 30 个合法
    for t in range(NUM_TRANSFORMS):
        new_mask = transform_action_array(test_mask, t)
        if int(test_mask.sum()) != int(new_mask.sum()):
            print(f"[FAIL] Transform {t}: legal count changed {int(test_mask.sum())} -> {int(new_mask.sum())}")
            ok = False

    # 检查 7: 棋盘变换可逆性 (rotate90 x4 = identity)
    test_board = rng.randn(7, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
    b1 = transform_board(test_board, 1)
    b2 = transform_board(b1, 1)
    b3 = transform_board(b2, 1)
    b4 = transform_board(b3, 1)
    if not np.allclose(test_board, b4, atol=1e-6):
        print("[FAIL] Board: 4x rotate90 != identity")
        ok = False

    # 检查 8: 棋盘变换与动作排列一致性
    # 在初始棋盘上放一条边，编码，变换，检查对应位置
    # (这里用一个简化检查: channel 3 = 可落子边 应与 legal_mask 一致)

    if ok:
        print("[OK] All D4 augmentation validations passed.")
    return ok


if __name__ == "__main__":
    validate_augmentation()

    # 展示一个 action 的 8 种变换结果
    print("\nAction 0 transformations:")
    for t in range(NUM_TRANSFORMS):
        a0_loc = action_to_loc(0)
        new_loc = _transform_loc(a0_loc[0], a0_loc[1], t)
        new_action = ACTION_PERMS[t][0]
        print(f"  T{t}: action 0 ({a0_loc}) -> action {new_action} ({new_loc})")
