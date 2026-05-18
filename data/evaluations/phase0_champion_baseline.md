# 阶段 0：Champion 基线记录

> 日期：2025-05-17
> 目的：冻结当前状态，记录各模型在 arena 中的表现

## 测试环境

- **MCTS 模式**：evaluation（无 Dirichlet 噪声，temperature=0）
- **模拟次数**：AZ_SIMULATIONS = 200，AZ_TIME_LIMIT_MS = 1000
- **每侧对局数**：5（共 10 局，交换先后手）
- **构建模式**：Release (-O3)

## 结果 1：Heuristic vs Heuristic

```
Games: 10
BLACK wins: 0  WHITE wins: 10  Unfinished: 0
A wins: 5  B wins: 5
Avg score BLACK: 12  WHITE: 13
```

- 启发式评估器完全确定性，WHITE 始终以 13:12 获胜。
- 这是所有 AI 的最低基线。

## 结果 2：weights.bin (MLP) vs Heuristic

```
A: data/models/weights.bin (257853 params)
B: heuristic

Games: 10
BLACK wins: 5  WHITE wins: 5  Unfinished: 0
A wins: 0  B wins: 10
A as BLACK wins: 0  A as WHITE wins: 0
B as BLACK wins: 5  B as WHITE wins: 5
Avg score BLACK: 12  WHITE: 13
```

- **当前官方 MLP 模型 0:10 不敌启发式**。
- MLP 作黑方：6:19 惨败；MLP 作白方（应有先手优势的后手位）：7:18 惨败。
- 结论：当前 weights.bin 尚未达到可用水平，heuristic 仍是实际 champion。

## 当前 Champion

| 模型 | 胜率 vs Heuristic | 状态 |
|---|---|---|
| `heuristic` | 基线 (50% 交换先后手) | **实际 champion** |
| `weights.bin` (MLP) | 0% | 待改进 |

## 改造后验证要点

阶段 0 改造（MCTS 模式分离）完成后，arena 结果应与此基线一致：
- Heuristic vs Heuristic 仍为 WHITE 13:12 × 10
- 旧功能不退化

## 阶段 0 改造内容

1. **MCTSConfig 结构体**：新增 `selfPlay()` / `evaluation()` 预设
2. **MCTS 模式分离**：Dirichlet 噪声仅在 `addRootNoise=true`（自对弈）时启用
3. **旧接口兼容**：`search(board, player, numSims, timeMs)` 自动使用 evaluation 模式
4. **单元测试**：27 个测试覆盖 action 映射、board、encoder、MCTS 配置与搜索
5. **旧 UCT 开关**：`USE_ALPHAZERO_AI` 保持在 `az_types.h`，可切换回旧 UCT
