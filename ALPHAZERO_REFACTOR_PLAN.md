# AlphaZero + 精确终局求解器改造方案

> 本文档用于记录 `Dots and Boxes` 项目的 AI 升级路线：将当前 `UCT + 随机模拟 + 手写后期策略` 改造成 `AlphaZero-style PUCT + 策略价值网络 + 精确终局求解器`。

---

## 1. 改造目标

当前项目 AI 的核心结构是：

```text
UCT 搜索
  + 随机模拟 / Filter Monte Carlo
  + 链、环、Double-Cross 等手写后期策略
```

目标结构是：

```text
开局 / 中局：
  AlphaZero-style PUCT 搜索
  + 策略价值网络 Policy/Value Network

后期：
  精确终局求解器
  + 长链、环、死链、死环、Double-Cross、理性状态分析
```

最终希望实现：

```text
gameTurnMove()
  |
  |-- 当前有 C 型格：先强制吃格
  |
  |-- 判断是否进入终局
  |     |
  |     |-- 是：调用精确终局求解器
  |     |
  |     |-- 否：调用 AlphaZeroMove()
  |
  |-- 返回 AI 本回合所有落子步骤
```

---

## 2. 为什么选择 AlphaZero + 终局求解器

纯 AlphaZero 可以通过自对弈学习整个游戏，但对点格棋来说，后期长链、环、Double-Cross 有明确的组合博弈论规律。项目中已经实现了大量后期分析逻辑，因此没必要让神经网络重新学习所有终局规则。

更适合本项目的方式是：

```text
神经网络负责复杂、不易手写规则的前中期判断；
精确终局求解器负责结构清晰、可推导的后期局面。
```

这样有三个优点：

- **稳定性更高**：后期不依赖神经网络猜测，避免在长链/环局面犯低级错误。
- **训练压力更低**：神经网络主要学习前中期策略，不必完整掌握所有终局理论。
- **实战强度更高**：学习能力和确定性求解结合，比单纯 UCT 或纯神经网络都更稳。

---

## 3. 当前代码中的接入点

### 3.1 AI 总入口

当前 AI 总入口位于：

```text
src/AI/UCT.cpp
```

函数：

```cpp
void gameTurnMove(Board &CB, int Player, int *status, vector<LOC> &pace);
```

当前逻辑大致为：

```cpp
Board Test = CB;
Test.eatAllCTypeBoxes(Player);
bool LatterSituation = (Test.getFilterMoveNum() == 0);

if (!LatterSituation)
    UCTMoveWithSacrifice(CB, Player, pace);
else
    latterSituationMove(CB, Player, pace);
```

改造后目标逻辑：

```cpp
Board Test = CB;
Test.eatAllCTypeBoxes(Player);
bool LatterSituation = (Test.getFilterMoveNum() == 0);

if (!LatterSituation)
    AlphaZeroMove(CB, Player, pace);
else
    latterSituationMove(CB, Player, pace);
```

### 3.2 需要保留的后期逻辑

后期求解相关代码主要包括：

| 文件 | 作用 |
|---|---|
| `src/AI/assess.h` / `assess.cpp` | 链、环、死链、死环、理性状态分析 |
| `src/AI/UCT.cpp` | `latterSituationMove()` 后期决策入口 |
| `src/AI/board.cpp` | `eatCBox()`、`eatAllCTypeBoxes()`、`getDoubleCrossLoc()` 等基础操作 |

这些逻辑第一阶段应尽量保留，不要立即重写。

---

## 4. 新增目录结构

建议新增：

```text
src/AI/az/
├── az_types.h          # AlphaZero 基础类型和参数
├── az_action.h
├── az_action.cpp       # 60 动作与 LOC 坐标的双向映射
├── az_encoder.h
├── az_encoder.cpp      # Board -> 神经网络输入特征
├── az_evaluator.h
├── az_evaluator.cpp    # 策略价值评估接口，初期可用启发式假网络
├── az_node.h
├── az_node.cpp         # PUCT 搜索节点
├── az_mcts.h
├── az_mcts.cpp         # PUCT 搜索主流程
├── az_move.h
└── az_move.cpp         # AlphaZeroMove() 对外入口
```

后续训练相关目录：

```text
train/
├── model.py            # PyTorch 网络结构
├── dataset.py          # 数据集读取
├── train.py            # 训练入口
├── export_onnx.py      # 导出 ONNX 模型
└── export_weights.py   # 导出手写推理权重

data/
├── selfplay/           # 自对弈数据
├── supervised/         # 旧 AI 生成的监督数据
└── models/             # 模型文件
```

---

## 5. 动作空间设计

点格棋 5×5 格共有 60 条边：

```text
横边：6 行 × 5 条 = 30
竖边：5 行 × 6 条 = 30
总动作数：60
```

神经网络的策略头输出固定为：

```text
policy[60]
```

建议动作编号规则：

```text
action 0 - 29：横边 HENG
action 30 - 59：竖边 SHU
```

需要实现三个核心函数：

```cpp
int locToAction(LOC loc);
LOC actionToLoc(int action);
bool isLegalAction(const Board& board, int action);
```

设计要求：

- 每条合法边必须映射到唯一 action。
- 每个 action 必须能还原到唯一 `LOC`。
- 已占边必须能通过 legal mask 屏蔽。
- 神经网络输出的非法动作概率需要置零并重新归一化。

---

## 6. 棋盘编码设计

当前棋盘使用：

```cpp
int map[11][11];
```

建议神经网络输入为：

```text
7 × 11 × 11
```

推荐 7 个通道：

| 通道 | 含义 |
|---|---|
| 0 | 当前玩家占据的格子 |
| 1 | 对手占据的格子 |
| 2 | 已被占据的边 |
| 3 | 当前可落子的边 |
| 4 | DOT 点位 |
| 5 | 当前玩家标识，全图填 `1` 或 `-1` |
| 6 | 格子自由度，归一化为 `freedom / 4.0` |

这样网络可以同时看到：

- 棋盘几何结构
- 边的占据情况
- 哪些边还能下
- 哪些格子接近被吃
- 当前轮到哪一方

---

## 7. 策略价值网络设计

### 7.1 输入输出

输入：

```text
7 × 11 × 11
```

输出：

```text
policy[60]：每条边的推荐概率
value：当前玩家视角的局面价值，范围 [-1, 1]
```

`value` 含义：

| value | 含义 |
|---|---|
| `+1` | 当前玩家最终大概率获胜 |
| `0` | 接近均势 |
| `-1` | 当前玩家最终大概率失败 |

### 7.2 推荐第一版网络

由于棋盘很小，第一版网络不宜过大：

```text
Input: 7 × 11 × 11

Conv 3×3, 64 channels
ReLU
ResidualBlock × 4

Policy Head:
  Conv 1×1, 2 channels
  Flatten
  FullyConnected -> 60

Value Head:
  Conv 1×1, 1 channel
  Flatten
  FullyConnected -> 128
  ReLU
  FullyConnected -> 1
  Tanh
```

第一版也可以先使用更简单的 MLP：

```text
Flatten(7×11×11)
FC 256
ReLU
FC 128
ReLU
Policy Head -> 60
Value Head -> 1
```

MLP 更容易手写 C++ 前向传播，适合作为初版验证。

---

## 8. PUCT 搜索设计

### 8.1 节点结构

每个搜索节点建议保存：

```cpp
struct AZNode {
    Board board;
    int player;
    LOC moveFromParent;
    float prior;
    float valueSum;
    int visits;
    std::vector<AZNode*> children;
};
```

其中：

| 字段 | 含义 |
|---|---|
| `board` | 当前局面 |
| `player` | 当前轮到的玩家 |
| `moveFromParent` | 从父节点走到该节点的动作 |
| `prior` | 神经网络给该动作的先验概率 `P` |
| `valueSum` | 累计价值 `W` |
| `visits` | 访问次数 `N` |
| `children` | 子节点列表 |

平均价值：

```text
Q = valueSum / visits
```

### 8.2 PUCT 公式

AlphaZero 使用的选择公式：

```text
score = Q + c_puct × P × sqrt(N_parent) / (1 + N_child)
```

参数建议：

```text
c_puct = 1.5 ~ 2.5
```

各项含义：

| 项 | 作用 |
|---|---|
| `Q` | 当前统计上这步棋有多好 |
| `P` | 神经网络认为这步棋有多好 |
| `sqrt(N_parent) / (1 + N_child)` | 鼓励探索访问较少的节点 |
| `c_puct` | 控制探索强度 |

### 8.3 一次搜索迭代流程

```text
1. Selection
   从根节点出发，根据 PUCT 分数一路选择子节点。

2. Evaluation
   到达叶节点后：
     如果局面已经进入后期或终局，调用精确终局求解器得到 value。
     否则，调用策略价值网络得到 policy 和 value。

3. Expansion
   根据 policy 扩展所有合法动作。

4. Backup
   把 value 沿路径回传，更新 visits 和 valueSum。
```

---

## 9. 点格棋连续行动规则处理

点格棋有特殊规则：如果当前玩家通过一条边完成了格子，则当前玩家继续行动；否则切换玩家。

状态转移逻辑：

```cpp
int earned = board.move(player, loc);
int nextPlayer = (earned > 0) ? player : -player;
```

不过当前项目已有一种抽象：AI 回合开始时先使用 `eatAllCTypeBoxes()` 把所有强制吃格走完，然后再选择一条非强制边。

因此建议分两阶段处理：

### 第一阶段：兼容当前抽象

```text
AlphaZeroMove() 开始时：
  先吃掉所有 C 型格；
  PUCT 只负责选择一个非强制边。
```

优点：

- 改动小
- 容易接入现有 UI 和 `pace` 记录
- 不容易破坏当前游戏流程

### 第二阶段：完整规则搜索

```text
PUCT 内部完整处理连续行动：
  如果吃格，则不换手；
  如果没吃格，则换手。
```

优点：

- 更接近真实点格棋规则
- 训练数据更自然
- 搜索上限更高

建议先实现第一阶段，验证稳定后再升级第二阶段。

---

## 10. 终局求解器接入方式

现有后期判断方式：

```cpp
Board Test = CB;
Test.eatAllCTypeBoxes(Player);
bool LatterSituation = (Test.getFilterMoveNum() == 0);
```

在 PUCT 叶节点评估时也应加入类似判断：

```text
如果局面进入后期：
  调用 BoxBoard::getBoardWinner() 或现有后期逻辑得到精确结果；
否则：
  调用神经网络评估。
```

value 转换方式：

```text
winner == 当前节点 player      -> value = +1
winner == 当前节点 opponent    -> value = -1
```

更平滑的方式是使用比分差：

```text
value = 当前玩家最终格子数差 / 25.0
```

例如当前玩家最终 15:10 获胜：

```text
value = (15 - 10) / 25 = 0.2
```

---

## 11. 训练数据格式

建议每条训练样本保存为 JSONL：

```json
{
  "board": "encoded_or_raw_board_state",
  "player": 1,
  "legal_mask": [1, 0, 1, 1, 0],
  "policy": [0.0, 0.0, 0.32],
  "value": 1.0
}
```

实际 `legal_mask` 和 `policy` 长度都应为 60。

字段含义：

| 字段 | 含义 |
|---|---|
| `board` | 当前棋盘状态 |
| `player` | 当前玩家，`BLACK = 1`，`WHITE = -1` |
| `legal_mask` | 60 维合法动作掩码 |
| `policy` | PUCT 根节点访问次数分布 |
| `value` | 最终胜负，当前玩家视角 |

---

## 12. 训练路线

### 12.1 阶段 A：监督预训练

先用当前 AI 或启发式 PUCT 自对弈生成数据。

流程：

```text
1. 当前 AI 自对弈若干盘。
2. 记录每个决策局面。
3. 记录当前 AI 最终选择的动作。
4. 记录最终胜负。
5. 训练网络模仿当前 AI。
```

policy target 可以先用 one-hot：

```text
AI 选择的动作概率为 1，其余为 0。
```

value target：

```text
当前玩家最终赢：+1
当前玩家最终输：-1
```

目标：

```text
让神经网络先达到接近当前 AI 的基础水平。
```

### 12.2 阶段 B：AlphaZero 自对弈训练

真正进入 AlphaZero 循环：

```text
1. 用当前网络 + PUCT 自对弈。
2. 每个局面记录：
   board
   player
   policy = 根节点子节点访问次数分布
   value = 最终胜负
3. 用这些数据训练新网络。
4. 新旧网络对战。
5. 如果新网络胜率超过阈值，则替换旧网络。
```

推荐替换标准：

```text
新网络 vs 旧网络，100 盘胜率 > 55%。
```

---

## 13. 损失函数

AlphaZero 训练目标：

```text
loss = value_loss + policy_loss + l2_regularization
```

具体形式：

```text
value_loss = (z - v)^2
policy_loss = -Σ π(a) log p(a)
```

含义：

| 符号 | 含义 |
|---|---|
| `z` | 最终真实胜负 |
| `v` | 网络预测价值 |
| `π` | PUCT 搜索得到的访问次数分布 |
| `p` | 网络输出策略概率 |

---

## 14. C++ 推理方案

### 14.1 第一阶段：启发式 Evaluator

先不接神经网络，而是写一个启发式 evaluator：

```text
policy：
  能立即得格子的边分数最高；
  安全边次之；
  不制造 C 型格的边次之；
  危险边最低。

value：
  当前分数差 + 安全边数量 + 链风险估计。
```

目的：

- 验证 PUCT 框架是否正确
- 验证动作映射是否正确
- 验证终局切换是否正确
- 验证 UI 是否能正常运行

### 14.2 第二阶段：手写小网络推理

训练 MLP 或小 CNN 后，将权重导出为文本或二进制文件，C++ 中手写前向传播。

优点：

- 无额外依赖
- 部署简单
- 推理速度快
- 适合小棋盘模型

### 14.3 第三阶段：ONNX Runtime

模型稳定后可切换到：

```text
PyTorch -> ONNX -> C++ ONNX Runtime
```

优点：

- 支持复杂网络
- 方便迭代模型结构
- 工程标准化程度更高

缺点：

- 需要引入新依赖
- CMake 配置更复杂

---

## 15. 分阶段实施计划

### 第 0 阶段：整理当前 AI 接口

目标：

```text
明确 gameTurnMove()、UCTMove()、latterSituationMove() 的调用关系。
```

验收标准：

- 项目仍可正常编译运行。
- 当前 AI 行为不变。

### 第 1 阶段：动作映射和棋盘编码器

目标：

```text
实现 60 动作映射和 Board -> Tensor 编码。
```

新增模块：

```text
az_action
az_encoder
```

验收标准：

- 所有合法边都能映射到唯一 action。
- action 能正确还原为 `LOC`。
- 非法边能被 mask 掉。
- 编码结果维度正确。

### 第 2 阶段：PUCT 框架

目标：

```text
实现 AlphaZero-style PUCT，初期 evaluator 使用启发式假网络。
```

新增模块：

```text
az_node
az_mcts
az_evaluator
```

推荐搜索参数：

| 参数 | 初始值 |
|---|---|
| `AZ_SIMULATIONS` | 800 |
| `AZ_C_PUCT` | 1.5 |
| `AZ_TIME_LIMIT_MS` | 1000 |
| `AZ_TEMPERATURE` | 实战 0.0，自对弈前期 1.0 |

验收标准：

- AI 能选择合法边。
- 不会选择已占边。
- 不会崩溃。
- 能正确进入后期求解器。

### 第 3 阶段：接入当前游戏入口

目标：

```text
让前中期从 UCTMoveWithSacrifice() 切换到 AlphaZeroMove()。
```

建议保留开关：

```cpp
constexpr bool USE_ALPHAZERO_AI = true;
```

这样可以随时回退：

```text
USE_ALPHAZERO_AI = false -> 使用旧 UCT
USE_ALPHAZERO_AI = true  -> 使用新 PUCT
```

验收标准：

- 人机对战正常。
- AI vs AI 正常。
- 旧 AI 可以通过开关恢复。

### 第 4 阶段：训练数据导出

目标：

```text
生成可被 Python 读取的训练数据。
```

建议输出格式：

```text
data/selfplay/*.jsonl
```

每条样本记录：

- 棋盘状态
- 当前玩家
- 合法动作 mask
- 搜索访问次数分布
- 最终胜负

验收标准：

- 能自动生成若干盘棋。
- 数据格式可被 Python 读取。
- 每条样本 policy 长度为 60。

### 第 5 阶段：PyTorch 训练

目标：

```text
训练第一个策略价值网络。
```

新增：

```text
train/model.py
train/dataset.py
train/train.py
```

验收标准：

- 训练 loss 稳定下降。
- policy accuracy 有明显提升。
- value accuracy 能判断胜负趋势。
- 能导出权重或 ONNX。

### 第 6 阶段：C++ 接入网络推理

目标：

```text
az_evaluator 从启发式版本切换为神经网络版本。
```

统一接口：

```cpp
NetworkOutput evaluate(const Board& board, int player);
```

验收标准：

- C++ 能加载模型。
- 同一局面 C++ 输出与 Python 输出一致。
- AI 能正常对弈。
- 推理速度可接受。

### 第 7 阶段：AlphaZero 自对弈迭代

目标：

```text
进入真正强化学习循环。
```

推荐初始参数：

| 项 | 初始值 |
|---|---|
| 每轮自对弈 | 1000 盘 |
| 每步模拟次数 | 400 - 800 |
| batch size | 256 |
| replay buffer | 最近 50k - 200k 样本 |
| 新旧模型评估 | 100 盘 |
| 替换阈值 | 胜率 > 55% |

验收标准：

- 新模型能稳定超过旧模型。
- 对旧 UCT AI 胜率持续上升。
- 中局决策质量明显提高。

### 第 8 阶段：增强终局求解器

目标：

```text
进一步提升后期确定性强度。
```

可选优化：

- Alpha-Beta 精确搜索
- 置换表缓存
- Bitboard 局面表示
- 链/环分析结果缓存
- Endgame tablebase

验收标准：

- 后期不依赖网络猜测。
- Double-Cross 决策稳定。
- 复杂链/环局面不犯明显错误。

---

## 16. 强度评估方案

建议建立固定对战评估：

```text
NewAI vs OldAI，1000 盘
双方轮流先手
统计胜率、平均得分、平均耗时
```

阶段性目标：

| 阶段 | 目标胜率 |
|---|---|
| PUCT + 启发式 evaluator | > 55% |
| 监督预训练网络 | > 60% |
| 自对弈 5 轮后 | > 70% |
| 加强终局后 | > 75% |

性能目标：

| 指标 | 目标 |
|---|---|
| 单步决策时间 | 约 1 秒 |
| C++ 单次网络推理 | < 1 ms |
| 每步 PUCT 模拟次数 | 400 - 1600 |
| 内存占用 | 不持续增长 |

---

## 17. 主要风险与解决方案

### 风险 1：训练数据质量不足

问题：

```text
如果一开始只模仿当前 AI，网络上限可能受旧 AI 限制。
```

解决方案：

- 先监督预训练，获得基础能力。
- 再通过 AlphaZero 自对弈强化学习逐步超过旧 AI。
- 保留终局求解器，避免网络学坏后期。

### 风险 2：连续行动规则处理错误

问题：

```text
点格棋吃格后不换手，和普通棋类不同。
```

解决方案：

- 第一阶段沿用 `eatAllCTypeBoxes()` 抽象。
- 第二阶段再把完整连续行动规则纳入 PUCT。

### 风险 3：C++ 推理依赖复杂

问题：

```text
ONNX Runtime 或 LibTorch 可能带来配置复杂度。
```

解决方案：

- 先使用启发式 evaluator。
- 再实现手写小网络前向传播。
- 最后再升级到 ONNX Runtime。

### 风险 4：搜索速度不足

问题：

```text
当前 Board 使用 11×11 int 数组，频繁复制较慢。
```

解决方案：

- 短期控制模拟次数。
- 中期使用节点内存池，减少 `new/delete`。
- 长期改为 Bitboard 表示。

---

## 18. 最推荐的落地顺序

建议按以下顺序实施：

```text
1. 新增 az_action 和 az_encoder。
2. 新增启发式 az_evaluator。
3. 新增 az_node 和 az_mcts。
4. 新增 AlphaZeroMove()。
5. 在 gameTurnMove() 中通过开关接入。
6. 编译运行，验证 AI 能正常下棋。
7. 生成训练数据。
8. 训练 PyTorch 策略价值网络。
9. C++ 接入神经网络推理。
10. 开始 AlphaZero 自对弈迭代。
11. 增强终局求解器。
```

不要一开始就训练模型。第一步应先把 PUCT 框架接入当前项目，并使用启发式 evaluator 跑通。

---

## 19. 最终版本目标

最终 AI 决策流程：

```text
AI 回合开始
  |
  |-- 强制吃掉所有 C 型格
  |
  |-- 判断是否进入终局
  |     |
  |     |-- 是：
  |     |     使用链/环/Double-Cross/AlphaBeta 终局求解
  |     |
  |     |-- 否：
  |           使用 AlphaZero PUCT
  |             |
  |             |-- 神经网络输出 policy/value
  |             |-- PUCT 搜索 N 次
  |             |-- 选择访问次数最多的动作
  |
  |-- 执行动作并记录 pace
  |
  |-- 返回 UI
```

---

## 20. 总结

本项目最推荐的 AI 升级路线不是纯 AlphaZero，而是：

```text
AlphaZero-style PUCT + 策略价值网络 + 精确终局求解器
```

这条路线兼具：

- AlphaZero 的自学习和强中局判断能力
- 现有后期链/环求解器的确定性与稳定性
- 可分阶段落地的工程可控性
- 较强的项目展示价值

第一步应优先完成：

```text
PUCT 框架 + 启发式 evaluator + 原有终局求解器接入
```

之后再逐步进入神经网络训练、自对弈迭代和终局求解器增强。
