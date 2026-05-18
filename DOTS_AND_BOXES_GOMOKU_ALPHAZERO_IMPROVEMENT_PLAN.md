# 点格棋 AlphaZero 长期改造方案：借鉴五子棋国家级项目

> 本方案基于对 `Dots and Boxes` 与 `五子棋` 两套代码的逐文件审阅，目标不是立刻修补单个 bug，而是在比赛时间充足的前提下，给出一条可验证、可回滚、可长期迭代的完整改造路线。
>
> 本文不要求立即实现。后续每一阶段都应单独开任务、单独验证、单独记录实验报告。

---

## 1. 总结结论

当前点格棋项目已经具备 AlphaZero 的雏形：

- **C++ 侧**：已有 `src/AI/az/` 下的动作编码、棋盘编码、PUCT MCTS、评估器接口、自对弈数据生成、手写 MLP 推理。
- **Python 侧**：已有 `train/model.py`、`train/dataset.py`、`train/train.py`、`export_weights.py`、`export_onnx.py`。
- **终局侧**：已有强大的链、环、死链、死环、Double-Cross、理性状态求解逻辑。
- **评估侧**：已有 `evaluate_ai_main.cpp`，可以进行 heuristic 与 neural net 的对战评估。

但是当前训练效果表明，继续做小修小补收益很低。已验证的 `teacher800`、文件级验证、置信度过滤、置信度加权实验都说明：

- **数据量仍然不足**：高置信样本过滤后更容易过拟合。
- **数据分布不够丰富**：点格棋还没有五子棋那样的 8 倍对称增强。
- **MLP 上限偏低**：便于 C++ 部署，但不足以作为长期主力网络。
- **训练闭环不完整**：缺少成熟 replay buffer、KL 监控、完整 checkpoint、模型晋级流程、可视化诊断。
- **自对弈/实战 MCTS 模式未完全分离**：例如根节点 Dirichlet 噪声应只在自对弈训练中启用，不应污染正式评估和实战。

因此，最推荐的长期方向是：

```text
保留点格棋终局求解器
+ 重构 AlphaZero 数据闭环
+ 引入五子棋式成熟训练管线
+ 建立可验证的 CNN 主力模型
+ 通过严格 arena 晋级再部署到 C++
```

---

## 2. 已审阅源码范围

### 2.1 点格棋 C++ 核心

| 文件 | 结论 |
|---|---|
| `src/AI/board.cpp/h` | 游戏状态、落子、撤销、C 型格、自由边、得分、终局判断、Double-Cross 基础函数，是最高优先级稳定边界。 |
| `src/AI/assess.cpp/h` | 链/环/死链/死环/理性状态分析，是点格棋项目区别于普通 AlphaZero 的核心资产。 |
| `src/AI/UCT.cpp/h` | 旧 UCT、随机模拟、终局策略、`gameTurnMove()` 总入口，后续必须通过开关保留回退。 |
| `src/AI/Node.cpp/h` | 旧 UCT 节点，后续仅作为 baseline，不应继续扩展为新框架。 |
| `src/AI/az/az_action.cpp/h` | 60 动作与 `LOC` 双向映射，必须保持 Python/C++ 一致。 |
| `src/AI/az/az_encoder.cpp/h` | 当前 7×11×11 编码，可作为 v1 输入，但需要加入增强/元数据体系。 |
| `src/AI/az/az_node.cpp/h` | PUCT 节点，已考虑点格棋同一玩家连续行动时的视角处理。 |
| `src/AI/az/az_mcts.cpp/h` | PUCT 主循环、扩展、回传、终局检测、访问分布，是后续重点重构对象。 |
| `src/AI/az/az_evaluator.cpp/h` | 启发式评估器与神经网络评估器接口，适合继续保留。 |
| `src/AI/az/az_neural_net.cpp/h` | 手写 MLP 前向推理，适合稳定部署候选，不适合承载长期 CNN 研究。 |
| `src/AI/az/az_selfplay.cpp/h` | 当前 JSONL 自对弈生成器，是训练闭环起点，但需要 schema、增强、并行、日志升级。 |
| `src/AI/az/az_move.cpp/h` | `AlphaZeroMove()` 对外入口，混合 C 型格处理、终局求解器与 PUCT。 |
| `src/selfplay_main.cpp` | 自对弈命令行入口。 |
| `src/evaluate_ai_main.cpp` | AI 对战评估入口，需要改造成正式 arena。 |
| `src/main.cpp` | SFML GUI、AI 异步计算、模型加载入口。 |
| `src/CJSON/datarecorder.cpp/h` | 棋谱记录基础，未来可升级为训练/复盘数据工具。 |
| `CMakeLists.txt` | 当前 C++ 构建边界，未来接 ONNX Runtime 时会改动。 |

### 2.2 点格棋 Python 训练

| 文件 | 结论 |
|---|---|
| `train/model.py` | 已有小型 ResNet CNN 与 MLP。CNN 只是初版，参数化程度不够；MLP 可继续作为 C++ 可部署模型。 |
| `train/dataset.py` | 已支持 JSONL、文件级验证、置信度过滤，但缺少增强、元数据、样本统计、按阶段评估。 |
| `train/train.py` | 已支持 policy temperature、验证集、文件级 split、样本加权、指标输出；仍缺少 AdamW、完整 checkpoint、KL、top-k、配置保存。 |
| `train/export_weights.py` | 可导出 MLP 二进制权重，已有保护正式权重思路，应继续强化。 |
| `train/export_onnx.py` | CNN 长期部署入口，但 C++ 尚未接 ONNX Runtime。 |

### 2.3 五子棋项目

| 文件 | 可迁移价值 |
|---|---|
| `五子棋/model_new.py` | 10 层残差块、256 通道、双头网络、AdamW、state_dict+optimizer 保存、JIT 保存、多分类价值头。 |
| `五子棋/自对弈模板.py` | 大 replay buffer、数据最低门槛、8 倍对称增强、自对弈后训练、KL 监控、日志、阶段模型保存。 |
| `五子棋/监督训练模板.py` | 批量监督训练、测试集监控、可视化记录、KL 计算。 |
| `五子棋/Visualization_error.py` | 损失、KL、解释方差等指标可视化。 |
| `五子棋/UI.py` | GUI 中展示 MCTS 搜索分布、网络价值、AI 用时、棋谱保存、赛制功能。 |
| `五子棋/棋谱/UI.py` | 棋谱保存、加载、复盘 UI，可作为点格棋复盘工具参考。 |

---

## 3. 当前点格棋架构诊断

### 3.1 当前优势

- **终局求解能力强**：`assess.cpp` 已经实现链、环、死链、死环、预备环、理性状态、Double-Cross 等组合博弈逻辑。
- **动作空间小**：5×5 格点格棋只有 60 条边，远小于五子棋 225 动作，适合做高质量搜索与数据增强。
- **C++/Python 边界已打通**：C++ 生成 JSONL，Python 训练，Python 导出权重，C++ 加载 MLP。
- **已有候选保护意识**：训练输出可用 `run_name`，导出权重已有不覆盖正式权重的机制。
- **已有评估工具**：`evaluate_ai_main.cpp` 可以扩展为模型晋级系统。

### 3.2 当前核心短板

- **无数据增强**：五子棋每条数据扩展 8 倍，点格棋目前没有对称增强，直接导致样本利用率低。
- **训练数据 schema 太薄**：只有 board、player、legal_mask、policy、value，缺少 `game_id`、`move_index`、`score`、`winner`、`phase`、`teacher`、`simulations`、`temperature` 等元数据。
- **value 目标过粗**：当前主要是 `+1/-1` 胜负，无法表达 13:12 与 25:0 的差异。点格棋天然适合用比分差。 
- **MCTS 模式不分离**：自对弈需要噪声和温度，正式实战需要确定性；当前 `AZMCTS::search()` 里根节点 Dirichlet 噪声应改成可配置。
- **训练 loop 不够成熟**：缺少完整 checkpoint、optimizer state、KL、top-k、按阶段指标、训练配置落盘。
- **CNN 没有进入闭环**：Python 有 CNN，但 C++ 当前只部署 MLP，长期强度会受限。
- **GUI 缺少 AlphaZero 可解释调试**：五子棋 UI 可展示 MCTS 与网络输出，点格棋 GUI 目前主要是下棋与基础记录。

---

## 4. 五子棋项目最值得迁移的设计

### 4.1 8 倍对称增强

五子棋通过：

```text
4 个旋转 × 每个旋转再左右镜像 = 8 倍数据
```

同时变换：

- 棋盘输入张量
- policy 分布
- value 保持不变

点格棋是正方形 5×5 格棋盘，同样具备 D4 对称群，可以迁移这个思想。但点格棋动作是“边”，不是“点”，因此必须用 `LOC -> transform -> action` 的方式重映射 policy 和 legal mask。

### 4.2 大 replay buffer 与训练最低门槛

五子棋自对弈脚本中：

- `buffer_max = 1000000`
- `data_mini = 70000`
- `batch_size = 256`
- 数据不足不训练

点格棋目前高置信过滤后样本偏少，容易过拟合。应采用类似原则：

```text
数据不足时，不急着训练；
训练必须等 replay buffer 覆盖足够多局面。
```

### 4.3 AdamW + 小学习率

五子棋长期使用：

- `AdamW`
- `lr = 6e-5`
- `weight_decay = 3e-5`

点格棋当前 `train.py` 使用 `Adam`，学习率默认 `1e-3`。这对小数据和高噪声 policy target 可能过激。长期应切到 AdamW，并记录多组学习率实验。

### 4.4 KL 散度监控

五子棋训练会比较训练前后网络输出：

```text
KL(old_policy || new_policy)
```

这对 AlphaZero 很重要：

- KL 太小：学习停滞。
- KL 合理：稳定进步。
- KL 太大：一次训练破坏旧策略，容易退化。

点格棋应把 KL 作为正式训练指标。

### 4.5 state_dict + optimizer + JIT 双保存

五子棋保存：

- `state_dict`：包含模型参数与优化器状态，适合继续训练。
- `torch.jit.script`：适合部署。

点格棋目前主要保存模型 state_dict，缺少 optimizer state 和训练配置。长期应保存完整 checkpoint。

### 4.6 GUI 搜索可视化

五子棋 UI 能显示：

- MCTS 搜索分布
- 网络策略概率
- 网络价值
- AI 用时
- 推荐点排名

点格棋也应加入类似调试层，否则很难判断模型为什么下某条边。

---

## 5. 不可破坏的核心不变量

后续任何重构都必须保护以下不变量。

### 5.1 动作映射不变量

- `AZ_ACTION_SIZE == 60`。
- 每条合法边唯一对应一个 action。
- `actionToLoc(locToAction(loc)) == loc`。
- Python 数据增强后的 policy 必须仍能被 C++ `actionToLoc()` 正确解释。

### 5.2 合法动作 mask 不变量

- 网络 loss 只能在合法动作上计算。
- MCTS expansion 只能扩展合法动作。
- C++ 推理 softmax 必须将非法动作概率置零并重新归一化。
- 数据增强后 `legal_mask` 必须和增强后的棋盘一致。

### 5.3 玩家视角不变量

- board encoding 必须始终以当前玩家为视角。
- value 必须明确是当前玩家视角。
- backup 时如果路径节点玩家与叶节点玩家不同，价值符号必须正确翻转。
- 点格棋吃格后同一玩家继续行动，不能简单按 ply 交替换手。

### 5.4 终局求解器不变量

- `assess.cpp` 的链环逻辑是项目核心资产，不应被神经网络替代。
- AlphaZero 只负责前中期不易手写的判断。
- 进入长链/环终局后，应优先调用确定性求解逻辑。
- 终局求解结果可以反过来生成高质量训练标签。

### 5.5 官方权重保护不变量

- 默认训练输出必须是 candidate，不覆盖 `data/models/weights.bin`。
- 未通过 arena 的模型不得 promote。
- promote 必须显式命令触发。
- promote 前必须保存评估报告。

---

## 6. 目标架构

推荐最终结构：

```text
C++ Game Core
├── Board / Rules / Endgame Solver
├── AlphaZero MCTS
│   ├── self-play mode: Dirichlet noise + temperature
│   └── evaluation mode: deterministic, no noise
├── Evaluator Interface
│   ├── Heuristic baseline
│   ├── MLP deployment
│   └── ONNX CNN deployment
└── Arena / GUI / Record tools

Python Training Core
├── Dataset v2
│   ├── JSONL schema metadata
│   ├── game/file split
│   ├── D4 augmentation
│   └── phase-aware sampling
├── Model Zoo
│   ├── MLP deployable baseline
│   ├── small ResNet CNN
│   └── stronger ResNet CNN
├── Trainer v2
│   ├── AdamW
│   ├── KL monitoring
│   ├── full checkpoint
│   ├── metrics and plots
│   └── candidate promotion gate
└── Experiment Reports
```

---

## 7. 数据系统改造方案

### 7.1 JSONL schema v2

当前每条样本只有：

```json
{
  "board": [],
  "player": 1,
  "legal_mask": [],
  "policy": [],
  "value": 1.0
}
```

建议升级为：

```json
{
  "schema_version": 2,
  "game_id": "selfplay_000123",
  "move_index": 17,
  "decision_index": 9,
  "player": 1,
  "board": [],
  "legal_mask": [],
  "policy": [],
  "value_winloss": 1.0,
  "value_margin": 0.28,
  "black_score_final": 16,
  "white_score_final": 9,
  "winner": 1,
  "phase": "midgame",
  "teacher": "az_puct_800_heuristic",
  "simulations": 800,
  "temperature": 1.0,
  "root_policy_entropy": 2.31,
  "root_policy_confidence": 0.42
}
```

重点新增字段：

- **`game_id`**：用于严格 game-level/file-level split。
- **`move_index`**：真实落子步数。
- **`decision_index`**：AlphaZero 决策点编号，排除强制吃格。
- **`value_margin`**：比分差目标，比单纯胜负更细。
- **`phase`**：opening/midgame/endgame/forced，用于按阶段评估。
- **`teacher`**：记录数据来源，避免混合数据不可追踪。
- **`simulations`**：不同强度 teacher 数据不能混在一起不标记。
- **`root_policy_entropy`**：后续可用于样本加权或过滤。

### 7.2 点格棋 D4 对称增强

必须实现 8 种变换：

```text
identity
rotate90
rotate180
rotate270
flip_horizontal
rotate90 + flip_horizontal
rotate180 + flip_horizontal
rotate270 + flip_horizontal
```

对每条样本同时变换：

- `board`
- `legal_mask`
- `policy`

value 不变。

实现建议：

```text
不要直接猜 action 下标如何旋转。
应使用：
  action -> LOC
  LOC -> transformed LOC
  transformed LOC -> action
```

原因：点格棋横边和竖边在 90 度旋转后会互换，直接 reshape policy 容易错。

验收测试：

- 任意合法 action 经 8 种增强后仍是合法 action。
- `sum(policy)` 增强前后保持为 1。
- `sum(legal_mask)` 增强前后相同。
- 增强 4 次 90 度后回到原 action。
- 所有增强后的样本能被 C++ `isLegalAction()` 验证。

### 7.3 replay buffer 策略

参考五子棋，不应每次只训练最新小数据。建议：

```text
buffer_max: 200k 起步，逐步到 1M
min_train_samples: 70k 起步
batch_size: 256
每轮自对弈后加入 buffer
训练时随机采样
```

点格棋动作空间小，但局面结构复杂。数据量不足时训练会快速记忆 teacher，而不是泛化。

### 7.4 样本过滤与加权

已验证简单 `max(policy)` 置信度加权效果不理想，因此后续不要只依赖单一置信度。

建议顺序：

1. **先做增强**：无增强时讨论过滤意义有限。
2. **再做分层评估**：按 opening/midgame/endgame 看 policy acc。
3. **最后做加权**：尝试 `confidence^2`、`1 - normalized_entropy`、phase-aware weighting。

推荐保留选项：

```text
--min_policy_confidence
--policy_weight_mode none/confidence/entropy/confidence2
--phase_weight opening=1.0 midgame=1.5 endgame=0.5
```

---

## 8. 网络架构改造方案

### 8.1 保留 MLP，但降级为部署 baseline

当前 MLP：

```text
Flatten(7×11×11=847)
-> FC 256
-> FC 128
-> policy 60
-> value 1
```

优点：

- C++ 手写推理已实现。
- 权重小，加载简单。
- 适合快速验证 C++ pipeline。

缺点：

- 空间归纳偏置弱。
- 不擅长学习链、环、局部几何结构。
- 长期强度上限低。

定位：

```text
MLP = 稳定可部署 baseline，不作为长期最强模型。
```

### 8.2 CNN 主力模型：点格棋 ResNet v2

参考五子棋 `model_new.py` 的 10 block、256 channel 思路，但点格棋棋盘更小，可以先从中等模型开始。

推荐三档：

| 模型 | 结构 | 用途 |
|---|---|---|
| `resnet_s` | 4 blocks × 64 channels | 快速调试、CPU 可跑。 |
| `resnet_m` | 6 blocks × 128 channels | 主力候选。 |
| `resnet_l` | 10 blocks × 256 channels | 长期强力模型，对齐五子棋经验。 |

输入仍为：

```text
7 × 11 × 11
```

输出：

```text
policy logits: 60
value head: value_margin 或 win/loss
```

### 8.3 value head 改造

当前 value 使用 `[-1, 1]` 胜负回归。点格棋更适合加上比分差。

推荐方案：

```text
value_margin = (current_player_final_boxes - opponent_final_boxes) / 25.0
```

例如：

```text
16:9 -> +0.28
13:12 -> +0.04
9:16 -> -0.28
```

这比纯 `+1/-1` 更能区分小胜与大胜。

长期可以做双价值头：

```text
value_winloss: 当前玩家胜负概率
value_margin: 当前玩家最终净胜格子比例
```

loss：

```text
loss = policy_loss + value_margin_loss + 0.5 * value_winloss_loss
```

### 8.4 可选辅助头

后续可以考虑辅助任务，但不要第一阶段就做：

- **score head**：预测最终黑白比分。
- **phase head**：预测 opening/midgame/endgame。
- **chain risk head**：预测是否即将进入长链/环结构。

辅助头只应在主流程稳定后加入。

---

## 9. 训练管线改造方案

### 9.1 Trainer v2 必备能力

当前 `train/train.py` 应长期演化为可复现实验脚本，必备能力：

- **配置落盘**：保存 args、模型结构、数据目录、git commit、时间戳。
- **完整 checkpoint**：保存 model、optimizer、scheduler、epoch、best metric。
- **AdamW**：默认优化器从 Adam 切到 AdamW。
- **KL 监控**：训练前后 policy KL。
- **Top-k 指标**：policy top-1、top-3、top-5。
- **value 指标**：MAE、sign acc、margin calibration。
- **文件级/游戏级验证**：默认不用 sample-level split。
- **阶段指标**：opening/midgame/endgame 分开统计。
- **实验报告**：每次训练自动输出到 `data/evaluations/`。

### 9.2 推荐训练默认参数

起步参数：

```text
optimizer: AdamW
lr: 6e-5 或 1e-4
weight_decay: 3e-5 ~ 1e-4
batch_size: 256
epochs: 50-200
scheduler: cosine 或 StepLR
split_mode: file/game
val_split: 0.1
policy_temperature: 1.0 起步，不急着 one-hot
```

### 9.3 policy target 策略

不要再把“训练 policy top-1 准确率”作为唯一指标。AlphaZero 的 policy target 是搜索分布，不一定需要 one-hot。

建议：

- **主训练**：使用 soft visit distribution。
- **诊断训练**：保留 one-hot 实验作为 teacher 清晰度检测。
- **报告同时输出**：CE、top-1、top-3、KL、entropy。

### 9.4 value target 策略

建议逐步替换：

| 阶段 | value target |
|---|---|
| v1 | `+1/-1` 胜负。 |
| v2 | `score_margin / 25`。 |
| v3 | `winloss + margin` 双头。 |

点格棋没有必要只学胜负，因为 13:12 与 20:5 的训练信号差别很大。

### 9.5 训练可视化

参考 `五子棋/Visualization_error.py`，点格棋应输出：

- train/val total loss
- policy loss
- value loss
- policy top-k
- value MAE
- value sign acc
- KL divergence
- policy entropy histogram
- confidence histogram
- phase-wise metrics

图像保存到：

```text
data/evaluations/plots/<run_name>/
```

---

## 10. MCTS 与自对弈改造方案

### 10.1 分离 self-play mode 与 evaluation mode

当前 `AZMCTS::search()` 内部会添加根节点 Dirichlet 噪声。应改成配置项：

```text
self-play:
  Dirichlet noise = on
  temperature = opening/midgame schedule

evaluation/gameplay:
  Dirichlet noise = off
  temperature = 0
```

这是高优先级改造。否则评估和实战会被随机噪声污染。

### 10.2 MCTS 参数配置化

当前很多参数在 `az_types.h` 常量中。长期建议改成结构体：

```text
MCTSConfig
  simulations
  time_limit_ms
  c_puct
  dirichlet_alpha
  dirichlet_frac
  add_root_noise
  temperature
  temperature_moves
  use_endgame_solver
```

这样 selfplay、arena、GUI 可以使用不同配置。

### 10.3 自对弈生成升级

当前 `az_selfplay.cpp` 每 50 盘写一个 JSONL 文件。建议升级：

- 每盘记录 `game_id`。
- 写入 schema v2。
- 记录 self-play 配置。
- 记录每盘最终比分、样本数、耗时。
- 支持多进程并行生成不同 shard。
- 支持从指定权重加载 teacher。
- 支持只生成数据不训练。

### 10.4 终局样本策略

当前自对弈在进入终局求解器后不生成训练样本。这降低了模型学习后期前缘局面的能力。

建议分三类：

- **forced samples**：强制吃格，不训练 policy，只可用于 value。
- **solver samples**：终局求解器可明确给出最佳动作，可作为高质量监督数据。
- **normal samples**：PUCT 搜索样本。

初期可以继续只训练 normal samples，但应在 schema 中标记并可选加入 solver samples。

### 10.5 树复用与性能

长期优化：

- 根节点复用。
- 节点内存池。
- 批量神经网络推理。
- transposition cache。
- 终局求解结果缓存。

这些不应在数据管线稳定前优先做。

---

## 11. C++ 推理与部署路线

### 11.1 第一阶段：继续使用 MLP

短期：

```text
Python MLP -> export_weights.py -> C++ MLPInference
```

用途：

- 保证 C++ pipeline 稳定。
- 保证比赛前有可部署模型。
- 给 CNN 研究留出安全回退。

### 11.2 第二阶段：ONNX Runtime 支持 CNN

长期主力 CNN 不适合手写 C++ 推理。建议：

```text
PyTorch CNN -> ONNX -> C++ ONNX Runtime Evaluator
```

要求：

- 保留 `NeuralNetEvaluator` 旧 MLP。
- 新增 `ONNXEvaluator`。
- 通过 config 选择 evaluator。
- Python 与 C++ 同一局面输出误差必须可测。

验收标准：

```text
max_abs(policy_logits_cpp - policy_logits_py) < 1e-4
abs(value_cpp - value_py) < 1e-4
```

### 11.3 正式模型晋级

模型流转应固定为：

```text
train candidate
-> export candidate
-> arena candidate vs champion
-> generate report
-> pass threshold
-> promote to official
```

禁止：

```text
训练结束直接覆盖 official weights
```

---

## 12. 评估体系改造方案

### 12.1 单元测试

必须先补齐基础测试：

- action bijection test
- legal mask test
- board move/unmove test
- encoder shape test
- augmentation action remap test
- Python/C++ weight consistency test
- MCTS no-illegal-action test
- endgame solver regression test

### 12.2 训练验证

训练报告必须包含：

- train/val split 方式。
- train/val 文件数量。
- train/val 样本数量。
- policy CE。
- policy top-1/top-3。
- value MAE。
- value sign acc。
- KL。
- best epoch。
- 是否建议导出。

### 12.3 Arena 对战

`evaluate_ai_main.cpp` 应升级为正式 arena：

```text
candidate vs champion
candidate vs heuristic
candidate vs old UCT
candidate vs previous best
```

每次评估：

- 交换先后手。
- 固定随机种子。
- 禁止 Dirichlet 噪声。
- 输出胜率、平均比分、平均耗时。
- 输出置信区间。
- 保存完整报告。

推荐晋级门槛：

```text
candidate vs champion >= 55% 胜率
且平均比分不低于 champion
且无超时/崩溃/非法动作
```

---

## 13. GUI 与工具改造方案

参考五子棋 UI，点格棋 GUI 应增加 AlphaZero 调试面板。

### 13.1 搜索可视化

显示：

- MCTS top actions。
- 每条边访问次数比例。
- 网络 policy 概率。
- Q value。
- 最终选择原因。
- 当前 value 评估。

### 13.2 棋谱与复盘

升级现有记录系统：

- 保存完整棋谱。
- 加载历史棋局。
- 每步回放。
- 对某一步重新运行 AI 分析。
- 标记 AI 推荐与实际选择差异。

### 13.3 训练数据调试器

新增工具：

```text
打开一条 JSONL 样本
显示 board
显示 legal_mask
显示 policy top actions
显示 value target
显示增强后的 8 个版本
```

这对防止增强写错非常重要。

---

## 14. 分阶段路线图

### 阶段 0：冻结当前状态与补测试

目标：建立安全网。

任务：

- 保留旧 UCT 开关。
- 为 action、encoder、board、MCTS 加基础测试。
- 修正 self-play/evaluation MCTS 模式分离设计。
- 记录当前 champion 表现。

验收：

- 旧功能不退化。
- 测试能验证动作映射和合法性。
- arena 能稳定跑完。

### 阶段 1：数据 schema v2 + 8 倍增强

目标：解决当前最大短板：数据利用率。

任务：

- 扩展 JSONL schema。
- 实现 D4 action remap。
- `dataset.py` 支持 on-the-fly augmentation。
- 增强前后样本可视化。
- 文件级/game-level split 默认开启。

验收：

- 训练样本有效扩大 8 倍。
- policy/ legal_mask 增强一致。
- 验证指标不因增强错误崩坏。

### 阶段 2：Trainer v2

目标：把训练变成可复现实验。

任务：

- AdamW。
- 完整 checkpoint。
- KL 监控。
- top-k 指标。
- phase-wise metrics。
- 自动报告与 plots。

验收：

- 每个 run 可复现。
- 每个 run 有报告。
- 不再只看 train loss。

### 阶段 3：value target 升级

目标：让网络学会比分质量。

任务：

- selfplay 写入最终比分。
- dataset 支持 `value_margin`。
- model 支持 margin head 或双 value head。
- 对比 win/loss 与 margin。

验收：

- value MAE 明显可解释。
- arena 中不再只追求险胜动作。

### 阶段 4：CNN 主力模型实验

目标：摆脱 MLP 上限。

任务：

- 参数化 `resnet_s/m/l`。
- 训练增强数据。
- 与 MLP 在同数据上对比。
- 输出验证指标与 arena 报告。

验收：

- CNN 在 file-level validation 上优于 MLP。
- CNN 在 arena 中至少不弱于 MLP。

### 阶段 5：自对弈闭环重构

目标：形成真正 AlphaZero 迭代。

任务：

- replay buffer。
- 多 shard 自对弈。
- candidate/champion 管理。
- 每轮自对弈、训练、评估、晋级。

验收：

- 新模型可周期性超过旧模型。
- 所有数据、模型、评估报告可追溯。

### 阶段 6：ONNX CNN 部署

目标：把长期主力模型接入 C++。

任务：

- `export_onnx.py` 固化输入输出名。
- C++ 新增 `ONNXEvaluator`。
- CMake 增加可选 ONNX Runtime。
- Python/C++ 输出一致性测试。

验收：

- GUI 可加载 CNN。
- arena 可评估 CNN。
- MLP 仍可作为回退。

### 阶段 7：GUI/复盘/比赛硬化

目标：比赛前稳定化。

任务：

- AI 分析面板。
- 棋谱复盘。
- 超时保护。
- 崩溃回退。
- 固定 champion。
- 生成比赛版包。

验收：

- 长时间 AI vs AI 无崩溃。
- 无非法动作。
- 决策时间可控。
- 模型与配置可一键加载。

---

## 15. 优先级最高的 10 个任务

1. **MCTS 模式分离**：根节点 Dirichlet 噪声只允许 self-play 使用。
2. **D4 数据增强**：实现 action-level remap，并加测试。
3. **schema v2**：加入 game_id、score、phase、teacher、simulations。
4. **Trainer v2 checkpoint**：保存 optimizer、scheduler、args、best metric。
5. **AdamW + KL**：迁移五子棋训练稳定性经验。
6. **value_margin**：从纯胜负升级到比分差。
7. **resnet_m**：建立点格棋主力 CNN 候选。
8. **arena 晋级系统**：候选模型必须通过对战才能 promote。
9. **ONNXEvaluator 设计**：为 CNN 部署铺路，但不急于替代 MLP。
10. **GUI 搜索可视化**：让 AI 决策可解释、可调试。

---

## 16. 不建议继续投入的方向

### 16.1 单纯调置信度阈值

已经验证：

- 置信度过滤会减少数据量。
- 简单置信度加权不能解决泛化问题。
- 没有增强和 replay buffer 前，继续调阈值收益有限。

### 16.2 继续押注 MLP 最强模型

MLP 适合部署，不适合长期追求强度。应保留但不应作为唯一方向。

### 16.3 直接引入大依赖替换所有 C++ 推理

ONNX Runtime 有价值，但不应在数据和训练不稳定时优先做。否则会把问题从“模型不强”变成“工程更复杂”。

### 16.4 抛弃终局求解器

点格棋终局是当前项目最大优势之一。纯神经网络学习终局规律没有必要，也不稳定。

---

## 17. 成功标准

### 17.1 工程成功标准

- 数据生成、训练、导出、评估、晋级全链路自动化。
- 任一模型都能追溯到训练数据、参数、评估报告。
- 官方权重不会被误覆盖。
- C++ 与 Python 推理一致性可测试。

### 17.2 训练成功标准

- file/game-level validation 稳定提升。
- policy top-3 明显高于随机与 MLP baseline。
- value margin MAE 可解释下降。
- KL 处于合理范围，训练不会突然破坏策略。

### 17.3 对战成功标准

- candidate 能稳定击败旧 champion。
- candidate 对 heuristic/old UCT 胜率持续提高。
- 平均比分优势扩大，而不只是险胜。
- 无非法动作、无超时、无崩溃。

### 17.4 比赛成功标准

- 有稳定 champion 模型。
- 有 MLP 回退模型。
- 有完整评估报告。
- GUI 可解释 AI 决策。
- 对异常局面可复盘。

---

## 18. 推荐的下一步

下一步不要直接训练，也不要直接改 CNN 部署。

最合理的第一个实施任务是：

```text
实现点格棋 D4 数据增强 + action remap 测试 + dataset on-the-fly augmentation
```

原因：

- 当前最大短板是数据不足和泛化差。
- 增强收益大、风险可控。
- 不会破坏 C++ 游戏逻辑。
- 可以立即复用现有 `teacher800` 数据重新验证。

完成后再做：

```text
Trainer v2 + AdamW + KL + checkpoint
```

最后再进入：

```text
CNN 主力模型与 ONNX 部署
```

---

## 19. 最终目标图

```text
                 ┌────────────────────┐
                 │  C++ Board/Rules    │
                 └─────────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │ Endgame Solver           │
              │ chain/circle/doublecross │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │ AlphaZero MCTS           │
              │ selfplay/eval modes      │
              └────────────┬────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │ Evaluator                              │
       │ heuristic / MLP / ONNX CNN             │
       └───────────────────┬───────────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │ JSONL schema v2 + D4 augmentation      │
       └───────────────────┬───────────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │ Trainer v2 + replay buffer + KL        │
       └───────────────────┬───────────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │ Arena evaluation + promotion gate       │
       └───────────────────┬───────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Champion model     │
                 └───────────────────┘
```

---

## 20. 最终结论

点格棋项目不缺 AlphaZero 雏形，也不缺终局棋理。真正缺的是五子棋项目已经具备的工程化训练闭环：

- 大规模 replay buffer
- 8 倍对称增强
- AdamW 小学习率稳定训练
- KL 与可视化监控
- 完整 checkpoint
- 严格模型晋级
- GUI 可解释分析

因此，本项目后续的核心路线应是：

```text
先把数据和训练闭环做成熟，
再把 CNN 做强，
最后把强模型安全部署进 C++。
```

如果只做一个最优先改造：

```text
点格棋 D4 对称增强 + action remap 测试
```

这是当前最可能立刻改善泛化、且不会破坏现有 AI 的改造。
