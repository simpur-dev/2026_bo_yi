# Dots and Boxes — Hybrid Expert + AlphaZero

> **世界最强点格棋 AI 项目**。采用 **Hybrid Expert + AlphaZero** 架构：精确组合博弈论规则处理可证明部分，神经网络 + MCTS 处理开局/中盘/未定结构。基于 BoxesZero (Entropy 2025) 论文的反向训练方法，结合自主扩展的 1-chain/2-chain 终局定理。

## 目录

- [项目简介](#项目简介)
- [当前状态](#当前状态)
- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [编译与运行](#编译与运行)
- [训练流程](#训练流程)
- [双模型架构](#双模型架构)
- [AI 算法详解](#ai-算法详解)
- [核心类与接口说明](#核心类与接口说明)
- [游戏操作说明](#游戏操作说明)
- [棋盘数据结构](#棋盘数据结构)

---

## 项目简介

点格棋（Dots and Boxes）是一个经典的两人回合制策略游戏：在 6×6 的点阵上，双方轮流连接相邻两点画一条边，当某条边使一个 1×1 格子的四条边全部被画出时，该格子归属当前玩家，并获得一次额外行动。所有 25 个格子被占满后，占格多的一方获胜。

本项目实现了：
- 基于 SFML 的图形化棋盘界面
- 人 vs 人、人 vs AI、AI vs AI 三种对局模式
- **Hybrid Expert + AlphaZero** 架构 AI
- **ONNX Runtime** 神经网络推理
- **反向训练 (Backward Training)** 自动化流水线
- Arena 评估工具 + 单元测试

## 当前状态

| 指标 | 值 |
|---|---|
| 最强模型 | resnet_m (1.83M) — iter22 内部 arena 75% |
| vs heuristic | 72% (ResNet-S 天花板, 待双模型突破) |
| 白方 (后手) | 100% vs iter18 基线 |
| 训练数据 | 41K+ 样本, 22 轮反向训练迭代 |
| 架构 | Hybrid Expert v1 + 1-chain/2-chain + 双模型 (black/white) |
| GPU 推理 | RTX 4070, ONNX CUDA, 3-17× 加速 |

---

## 环境配置

详细环境配置指南见 **[SETUP.md](SETUP.md)**。涵盖：

- GCC / CMake / SFML / ONNX Runtime C++ SDK 安装
- Python PyTorch + ONNX Runtime GPU 配置
- GPU CUDA DLL 自动部署
- 编译与验证步骤

---

## 项目结构

```
├── CMakeLists.txt              # CMake 构建脚本 (ONNX/非ONNX双模式)
├── README.md
├── .gitignore
├── 进度.md                     # 详细开发进度
├── 3rdparty/                   # 第三方库
│   ├── SFML_new/               #   SFML 2.6.2 (GCC 15.2 编译)
│   └── onnxruntime/            #   ONNX Runtime C++ SDK
├── res/                        # 资源文件
├── src/                        # C++ 源代码
│   ├── main.cpp                #   GUI 主循环 (SFML)
│   ├── evaluate_ai_main.cpp    #   Arena 评估工具
│   ├── selfplay_main.cpp       #   前向自对弈 CLI
│   ├── backward_selfplay_main.cpp # 反向自对弈 CLI
│   ├── AI/                     #   AI 核心
│   │   ├── define.h            #     全局常量/枚举/类型
│   │   ├── board.h/cpp         #     棋盘类
│   │   ├── Node.h/cpp          #     UCT 搜索树节点 (旧)
│   │   ├── UCT.h/cpp           #     UCT 搜索 + 终局求解器
│   │   ├── assess.h/cpp        #     链/环分析 + 理性状态计算
│   │   └── az/                 #     AlphaZero 模块
│   │       ├── az_types.h      #       MCTSConfig, SelfPlaySample
│   │       ├── az_action.h/cpp #       动作编解码
│   │       ├── az_encoder.h/cpp#       棋盘特征编码 (7×11×11)
│   │       ├── az_node.h/cpp   #       MCTS 节点
│   │       ├── az_mcts.h/cpp   #       PUCT 搜索 + Filter pruning
│   │       ├── az_move.h/cpp   #       AlphaZeroMove (GUI入口)
│   │       ├── az_selfplay.h/cpp#      自对弈引擎 (forward/backward)
│   │       ├── az_evaluator.h/cpp#     评估器接口 (Heuristic/MLP/ONNX)
│   │       ├── az_onnx_evaluator.h/cpp# ONNX Runtime 推理
│   │       └── az_expert.h/cpp #       **统一专家层** (Hybrid Expert核心)
│   ├── element/                #   辅助组件
│   └── CJSON/                  #   棋谱 JSON 导出
├── train/                      # Python 训练管线
│   ├── model.py                #   策略价值网络 (ResNet-S/M/L, MLP)
│   ├── train.py                #   Trainer v2 (AdamW, KL监控, JSON报告)
│   ├── dataset.py              #   JSONL 数据集 + D4 增强
│   ├── augmentation.py         #   D4 对称群 (8种变换)
│   ├── backward_training.py    #   反向训练编排脚本
│   ├── export_onnx.py          #   PyTorch → ONNX 导出
│   └── test_onnx_consistency.py#   ONNX 一致性验证
├── test/                       # C++ 单元测试
│   └── az_tests.cpp            #   27 个 AlphaZero 模块测试
└── scripts/                    # 辅助脚本
    └── setup_onnxruntime.py    #   ONNX Runtime SDK 下载
```

---

## 环境要求

- **操作系统**：Windows
- **编译器**：MinGW-w64 GCC（推荐 11.2+），需要支持 C++17
- **构建工具**：CMake 3.22+
- **图形库**：SFML 2.5+（项目已在 `3rdparty/` 中内置 SFML 2.5.1 的 MSVCRT 版本）

> **重要提示**：项目内置的 SFML DLL 使用 MSVCRT 运行时编译。如果你使用 MSYS2 UCRT64 环境的 GCC，需通过系统包管理器安装 SFML 并在 CMake 中指定路径（详见下方说明）。

---

## 编译与运行

### 方法一：使用 CMake（推荐）

首先安装 [CMake](https://cmake.org/download/)（3.22 或以上版本）。

#### 使用 3rdparty 内置 SFML（MSVCRT MinGW）

```bash
cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build
./dots_and_boxes.exe
```

#### 使用 MSYS2 UCRT64 系统 SFML

如果你的编译器是 MSYS2 UCRT64 版本，需要先安装 SFML：

```bash
pacman -S mingw-w64-ucrt-x86_64-sfml
```

然后通过 `-DSFML_DIR` 指向系统安装的 SFML：

```bash
cmake -G "MinGW Makefiles" -B build -DCMAKE_BUILD_TYPE=Release -DSFML_DIR="<MSYS2安装路径>/ucrt64/lib/cmake/SFML"
cmake --build build
cd build
./dots_and_boxes.exe
```

#### CLion

1. 打开 CLion，选择从 VCS 导入项目
2. 等待加载，弹出配置 CMake 的窗口后直接点确定即可

> CLion 自带 CMake 和 MinGW，可以开箱即用。

#### VS Code

1. 安装 [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) 插件
2. 重启 VS Code，打开项目文件夹
3. 选择工具包时选择对应的 GCC 版本
4. 点击下方状态栏的运行按钮即可

`.vscode/tasks.json`：
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "cmake",
            "label": "CMake: build",
            "command": "build",
            "targets": ["dots_and_boxes"],
            "group": { "kind": "build", "isDefault": true },
            "problemMatcher": [],
            "detail": "CMake template build task"
        }
    ]
}
```

`.vscode/launch.json`：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) 启动",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/dots_and_boxes.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build/",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                { "description": "启用整齐打印", "text": "-enable-pretty-printing", "ignoreFailures": true },
                { "description": "设置 Intel 反汇编风格", "text": "-gdb-set disassembly-flavor intel", "ignoreFailures": true }
            ],
            "preLaunchTask": "CMake: build"
        }
    ]
}
```

### 方法二：命令行直接编译（不使用 CMake）

```bash
mkdir build
# 复制 DLL 和资源
copy 3rdparty\SFML\bin\*.dll build\
xcopy res build\res\ /E
# 编译
g++ src/main.cpp src/AI/*.cpp src/element/*.cpp src/CJSON/*.cpp src/CJSON/*.c -o build/dots_and_boxes.exe -I ./3rdparty/SFML/include -L./3rdparty/SFML/lib -lsfml-graphics -lsfml-window -lsfml-system -std=c++17
# 运行
cd build
./dots_and_boxes.exe
```

---

## 训练流程

### 反向训练 (Backward Training)

基于 BoxesZero 论文 (Entropy 2025, 中国计算机博弈大赛冠军)：

```bash
cd train
# 阶段 1: Value Network Reinforcement (从终局向开局推进)
python backward_training.py --arch resnet_s --simulations 800 --start_st 50 --beta 5

# 阶段 2: Policy Network Reinforcement (全程 MCTS)
python backward_training.py --arch resnet_s --simulations 800 --max_iterations 3

# 使用更大模型
python backward_training.py --arch resnet_m --simulations 1600 --max_iterations 1
```

### Arena 评估

```bash
# 模型 vs 启发式 100 局基准测试
./build_onnx/evaluate_ai.exe 50 data/models/backward_model.onnx heuristic --sims 800 --temp 4

# 候选模型 vs 当前模型
./build_onnx/evaluate_ai.exe 50 candidate.onnx current.onnx --sims 800 --temp 4
```

### 自对弈数据生成

```bash
# 前向自对弈
./build_onnx/selfplay.exe 100 800 data/selfplay_output data/models/backward_model.onnx

# 反向自对弈 (从第50步开始MCTS)
./build_onnx/backward_selfplay.exe 50 800 50 data/backward_output data/models/backward_model.onnx
```

---

## 游戏操作说明

程序窗口分为左右两部分：

| 区域 | 说明 |
|------|------|
| **左侧（1000×1000）** | 6×6 点阵棋盘，鼠标点击相邻点之间的边进行落子 |
| **右侧信息栏** | 显示先后手分数、用时，以及操作按钮 |

### 右侧按钮功能

| 按钮 | 功能 |
|------|------|
| **玩家 / 机器** | 分别切换先手（红方）和后手（蓝方）为人类玩家或 AI |
| **开始游戏** | 开始对局；对局中变为"结束游戏"用于重置 |
| **undo / redo** | 悔棋 / 重做（会暂停计时） |
| **打印棋盘** | 将当前棋谱导出为 JSON 格式 |
| **加载棋局** | 加载已保存的棋局（预留功能） |

### 配置文件

在程序运行目录下创建 `config.ini`，写入两行文本分别作为先手和后手的名字：

```
先手名字
后手名字
```

---

## 棋盘数据结构

### 棋盘表示

棋盘使用 `int map[11][11]` 二维数组存储，映射到 6×6 的点阵（6 个点 × 5 条边）：

```
坐标 (i, j) 的奇偶性决定元素类型：
  i偶 j偶 → 点（DOT = 10）
  i偶 j奇 → 横线（HENG = 20，被占后变为 OCCLINE = 40）
  i奇 j偶 → 竖线（SHU = 30，被占后变为 OCCLINE = 40）
  i奇 j奇 → 格子（BOX = 0，被先手占为 BLACK = 1，后手占为 WHITE = -1）
```

**可视化示意（5×5 格子的棋盘，对应 11×11 数组）**：

```
  DOT ─ HENG ─ DOT ─ HENG ─ DOT ─ HENG ─ ...
   |           |           |           |
  SHU   BOX  SHU   BOX  SHU   BOX   SHU
   |           |           |           |
  DOT ─ HENG ─ DOT ─ HENG ─ DOT ─ HENG ─ ...
  ...
```

### 格子分类

格子根据周围空边数（自由度，Freedom）分为四类：

| 类型 | 常量 | 自由度 | 含义 |
|------|------|--------|------|
| 满格 FULLBOX | 0 | 0 | 四边全部被占，已归属某玩家 |
| 死格 DEADBOX | 1 | 1 | 仅剩一条空边，下一手必被占（即 C型格） |
| 链格 CHAINBOX | 2 | 2 | 剩余两条空边，构成链或环的中间节点 |
| 自由格 FREEBOX | 3 或 4 | 3~4 | 空边充足，不会被立即占领 |

### 坐标类型

```cpp
using LOC = std::pair<int, int>;  // (行, 列)，基于 11×11 棋盘数组
```

### 常用判断

```cpp
// 判断一条边是否未被占据
bool isFreeLine = (map[i][j] == HENG || map[i][j] == SHU);

// 判断一个格子是否空（未被任何玩家占领）
bool isEmpty = (map[i][j] == BOX);  // BOX == EMPTY == 0
```

---

## 双模型架构

针对 5×5 点格棋先手（黑方）系统性劣势，**黑白双方各自拥有独立的神经网络模型**。

```
selfplay: black_model (BLACK) vs white_model (WHITE)
training: black_model ← BLACK 视角样本, white_model ← WHITE 视角样本
arena:    black arena (fixed-color) + white arena (fixed-color)独立晋升
```

| 组件 | 单模型模式 | 双模型模式 |
|---|---|---|
| 模型文件 | `backward_model.onnx` | `backward_model_black.onnx` + `backward_model_white.onnx` |
| selfplay | `backward_selfplay ... model.onnx` | `backward_selfplay ... --black-model b.onnx --white-model w.onnx` |
| arena | `evaluate_ai 50 A B` | `evaluate_ai 25 A B --fixed-color` |
| 训练 | `python backward_training.py` | `python backward_training.py --dual` |

**初始化：** 首次启用 `--dual` 时，自动从当前最佳单模型复制权重到黑白两个模型。

---

## AI 算法详解

AI 采用 **Hybrid Expert + AlphaZero** 架构，核心思想：精确组合博弈论规则处理可证明部分，神经网络 + MCTS 专注处理开局/中盘/未定结构。

### 架构总览

```
gameTurnMove()
 ├── az_expert::applyPreSearch()  ── 确定性专家规则
 │    ├── eatAllCTypeBoxes()      ── C 型格强制吃
 │    ├── isEndgameAfterForcedCaptures() ── 终局检测
 │    └── Dead Chain/Circle       ── Double-Cross 决策
 └── AZMCTS::search()             ── PUCT 搜索 + NN 评估
      ├── getFilteredActions()    ── 安全边过滤
      ├── ONNXEvaluator::evaluate() ── ResNet 推理
      └── expand()                ── 子节点 + az_expert::normalizeToSearch()
```

### 1. 统一专家层 (`az_expert`)

**所有入口点共用同一套确定性规则**，确保训练/部署完全一致：

| API | 语义 | 入口 |
|---|---|---|
| `applyPreSearch()` | 单次专家处理 | GUI `AlphaZeroMove` |
| `normalizeToSearch()` | 循环执行直到需要 NN+MCTS | selfplay, evaluate_ai, MCTS expand |

处理的专家规则（按执行顺序）：
1. **C 型格强制吃**：自由度=1 的格子立即捕获
2. **终局检测**：吃尽 C 型格后若无安全边 → 进入终局求解器
3. **死链/死环 Double-Cross**：计算"全吃 vs 牺牲"的理性状态收益对比

### 2. PUCT 搜索 (`az_mcts`)

采用 AlphaZero 标准 PUCT 公式：

```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

**领域优化**：
- **Filter Pruning**：只扩展安全边（不产生死链），减小分支因子 ~6×
- **Instant Terminal Detection**：扩展子节点时立即标记终局
- **Terminal Propagation Cascade**：全部子节点终局时父节点级联标记

### 3. 神经网络评估 (`model.py`)

输入 7×11×11 棋盘特征张量，输出 policy (60 维) + value (标量)。

| 架构 | 参数量 | 状态 |
|---|---|---|
| resnet_s | 330K | 已确认天花板 (72% vs heuristic) |
| **resnet_m** | **1.83M** | 🔬 实验中 |
| resnet_l | 11.9M | 待数据扩充后启用 |

### 4. 链式结构分析 (`assess`)

| 类型 | 定义 | 牺牲代价 |
|---|---|---|
| **长链** LongChain | ≥3 格链 | 2 格 |
| **短链** ShortChain | 1-2 格链 (BoxesZero 扩展) | 1-2 格 |
| **环** Circle | ≥4 格环 | 4 格 |
| **预备环** PreCircle | 两条长链共享自由格 | 4 格 |
| **死链** DeadChain | 起始于 C 型格的链 | — |
| **死环** DeadCircle | 起始于 C 型格的环 | — |

### 5. 终局求解器 (`latterSituationMove`)

基于 Nim-like 组合博弈论：

```
控制者得分 = 总格子 - 牺牲格子数
牺牲 = Σ环×4 + Σ链×2 - 最后一结构的代价
```

**打开策略 (`openPolicy`)**：
- 无长链 → 开最短环
- 无环 → 开最短长链
- 有环+无 3-链 → 开最短环
- 有 3-链 → 优先开 3-链

---

## 核心类与接口说明

### `Board` (`board.h`)

棋盘基础操作。

| 方法 | 说明 |
|---|---|
| `move(player, loc)` | 落子，返回占得格子数 |
| `eatAllCTypeBoxes(player)` | 贪心吃掉所有 C 型格（自由度=1） |
| `getFilterMoveNum()` | 安全边数量（不产生死链） |
| `getDoubleCrossLoc(player)` | 找到 Double-Cross 目标边 |
| `ifEnd()` / `getWinner()` | 终局/胜负判定 |

### `BoxBoard` (`assess.h`, 继承 `Board`)

高级局面评估。

| 方法 | 说明 |
|---|---|
| `defineAllChains()` | 识别所有链/环/短链结构 |
| `getRationalStateBoxNum()` | 理性状态双方预期得分 |
| `getFilterMoves()` | 安全边列表 + 等价裁剪 |
| `openPolicy()` | 终局打开策略 |

### `az_expert` (`az_expert.h`) — **统一专家层**

| 函数 | 说明 |
|---|---|
| `applyPreSearch(board, player)` | 单次专家处理 → `Decision` (SearchNeeded/GameEnded/TurnEnded/ContinueSamePlayer) |
| `normalizeToSearch(board, player)` | 循环执行直到需要 NN+MCTS |
| `isEndgameAfterForcedCaptures()` | 终局判定 |

### `AZMCTS` (`az_mcts.h`)

| 方法 | 说明 |
|---|---|
| `search(board, player, config)` | PUCT 搜索，返回根节点 |
| `selectAction(root, temperature)` | 按访问分布选择动作 |
| `getVisitDistribution(root)` | 获取 policy 目标 (训练用) |

### `ONNXEvaluator` (`az_onnx_evaluator.h`)

| 方法 | 说明 |
|---|---|
| `evaluate(board)` | ONNX Runtime 推理 → policy_logits + value |
| 输入 | 7×11×11 张量 |
| 输出 | 60 维 policy logits + 1 维 value [-1,1] |

### 全局函数

| 函数 | 说明 |
|---|---|
| `gameTurnMove(CB, Player, status, pace)` | **AI 决策总入口** |
| `latterSituationMove(CB, Player, pace)` | 终局精确求解 |
| `AlphaZeroMove(CB, Player, pace)` | GUI AlphaZero 回合循环 |
| `SelfPlayEngine::playOneGame()` | 前向自对弈（采样根节点） |
| `SelfPlayEngine::playOneGameBackward()` | 反向自对弈（curriculum） |
