# Heap Overflow DotsAndBoxes

> 一个基于 **UCT（Upper Confidence Bounds Applied to Trees）蒙特卡洛树搜索** 的点格棋（Dots and Boxes）AI 对弈程序，使用 C++17 和 SFML 图形库构建。

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [编译与运行](#编译与运行)
- [游戏操作说明](#游戏操作说明)
- [棋盘数据结构](#棋盘数据结构)
- [AI 算法详解](#ai-算法详解)
- [核心类与接口说明](#核心类与接口说明)

---

## 项目简介

点格棋（Dots and Boxes）是一个经典的两人回合制策略游戏：在 6×6 的点阵上，双方轮流连接相邻两点画一条边，当某条边使一个 1×1 格子的四条边全部被画出时，该格子归属当前玩家，并获得一次额外行动。所有 25 个格子被占满后，占格多的一方获胜。

本项目实现了：
- 基于 SFML 的图形化棋盘界面（1500×1000 窗口）
- 人 vs 人、人 vs AI、AI vs AI 三种对局模式
- 基于 **UCT 蒙特卡洛树搜索 + 链式结构分析 + 理性状态博弈论终局求解** 的强力 AI
- 棋局回放（Undo / Redo）与棋谱导出（JSON 格式）功能
- 计时器功能，分别统计双方用时

---

## 项目结构

```
dots_and_boxes/
├── CMakeLists.txt              # CMake 构建脚本
├── README.md                   # 本文档
├── 3rdparty/                   # 第三方库
│   ├── SFML/                   #   SFML 2.5 图形库（头文件、库、DLL）
│   └── nowide/                 #   Boost.Nowide（UTF-8 支持）
├── res/                        # 资源文件
│   └── LXGWWenKai-Bold.ttf    #   霞鹜文楷粗体字体
├── src/                        # 源代码
│   ├── main.cpp                #   程序入口、SFML 窗口主循环、UI 逻辑
│   ├── AI/                     #   AI 核心算法
│   │   ├── define.h            #     全局常量、枚举、类型定义
│   │   ├── board.h / board.cpp #     棋盘类：落子、悔棋、自由度、C型格等基础操作
│   │   ├── Node.h / Node.cpp   #     UCT 搜索树节点类
│   │   ├── UCT.h / UCT.cpp     #     UCT 搜索主流程、前中后期决策调度
│   │   └── assess.h / assess.cpp #   高级局面评估：链/环识别、理性状态计算、等价裁边
│   ├── element/                #   辅助组件
│   │   └── Time.h / Time.cpp   #     计时器
│   └── CJSON/                  #   棋谱记录
│       ├── cJSON.h / cJSON.c   #     cJSON 库
│       └── datarecorder.h / datarecorder.cpp # 棋谱 JSON 生成
└── test/                       # 测试代码
    └── xieyi.cpp               #   协议测试
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

## AI 算法详解

AI 的总体决策框架分为 **前中期（UCT 搜索）** 和 **后期（链式结构博弈论求解）** 两个阶段，入口函数为 `gameTurnMove()`。

### 1. 阶段划分

```
gameTurnMove()
 ├── 判断是否进入后期：吃掉所有 C型格 后，是否还存在 Filter 可行边
 ├── 前中期 → UCTMoveWithSacrifice()  ── 预处理 + UCT 搜索
 └── 后期   → latterSituationMove()   ── 纯链式结构博弈论决策
```

**前中期判定条件**：模拟吃掉所有 C型格（自由度为 1 的格子）后，检查是否还存在"过滤可行边"（Filter Move，即不会产生新的死链的安全着法）。若存在，则仍在前中期；否则进入后期。

### 2. UCT 蒙特卡洛树搜索（前中期核心）

UCT（Upper Confidence Bounds Applied to Trees）是 MCTS 的一种变体，核心流程：

#### 2.1 搜索流程 `UCTProcess()`

```
每次迭代：
  1. 选择（Selection）
     └─ 从根节点出发，每层用 UCB 公式选择最优子节点，直至：
        (a) 遇到未完全扩展的节点 → 进入扩展
        (b) 遇到游戏结束的叶节点 → 直接回传胜负

  2. 扩展（Expansion）
     └─ 创建一个新的子节点，对应一个尚未尝试的着法

  3. 模拟（Simulation）
     └─ 对新节点执行一次蒙特卡洛模拟（FilterMC），随机下完整盘并返回胜负结果

  4. 反向传播（Backpropagation）
     └─ 将模拟结果沿路径向上更新每个节点的平均收益 AvgValue
```

#### 2.2 UCB 公式

节点选择采用 UCB1 公式平衡 **利用（exploitation）** 与 **探索（exploration）**：

```
UCB(i) = AvgValue(i) + sqrt( 2 × ln(Total) / Times(i) )
```

- `AvgValue(i)`：子节点 i 的平均收益（对父节点而言）
- `Total`：总迭代次数
- `Times(i)`：子节点 i 被访问的次数
- 探索常数 C = 1（定义在 `define.h`）

**平均收益更新**：父节点收益 = `1 - Σ(子节点收益 × 子节点访问占比)`，体现零和博弈的对抗性。

#### 2.3 蒙特卡洛模拟 `getFilterMCWinner()`

模拟阶段并非完全随机，而是采用 **过滤随机策略**：

1. 每步先贪心吃掉所有 C型格（自由度为 1 的必得格子）
2. 当自由边数量 < 30 时，仅在 **Filter Move**（不会产生死链的安全着法）中随机选择
3. 当自由边数量 ≥ 30 时，在所有自由边中随机选择（早期过滤代价过高）
4. 模拟直到进入后期，然后用博弈论终局求解器判定胜负

#### 2.4 关键参数

| 参数 | 值 | 含义 |
|------|----|------|
| `UCT_TIMES` | 15,000,000 | UCT 最大迭代次数 |
| `UCT_LIMIT_TIME` | 1 秒 | UCT 搜索时间上限 |
| `UCT_MC_TIMES` | 1 | 每次扩展的蒙特卡洛模拟次数 |
| `UCT_FILTER_RANGE` | 30 | 自由边少于此值时启用 Filter 过滤 |

### 3. 等价裁边（搜索空间优化）

为降低搜索分支因子，AI 在生成可行着法时执行两种裁边：

#### 3.1 死链过滤 `getFilterMoves()`

对每条候选边模拟落子后，检查是否会产生"长死格链"（即一个 C型格通过空边连接到一个链格，形成对手可连续吃掉的结构）。会产生死链的着法被过滤掉。

#### 3.2 等价边裁剪

- **一格短链裁剪** `getSaveChainEdgeBool()`：当一个格子构成仅含 1 个格子的短链时，其多条空边在策略上等价，只保留一条代表。
- **角落格裁剪** `getSaveAngleEdgeBool()`：棋盘四角的自由格，其两条外边界边等价，只保留一条。

### 4. 链式结构分析（后期核心）

当棋局只剩下长链和环时，AI 基于 **组合博弈论的链式结构理论** 进行精确决策。

#### 4.1 结构类型

| 类型 | 枚举值 | 定义 |
|------|--------|------|
| **短链** ShortChain | 长度 ≤ 2 的链 | 由一端自由格出发，经过 1~2 个链格到达另一端 |
| **长链** LongChain | 长度 ≥ 3 的链 | 同上但经过 ≥ 3 个链格 |
| **环** Circle | 首尾相连的链格环 | 搜索回到起点形成闭合 |
| **预备环** PreCircle | 两条长链共享首尾自由格 | 等价于一个大环 |
| **死链** DeadChain | 起始于 C型格的链 | 已打开、对手可连续吃 |
| **死环** DeadCircle | 起始于 C型格的环 | 已打开的环形死结构 |

#### 4.2 链的识别流程 `defineAllChains()`

1. **分类所有格子**：根据自由度将 5×5 格子标记为满格、死格、链格、自由格
2. **从自由格出发搜索链**：沿空边方向，逐格追踪链格序列，直到遇到非链格节点
3. **从边界出发搜索**：处理一端连接棋盘边界的链
4. **识别环**：对未被归属的链格，尝试追踪闭合路径
5. **ChainPlus 模式**：合并共享端点的长链为预备环，处理自由格连接的多条链

#### 4.3 理性状态求解 `getRationalStateBoxNum()`

基于组合博弈论，假设双方都采取最优策略：

- **控制权**：后走方（Latter Player）拥有控制权，决定打开哪条链
- **牺牲策略**：通过 Double-Cross 手法牺牲少量格子来维持控制权
- **牺牲代价**：打开一条长链牺牲 2 格，打开一个环牺牲 4 格

```
控制者得到的格子 = 总格子 - 牺牲格子数
牺牲格子数 = 环数 × 4 + 长链数 × 2 - 最后一个结构的代价
```

判断是否应该放弃控制（全吃）还是牺牲保持控制，取决于：
> 若 `控制者剩余格子 - 打开者获得格子 ≤ 牺牲代价`，则全吃更优。

#### 4.4 后期决策流程 `latterSituationMove()`

```
后期决策：
  ├── 存在已打开的长死链？
  │    ├── 计算全吃 vs 牺牲的收益对比
  │    ├── 全吃更优 → eatAllCTypeBoxes() 全部吃掉
  │    └── 牺牲更优 → 贪心吃到临界点 → Double-Cross → 限制性吃格
  └── 无已打开的链
       └── openPolicy() 选择打开哪条链/环：
           ├── 无长链 → 打开最短的环
           ├── 无环   → 打开最短的长链
           ├── 有环但无3-链 → 打开最短的环
           └── 有3-链 → 优先打开3格长链
```

#### 4.5 Double-Cross 手法

当决定牺牲时，AI 使用 Double-Cross（双交叉）技巧：

1. 贪心吃掉 C型格直到即将消灭当前死链/死环
2. 找到死链末端链格的一条空边（非公共边）
3. 落子该边，故意留下 2 个格子（长链）或 4 个格子（环）给对方
4. 对方吃完后被迫打开下一条链，己方重新获得控制权

### 5. 前中期预处理 `UCTMoveWithSacrifice()`

在执行 UCT 搜索之前，先检查是否存在死链/死环：

1. 若存在 → 评估是否需要牺牲来保持控制
2. 牺牲更优时 → 执行 Double-Cross 手法后结束本轮
3. 全吃更优 / 无死结构 → 吃掉所有 C型格后进入 UCT 搜索

---

## 核心类与接口说明

### `Board` 类（`board.h`）

棋盘基础操作类。

| 方法 | 说明 |
|------|------|
| `move(player, loc)` | 在指定位置落子，返回占得的格子数 |
| `unmove(loc)` | 撤销落子（用于搜索回溯） |
| `getFreedom(x, y)` | 获取格子 `(x,y)` 的自由度（周围空边数） |
| `ifEnd()` | 判断游戏是否结束（25 格全满） |
| `getWinner()` | 返回当前胜者（BLACK / WHITE / EMPTY） |
| `eatCBox(player)` | 吃掉一个 C型格，返回所占边的坐标 |
| `eatAllCTypeBoxes(player, pace)` | 贪心吃掉所有 C型格 |
| `ifMakeCBox(loc)` | 判断在 loc 落子是否会制造新的 C型格 |
| `ifHaveSafeEdge()` | 是否存在不制造 C型格的安全着法 |
| `getDoubleCrossLoc(player)` | 找到执行 Double-Cross 的目标边坐标 |

### `Node` 类（`Node.h`，继承自 `Board`）

UCT 搜索树节点。

| 成员 | 说明 |
|------|------|
| `Owner` | 节点所属玩家 |
| `Times` | 被访问次数 |
| `AvgValue` | 平均收益（对父节点而言） |
| `BoardWinner` | 该局面的预测胜者（0 表示未结束） |
| `NodeMoves[60]` | 所有候选着法 |
| `ChildNodes[60]` | 子节点指针数组 |
| `expandUCTNode()` | 扩展一个新的子节点并进行初始 MC 评估 |
| `getUCBValue(Total)` | 计算 UCB 值用于节点选择 |
| `refreshAvgValue()` | 根据子节点更新平均收益 |

### `BoxBoard` 类（`assess.h`，继承自 `Board`）

高级局面评估类，用于链式结构分析和后期决策。

| 方法 | 说明 |
|------|------|
| `defineAllChains(ChainPlus)` | 识别棋盘上所有链和环结构 |
| `defineDeadChain()` | 识别所有死链和死环 |
| `getRationalStateBoxNum()` | 计算理性状态下双方可获得的格子数 |
| `getBoardWinner(LatterPlayer)` | 预测当前局面最终胜者 |
| `getFilterMoves(Moves)` | 获取所有过滤后的安全可行边 |
| `openPolicy()` | 后期打开策略（选择打开哪条链/环） |
| `captualShortestChain(player)` | 模拟吞并最短的长链/环 |

### 全局函数（`UCT.h`）

| 函数 | 说明 |
|------|------|
| `gameTurnMove(CB, Player, status, pace)` | **AI 决策总入口**，自动判断前后期并执行对应策略 |
| `UCTMove(CB, Player, pace)` | 执行 UCT 搜索并落子 |
| `UCTMoveWithSacrifice(CB, Player, pace)` | 带牺牲预处理的 UCT 决策 |
| `latterSituationMove(CB, Player, pace)` | 后期纯链式结构决策 |
| `UCTProcess(Node, Total)` | UCT 单次迭代（选择→扩展→模拟→回传） |
| `rndFilterTurn(CB, Player)` | 模拟中的单步过滤随机着法 |
