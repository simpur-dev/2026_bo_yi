#pragma once
#include "../board.h"
#include "az_types.h"
#include <vector>
#include <string>

// ========== PUCT 搜索树节点 ==========
//
// 核心字段说明:
//   board         当前局面（该节点动作已执行，C 型格已吃完）
//   player        该局面下轮到谁走
//   action        从父节点到该节点的动作索引（-1 = 根节点）
//   prior         神经网络给该动作的先验概率 P
//   valueSum      累计价值 W（从 player 视角）
//   visits        访问次数 N
//   expanded      是否已扩展（生成了子节点）
//   terminal      是否为终局/游戏结束节点（精确值已缓存）
//   terminalValue 终局精确价值（从 player 视角，仅 terminal 时有效）
//
// 价值约定:
//   Q = W / N，含义是"从本节点 player 视角的平均收益"
//   +1 表示 player 大概率赢，-1 表示 player 大概率输
//
// PUCT 公式 (从父节点视角选择子节点):
//   score = exploitation + exploration
//
//   exploitation:
//     若 child.player == parent.player → +Q  (同视角)
//     若 child.player != parent.player → -Q  (对立视角)
//
//   exploration = c_puct × P × sqrt(N_parent) / (1 + N_child)
//
//   ★ 关键: 点格棋中吃到格子后同一玩家继续行动，
//     因此父子节点的 player 可能相同，不能像围棋那样一律取 -Q

struct AZNode
{
    Board board;
    int player;
    int action;
    float prior;
    float valueSum;
    int visits;
    bool expanded;
    bool terminal;
    float terminalValue;

    std::vector<AZNode *> children;

    AZNode();
    AZNode(const Board &b, int p, int act, float pr);
    ~AZNode();

    // 平均价值 Q = W / N（从本节点 player 视角）
    // 未被访问时返回 0
    float Q() const;

    // PUCT 选择分数（从父节点视角）
    // parentPlayer: 父节点的 player
    // parentVisits: 父节点的总访问次数
    float puctScore(int parentPlayer, int parentVisits) const;

    // 选择 PUCT 分数最高的子节点
    AZNode *selectChild() const;

    // 是否为叶节点（需要评估/扩展）
    bool isLeaf() const;
};

// ========== 搜索统计 ==========

struct AZSearchStats
{
    int totalSimulations = 0;   // 实际完成的模拟次数
    int elapsedMs = 0;          // 耗时（毫秒）
    int treeNodes = 0;          // 搜索树总节点数
    float rootQ = 0.0f;         // 根节点 Q 值
    int bestAction = -1;        // 最佳动作
    int bestVisits = 0;         // 最佳动作访问次数
    float bestQ = 0.0f;         // 最佳动作 Q 值

    // 输出为可读字符串
    std::string toString() const;
};

// ========== 工具函数 ==========

// 递归释放整棵子树内存（不释放根节点本身）
void deleteAZTree(AZNode *node);

// 递归统计树的总节点数
int countTreeNodes(const AZNode *node);
