#pragma once
#include "../board.h"
#include "az_types.h"
#include <vector>

// PUCT 搜索树节点
struct AZNode
{
    Board board;          // 当前局面（已吃完 C 型格）
    int player;           // 当前轮到的玩家
    int action;           // 从父节点走到该节点的动作索引（-1 表示根节点）
    float prior;          // 神经网络给该动作的先验概率 P
    float valueSum;       // 累计价值 W（从本节点 player 视角）
    int visits;           // 访问次数 N
    bool expanded;        // 是否已扩展

    std::vector<AZNode *> children; // 子节点列表

    AZNode();
    AZNode(const Board &b, int p, int act, float pr);
    ~AZNode();

    // 平均价值 Q（从本节点 player 视角）
    float Q() const;

    // PUCT 分数（从父节点视角选择子节点）
    // 注意：父节点与子节点 player 相反，所以用 -Q()
    float puctScore(int parentVisits) const;

    // 选择 PUCT 分数最高的子节点
    AZNode *selectChild() const;

    // 是否为叶节点（未扩展或无子节点）
    bool isLeaf() const;
};

// 释放整棵子树内存
void deleteAZTree(AZNode *node);
