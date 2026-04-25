#include "az_node.h"
#include <cmath>
#include <limits>

AZNode::AZNode()
    : player(0), action(-1), prior(0.0f), valueSum(0.0f), visits(0), expanded(false)
{
}

AZNode::AZNode(const Board &b, int p, int act, float pr)
    : board(b), player(p), action(act), prior(pr), valueSum(0.0f), visits(0), expanded(false)
{
}

AZNode::~AZNode()
{
    // 析构时不递归删除子节点，由 deleteAZTree 统一处理
}

float AZNode::Q() const
{
    if (visits == 0)
        return 0.0f;
    return valueSum / static_cast<float>(visits);
}

float AZNode::puctScore(int parentVisits) const
{
    // 从父节点视角：父节点 player 与本节点 player 相反
    // 父节点希望选择对自己最有利的子节点 => 选 -Q() 最大的
    float exploitation = -Q();
    float exploration = AZ_C_PUCT * prior * std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(visits));
    return exploitation + exploration;
}

AZNode *AZNode::selectChild() const
{
    AZNode *best = nullptr;
    float bestScore = -std::numeric_limits<float>::infinity();

    for (auto *child : children)
    {
        float score = child->puctScore(visits);
        if (score > bestScore)
        {
            bestScore = score;
            best = child;
        }
    }
    return best;
}

bool AZNode::isLeaf() const
{
    return !expanded;
}

void deleteAZTree(AZNode *node)
{
    if (!node)
        return;
    for (auto *child : node->children)
    {
        deleteAZTree(child);
        delete child;
    }
    node->children.clear();
}
