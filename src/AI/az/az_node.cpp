#include "az_node.h"
#include "az_action.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

// ========== AZNode 实现 ==========

AZNode::AZNode()
    : player(0), action(-1), prior(0.0f),
      valueSum(0.0f), visits(0),
      expanded(false), terminal(false), terminalValue(0.0f)
{
}

AZNode::AZNode(const Board &b, int p, int act, float pr)
    : board(b), player(p), action(act), prior(pr),
      valueSum(0.0f), visits(0),
      expanded(false), terminal(false), terminalValue(0.0f)
{
}

AZNode::~AZNode()
{
}

float AZNode::Q() const
{
    if (visits == 0)
        return 0.0f;
    return valueSum / static_cast<float>(visits);
}

float AZNode::puctScore(int parentPlayer, int parentVisits) const
{
    // ---- Exploitation ----
    // Q 是从本节点 player 视角的平均价值
    // 父节点想最大化自己的收益:
    //   若 child.player == parent.player → 同视角 → +Q
    //   若 child.player != parent.player → 对立视角 → -Q
    float exploitation;
    if (player == parentPlayer)
        exploitation = Q();
    else
        exploitation = -Q();

    // ---- Exploration ----
    // c_puct × P × sqrt(N_parent) / (1 + N_child)
    float exploration = AZ_C_PUCT * prior *
                        std::sqrt(static_cast<float>(parentVisits)) /
                        (1.0f + static_cast<float>(visits));

    return exploitation + exploration;
}

AZNode *AZNode::selectChild() const
{
    if (children.empty())
        return nullptr;

    AZNode *best = nullptr;
    float bestScore = -std::numeric_limits<float>::infinity();

    for (auto *child : children)
    {
        float score = child->puctScore(player, visits);
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

// ========== AZSearchStats 实现 ==========

std::string AZSearchStats::toString() const
{
    std::ostringstream ss;
    LOC loc = (bestAction >= 0) ? actionToLoc(bestAction) : LOC{-1, -1};

    ss << "[PUCT] sims=" << totalSimulations
       << " time=" << elapsedMs << "ms"
       << " nodes=" << treeNodes
       << " rootQ=" << std::fixed << std::setprecision(3) << rootQ
       << " best=action" << bestAction
       << "(" << loc.first << "," << loc.second << ")"
       << " visits=" << bestVisits
       << " Q=" << std::fixed << std::setprecision(3) << bestQ;

    if (elapsedMs > 0)
    {
        float sps = static_cast<float>(totalSimulations) * 1000.0f / static_cast<float>(elapsedMs);
        ss << " speed=" << static_cast<int>(sps) << "sims/s";
    }

    return ss.str();
}

// ========== 工具函数 ==========

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

int countTreeNodes(const AZNode *node)
{
    if (!node)
        return 0;
    int count = 1;
    for (const auto *child : node->children)
        count += countTreeNodes(child);
    return count;
}
