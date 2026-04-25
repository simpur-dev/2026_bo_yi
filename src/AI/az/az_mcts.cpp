#include "az_mcts.h"
#include "az_action.h"
#include "az_evaluator.h"
#include "../assess.h"
#include "../define.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

static std::mt19937 azRng(std::random_device{}());

// ========== 辅助函数 ==========

// 添加 Dirichlet 噪声到根节点的先验概率
static void addDirichletNoise(AZNode *root)
{
    if (root->children.empty())
        return;

    int n = static_cast<int>(root->children.size());
    std::vector<float> noise(n);
    std::gamma_distribution<float> gamma(AZ_DIRICHLET_ALPHA, 1.0f);

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        noise[i] = gamma(azRng);
        sum += noise[i];
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < n; i++)
            noise[i] /= sum;
    }

    for (int i = 0; i < n; i++)
    {
        root->children[i]->prior =
            (1.0f - AZ_DIRICHLET_FRAC) * root->children[i]->prior +
            AZ_DIRICHLET_FRAC * noise[i];
    }
}

// ========== AZMCTS 实现 ==========

AZNode *AZMCTS::search(const Board &board, int player, int numSimulations, int timeLimitMs)
{
    // 创建根节点
    AZNode *root = new AZNode(board, player, -1, 0.0f);

    // 先检查是否已经是终局
    if (isEndgame(board, player))
    {
        // 终局不搜索，直接返回
        return root;
    }

    // 对根节点做初次评估和扩展
    NetworkOutput rootOutput = getEvaluator().evaluate(board, player);
    expand(root, rootOutput);

    // 对根节点添加 Dirichlet 噪声以增加探索
    addDirichletNoise(root);

    // 开始搜索
    auto startTime = std::chrono::steady_clock::now();

    for (int sim = 0; sim < numSimulations; sim++)
    {
        // 检查时间
        if (timeLimitMs > 0)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed >= timeLimitMs)
                break;
        }

        simulate(root);
    }

    return root;
}

void AZMCTS::simulate(AZNode *root)
{
    std::vector<AZNode *> path;
    AZNode *node = root;
    path.push_back(node);

    // === 选择阶段 ===
    while (!node->isLeaf())
    {
        node = node->selectChild();
        if (!node)
            break;
        path.push_back(node);
    }

    if (!node)
        return;

    // === 评估阶段 ===
    float value;

    // 检查游戏是否结束
    if (node->board.ifEnd())
    {
        int winner = node->board.getWinner();
        if (winner == node->player)
            value = 1.0f;
        else if (winner == -node->player)
            value = -1.0f;
        else
            value = 0.0f;
    }
    // 检查是否进入终局（只剩长链/环）
    else if (isEndgame(node->board, node->player))
    {
        value = endgameEvaluate(node->board, node->player);
    }
    // 正常评估 + 扩展
    else
    {
        NetworkOutput output = getEvaluator().evaluate(node->board, node->player);
        expand(node, output);
        value = output.value;
    }

    // === 回传阶段 ===
    backup(path, value, node->player);
}

void AZMCTS::expand(AZNode *node, const NetworkOutput &output)
{
    if (node->expanded)
        return;

    auto legalActions = getLegalActions(node->board);
    if (legalActions.empty())
    {
        node->expanded = true;
        return;
    }

    // 对 policy 做合法动作 mask 和重新归一化
    float policySum = 0.0f;
    for (int a : legalActions)
        policySum += output.policy[a];

    for (int a : legalActions)
    {
        float prior = (policySum > 0.0f) ? output.policy[a] / policySum : 1.0f / static_cast<float>(legalActions.size());

        // 创建子节点：执行动作 -> 吃掉 C 型格 -> 切换玩家
        Board childBoard = node->board;
        int earned = childBoard.move(node->player, actionToLoc(a));

        // 下一个玩家：如果吃到了格子，当前玩家继续；否则切换
        int nextPlayer;
        if (earned > 0)
        {
            // 当前玩家继续行动，先吃完所有 C 型格
            childBoard.eatAllCTypeBoxes(node->player);
            nextPlayer = node->player;
        }
        else
        {
            // 切换到对手，对手先吃完所有 C 型格
            nextPlayer = -node->player;
            childBoard.eatAllCTypeBoxes(nextPlayer);
        }

        AZNode *child = new AZNode(childBoard, nextPlayer, a, prior);
        node->children.push_back(child);
    }

    node->expanded = true;
}

void AZMCTS::backup(const std::vector<AZNode *> &path, float value, int leafPlayer)
{
    // value 是从 leafPlayer 视角的评估值
    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
        AZNode *node = *it;
        node->visits++;
        // valueSum 存储的是从该节点 player 视角的累计价值
        if (node->player == leafPlayer)
            node->valueSum += value;
        else
            node->valueSum -= value;
    }
}

int AZMCTS::selectAction(const AZNode *root, float temperature) const
{
    if (root->children.empty())
        return -1;

    if (temperature < 1e-6f)
    {
        // 贪心：选择访问次数最多的
        int bestAction = -1;
        int maxVisits = -1;
        for (auto *child : root->children)
        {
            if (child->visits > maxVisits)
            {
                maxVisits = child->visits;
                bestAction = child->action;
            }
        }
        return bestAction;
    }
    else
    {
        // 按温度采样
        std::vector<float> probs;
        probs.reserve(root->children.size());
        float sum = 0.0f;

        for (auto *child : root->children)
        {
            float p = std::pow(static_cast<float>(child->visits), 1.0f / temperature);
            probs.push_back(p);
            sum += p;
        }

        if (sum > 0.0f)
        {
            for (auto &p : probs)
                p /= sum;
        }

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(azRng);
        return root->children[idx]->action;
    }
}

std::array<float, AZ_ACTION_SIZE> AZMCTS::getVisitDistribution(const AZNode *root) const
{
    std::array<float, AZ_ACTION_SIZE> dist{};
    int totalVisits = 0;
    for (auto *child : root->children)
        totalVisits += child->visits;

    if (totalVisits > 0)
    {
        for (auto *child : root->children)
            dist[child->action] = static_cast<float>(child->visits) / static_cast<float>(totalVisits);
    }
    return dist;
}

bool AZMCTS::isEndgame(const Board &board, int player) const
{
    // 复制局面，模拟吃完所有 C 型格后检查是否还有 Filter 可行边
    Board test = board;
    test.eatAllCTypeBoxes(player);
    return (test.getFilterMoveNum() == 0);
}

float AZMCTS::endgameEvaluate(const Board &board, int player) const
{
    // 使用现有精确终局求解器
    Board boardCopy = board;
    BoxBoard advanced(boardCopy);
    int winner = advanced.getBoardWinner(-player); // latterPlayer 是对手

    if (winner == player)
        return 1.0f;
    else if (winner == -player)
        return -1.0f;
    else
        return 0.0f;
}
