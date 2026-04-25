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
#include <iostream>

static std::mt19937 azRng(std::random_device{}());

// ========== 辅助函数 ==========

// 添加 Dirichlet 噪声到根节点的先验概率（增加探索多样性）
// 混合后: P' = (1 - frac) × P + frac × noise
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

// ========== AZMCTS::search ==========

AZNode *AZMCTS::search(const Board &board, int player, int numSimulations, int timeLimitMs)
{
    AZNode *root = new AZNode(board, player, -1, 0.0f);

    // 检查根节点是否已是终局
    if (checkAndMarkTerminal(root))
        return root;

    // 对根节点执行初次评估和扩展
    NetworkOutput rootOutput = getEvaluator().evaluate(board, player);
    expand(root, rootOutput);

    // 对根节点添加 Dirichlet 噪声
    addDirichletNoise(root);

    // 搜索主循环
    auto startTime = std::chrono::steady_clock::now();
    constexpr int TIME_CHECK_INTERVAL = 64; // 每 64 次迭代检查一次时间

    int sim = 0;
    for (; sim < numSimulations; sim++)
    {
        // 定期检查时间限制（减少 chrono 调用开销）
        if (timeLimitMs > 0 && (sim & (TIME_CHECK_INTERVAL - 1)) == 0 && sim > 0)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed >= timeLimitMs)
                break;
        }

        simulate(root);
    }

    // 输出搜索统计
    auto endTime = std::chrono::steady_clock::now();
    int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
    AZSearchStats stats = getSearchStats(root, elapsed);
    std::cerr << stats.toString() << "\n";

    return root;
}

// ========== AZMCTS::simulate ==========
//
// 一次完整的 PUCT 迭代:
//   Selection → Evaluation → Expansion → Backup

void AZMCTS::simulate(AZNode *root)
{
    // ---- 1. Selection ----
    // 从根节点沿 PUCT 分数最高的路径下降，直到遇到叶节点或终局节点
    std::vector<AZNode *> path;
    AZNode *node = root;
    path.push_back(node);

    while (node->expanded && !node->terminal)
    {
        AZNode *child = node->selectChild();
        if (!child)
            break; // 无子节点（不应该发生）
        path.push_back(child);
        node = child;
    }

    // ---- 2. Evaluation ----
    float value;

    if (node->terminal)
    {
        // 终局节点：使用缓存的精确价值（避免重复计算）
        value = node->terminalValue;
    }
    else if (checkAndMarkTerminal(node))
    {
        // 首次发现的终局节点：标记并缓存
        value = node->terminalValue;
    }
    else
    {
        // ---- 3. Expansion ----
        // 正常局面：调用评估器获取 policy + value，然后扩展
        NetworkOutput output = getEvaluator().evaluate(node->board, node->player);
        expand(node, output);
        value = output.value;
    }

    // ---- 4. Backup ----
    // 将 value（从叶节点 player 视角）沿路径回传
    backup(path, value, node->player);
}

// ========== AZMCTS::expand ==========

void AZMCTS::expand(AZNode *node, const NetworkOutput &output)
{
    if (node->expanded || node->terminal)
        return;

    auto legalActions = getLegalActions(node->board);

    if (legalActions.empty())
    {
        // 无合法动作 → 游戏结束，标记为终局
        node->expanded = true;
        node->terminal = true;
        int winner = node->board.getWinner();
        if (winner == node->player)
            node->terminalValue = 1.0f;
        else if (winner == -node->player)
            node->terminalValue = -1.0f;
        else
            node->terminalValue = 0.0f;
        return;
    }

    // 合法动作的 policy 归一化
    float policySum = 0.0f;
    for (int a : legalActions)
        policySum += output.policy[a];

    // 为每个合法动作创建子节点
    node->children.reserve(legalActions.size());

    for (int a : legalActions)
    {
        // 先验概率：归一化后的 policy；如果全为 0 则均匀分布
        float prior = (policySum > 1e-8f)
                          ? output.policy[a] / policySum
                          : 1.0f / static_cast<float>(legalActions.size());

        // 执行动作
        Board childBoard = node->board;
        int earned = childBoard.move(node->player, actionToLoc(a));

        // 确定下一个玩家:
        //   吃到格子 → 当前玩家继续，并贪心吃完所有 C 型格
        //   没吃到   → 切换到对手，对手先吃完 C 型格
        int nextPlayer;
        if (earned > 0)
        {
            childBoard.eatAllCTypeBoxes(node->player);
            nextPlayer = node->player;
        }
        else
        {
            nextPlayer = -node->player;
            childBoard.eatAllCTypeBoxes(nextPlayer);
        }

        AZNode *child = new AZNode(childBoard, nextPlayer, a, prior);
        node->children.push_back(child);
    }

    node->expanded = true;
}

// ========== AZMCTS::backup ==========

void AZMCTS::backup(const std::vector<AZNode *> &path, float value, int leafPlayer)
{
    // value 是从 leafPlayer 视角的评估值
    // 回传到每个节点: 根据该节点的 player 决定加还是减
    for (auto it = path.rbegin(); it != path.rend(); ++it)
    {
        AZNode *node = *it;
        node->visits++;

        if (node->player == leafPlayer)
            node->valueSum += value;
        else
            node->valueSum -= value;
    }
}

// ========== AZMCTS::checkAndMarkTerminal ==========

bool AZMCTS::checkAndMarkTerminal(AZNode *node) const
{
    if (node->terminal)
        return true;

    // 情况 1: 游戏已结束（所有边都已占据）
    if (node->board.ifEnd())
    {
        int winner = node->board.getWinner();
        if (winner == node->player)
            node->terminalValue = 1.0f;
        else if (winner == -node->player)
            node->terminalValue = -1.0f;
        else
            node->terminalValue = 0.0f;
        node->terminal = true;
        node->expanded = true;
        return true;
    }

    // 情况 2: 进入终局（吃完 C 型格后无 Filter 可行边，只剩长链/环/预备环）
    Board test = node->board;
    test.eatAllCTypeBoxes(node->player);
    if (test.getFilterMoveNum() == 0)
    {
        node->terminalValue = endgameEvaluate(node->board, node->player);
        node->terminal = true;
        node->expanded = true;
        return true;
    }

    return false;
}

// ========== AZMCTS::endgameEvaluate ==========

float AZMCTS::endgameEvaluate(const Board &board, int player) const
{
    // 使用现有精确终局求解器
    // getBoardWinner(latterPlayer) 中 latterPlayer 是"后手"（即将开长链/环的一方）
    Board boardCopy = board;
    BoxBoard advanced(boardCopy);
    int winner = advanced.getBoardWinner(-player);

    if (winner == player)
        return 1.0f;
    else if (winner == -player)
        return -1.0f;
    else
        return 0.0f;
}

// ========== AZMCTS::selectAction ==========

int AZMCTS::selectAction(const AZNode *root, float temperature) const
{
    if (root->children.empty())
        return -1;

    if (temperature < 1e-6f)
    {
        // 贪心：选择访问次数最多的子节点
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
        // 温度采样: prob ∝ visits^(1/T)
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

// ========== AZMCTS::getVisitDistribution ==========

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

// ========== AZMCTS::getSearchStats ==========

AZSearchStats AZMCTS::getSearchStats(const AZNode *root, int elapsedMs) const
{
    AZSearchStats stats;
    stats.totalSimulations = root->visits;
    stats.elapsedMs = elapsedMs;
    stats.treeNodes = countTreeNodes(root);
    stats.rootQ = root->Q();

    // 找出访问最多的子节点
    for (auto *child : root->children)
    {
        if (child->visits > stats.bestVisits)
        {
            stats.bestVisits = child->visits;
            stats.bestAction = child->action;

            // 从根节点视角显示 Q
            if (child->player == root->player)
                stats.bestQ = child->Q();
            else
                stats.bestQ = -child->Q();
        }
    }

    return stats;
}
