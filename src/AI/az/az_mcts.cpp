#include "az_mcts.h"
#include "az_action.h"
#include "az_evaluator.h"
#include "az_expert.h"
#include "../assess.h"
#include "../define.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <cstring>

static std::mt19937 azRng(std::random_device{}());

// ========== 辅助函数 ==========

// 添加 Dirichlet 噪声到根节点的先验概率（增加探索多样性）
// 混合后: P' = (1 - frac) × P + frac × noise
static void addDirichletNoise(AZNode *root, float alpha, float frac)
{
    if (root->children.empty())
        return;

    int n = static_cast<int>(root->children.size());
    std::vector<float> noise(n);
    std::gamma_distribution<float> gamma(alpha, 1.0f);

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
            (1.0f - frac) * root->children[i]->prior +
            frac * noise[i];
    }
}

// ========== AZMCTS::search ==========

AZNode *AZMCTS::search(const Board &board, int player, const MCTSConfig &config)
{
    config_ = config;

    AZNode *root = new AZNode(board, player, -1, 0.0f);

    // 检查根节点是否已是终局
    if (checkAndMarkTerminal(root))
        return root;

    // 对根节点执行初次评估和扩展
    // 双模型模式: 按当前玩家颜色选择对应评估器 (单模型模式下为 no-op)
    selectEvaluatorForColor(player);
    NetworkOutput rootOutput = getEvaluator().evaluate(board, player);
    expand(root, rootOutput);

    // 仅在自对弈模式下添加 Dirichlet 噪声
    if (config_.addRootNoise)
        addDirichletNoise(root, config_.dirichletAlpha, config_.dirichletFrac);

    // 搜索主循环
    auto startTime = std::chrono::steady_clock::now();
    constexpr int TIME_CHECK_INTERVAL = 64; // 每 64 次迭代检查一次时间

    int sim = 0;
    for (; sim < config_.simulations; sim++)
    {
        // 定期检查时间限制（减少 chrono 调用开销）
        if (config_.timeLimitMs > 0 && (sim & (TIME_CHECK_INTERVAL - 1)) == 0 && sim > 0)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed >= config_.timeLimitMs)
                break;
        }

        simulate(root);

        // 终局传播导致根节点已精确求解 → 立即退出，不浪费剩余模拟
        if (root->terminal)
            break;
    }

    // 输出搜索统计
    auto endTime = std::chrono::steady_clock::now();
    int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
    AZSearchStats stats = getSearchStats(root, elapsed);
    std::cerr << stats.toString() << "\n";

    return root;
}

// 兼容旧接口：默认 evaluation 模式（无噪声）
AZNode *AZMCTS::search(const Board &board, int player, int numSimulations, int timeLimitMs)
{
    return search(board, player, MCTSConfig::evaluation(numSimulations, timeLimitMs));
}

// ========== 子树复用 ==========
//
// 传统 AlphaZero 每步重建整棵树，浪费大量计算。
// 这里实现简化版子树复用：将上一步所选动作对应的子树提升为新根节点，
// 只对新分支和旧子树的补充进行搜索，其余保持不变。
//
// 注意：由于 Board 对象按值传递且可能因 C 型格归一化而改变，
// 这里采用"引用匹配"策略：检查上一步根节点的子节点中，
// 是否有子节点的 lastAction == prevAction，且子节点棋盘与当前棋盘一致。
// 如果一致，直接复用该子树（已包含历史搜索信息）。
// 如果棋盘因专家归一化而不一致，安全回退到新建树。

AZNode *AZMCTS::reuseSubtree(const Board &board, int player,
                              AZNode *&previousRoot, int prevAction)
{
    if (!previousRoot || prevAction < 0)
    {
        AZNode *newRoot = new AZNode(board, player, -1, 0.0f);
        deleteAZTree(previousRoot);
        delete previousRoot;
        previousRoot = nullptr;
        return newRoot;
    }

    AZNode *reused = nullptr;
    for (auto *child : previousRoot->children)
    {
        if (child->action == prevAction)
        {
            reused = child;
            break;
        }
    }

    if (!reused)
    {
        AZNode *newRoot = new AZNode(board, player, -1, 0.0f);
        deleteAZTree(previousRoot);
        delete previousRoot;
        previousRoot = nullptr;
        return newRoot;
    }

    if (std::memcmp(reused->board.map, board.map, sizeof(board.map)) != 0)
    {
        AZNode *newRoot = new AZNode(board, player, -1, 0.0f);
        deleteAZTree(previousRoot);
        delete previousRoot;
        previousRoot = nullptr;
        return newRoot;
    }

    // 棋盘匹配！将子树提升为新根节点
    AZNode *newRoot = reused;

    // 将子树从旧根节点中脱离（避免双重释放）
    // 通过遍历移除（避免 vector erase 的 O(n) 查找）
    auto &oldChildren = previousRoot->children;
    for (auto it = oldChildren.begin(); it != oldChildren.end(); ++it)
    {
        if (*it == reused)
        {
            oldChildren.erase(it);
            break;
        }
    }

    // 更新新根节点的 action 为 -1（表示它是搜索的起点）
    newRoot->action = -1;

    // 消耗旧根节点（子树已被提升，不再需要旧根）
    deleteAZTree(previousRoot);
    delete previousRoot;
    previousRoot = nullptr;

    return newRoot;
}

// 带子树复用的搜索重载
AZNode *AZMCTS::search(const Board &board, int player, const MCTSConfig &config,
                       AZNode *&previousRoot, int prevAction)
{
    config_ = config;

    // 尝试复用子树
    AZNode *root = reuseSubtree(board, player, previousRoot, prevAction);

    // 重置根节点统计（子树的 visits/valueSum 保留历史信息，这是期望行为）
    // 如果返回的是全新节点，visits=0/valueSum=0，自然从零开始

    // 检查根节点是否已是终局
    if (checkAndMarkTerminal(root))
        return root;

    // 如果根节点尚未展开（全新节点或复用但未展开），执行初次评估和扩展
    if (!root->expanded)
    {
        selectEvaluatorForColor(player);
        NetworkOutput rootOutput = getEvaluator().evaluate(board, player);
        expand(root, rootOutput);
    }
    else if (root->player != player)
    {
        // 复用子树但玩家不同（理论上不应发生，因为棋盘匹配时玩家相同）
        // 安全起见：标记为需重新展开
        root->expanded = false;
        root->terminal = false;
        root->terminalValue = 0.0f;
        root->children.clear();
        selectEvaluatorForColor(player);
        NetworkOutput rootOutput = getEvaluator().evaluate(board, player);
        expand(root, rootOutput);
    }

    // 仅在自对弈模式下添加 Dirichlet 噪声
    if (config_.addRootNoise && root->children.size() > 0)
        addDirichletNoise(root, config_.dirichletAlpha, config_.dirichletFrac);

    // 搜索主循环
    auto startTime = std::chrono::steady_clock::now();
    constexpr int TIME_CHECK_INTERVAL = 64;

    int sim = 0;
    for (; sim < config_.simulations; sim++)
    {
        if (config_.timeLimitMs > 0 && (sim & (TIME_CHECK_INTERVAL - 1)) == 0 && sim > 0)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed >= config_.timeLimitMs)
                break;
        }

        simulate(root);

        if (root->terminal)
            break;
    }

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
        // ★ 关键: 双模型模式下，每个叶节点必须按 leaf->player 切换评估器
        //   单模型模式下 selectEvaluatorForColor 为 no-op，行为不变
        //   旧代码 bug: 全局 evaluator 只在外层 turn 切换时按根节点设置一次，
        //   MCTS 内部所有 white-to-move 节点都被错误地用 black 模型评估，反之亦然
        selectEvaluatorForColor(node->player);
        NetworkOutput output = getEvaluator().evaluate(node->board, node->player);
        expand(node, output);

        // 若扩展后节点因终局传播变为终局，使用精确值替代 NN 估值
        if (node->terminal)
            value = node->terminalValue;
        else
            value = output.value;
    }

    // ---- 4. Backup ----
    // 将 value（从叶节点 player 视角）沿路径回传
    backup(path, value, node->player);

    // ---- 5. 终局传播 (向上级联) ----
    // expand 可能使叶节点变为终局; 检查路径上的祖先是否也因此可精确求解
    if (node->terminal && path.size() >= 2)
    {
        // 从叶节点的父节点开始向上检查
        for (int i = static_cast<int>(path.size()) - 2; i >= 0; i--)
        {
            AZNode *parent = path[i];
            if (parent->terminal)
                continue; // 已经是终局了
            if (!parent->expanded || parent->children.empty())
                break;

            // 检查 parent 的所有子节点是否均为终局
            bool allTerminal = true;
            for (auto *ch : parent->children)
            {
                if (!ch->terminal)
                {
                    allTerminal = false;
                    break;
                }
            }
            if (!allTerminal)
                break;

            // 所有子节点均为终局 → parent 精确求解
            float bestValue = -2.0f;
            for (auto *ch : parent->children)
            {
                float v;
                if (ch->player == parent->player)
                    v = ch->terminalValue;
                else
                    v = -ch->terminalValue;
                if (v > bestValue)
                    bestValue = v;
            }
            parent->terminal = true;
            parent->terminalValue = bestValue;
        }
    }
}

// ========== 过滤动作辅助 ==========
//
// 使用 assess.cpp 的 getFilterMoves 获取安全边（不产生死链）
// 减少搜索分支因子，避免探索明显坏招

static std::vector<int> getFilteredActions(const Board &board)
{
    Board boardCopy = board;
    BoxBoard bb(boardCopy);
    LOC moves[60];
    int n = bb.getFilterMoves(moves);

    std::vector<int> actions;
    actions.reserve(n);
    for (int i = 0; i < n; i++)
    {
        int a = locToAction(moves[i]);
        if (a >= 0)
            actions.push_back(a);
    }
    return actions;
}

// ========== AZMCTS::expand ==========
//
// 三大优化:
//   1. 过滤动作剪枝: 只扩展安全边 (不产生死链), 减小分支因子
//   2. 子节点即时终局检测: 创建后立即检查是否终局, 避免浪费 sim
//   3. 终局传播: 若全部子节点均为终局, 父节点也精确求解
//      (形成级联，将大量终局子树不经神经网络直接求解)

void AZMCTS::expand(AZNode *node, const NetworkOutput &output)
{
    if (node->expanded || node->terminal)
        return;

    // 优化 1: 使用过滤动作 (安全边)，减少分支因子
    // 先尝试过滤招法, 若没有安全边则回退到全部合法招法
    auto filteredActions = getFilteredActions(node->board);
    auto allActions = getLegalActions(node->board);
    auto &expandActions = filteredActions.empty() ? allActions : filteredActions;

    if (allActions.empty())
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
    for (int a : expandActions)
        policySum += output.policy[a];

    // 为每个动作创建子节点
    node->children.reserve(expandActions.size());
    int terminalChildCount = 0;

    for (int a : expandActions)
    {
        // 先验概率：归一化后的 policy
        float prior = (policySum > 1e-8f)
                          ? output.policy[a] / policySum
                          : 1.0f / static_cast<float>(expandActions.size());

        // 执行动作
        Board childBoard = node->board;
        int earned = childBoard.move(node->player, actionToLoc(a));

        int nextPlayer = (earned > 0) ? node->player : -node->player;
        az_expert::normalizeToSearch(childBoard, nextPlayer, nullptr, az_expert::Options{true});

        AZNode *child = new AZNode(childBoard, nextPlayer, a, prior);

        // 优化 2: 子节点即时终局检测
        // 创建后立即检查是否是终局节点, 避免浪费后续模拟
        if (checkAndMarkTerminal(child))
            terminalChildCount++;

        node->children.push_back(child);
    }

    node->expanded = true;

    // 优化 3: 终局传播
    // 若全部子节点都是终局, 则父节点的价值精确已知:
    // 当前玩家可以选择最优子节点 → value = max(子节点 value from 父视角)
    if (terminalChildCount == static_cast<int>(node->children.size()))
    {
        float bestValue = -2.0f;
        for (auto *child : node->children)
        {
            // 将子节点的 terminalValue 转换到父节点视角
            float v;
            if (child->player == node->player)
                v = child->terminalValue;     // 同视角
            else
                v = -child->terminalValue;    // 对手视角, 取反
            if (v > bestValue)
                bestValue = v;
        }
        node->terminal = true;
        node->terminalValue = bestValue;
    }
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
