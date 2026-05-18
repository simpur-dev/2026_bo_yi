#include "AI/az/az_action.h"
#include "AI/az/az_encoder.h"
#include "AI/az/az_evaluator.h"
#include "AI/az/az_mcts.h"
#include "AI/az/az_node.h"
#include "AI/az/az_types.h"
#include "AI/board.h"
#include "AI/define.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

// ========== 测试框架 ==========

static int totalTests = 0;
static int passedTests = 0;
static int failedTests = 0;

#define TEST(name)                                                       \
    static void test_##name();                                           \
    struct TestReg_##name                                                 \
    {                                                                    \
        TestReg_##name() { registerTest(#name, test_##name); }           \
    };                                                                   \
    static TestReg_##name reg_##name;                                    \
    static void test_##name()

#define ASSERT_TRUE(expr)                                                \
    do                                                                   \
    {                                                                    \
        if (!(expr))                                                     \
        {                                                                \
            std::cerr << "  FAIL: " << #expr << " at line " << __LINE__ \
                      << "\n";                                           \
            throw std::runtime_error("assertion failed");                \
        }                                                                \
    } while (0)

#define ASSERT_EQ(a, b)                                                  \
    do                                                                   \
    {                                                                    \
        if ((a) != (b))                                                  \
        {                                                                \
            std::cerr << "  FAIL: " << #a << " == " << #b               \
                      << " (" << (a) << " vs " << (b) << ")"            \
                      << " at line " << __LINE__ << "\n";                \
            throw std::runtime_error("assertion failed");                \
        }                                                                \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                           \
    do                                                                   \
    {                                                                    \
        if (std::abs((a) - (b)) > (eps))                                 \
        {                                                                \
            std::cerr << "  FAIL: |" << #a << " - " << #b << "| <= "    \
                      << (eps) << " (" << (a) << " vs " << (b) << ")"   \
                      << " at line " << __LINE__ << "\n";                \
            throw std::runtime_error("assertion failed");                \
        }                                                                \
    } while (0)

struct TestEntry
{
    const char *name;
    void (*func)();
};

static std::vector<TestEntry> &getTests()
{
    static std::vector<TestEntry> tests;
    return tests;
}

void registerTest(const char *name, void (*func)())
{
    getTests().push_back({name, func});
}

// ========== 动作映射测试 ==========

TEST(action_bijection)
{
    // 所有 60 个 action 都能往返转换
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        LOC loc = actionToLoc(a);
        int roundtrip = locToAction(loc);
        ASSERT_EQ(roundtrip, a);
    }
}

TEST(action_loc_ranges)
{
    // 横边 action 0-29: i 偶数, j 奇数
    for (int a = 0; a < 30; a++)
    {
        LOC loc = actionToLoc(a);
        ASSERT_TRUE(loc.first >= 0 && loc.first <= 10);
        ASSERT_TRUE(loc.second >= 1 && loc.second <= 9);
        ASSERT_TRUE(isEven(loc.first));
        ASSERT_TRUE(!isEven(loc.second));
    }
    // 竖边 action 30-59: i 奇数, j 偶数
    for (int a = 30; a < 60; a++)
    {
        LOC loc = actionToLoc(a);
        ASSERT_TRUE(loc.first >= 1 && loc.first <= 9);
        ASSERT_TRUE(loc.second >= 0 && loc.second <= 10);
        ASSERT_TRUE(!isEven(loc.first));
        ASSERT_TRUE(isEven(loc.second));
    }
}

TEST(action_unique_locs)
{
    // 所有 60 个 action 映射到不同的 LOC
    std::set<std::pair<int, int>> locs;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        LOC loc = actionToLoc(a);
        ASSERT_TRUE(locs.find(loc) == locs.end());
        locs.insert(loc);
    }
    ASSERT_EQ(static_cast<int>(locs.size()), AZ_ACTION_SIZE);
}

TEST(action_invalid_input)
{
    // 无效 action 返回 {-1, -1}
    LOC bad1 = actionToLoc(-1);
    ASSERT_EQ(bad1.first, -1);
    ASSERT_EQ(bad1.second, -1);

    LOC bad2 = actionToLoc(60);
    ASSERT_EQ(bad2.first, -1);
    ASSERT_EQ(bad2.second, -1);

    // DOT 坐标应返回 -1
    int bad3 = locToAction({0, 0});
    ASSERT_EQ(bad3, -1);
}

TEST(validate_action_mapping)
{
    // 使用已有的 validateActionMapping 函数
    ASSERT_TRUE(validateActionMapping());
}

// ========== 棋盘测试 ==========

TEST(board_initial_state)
{
    Board board;
    // 初始无得分
    ASSERT_EQ(board.blackBox, 0);
    ASSERT_EQ(board.whiteBox, 0);
    // 未结束
    ASSERT_TRUE(!board.ifEnd());
    // 无赢家
    ASSERT_EQ(board.getWinner(), 0);
}

TEST(board_all_edges_free_initially)
{
    Board board;
    // 所有 60 条边都是自由的
    int freeCount = 0;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (isLegalAction(board, a))
            freeCount++;
    }
    ASSERT_EQ(freeCount, AZ_ACTION_SIZE);
}

TEST(board_move_occupies_edge)
{
    Board board;
    LOC loc = actionToLoc(0); // 第一条横边
    ASSERT_TRUE(board.isFreeLine(loc));
    board.move(BLACK, loc);
    ASSERT_TRUE(!board.isFreeLine(loc));
}

TEST(board_move_earn_box)
{
    Board board;
    // 手动构造一个格子的三条边，第四条吃格
    // 格子 (1,1) 的四条边: 上(0,1) 下(2,1) 左(1,0) 右(1,2)
    board.move(BLACK, {0, 1}); // 上
    board.move(WHITE, {2, 1}); // 下
    board.move(BLACK, {1, 0}); // 左

    int earned = board.move(WHITE, {1, 2}); // 右 → 应该吃到格子
    ASSERT_TRUE(earned > 0);
}

TEST(board_legal_mask_consistency)
{
    Board board;
    auto mask = getLegalMask(board);
    auto actions = getLegalActions(board);

    // mask 中 1 的数量应等于 actions 的大小
    int maskCount = 0;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (mask[a] > 0.5f)
            maskCount++;
    }
    ASSERT_EQ(maskCount, static_cast<int>(actions.size()));
}

TEST(board_legal_mask_after_move)
{
    Board board;
    auto actionsBefore = getLegalActions(board);
    int a = actionsBefore[0];
    board.move(BLACK, actionToLoc(a));

    auto actionsAfter = getLegalActions(board);
    // 少了一条合法边
    ASSERT_EQ(static_cast<int>(actionsAfter.size()),
              static_cast<int>(actionsBefore.size()) - 1);

    // 该 action 不再合法
    ASSERT_TRUE(!isLegalAction(board, a));
}

TEST(board_game_ends)
{
    Board board;
    // 占据所有 60 条边后游戏结束
    int player = BLACK;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        LOC loc = actionToLoc(a);
        if (board.isFreeLine(loc))
        {
            int earned = board.move(player, loc);
            if (earned == 0)
                player = -player;
        }
    }
    ASSERT_TRUE(board.ifEnd());
    ASSERT_EQ(board.blackBox + board.whiteBox, BOXNUM);
}

// ========== 编码器测试 ==========

TEST(encoder_output_shape)
{
    Board board;
    Tensor t = encodeBoard(board, BLACK);
    ASSERT_EQ(static_cast<int>(t.size()), AZ_CHANNELS * AZ_BOARD_SIZE * AZ_BOARD_SIZE);
}

TEST(encoder_player_channel)
{
    Board board;
    // BLACK 编码: 通道 5 全为 1.0
    Tensor tBlack = encodeBoard(board, BLACK);
    for (int i = 0; i < AZ_BOARD_SIZE; i++)
        for (int j = 0; j < AZ_BOARD_SIZE; j++)
            ASSERT_NEAR(tensorAt(tBlack, 5, i, j), 1.0f, 1e-6f);

    // WHITE 编码: 通道 5 全为 -1.0
    Tensor tWhite = encodeBoard(board, WHITE);
    for (int i = 0; i < AZ_BOARD_SIZE; i++)
        for (int j = 0; j < AZ_BOARD_SIZE; j++)
            ASSERT_NEAR(tensorAt(tWhite, 5, i, j), -1.0f, 1e-6f);
}

TEST(encoder_initial_free_edges)
{
    Board board;
    Tensor t = encodeBoard(board, BLACK);
    // 通道 3: 可落子边 = 所有 HENG 和 SHU 位置
    int freeEdgeCount = 0;
    for (int i = 0; i < AZ_BOARD_SIZE; i++)
        for (int j = 0; j < AZ_BOARD_SIZE; j++)
            if (tensorAt(t, 3, i, j) > 0.5f)
                freeEdgeCount++;
    ASSERT_EQ(freeEdgeCount, AZ_ACTION_SIZE);
}

TEST(encoder_occupied_edge)
{
    Board board;
    LOC loc = actionToLoc(0);
    board.move(BLACK, loc);

    Tensor t = encodeBoard(board, BLACK);
    // 通道 2: 已被占据的边
    ASSERT_NEAR(tensorAt(t, 2, loc.first, loc.second), 1.0f, 1e-6f);
    // 通道 3: 不再是可落子边
    ASSERT_NEAR(tensorAt(t, 3, loc.first, loc.second), 0.0f, 1e-6f);
}

TEST(encoder_box_freedom)
{
    Board board;
    Tensor t = encodeBoard(board, BLACK);
    // 初始时每个格子自由度为 4, 归一化后为 1.0
    for (int bi = 0; bi < BOXLEN; bi++)
    {
        for (int bj = 0; bj < BOXLEN; bj++)
        {
            int i = bi * 2 + 1;
            int j = bj * 2 + 1;
            ASSERT_NEAR(tensorAt(t, 6, i, j), 1.0f, 1e-6f);
        }
    }
}

// ========== MCTS 配置测试 ==========

TEST(mcts_config_selfplay_has_noise)
{
    MCTSConfig cfg = MCTSConfig::selfPlay(800);
    ASSERT_TRUE(cfg.addRootNoise);
    ASSERT_EQ(cfg.simulations, 800);
    ASSERT_TRUE(cfg.temperature > 0.0f);
    ASSERT_TRUE(cfg.temperatureMoves > 0);
    ASSERT_EQ(cfg.timeLimitMs, 0);
}

TEST(mcts_config_evaluation_no_noise)
{
    MCTSConfig cfg = MCTSConfig::evaluation(200, 1000);
    ASSERT_TRUE(!cfg.addRootNoise);
    ASSERT_EQ(cfg.simulations, 200);
    ASSERT_NEAR(cfg.temperature, 0.0f, 1e-6f);
    ASSERT_EQ(cfg.temperatureMoves, 0);
    ASSERT_EQ(cfg.timeLimitMs, 1000);
}

TEST(mcts_config_default_is_eval)
{
    MCTSConfig cfg;
    ASSERT_TRUE(!cfg.addRootNoise);
    ASSERT_NEAR(cfg.temperature, 0.0f, 1e-6f);
}

// ========== MCTS 搜索测试 ==========

TEST(mcts_search_returns_valid_root)
{
    Board board;
    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, MCTSConfig::evaluation(10, 0));
    ASSERT_TRUE(root != nullptr);
    ASSERT_TRUE(root->expanded);
    ASSERT_TRUE(!root->children.empty());

    deleteAZTree(root);
    delete root;
}

TEST(mcts_search_all_children_legal)
{
    Board board;
    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, MCTSConfig::evaluation(10, 0));

    for (auto *child : root->children)
    {
        ASSERT_TRUE(child->action >= 0 && child->action < AZ_ACTION_SIZE);
        ASSERT_TRUE(isLegalAction(board, child->action));
    }

    deleteAZTree(root);
    delete root;
}

TEST(mcts_visit_distribution_sums_to_one)
{
    Board board;
    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, MCTSConfig::evaluation(50, 0));

    auto dist = mcts.getVisitDistribution(root);
    float sum = 0.0f;
    for (float v : dist)
        sum += v;
    ASSERT_NEAR(sum, 1.0f, 1e-4f);

    // 非法动作概率为 0
    auto mask = getLegalMask(board);
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (mask[a] < 0.5f)
            ASSERT_NEAR(dist[a], 0.0f, 1e-6f);
    }

    deleteAZTree(root);
    delete root;
}

TEST(mcts_select_action_greedy)
{
    Board board;
    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, MCTSConfig::evaluation(50, 0));

    int action = mcts.selectAction(root, 0.0f);
    ASSERT_TRUE(action >= 0 && action < AZ_ACTION_SIZE);
    ASSERT_TRUE(isLegalAction(board, action));

    // 贪心选择应是访问最多的
    int maxVisits = 0;
    int maxAction = -1;
    for (auto *child : root->children)
    {
        if (child->visits > maxVisits)
        {
            maxVisits = child->visits;
            maxAction = child->action;
        }
    }
    ASSERT_EQ(action, maxAction);

    deleteAZTree(root);
    delete root;
}

TEST(mcts_no_noise_in_eval_mode)
{
    Board board;
    // 运行两次 evaluation 模式搜索，确认结果确定性（无噪声）
    // 注意：因为启发式评估器是确定性的，没有噪声时搜索应是确定性的
    AZMCTS mcts1;
    AZNode *root1 = mcts1.search(board, BLACK, MCTSConfig::evaluation(50, 0));
    auto dist1 = mcts1.getVisitDistribution(root1);

    AZMCTS mcts2;
    AZNode *root2 = mcts2.search(board, BLACK, MCTSConfig::evaluation(50, 0));
    auto dist2 = mcts2.getVisitDistribution(root2);

    // evaluation 模式无噪声，使用启发式评估器时结果应一致
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
        ASSERT_NEAR(dist1[a], dist2[a], 1e-4f);

    deleteAZTree(root1);
    delete root1;
    deleteAZTree(root2);
    delete root2;
}

TEST(mcts_terminal_board)
{
    // 构造一个游戏结束的棋盘
    Board board;
    int player = BLACK;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        LOC loc = actionToLoc(a);
        if (board.isFreeLine(loc))
        {
            int earned = board.move(player, loc);
            if (earned == 0)
                player = -player;
        }
    }
    ASSERT_TRUE(board.ifEnd());

    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, MCTSConfig::evaluation(10, 0));
    ASSERT_TRUE(root->terminal);

    deleteAZTree(root);
    delete root;
}

TEST(mcts_old_interface_compat)
{
    // 旧接口应等价于 evaluation 模式
    Board board;
    AZMCTS mcts;
    AZNode *root = mcts.search(board, BLACK, 10, 0);
    ASSERT_TRUE(root != nullptr);
    ASSERT_TRUE(root->expanded || root->terminal);

    deleteAZTree(root);
    delete root;
}

// ========== 主函数 ==========

int main()
{
    std::cerr << "=== Dots & Boxes AlphaZero Unit Tests ===\n\n";

    for (auto &test : getTests())
    {
        totalTests++;
        std::cerr << "[RUN ] " << test.name << "\n";
        try
        {
            test.func();
            passedTests++;
            std::cerr << "[PASS] " << test.name << "\n";
        }
        catch (const std::exception &e)
        {
            failedTests++;
            std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
        }
    }

    std::cerr << "\n=== Results: " << passedTests << "/" << totalTests
              << " passed, " << failedTests << " failed ===\n";

    return failedTests > 0 ? 1 : 0;
}
