#include "az_move.h"
#include "az_mcts.h"
#include "az_action.h"
#include "az_node.h"
#include "az_types.h"
#include "../assess.h"
#include "../define.h"
#include <iostream>
#include <chrono>

// 前向声明后期决策函数（定义在 UCT.cpp 中）
void latterSituationMove(Board &CB, int Player, std::vector<LOC> &pace);

void AlphaZeroMove(Board &board, int player, std::vector<LOC> &pace)
{
    // === 第一步：先贪心吃掉所有 C 型格 ===
    board.eatAllCTypeBoxes(player, pace);

    // === 第二步：检查是否已进入终局 ===
    Board test = board;
    test.eatAllCTypeBoxes(player);
    bool isLatter = (test.getFilterMoveNum() == 0);

    if (isLatter)
    {
        // 终局：使用现有精确终局求解器
        std::cerr << "[AZ] Entering endgame solver...\n";
        auto t0 = std::chrono::steady_clock::now();
        latterSituationMove(board, player, pace);
        auto t1 = std::chrono::steady_clock::now();
        int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        std::cerr << "[AZ] Endgame solver done, " << ms << "ms, " << pace.size() << " moves\n";
        return;
    }

    // === 第三步：检查是否存在死链/死环需要预处理 ===
    BoxBoard dead(board);
    bool deadChain = dead.getDeadChainExist();
    bool deadCircle = dead.getDeadCircleExist();

    if (deadCircle || deadChain)
    {
        std::cerr << "[AZ] Dead chain/circle detected, handling...\n";
        auto dcStart = std::chrono::steady_clock::now();
        int sacrificeBoxNum = deadCircle ? 4 : 2;

        // 模拟全吃后，评估理性状态
        BoxBoard sim(board);
        sim.eatAllCTypeBoxes(player);
        LOC boxNum = sim.getEarlyRationalBoxNum();

        if (boxNum.first - boxNum.second <= sacrificeBoxNum)
        {
            // 全吃更优：放弃控制
            board.eatAllCTypeBoxes(player, pace);
        }
        else
        {
            // 牺牲更优：执行 Double-Cross
            if (sacrificeBoxNum == 2)
            {
                // 吃到贪婪临界点
                for (;;)
                {
                    Board testBoard = board;
                    testBoard.eatCBox(player);
                    BoxBoard deadTest(testBoard);
                    if (deadTest.getDeadChainExist())
                    {
                        LOC t = board.eatCBox(player);
                        pace.emplace_back(t);
                    }
                    else
                        break;
                }
            }
            else
            {
                for (;;)
                {
                    Board testBoard = board;
                    testBoard.eatCBox(player);
                    BoxBoard deadTest(testBoard);
                    if (deadTest.getDeadCircleExist())
                    {
                        LOC t = board.eatCBox(player);
                        pace.emplace_back(t);
                    }
                    else
                        break;
                }
            }
            // Double-Cross
            LOC dcMove = board.getDoubleCrossLoc(player);
            board.move(player, dcMove);
            pace.emplace_back(dcMove);
            for (;;)
            {
                if (!board.getCTypeBoxLimit(player, pace))
                    break;
            }
            return; // 牺牲结束
        }
    }

    // === 第四步：执行 PUCT 搜索 ===
    // 检查游戏是否已经结束
    if (board.ifEnd())
        return;

    // 再次检查是否进入终局（因为上面可能吃了更多格子）
    Board test2 = board;
    test2.eatAllCTypeBoxes(player);
    if (test2.getFilterMoveNum() == 0)
    {
        latterSituationMove(board, player, pace);
        return;
    }

    AZMCTS mcts;
    AZNode *root = mcts.search(board, player, AZ_SIMULATIONS, AZ_TIME_LIMIT_MS);

    // 选择最佳动作（贪心，temperature=0）
    int bestAction = mcts.selectAction(root, 0.0f);

    if (bestAction >= 0)
    {
        LOC loc = actionToLoc(bestAction);
        board.move(player, loc);
        pace.emplace_back(loc);
    }

    // 释放搜索树
    deleteAZTree(root);
    delete root;
}
