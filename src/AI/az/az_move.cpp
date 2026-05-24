#include "az_move.h"
#include "az_mcts.h"
#include "az_action.h"
#include "az_node.h"
#include "az_types.h"
#include "az_expert.h"
#include "../assess.h"
#include "../define.h"
#include <iostream>
#include <chrono>

void AlphaZeroMove(Board &board, int player, std::vector<LOC> &pace)
{
    for (;;)
    {
        az_expert::Result expert = az_expert::applyPreSearch(board, player, &pace);
        if (expert.decision == az_expert::Decision::GameEnded ||
            expert.decision == az_expert::Decision::TurnEnded)
            return;
        if (expert.decision == az_expert::Decision::ContinueSamePlayer)
            continue;

        AZMCTS mcts;
        AZNode *root = mcts.search(board, player, AZ_SIMULATIONS, AZ_TIME_LIMIT_MS);

        // 选择最佳动作（贪心，temperature=0）
        int bestAction = mcts.selectAction(root, 0.0f);

        if (bestAction >= 0)
        {
            LOC loc = actionToLoc(bestAction);
            int earned = board.move(player, loc);
            pace.emplace_back(loc);

            // 释放搜索树
            deleteAZTree(root);
            delete root;

            // 吃到格子 → 同一玩家继续（回到循环顶部吃 C 型格 + 再次决策）
            if (earned > 0 && !board.ifEnd())
                continue;
        }
        else
        {
            deleteAZTree(root);
            delete root;
        }

        return; // 没吃到格子或无合法动作，回合结束
    }
}
