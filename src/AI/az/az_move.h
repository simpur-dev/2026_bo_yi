#pragma once
#include "../board.h"
#include <vector>

// AlphaZero 风格的 AI 移动入口
// 替代原有的 UCTMoveWithSacrifice()
// board: 当前棋盘状态
// player: 当前玩家
// pace: 输出参数，记录本回合所有落子步骤
void AlphaZeroMove(Board &board, int player, std::vector<LOC> &pace);
