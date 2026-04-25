#pragma once
#include "../board.h"
#include "az_types.h"
#include <array>

// 棋盘编码器：将 Board 转换为神经网络输入张量
// 输出维度: AZ_CHANNELS × AZ_BOARD_SIZE × AZ_BOARD_SIZE = 7 × 11 × 11
//
// 通道含义:
//   0: 当前玩家占据的格子 (1.0 / 0.0)
//   1: 对手占据的格子     (1.0 / 0.0)
//   2: 已被占据的边       (1.0 / 0.0)
//   3: 当前可落子的边     (1.0 / 0.0)
//   4: DOT 点位           (1.0 / 0.0)
//   5: 当前玩家标识       (全填 1.0 表示 BLACK, -1.0 表示 WHITE)
//   6: 格子自由度归一化   (freedom / 4.0)

using Tensor = std::array<float, AZ_CHANNELS * AZ_BOARD_SIZE * AZ_BOARD_SIZE>;

// 编码棋盘状态
Tensor encodeBoard(const Board &board, int currentPlayer);

// 辅助：获取张量中某通道某位置的引用
inline float &tensorAt(Tensor &t, int c, int i, int j)
{
    return t[c * AZ_BOARD_SIZE * AZ_BOARD_SIZE + i * AZ_BOARD_SIZE + j];
}

inline float tensorAt(const Tensor &t, int c, int i, int j)
{
    return t[c * AZ_BOARD_SIZE * AZ_BOARD_SIZE + i * AZ_BOARD_SIZE + j];
}
