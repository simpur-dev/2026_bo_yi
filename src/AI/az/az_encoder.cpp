#include "az_encoder.h"
#include "../define.h"

// 直接从 map 计算格子自由度，避免对 Board 的 const_cast
// 格子 (i,j) 的四条边: 上 (i-1,j)、下 (i+1,j)、左 (i,j-1)、右 (i,j+1)
// 未被占据的边（HENG 或 SHU）计为一度自由
static int computeFreedom(const int map[LEN][LEN], int i, int j)
{
    int freedom = 0;
    if (map[i - 1][j] != OCCLINE) freedom++; // 上
    if (map[i + 1][j] != OCCLINE) freedom++; // 下
    if (map[i][j - 1] != OCCLINE) freedom++; // 左
    if (map[i][j + 1] != OCCLINE) freedom++; // 右
    return freedom;
}

Tensor encodeBoard(const Board &board, int currentPlayer)
{
    Tensor tensor{};
    int opponent = -currentPlayer;

    for (int i = 0; i < AZ_BOARD_SIZE; i++)
    {
        for (int j = 0; j < AZ_BOARD_SIZE; j++)
        {
            int val = board.map[i][j];
            bool isBoxPos = (i % 2 == 1) && (j % 2 == 1); // 格子位置：行列均为奇数
            bool isEdgePos = (val == HENG || val == SHU || val == OCCLINE);
            bool isDotPos = (i % 2 == 0) && (j % 2 == 0);

            // 通道 0: 当前玩家占据的格子
            if (isBoxPos && val == currentPlayer)
                tensorAt(tensor, 0, i, j) = 1.0f;

            // 通道 1: 对手占据的格子
            if (isBoxPos && val == opponent)
                tensorAt(tensor, 1, i, j) = 1.0f;

            // 通道 2: 已被占据的边
            if (val == OCCLINE)
                tensorAt(tensor, 2, i, j) = 1.0f;

            // 通道 3: 当前可落子的边（未被占据的横边或竖边）
            if (val == HENG || val == SHU)
                tensorAt(tensor, 3, i, j) = 1.0f;

            // 通道 4: DOT 点位
            if (isDotPos)
                tensorAt(tensor, 4, i, j) = 1.0f;

            // 通道 5: 当前玩家标识（全图统一值）
            tensorAt(tensor, 5, i, j) = (currentPlayer == BLACK) ? 1.0f : -1.0f;

            // 通道 6: 格子自由度归一化 (freedom / 4.0)
            // 仅对未被占据的格子位置填写；已被占据的格子自由度为 0
            if (isBoxPos)
            {
                if (val == BLACK || val == WHITE)
                    tensorAt(tensor, 6, i, j) = 0.0f; // 已被占据
                else
                    tensorAt(tensor, 6, i, j) = static_cast<float>(computeFreedom(board.map, i, j)) / 4.0f;
            }
        }
    }
    return tensor;
}
