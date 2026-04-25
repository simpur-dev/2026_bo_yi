#include "az_encoder.h"
#include "../define.h"

Tensor encodeBoard(const Board &board, int currentPlayer)
{
    Tensor tensor{};
    int opponent = -currentPlayer;

    for (int i = 0; i < AZ_BOARD_SIZE; i++)
    {
        for (int j = 0; j < AZ_BOARD_SIZE; j++)
        {
            int val = board.map[i][j];

            // 通道 0: 当前玩家占据的格子
            if (isOdd(i) && isOdd(j) && val == currentPlayer)
                tensorAt(tensor, 0, i, j) = 1.0f;

            // 通道 1: 对手占据的格子
            if (isOdd(i) && isOdd(j) && val == opponent)
                tensorAt(tensor, 1, i, j) = 1.0f;

            // 通道 2: 已被占据的边
            if (val == OCCLINE)
                tensorAt(tensor, 2, i, j) = 1.0f;

            // 通道 3: 当前可落子的边
            if (val == HENG || val == SHU)
                tensorAt(tensor, 3, i, j) = 1.0f;

            // 通道 4: DOT 点位
            if (isEven(i) && isEven(j))
                tensorAt(tensor, 4, i, j) = 1.0f;

            // 通道 5: 当前玩家标识
            tensorAt(tensor, 5, i, j) = (currentPlayer == BLACK) ? 1.0f : -1.0f;

            // 通道 6: 格子自由度归一化
            if (isOdd(i) && isOdd(j) && val != BLACK && val != WHITE)
            {
                // Board::getFreedom 不是 const，所以需要 const_cast
                Board &mutableBoard = const_cast<Board &>(board);
                int freedom = mutableBoard.getFreedom(i, j);
                tensorAt(tensor, 6, i, j) = static_cast<float>(freedom) / 4.0f;
            }
        }
    }
    return tensor;
}
