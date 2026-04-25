#include "az_action.h"

// 横边 HENG: map[i][j], i 偶数 (0,2,4,6,8,10), j 奇数 (1,3,5,7,9)
// action = (i/2)*5 + (j-1)/2，范围 0-29
//
// 竖边 SHU: map[i][j], i 奇数 (1,3,5,7,9), j 偶数 (0,2,4,6,8,10)
// action = 30 + ((i-1)/2)*6 + j/2，范围 30-59

int locToAction(LOC loc)
{
    int i = loc.first;
    int j = loc.second;

    if (isEven(i) && !isEven(j)) // 横边
    {
        return (i / 2) * 5 + (j - 1) / 2;
    }
    else if (!isEven(i) && isEven(j)) // 竖边
    {
        return 30 + ((i - 1) / 2) * 6 + j / 2;
    }
    return -1; // 非法坐标
}

LOC actionToLoc(int action)
{
    if (action < 0 || action >= AZ_ACTION_SIZE)
        return {-1, -1};

    if (action < 30) // 横边
    {
        int i = (action / 5) * 2;
        int j = (action % 5) * 2 + 1;
        return {i, j};
    }
    else // 竖边
    {
        int a = action - 30;
        int i = (a / 6) * 2 + 1;
        int j = (a % 6) * 2;
        return {i, j};
    }
}

bool isLegalAction(const Board &board, int action)
{
    if (action < 0 || action >= AZ_ACTION_SIZE)
        return false;
    LOC loc = actionToLoc(action);
    return board.isFreeLine(loc);
}

std::array<float, AZ_ACTION_SIZE> getLegalMask(const Board &board)
{
    std::array<float, AZ_ACTION_SIZE> mask{};
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        mask[a] = isLegalAction(board, a) ? 1.0f : 0.0f;
    }
    return mask;
}

std::vector<int> getLegalActions(const Board &board)
{
    std::vector<int> actions;
    actions.reserve(AZ_ACTION_SIZE);
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (isLegalAction(board, a))
            actions.push_back(a);
    }
    return actions;
}
