#include "az_action.h"
#include <iostream>
#include <set>

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

bool validateActionMapping()
{
    bool ok = true;
    std::set<int> seenActions;
    std::set<std::pair<int, int>> seenLocs;

    // 检查 1: 遍历所有棋盘边位置，确认 locToAction -> actionToLoc 双向一致
    // 横边: i 偶数 (0,2,4,6,8,10)，j 奇数 (1,3,5,7,9) => 6×5 = 30
    for (int i = 0; i <= 10; i += 2)
    {
        for (int j = 1; j <= 9; j += 2)
        {
            LOC loc = {i, j};
            int action = locToAction(loc);
            if (action < 0 || action >= 30)
            {
                std::cerr << "[FAIL] HENG (" << i << "," << j << ") -> action " << action << " out of range [0,29]\n";
                ok = false;
                continue;
            }
            LOC restored = actionToLoc(action);
            if (restored != loc)
            {
                std::cerr << "[FAIL] HENG (" << i << "," << j << ") -> action " << action
                          << " -> (" << restored.first << "," << restored.second << ")\n";
                ok = false;
            }
            seenActions.insert(action);
            seenLocs.insert({i, j});
        }
    }

    // 竖边: i 奇数 (1,3,5,7,9)，j 偶数 (0,2,4,6,8,10) => 5×6 = 30
    for (int i = 1; i <= 9; i += 2)
    {
        for (int j = 0; j <= 10; j += 2)
        {
            LOC loc = {i, j};
            int action = locToAction(loc);
            if (action < 30 || action >= 60)
            {
                std::cerr << "[FAIL] SHU (" << i << "," << j << ") -> action " << action << " out of range [30,59]\n";
                ok = false;
                continue;
            }
            LOC restored = actionToLoc(action);
            if (restored != loc)
            {
                std::cerr << "[FAIL] SHU (" << i << "," << j << ") -> action " << action
                          << " -> (" << restored.first << "," << restored.second << ")\n";
                ok = false;
            }
            seenActions.insert(action);
            seenLocs.insert({i, j});
        }
    }

    // 检查 2: 总共恰好 60 个不同的 action
    if (seenActions.size() != AZ_ACTION_SIZE)
    {
        std::cerr << "[FAIL] Expected " << AZ_ACTION_SIZE << " unique actions, got " << seenActions.size() << "\n";
        ok = false;
    }

    // 检查 3: 总共恰好 60 个不同的 LOC
    if (seenLocs.size() != AZ_ACTION_SIZE)
    {
        std::cerr << "[FAIL] Expected " << AZ_ACTION_SIZE << " unique LOCs, got " << seenLocs.size() << "\n";
        ok = false;
    }

    // 检查 4: 从 action 反查也全部正确
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        LOC loc = actionToLoc(a);
        int roundtrip = locToAction(loc);
        if (roundtrip != a)
        {
            std::cerr << "[FAIL] action " << a << " -> (" << loc.first << "," << loc.second
                      << ") -> action " << roundtrip << "\n";
            ok = false;
        }
    }

    if (ok)
        std::cerr << "[OK] All 60 action mappings validated successfully.\n";

    return ok;
}
