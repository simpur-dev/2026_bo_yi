#pragma once
#include "../define.h"
#include "../board.h"
#include "az_types.h"
#include <array>
#include <vector>

// 动作空间映射
// action  0-29: 横边 HENG (6行×5列)
// action 30-59: 竖边 SHU  (5行×6列)

// LOC -> action index
int locToAction(LOC loc);

// action index -> LOC
LOC actionToLoc(int action);

// 判断某个 action 在当前棋盘上是否合法（即对应的边未被占据）
bool isLegalAction(const Board &board, int action);

// 获取当前棋盘所有合法动作的 mask（长度 60，1 为合法，0 为非法）
std::array<float, AZ_ACTION_SIZE> getLegalMask(const Board &board);

// 获取所有合法动作的索引列表
std::vector<int> getLegalActions(const Board &board);
