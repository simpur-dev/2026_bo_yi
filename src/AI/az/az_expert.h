#pragma once
#include "../board.h"
#include <vector>

namespace az_expert
{

enum class Decision
{
    SearchNeeded,
    ContinueSamePlayer,
    TurnEnded,
    GameEnded
};

struct Options
{
    bool suppressOutput = false;
};

struct Result
{
    Decision decision = Decision::SearchNeeded;
    int moveCount = 0;
    bool usedForcedCapture = false;
    bool usedEndgameSolver = false;
    bool usedDoubleCross = false;
};

bool isEndgameAfterForcedCaptures(const Board &board, int player);
Result applyPreSearch(Board &board, int &player, std::vector<LOC> *pace = nullptr, const Options &options = Options{});
Result normalizeToSearch(Board &board, int &player, std::vector<LOC> *pace = nullptr, const Options &options = Options{});

}
