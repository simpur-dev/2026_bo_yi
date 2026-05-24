#include "az_expert.h"
#include "../assess.h"
#include <iostream>
#include <streambuf>

void latterSituationMove(Board &CB, int Player, std::vector<LOC> &pace);

namespace az_expert
{

namespace
{
class NullBuffer : public std::streambuf
{
  public:
    int overflow(int c) override { return c; }
};

void appendPace(std::vector<LOC> *dst, const std::vector<LOC> &src)
{
    if (!dst)
        return;
    dst->insert(dst->end(), src.begin(), src.end());
}

void runEndgameSolver(Board &board, int player, std::vector<LOC> &pace, const Options &options)
{
    if (!options.suppressOutput)
    {
        latterSituationMove(board, player, pace);
        return;
    }

    NullBuffer nullBuffer;
    std::streambuf *oldBuf = std::cerr.rdbuf(&nullBuffer);
    latterSituationMove(board, player, pace);
    std::cerr.rdbuf(oldBuf);
}
}

bool isEndgameAfterForcedCaptures(const Board &board, int player)
{
    Board test = board;
    test.eatAllCTypeBoxes(player);
    return test.getFilterMoveNum() == 0;
}

Result applyPreSearch(Board &board, int &player, std::vector<LOC> *pace, const Options &options)
{
    Result result;

    std::vector<LOC> forcedPace;
    board.eatAllCTypeBoxes(player, forcedPace);
    appendPace(pace, forcedPace);
    result.moveCount += static_cast<int>(forcedPace.size());
    result.usedForcedCapture = !forcedPace.empty();

    if (board.ifEnd())
    {
        result.decision = Decision::GameEnded;
        return result;
    }

    if (isEndgameAfterForcedCaptures(board, player))
    {
        std::vector<LOC> endgamePace;
        runEndgameSolver(board, player, endgamePace, options);
        appendPace(pace, endgamePace);
        result.moveCount += static_cast<int>(endgamePace.size());
        result.usedEndgameSolver = true;
        player = -player;
        result.decision = board.ifEnd() ? Decision::GameEnded : Decision::TurnEnded;
        return result;
    }

    BoxBoard dead(board);
    bool deadChain = dead.getDeadChainExist();
    bool deadCircle = dead.getDeadCircleExist();
    if (!(deadCircle || deadChain))
        return result;

    int sacrificeBoxNum = deadCircle ? 4 : 2;
    BoxBoard sim(board);
    sim.eatAllCTypeBoxes(player);
    LOC boxNum = sim.getEarlyRationalBoxNum();

    if (boxNum.first - boxNum.second <= sacrificeBoxNum)
    {
        std::vector<LOC> eatPace;
        board.eatAllCTypeBoxes(player, eatPace);
        appendPace(pace, eatPace);
        result.moveCount += static_cast<int>(eatPace.size());
        result.usedForcedCapture = result.usedForcedCapture || !eatPace.empty();
        if (board.ifEnd())
            result.decision = Decision::GameEnded;
        else if (!eatPace.empty())
            result.decision = Decision::ContinueSamePlayer;
        else
            result.decision = Decision::SearchNeeded;
        return result;
    }

    if (sacrificeBoxNum == 2)
    {
        for (;;)
        {
            Board testBoard = board;
            testBoard.eatCBox(player);
            BoxBoard deadTest(testBoard);
            if (deadTest.getDeadChainExist())
            {
                LOC t = board.eatCBox(player);
                if (t.first >= 0)
                {
                    if (pace)
                        pace->emplace_back(t);
                    result.moveCount++;
                }
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
                if (t.first >= 0)
                {
                    if (pace)
                        pace->emplace_back(t);
                    result.moveCount++;
                }
            }
            else
                break;
        }
    }

    LOC dcMove = board.getDoubleCrossLoc(player);
    board.move(player, dcMove);
    if (pace)
        pace->emplace_back(dcMove);

    std::vector<LOC> tempPace;
    for (;;)
    {
        if (!board.getCTypeBoxLimit(player, tempPace))
            break;
    }
    appendPace(pace, tempPace);

    result.moveCount += 1 + static_cast<int>(tempPace.size());
    result.usedDoubleCross = true;
    player = -player;
    result.decision = board.ifEnd() ? Decision::GameEnded : Decision::TurnEnded;
    return result;
}

Result normalizeToSearch(Board &board, int &player, std::vector<LOC> *pace, const Options &options)
{
    Result total;
    for (;;)
    {
        Result step = applyPreSearch(board, player, pace, options);
        total.moveCount += step.moveCount;
        total.usedForcedCapture = total.usedForcedCapture || step.usedForcedCapture;
        total.usedEndgameSolver = total.usedEndgameSolver || step.usedEndgameSolver;
        total.usedDoubleCross = total.usedDoubleCross || step.usedDoubleCross;

        if (step.decision == Decision::SearchNeeded ||
            step.decision == Decision::GameEnded)
        {
            total.decision = step.decision;
            return total;
        }
    }
}

}
