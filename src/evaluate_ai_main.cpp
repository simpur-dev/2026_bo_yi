#include "AI/az/az_move.h"
#include "AI/az/az_evaluator.h"
#include "AI/board.h"
#include "AI/define.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <streambuf>
#include <vector>

struct PlayerConfig
{
    std::string name;
    std::unique_ptr<AZEvaluator> evaluator;
};

struct MatchStats
{
    int games = 0;
    int blackWins = 0;
    int whiteWins = 0;
    int unfinished = 0;
    int aWins = 0;
    int bWins = 0;
    int aAsBlackWins = 0;
    int bAsBlackWins = 0;
    int aAsWhiteWins = 0;
    int bAsWhiteWins = 0;
    int blackScoreSum = 0;
    int whiteScoreSum = 0;
};

class NullBuffer : public std::streambuf
{
  public:
    int overflow(int c) override { return c; }
};

std::unique_ptr<AZEvaluator> createEvaluator(const std::string &spec)
{
    if (spec == "heuristic" || spec == "HEURISTIC")
        return std::make_unique<HeuristicEvaluator>();

    auto evaluator = std::make_unique<NeuralNetEvaluator>();
    if (!evaluator->loadModel(spec))
        return nullptr;
    return evaluator;
}

int playOneGame(AZEvaluator &blackEvaluator,
                AZEvaluator &whiteEvaluator,
                int &blackScore,
                int &whiteScore,
                bool verbose)
{
    Board board;
    int player = BLACK;
    int guard = 0;
    NullBuffer nullBuffer;

    while (!board.ifEnd() && guard < 200)
    {
        std::vector<LOC> pace;
        if (player == BLACK)
            setEvaluator(&blackEvaluator);
        else
            setEvaluator(&whiteEvaluator);

        std::streambuf *oldBuffer = nullptr;
        if (!verbose)
            oldBuffer = std::cerr.rdbuf(&nullBuffer);
        AlphaZeroMove(board, player, pace);
        if (!verbose)
            std::cerr.rdbuf(oldBuffer);

        if (pace.empty() && !board.ifEnd())
            break;

        player = -player;
        guard++;
    }

    blackScore = board.blackBox;
    whiteScore = board.whiteBox;
    return board.getWinner();
}

void runGame(MatchStats &stats,
             PlayerConfig &a,
             PlayerConfig &b,
             bool aIsBlack,
             int gameIndex,
             bool verbose)
{
    int blackScore = 0;
    int whiteScore = 0;
    AZEvaluator &blackEvaluator = aIsBlack ? *a.evaluator : *b.evaluator;
    AZEvaluator &whiteEvaluator = aIsBlack ? *b.evaluator : *a.evaluator;
    int winner = playOneGame(blackEvaluator, whiteEvaluator, blackScore, whiteScore, verbose);

    stats.games++;
    stats.blackScoreSum += blackScore;
    stats.whiteScoreSum += whiteScore;

    bool aWon = (aIsBlack && winner == BLACK) || (!aIsBlack && winner == WHITE);
    bool bWon = (aIsBlack && winner == WHITE) || (!aIsBlack && winner == BLACK);

    if (winner == BLACK)
        stats.blackWins++;
    else if (winner == WHITE)
        stats.whiteWins++;
    else
        stats.unfinished++;

    if (aWon)
    {
        stats.aWins++;
        if (aIsBlack)
            stats.aAsBlackWins++;
        else
            stats.aAsWhiteWins++;
    }
    if (bWon)
    {
        stats.bWins++;
        if (aIsBlack)
            stats.bAsWhiteWins++;
        else
            stats.bAsBlackWins++;
    }

    std::cerr << "[Eval] Game " << gameIndex
              << "  " << (aIsBlack ? "A=BLACK B=WHITE" : "B=BLACK A=WHITE")
              << "  score=" << blackScore << ":" << whiteScore
              << "  winner=" << (winner == BLACK ? "BLACK" : winner == WHITE ? "WHITE" : "NONE")
              << "  modelWinner=" << (aWon ? "A" : bWon ? "B" : "NONE") << "\n";
}

void printSummary(const MatchStats &stats)
{
    std::cout << "\n=== Evaluation Summary ===\n";
    std::cout << "Games: " << stats.games << "\n";
    std::cout << "BLACK wins: " << stats.blackWins
              << "  WHITE wins: " << stats.whiteWins
              << "  Unfinished: " << stats.unfinished << "\n";
    std::cout << "A wins: " << stats.aWins
              << "  B wins: " << stats.bWins << "\n";
    std::cout << "A as BLACK wins: " << stats.aAsBlackWins << "\n";
    std::cout << "A as WHITE wins: " << stats.aAsWhiteWins << "\n";
    std::cout << "B as BLACK wins: " << stats.bAsBlackWins << "\n";
    std::cout << "B as WHITE wins: " << stats.bAsWhiteWins << "\n";
    if (stats.games > 0)
    {
        std::cout << "Avg score BLACK: " << static_cast<double>(stats.blackScoreSum) / stats.games << "\n";
        std::cout << "Avg score WHITE: " << static_cast<double>(stats.whiteScoreSum) / stats.games << "\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: evaluate_ai <games_per_side> <modelA|heuristic> <modelB|heuristic> [--verbose]\n";
        return 1;
    }

    int gamesPerSide = std::atoi(argv[1]);
    if (gamesPerSide <= 0)
        gamesPerSide = 10;
    bool verbose = (argc > 4 && std::string(argv[4]) == "--verbose");

    PlayerConfig a;
    PlayerConfig b;
    a.name = argv[2];
    b.name = argv[3];
    a.evaluator = createEvaluator(a.name);
    b.evaluator = createEvaluator(b.name);

    if (!a.evaluator || !b.evaluator)
    {
        std::cerr << "Failed to create evaluator\n";
        return 1;
    }

    std::cout << "=== Dots & Boxes AI Evaluation ===\n";
    std::cout << "Games per side: " << gamesPerSide << "\n";
    std::cout << "A: " << a.name << "\n";
    std::cout << "B: " << b.name << "\n";

    MatchStats stats;
    int gameIndex = 1;
    for (int i = 0; i < gamesPerSide; i++)
        runGame(stats, a, b, true, gameIndex++, verbose);
    for (int i = 0; i < gamesPerSide; i++)
        runGame(stats, a, b, false, gameIndex++, verbose);

    printSummary(stats);
    return 0;
}
