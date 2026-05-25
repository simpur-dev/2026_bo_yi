#include "AI/az/az_mcts.h"
#include "AI/az/az_node.h"
#include "AI/az/az_action.h"
#include "AI/az/az_types.h"
#include "AI/az/az_evaluator.h"
#include "AI/az/az_onnx_evaluator.h"
#include "AI/az/az_expert.h"
#include "AI/assess.h"
#include "AI/board.h"
#include "AI/define.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <iomanip>
#include <streambuf>
#include <vector>

// 前向声明终局求解器（定义在 UCT.cpp 中）
void latterSituationMove(Board &CB, int Player, std::vector<LOC> &pace);

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
    int simulations = 800;
    int tempMoves = 4;
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

    // 检测文件扩展名
    std::string ext;
    auto dotPos = spec.rfind('.');
    if (dotPos != std::string::npos)
        ext = spec.substr(dotPos);

#ifdef USE_ONNX
    if (ext == ".onnx")
    {
        auto evaluator = std::make_unique<ONNXEvaluator>();
        if (!evaluator->loadModel(spec))
            return nullptr;
        return evaluator;
    }
#else
    if (ext == ".onnx")
    {
        std::cerr << "[Error] ONNX model specified but USE_ONNX not enabled at compile time\n";
        return nullptr;
    }
#endif

    // 默认: MLP 二进制权重
    auto evaluator = std::make_unique<NeuralNetEvaluator>();
    if (!evaluator->loadModel(spec))
        return nullptr;
    return evaluator;
}

int playOneGame(AZEvaluator &blackEvaluator,
                AZEvaluator &whiteEvaluator,
                int &blackScore,
                int &whiteScore,
                int simulations,
                int tempMoves,
                bool verbose)
{
    Board board;
    int player = BLACK;
    int moveCount = 0;
    NullBuffer nullBuffer;

    while (!board.ifEnd() && moveCount < 500)
    {
        {
            az_expert::Result expert = az_expert::normalizeToSearch(
                board, player, nullptr, az_expert::Options{!verbose});
            moveCount += expert.moveCount;
            if (expert.decision == az_expert::Decision::GameEnded)
                break;
        }

        // 5. 设置评估器
        if (player == BLACK)
            setEvaluator(&blackEvaluator);
        else
            setEvaluator(&whiteEvaluator);

        // 6. MCTS 搜索
        MCTSConfig config = MCTSConfig::evaluation(simulations, 0);
        // 前几步加入轻微噪声以增加多样性
        if (tempMoves > 0 && moveCount < tempMoves)
        {
            config.addRootNoise = true;
            config.dirichletAlpha = 0.3f;
            config.dirichletFrac = 0.15f;
        }

        AZMCTS mcts;
        std::streambuf *oldBuf = nullptr;
        if (!verbose)
            oldBuf = std::cerr.rdbuf(&nullBuffer);
        AZNode *root = mcts.search(board, player, config);
        if (!verbose && oldBuf)
            std::cerr.rdbuf(oldBuf);

        // 选择动作：前几步温度采样，之后贪心
        float temp = (tempMoves > 0 && moveCount < tempMoves) ? 0.5f : 0.0f;
        int action = mcts.selectAction(root, temp);

        deleteAZTree(root);
        delete root;

        if (action < 0)
        {
            player = -player;
            continue;
        }

        // 7. 执行动作
        LOC loc = actionToLoc(action);
        int earned = board.move(player, loc);
        moveCount++;

        if (verbose)
            std::cerr << "[Move] #" << moveCount
                      << " player=" << (player == BLACK ? "B" : "W")
                      << " action=" << action
                      << " earned=" << earned << "\n";

        // 8. 玩家切换：吃到格子则继续，否则换手
        if (earned == 0)
            player = -player;
        // else: 同一玩家继续（回到循环顶部吃 C 型格）
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
    int winner = playOneGame(blackEvaluator, whiteEvaluator, blackScore, whiteScore,
                              stats.simulations, stats.tempMoves, verbose);

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
    std::cout << "Games: " << stats.games
              << "  Sims: " << stats.simulations
              << "  TempMoves: " << stats.tempMoves << "\n";
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
        double aWinRate = 100.0 * stats.aWins / stats.games;
        std::cout << "A winrate: " << std::fixed << std::setprecision(1) << aWinRate << "%\n";
        std::cout << "Avg score BLACK: " << static_cast<double>(stats.blackScoreSum) / stats.games << "\n";
        std::cout << "Avg score WHITE: " << static_cast<double>(stats.whiteScoreSum) / stats.games << "\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: evaluate_ai <games_per_side> <modelA|heuristic> <modelB|heuristic> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --sims N        MCTS simulations per move (default: 800)\n";
        std::cerr << "  --temp N        Temperature moves for diversity (default: 4, 0=deterministic)\n";
        std::cerr << "  --verbose       Print detailed move info\n";
        std::cerr << "  --fixed-color   A always BLACK, B always WHITE (no side swap)\n";
        return 1;
    }

    int gamesPerSide = std::atoi(argv[1]);
    if (gamesPerSide <= 0)
        gamesPerSide = 10;

    // 解析可选参数
    int simulations = 800;
    int tempMoves = 4;
    bool verbose = false;
    bool fixedColor = false;
    for (int i = 4; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--sims" && i + 1 < argc)
            simulations = std::atoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc)
            tempMoves = std::atoi(argv[++i]);
        else if (arg == "--verbose")
            verbose = true;
        else if (arg == "--fixed-color")
            fixedColor = true;
    }

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
    std::cout << "Simulations: " << simulations << "\n";
    std::cout << "Temp moves: " << tempMoves << "\n";
    std::cout << "A: " << a.name << "\n";
    std::cout << "B: " << b.name << "\n";

    MatchStats stats;
    stats.simulations = simulations;
    stats.tempMoves = tempMoves;
    int gameIndex = 1;
    if (fixedColor)
    {
        // A always BLACK, B always WHITE — for dual model per-color arena
        std::cout << "Mode: fixed-color (A=BLACK, B=WHITE)\n";
        for (int i = 0; i < gamesPerSide; i++)
            runGame(stats, a, b, true, gameIndex++, verbose);
    }
    else
    {
        for (int i = 0; i < gamesPerSide; i++)
            runGame(stats, a, b, true, gameIndex++, verbose);
        for (int i = 0; i < gamesPerSide; i++)
            runGame(stats, a, b, false, gameIndex++, verbose);
    }

    printSummary(stats);
    return 0;
}
