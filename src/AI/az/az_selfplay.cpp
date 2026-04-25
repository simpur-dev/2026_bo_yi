#include "az_selfplay.h"
#include "az_mcts.h"
#include "az_action.h"
#include "az_node.h"
#include "../assess.h"
#include "../define.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>

// 前向声明后期决策函数（定义在 UCT.cpp 中）
void latterSituationMove(Board &CB, int Player, std::vector<LOC> &pace);

// ========== SelfPlayEngine::run ==========

void SelfPlayEngine::run(int numGames, int numSimulations, const std::string &outputDir)
{
    // 确保输出目录存在
    std::filesystem::create_directories(outputDir);

    std::vector<SelfPlaySample> allSamples;
    allSamples.reserve(numGames * 40); // 预估每盘约 40 个决策点

    int totalSamples = 0;
    int blackWins = 0, whiteWins = 0, draws = 0;

    auto totalStart = std::chrono::steady_clock::now();

    for (int g = 0; g < numGames; g++)
    {
        auto gameStart = std::chrono::steady_clock::now();

        std::vector<SelfPlaySample> gameSamples;
        SelfPlayResult result = playOneGame(numSimulations, 1.0f, 20, gameSamples);

        auto gameEnd = std::chrono::steady_clock::now();
        int gameMs = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(gameEnd - gameStart).count());

        // 统计
        if (result.winner == BLACK)
            blackWins++;
        else if (result.winner == WHITE)
            whiteWins++;
        else
            draws++;
        totalSamples += result.numSamples;

        std::cerr << "[SelfPlay] Game " << (g + 1) << "/" << numGames
                  << "  score=" << result.blackScore << ":" << result.whiteScore
                  << "  winner=" << (result.winner == BLACK ? "BLACK" : result.winner == WHITE ? "WHITE" : "DRAW")
                  << "  samples=" << result.numSamples
                  << "  moves=" << result.numMoves
                  << "  time=" << gameMs << "ms\n";

        allSamples.insert(allSamples.end(), gameSamples.begin(), gameSamples.end());

        // 每 50 盘写一次文件（减少内存占用）
        if ((g + 1) % 50 == 0 || g == numGames - 1)
        {
            std::ostringstream filename;
            filename << outputDir << "/selfplay_" << std::setfill('0') << std::setw(6) << gamesPlayed << ".jsonl";
            writeSamples(filename.str(), allSamples);
            std::cerr << "[SelfPlay] Wrote " << allSamples.size() << " samples to " << filename.str() << "\n";
            allSamples.clear();
        }

        gamesPlayed++;
    }

    auto totalEnd = std::chrono::steady_clock::now();
    int totalMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count());

    std::cerr << "\n[SelfPlay] === Summary ===\n"
              << "  Games:   " << numGames << "\n"
              << "  Samples: " << totalSamples << "\n"
              << "  BLACK wins: " << blackWins << "  WHITE wins: " << whiteWins << "  Draws: " << draws << "\n"
              << "  Total time: " << totalMs / 1000 << "s\n"
              << "  Avg time/game: " << totalMs / numGames << "ms\n";
}

// ========== SelfPlayEngine::playOneGame ==========

SelfPlayResult SelfPlayEngine::playOneGame(int numSimulations,
                                           float temperature, int temperatureMoves,
                                           std::vector<SelfPlaySample> &samples)
{
    Board board;
    int currentPlayer = BLACK;
    int moveCount = 0;
    int sampleStart = static_cast<int>(samples.size());

    while (!board.ifEnd())
    {
        // 1. 强制吃掉所有 C 型格
        board.eatAllCTypeBoxes(currentPlayer);

        if (board.ifEnd())
            break;

        // 2. 检查是否进入终局（只剩长链/环）
        {
            Board test = board;
            test.eatAllCTypeBoxes(currentPlayer);
            if (test.getFilterMoveNum() == 0)
            {
                // 终局：使用精确求解器，不生成训练样本
                std::vector<LOC> pace;
                latterSituationMove(board, currentPlayer, pace);
                moveCount += static_cast<int>(pace.size());
                currentPlayer = -currentPlayer;
                continue;
            }
        }

        // 3. 记录训练样本的输入特征
        SelfPlaySample sample;
        sample.boardTensor = encodeBoard(board, currentPlayer);
        sample.player = currentPlayer;
        sample.legalMask = getLegalMask(board);
        sample.value = 0.0f; // 稍后回填

        // 4. PUCT 搜索
        AZMCTS mcts;
        AZNode *root = mcts.search(board, currentPlayer, numSimulations, 0);

        // 记录策略（访问次数分布）
        sample.policy = mcts.getVisitDistribution(root);

        // 5. 选择动作（前 N 步温度采样，之后贪心）
        float temp = (moveCount < temperatureMoves) ? temperature : 0.0f;
        int action = mcts.selectAction(root, temp);

        // 释放搜索树
        deleteAZTree(root);
        delete root;

        if (action < 0)
        {
            // 无合法动作（不应该发生）
            currentPlayer = -currentPlayer;
            continue;
        }

        // 保存样本
        samples.push_back(sample);

        // 6. 执行动作
        LOC loc = actionToLoc(action);
        int earned = board.move(currentPlayer, loc);
        moveCount++;

        // 7. 确定下一个玩家
        if (earned > 0)
        {
            // 吃到格子 → 同一玩家继续（回到循环顶部吃 C 型格）
        }
        else
        {
            // 没吃到 → 换手
            currentPlayer = -currentPlayer;
        }
    }

    // 8. 对局结束，用最终胜负回填所有样本的 value
    int winner = board.getWinner();
    int sampleEnd = static_cast<int>(samples.size());

    for (int i = sampleStart; i < sampleEnd; i++)
    {
        if (winner == 0)
            samples[i].value = 0.0f;
        else if (samples[i].player == winner)
            samples[i].value = 1.0f;
        else
            samples[i].value = -1.0f;
    }

    // 返回结果
    SelfPlayResult result;
    result.blackScore = board.blackBox;
    result.whiteScore = board.whiteBox;
    result.winner = winner;
    result.numSamples = sampleEnd - sampleStart;
    result.numMoves = moveCount;
    return result;
}

// ========== SelfPlayEngine::writeSamples ==========

void SelfPlayEngine::writeSamples(const std::string &filepath,
                                  const std::vector<SelfPlaySample> &samples)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[SelfPlay] ERROR: Cannot open " << filepath << "\n";
        return;
    }

    constexpr int TENSOR_SIZE = AZ_CHANNELS * AZ_BOARD_SIZE * AZ_BOARD_SIZE;

    for (const auto &s : samples)
    {
        file << "{\"board\":[";
        for (int i = 0; i < TENSOR_SIZE; i++)
        {
            if (i > 0) file << ',';
            // 用短格式输出：0 和 1 不加小数点，其他保留 4 位
            if (s.boardTensor[i] == 0.0f)
                file << '0';
            else if (s.boardTensor[i] == 1.0f)
                file << '1';
            else if (s.boardTensor[i] == -1.0f)
                file << "-1";
            else
                file << std::fixed << std::setprecision(4) << s.boardTensor[i];
        }

        file << "],\"player\":" << s.player;

        file << ",\"legal_mask\":[";
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
        {
            if (i > 0) file << ',';
            file << static_cast<int>(s.legalMask[i]);
        }

        file << "],\"policy\":[";
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
        {
            if (i > 0) file << ',';
            if (s.policy[i] == 0.0f)
                file << '0';
            else
                file << std::fixed << std::setprecision(6) << s.policy[i];
        }

        file << "],\"value\":" << std::fixed << std::setprecision(1) << s.value;
        file << "}\n";
    }

    file.close();
}
