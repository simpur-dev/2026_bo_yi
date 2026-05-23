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
#include <cmath>
#include <filesystem>

// 前向声明后期决策函数（定义在 UCT.cpp 中）
void latterSituationMove(Board &CB, int Player, std::vector<LOC> &pace);

// ========== 死链/死环 + Double-Cross 启发式 ==========
//
// 与 evaluate_ai_main.cpp 中 playOneGame() 步骤 3 完全一致的预处理。
// 在 selfplay 中复用，确保训练数据分布与部署一致：
//
//   - 若局面存在死链/死环：根据"全吃"vs"牺牲"得分差选择更优方案；
//     该启发式会取代 MCTS 决策，因此 selfplay 在这种局面下不应采样
//     训练样本（模型若在此学习，部署时该决策被覆盖→训练信号失效）。
//
// 返回值:
//   0  - 局面不存在死链/死环, 调用方应继续走 MCTS/greedy 决策
//   1  - 已执行"全吃"路径 (玩家不变, 调用方应 continue 回循环顶)
//   2  - 已执行"牺牲" Double-Cross 路径 (玩家已切换, 调用方应 continue)
static int applyDoubleCrossHeuristic(Board &board, int &player, int &moveCount)
{
    BoxBoard dead(board);
    bool deadChain = dead.getDeadChainExist();
    bool deadCircle = dead.getDeadCircleExist();
    if (!(deadChain || deadCircle))
        return 0;

    int sacrificeBoxNum = deadCircle ? 4 : 2;
    BoxBoard sim(board);
    sim.eatAllCTypeBoxes(player);
    LOC boxNum = sim.getEarlyRationalBoxNum();

    if (boxNum.first - boxNum.second <= sacrificeBoxNum)
    {
        // 全吃更优 — 不换手, caller continue 后下一轮可能进入 MCTS
        std::vector<LOC> eatPace;
        board.eatAllCTypeBoxes(player, eatPace);
        moveCount += static_cast<int>(eatPace.size());
        return 1;
    }

    // 牺牲更优 — 执行 Double-Cross + 后续吃 C 格 + 换手
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
                    moveCount++;
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
                    moveCount++;
            }
            else
                break;
        }
    }
    LOC dcMove = board.getDoubleCrossLoc(player);
    board.move(player, dcMove);
    std::vector<LOC> tempPace;
    for (;;)
    {
        if (!board.getCTypeBoxLimit(player, tempPace))
            break;
    }
    moveCount += 1 + static_cast<int>(tempPace.size());
    player = -player;
    return 2;
}

// ========== 局面阶段检测 ==========

std::string SelfPlayEngine::detectPhase(const Board &board)
{
    // 统计自由边数量 (横边 + 竖边)
    int freeEdges = 0;
    // 横边: i 偶数 (0,2,...,10), j 奇数 (1,3,...,9)
    for (int i = 0; i <= 10; i += 2)
        for (int j = 1; j <= 9; j += 2)
            if (board.map[i][j] != OCCLINE)
                freeEdges++;
    // 竖边: i 奇数 (1,3,...,9), j 偶数 (0,2,...,10)
    for (int i = 1; i <= 9; i += 2)
        for (int j = 0; j <= 10; j += 2)
            if (board.map[i][j] != OCCLINE)
                freeEdges++;

    if (freeEdges >= 40)
        return "opening";
    else if (freeEdges >= 20)
        return "midgame";
    else
        return "endgame";
}

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
    int decisionCount = 0;
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

        // 2.5 死链/死环 + Double-Cross 启发式 (与推理端对齐, 不采样)
        {
            int hr = applyDoubleCrossHeuristic(board, currentPlayer, moveCount);
            if (hr != 0)
                continue;
        }

        // 2.6 启发式后再次检查终局
        if (board.ifEnd())
            break;
        {
            Board test2 = board;
            test2.eatAllCTypeBoxes(currentPlayer);
            if (test2.getFilterMoveNum() == 0)
            {
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
        sample.moveIndex = moveCount;
        sample.decisionIndex = decisionCount;
        sample.phase = detectPhase(board);

        // 4. PUCT 搜索（自对弈模式：启用 Dirichlet 噪声）
        MCTSConfig config = MCTSConfig::selfPlay(numSimulations);
        AZMCTS mcts;
        AZNode *root = mcts.search(board, currentPlayer, config);

        // 记录策略（访问次数分布）
        sample.policy = mcts.getVisitDistribution(root);

        // 记录 policy 熔和置信度和 rootQ
        float maxP = 0.0f, entropy = 0.0f;
        for (int a = 0; a < AZ_ACTION_SIZE; a++)
        {
            if (sample.policy[a] > maxP)
                maxP = sample.policy[a];
            if (sample.policy[a] > 1e-8f)
                entropy -= sample.policy[a] * std::log(sample.policy[a]);
        }
        sample.rootConfidence = maxP;
        sample.rootEntropy = entropy;
        sample.rootQ = root->Q();

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
        decisionCount++;

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

    // 8. 对局结束，用最终胜负回填所有样本的 value 和 value_margin
    int winner = board.getWinner();
    int sampleEnd = static_cast<int>(samples.size());
    int blackFinal = board.blackBox;
    int whiteFinal = board.whiteBox;

    // 生成 game_id
    std::ostringstream gidss;
    gidss << "selfplay_" << std::setfill('0') << std::setw(6) << gamesPlayed;
    std::string gameId = gidss.str();

    for (int i = sampleStart; i < sampleEnd; i++)
    {
        samples[i].gameId = gameId;
        samples[i].blackScoreFinal = blackFinal;
        samples[i].whiteScoreFinal = whiteFinal;
        samples[i].winner = winner;

        // value: +1/-1/0 胜负
        if (winner == 0)
            samples[i].value = 0.0f;
        else if (samples[i].player == winner)
            samples[i].value = 1.0f;
        else
            samples[i].value = -1.0f;

        // value_margin: 当前玩家视角的比分差 / 25.0
        int myScore = (samples[i].player == BLACK) ? blackFinal : whiteFinal;
        int oppScore = (samples[i].player == BLACK) ? whiteFinal : blackFinal;
        samples[i].valueMargin = static_cast<float>(myScore - oppScore) / 25.0f;
    }

    // 返回结果
    SelfPlayResult result;
    result.blackScore = blackFinal;
    result.whiteScore = whiteFinal;
    result.winner = winner;
    result.numSamples = sampleEnd - sampleStart;
    result.numMoves = moveCount;
    return result;
}

// ========== SelfPlayEngine::runBackward ==========

void SelfPlayEngine::runBackward(int numGames, int numSimulations,
                                  int startSearchStep, const std::string &outputDir)
{
    std::filesystem::create_directories(outputDir);
    std::vector<SelfPlaySample> allSamples;
    allSamples.reserve(numGames * 40);

    int totalSamples = 0;
    int blackWins = 0, whiteWins = 0, draws = 0;

    auto totalStart = std::chrono::steady_clock::now();

    for (int g = 0; g < numGames; g++)
    {
        auto gameStart = std::chrono::steady_clock::now();

        std::vector<SelfPlaySample> gameSamples;
        SelfPlayResult result = playOneGameBackward(
            numSimulations, startSearchStep, 1.0f, 10, gameSamples);

        auto gameEnd = std::chrono::steady_clock::now();
        int gameMs = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(gameEnd - gameStart).count());

        if (result.winner == BLACK) blackWins++;
        else if (result.winner == WHITE) whiteWins++;
        else draws++;
        totalSamples += result.numSamples;

        std::cerr << "[BackwardSP] Game " << (g + 1) << "/" << numGames
                  << "  st=" << startSearchStep
                  << "  score=" << result.blackScore << ":" << result.whiteScore
                  << "  winner=" << (result.winner == BLACK ? "BLACK" : result.winner == WHITE ? "WHITE" : "DRAW")
                  << "  samples=" << result.numSamples
                  << "  moves=" << result.numMoves
                  << "  time=" << gameMs << "ms\n";

        allSamples.insert(allSamples.end(), gameSamples.begin(), gameSamples.end());

        if ((g + 1) % 50 == 0 || g == numGames - 1)
        {
            std::ostringstream filename;
            filename << outputDir << "/backward_st" << startSearchStep
                     << "_" << std::setfill('0') << std::setw(6) << gamesPlayed << ".jsonl";
            writeSamples(filename.str(), allSamples);
            std::cerr << "[BackwardSP] Wrote " << allSamples.size() << " samples to " << filename.str() << "\n";
            allSamples.clear();
        }

        gamesPlayed++;
    }

    auto totalEnd = std::chrono::steady_clock::now();
    int totalMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart).count());

    std::cerr << "\n[BackwardSP] === Summary (st=" << startSearchStep << ") ===\n"
              << "  Games:   " << numGames << "\n"
              << "  Samples: " << totalSamples << "\n"
              << "  BLACK wins: " << blackWins << "  WHITE wins: " << whiteWins << "  Draws: " << draws << "\n"
              << "  Total time: " << totalMs / 1000 << "s\n"
              << "  Avg time/game: " << (numGames > 0 ? totalMs / numGames : 0) << "ms\n";
}

// ========== SelfPlayEngine::playOneGameBackward ==========
//
// BoxesZero 反向训练核心:
//   step < startSearchStep → 贪心启发式 (不收集数据)
//   step >= startSearchStep → MCTS 搜索 (收集训练数据)
//
// 贪心策略:
//   优先吃第4条边 (完成格子) > 1/2条边 (安全边) > 第3条边 (危险边)
//   这与 BoxesZero 论文中的 handcrafted greedy policy 一致

SelfPlayResult SelfPlayEngine::playOneGameBackward(int numSimulations,
                                                    int startSearchStep,
                                                    float temperature, int temperatureMoves,
                                                    std::vector<SelfPlaySample> &samples)
{
    Board board;
    int currentPlayer = BLACK;
    int moveCount = 0;       // 落子步数 (边的数量)
    int decisionCount = 0;   // MCTS 决策点
    int sampleStart = static_cast<int>(samples.size());

    while (!board.ifEnd())
    {
        // 1. 强制吃掉所有 C 型格
        board.eatAllCTypeBoxes(currentPlayer);

        if (board.ifEnd())
            break;

        // 2. 检查终局
        {
            Board test = board;
            test.eatAllCTypeBoxes(currentPlayer);
            if (test.getFilterMoveNum() == 0)
            {
                std::vector<LOC> pace;
                latterSituationMove(board, currentPlayer, pace);
                moveCount += static_cast<int>(pace.size());
                currentPlayer = -currentPlayer;
                continue;
            }
        }

        // 2.5 死链/死环 + Double-Cross 启发式 (与推理端对齐, 不采样)
        {
            int hr = applyDoubleCrossHeuristic(board, currentPlayer, moveCount);
            if (hr != 0)
                continue;
        }

        // 2.6 启发式后再次检查终局
        if (board.ifEnd())
            break;
        {
            Board test2 = board;
            test2.eatAllCTypeBoxes(currentPlayer);
            if (test2.getFilterMoveNum() == 0)
            {
                std::vector<LOC> pace;
                latterSituationMove(board, currentPlayer, pace);
                moveCount += static_cast<int>(pace.size());
                currentPlayer = -currentPlayer;
                continue;
            }
        }

        // 3. 判断是否在 MCTS 搜索阶段
        if (moveCount < startSearchStep)
        {
            // === 贪心阶段: 不搜索, 不收集数据 ===
            // 贪心策略: 优先吃格 > 安全边 > 危险边
            auto legalMask = getLegalMask(board);
            int bestAction = -1;
            int bestPriority = -1;

            for (int a = 0; a < AZ_ACTION_SIZE; a++)
            {
                if (legalMask[a] < 0.5f) continue;

                LOC loc = actionToLoc(a);
                int priority = 0;

                // 检查能否完成格子 (第4条边)
                Board testBoard = board;
                int earned = testBoard.move(currentPlayer, loc);
                if (earned > 0)
                    priority = 3; // 最高优先
                else if (!board.ifMakeCBox(loc))
                    priority = 2; // 安全边
                else
                    priority = 1; // 危险边 (第3条边)

                if (priority > bestPriority)
                {
                    bestPriority = priority;
                    bestAction = a;
                }
            }

            if (bestAction < 0)
            {
                currentPlayer = -currentPlayer;
                continue;
            }

            // 执行贪心动作
            LOC loc = actionToLoc(bestAction);
            int earned = board.move(currentPlayer, loc);
            moveCount++;

            if (earned <= 0)
                currentPlayer = -currentPlayer;
        }
        else
        {
            // === MCTS 搜索阶段: 收集训练数据 ===
            SelfPlaySample sample;
            sample.boardTensor = encodeBoard(board, currentPlayer);
            sample.player = currentPlayer;
            sample.legalMask = getLegalMask(board);
            sample.value = 0.0f;
            sample.moveIndex = moveCount;
            sample.decisionIndex = decisionCount;
            sample.phase = detectPhase(board);

            MCTSConfig config = MCTSConfig::selfPlay(numSimulations);
            AZMCTS mcts;
            AZNode *root = mcts.search(board, currentPlayer, config);

            sample.policy = mcts.getVisitDistribution(root);

            float maxP = 0.0f, entropy = 0.0f;
            for (int a = 0; a < AZ_ACTION_SIZE; a++)
            {
                if (sample.policy[a] > maxP)
                    maxP = sample.policy[a];
                if (sample.policy[a] > 1e-8f)
                    entropy -= sample.policy[a] * std::log(sample.policy[a]);
            }
            sample.rootConfidence = maxP;
            sample.rootEntropy = entropy;
            sample.rootQ = root->Q();

            // 选择动作
            int mctsDecisions = decisionCount; // MCTS 决策序号
            float temp = (mctsDecisions < temperatureMoves) ? temperature : 0.0f;
            int action = mcts.selectAction(root, temp);

            deleteAZTree(root);
            delete root;

            if (action < 0)
            {
                currentPlayer = -currentPlayer;
                continue;
            }

            samples.push_back(sample);
            decisionCount++;

            LOC loc = actionToLoc(action);
            int earned = board.move(currentPlayer, loc);
            moveCount++;

            if (earned <= 0)
                currentPlayer = -currentPlayer;
        }
    }

    // 对局结束，回填
    int winner = board.getWinner();
    int sampleEnd = static_cast<int>(samples.size());
    int blackFinal = board.blackBox;
    int whiteFinal = board.whiteBox;

    std::ostringstream gidss;
    gidss << "backward_st" << startSearchStep << "_"
          << std::setfill('0') << std::setw(6) << gamesPlayed;
    std::string gameId = gidss.str();

    for (int i = sampleStart; i < sampleEnd; i++)
    {
        samples[i].gameId = gameId;
        samples[i].blackScoreFinal = blackFinal;
        samples[i].whiteScoreFinal = whiteFinal;
        samples[i].winner = winner;

        if (winner == 0)
            samples[i].value = 0.0f;
        else if (samples[i].player == winner)
            samples[i].value = 1.0f;
        else
            samples[i].value = -1.0f;

        int myScore = (samples[i].player == BLACK) ? blackFinal : whiteFinal;
        int oppScore = (samples[i].player == BLACK) ? whiteFinal : blackFinal;
        samples[i].valueMargin = static_cast<float>(myScore - oppScore) / 25.0f;
    }

    SelfPlayResult result;
    result.blackScore = blackFinal;
    result.whiteScore = whiteFinal;
    result.winner = winner;
    result.numSamples = sampleEnd - sampleStart;
    result.numMoves = moveCount;
    return result;
}

// ========== SelfPlayEngine::writeSamples ==========

void SelfPlayEngine::writeSamples(const std::string &filepath,
                                  const std::vector<SelfPlaySample> &samples) const
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "[SelfPlay] ERROR: Cannot open " << filepath << "\n";
        return;
    }

    constexpr int TENSOR_SIZE = AZ_CHANNELS * AZ_BOARD_SIZE * AZ_BOARD_SIZE;

    // 从样本中取最终比分（所有样本来自同一批对局，但可能混合多局）
    for (const auto &s : samples)
    {
        // schema v2 元数据
        file << "{\"schema_version\":2";
        file << ",\"game_id\":\"" << s.gameId << "\"";
        file << ",\"move_index\":" << s.moveIndex;
        file << ",\"decision_index\":" << s.decisionIndex;
        file << ",\"player\":" << s.player;

        // 棋盘张量
        file << ",\"board\":[";
        for (int i = 0; i < TENSOR_SIZE; i++)
        {
            if (i > 0) file << ',';
            if (s.boardTensor[i] == 0.0f)
                file << '0';
            else if (s.boardTensor[i] == 1.0f)
                file << '1';
            else if (s.boardTensor[i] == -1.0f)
                file << "-1";
            else
                file << std::fixed << std::setprecision(4) << s.boardTensor[i];
        }

        file << "],\"legal_mask\":[";
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

        // value 和 value_margin
        file << "],\"value\":" << std::fixed << std::setprecision(1) << s.value;
        file << ",\"value_margin\":" << std::fixed << std::setprecision(4) << s.valueMargin;

        // 最终比分与胜者
        file << ",\"black_score_final\":" << s.blackScoreFinal;
        file << ",\"white_score_final\":" << s.whiteScoreFinal;
        file << ",\"winner\":" << s.winner;

        // 阶段和教师信息
        file << ",\"phase\":\"" << s.phase << "\"";
        file << ",\"teacher\":\"" << teacherName << "\"";
        file << ",\"simulations\":" << AZ_SIMULATIONS;
        file << ",\"temperature\":1.0";
        file << ",\"root_policy_entropy\":" << std::fixed << std::setprecision(4) << s.rootEntropy;
        file << ",\"root_policy_confidence\":" << std::fixed << std::setprecision(4) << s.rootConfidence;
        file << ",\"root_q\":" << std::fixed << std::setprecision(4) << s.rootQ;

        file << "}\n";
    }

    file.close();
}
