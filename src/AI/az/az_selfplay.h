#pragma once
#include "../board.h"
#include "az_types.h"
#include "az_encoder.h"
#include <vector>
#include <string>
#include <array>

// ========== 自对弈训练数据生成 ==========
//
// 流程:
//   1. 两个 AlphaZero AI 对弈一盘完整的棋
//   2. 每个 PUCT 决策点记录: 棋盘编码、当前玩家、合法动作、访问分布
//   3. 对局结束后，用最终胜负回填每条样本的 value
//   4. 所有样本写入 JSONL 文件，供 Python 训练脚本读取
//
// 输出格式 (schema v2, JSONL):
//   {
//     "schema_version": 2,
//     "game_id": "selfplay_000123",
//     "move_index": 17,
//     "decision_index": 9,
//     "player": 1,
//     "board": [847 floats],
//     "legal_mask": [60 floats],
//     "policy": [60 floats],
//     "value": 1.0,
//     "value_margin": 0.28,
//     "black_score_final": 16,
//     "white_score_final": 9,
//     "winner": 1,
//     "phase": "midgame",
//     "teacher": "heuristic_800",
//     "simulations": 800,
//     "temperature": 1.0,
//     "root_policy_entropy": 2.31,
//     "root_policy_confidence": 0.42
//   }

struct SelfPlaySample
{
    // schema v1 字段
    Tensor boardTensor;                              // 编码后的棋盘特征
    int player;                                      // 当前玩家
    std::array<float, AZ_ACTION_SIZE> legalMask;     // 合法动作 mask
    std::array<float, AZ_ACTION_SIZE> policy;        // PUCT 访问分布
    float value;                                     // 最终胜负（对局结束后回填）

    // schema v2 新增字段
    std::string gameId;                              // 对局唯一标识
    int moveIndex = 0;                               // 真实落子步数
    int decisionIndex = 0;                           // AlphaZero 决策点编号
    float valueMargin = 0.0f;                        // 比分差目标 (对局结束后回填)
    int blackScoreFinal = 0;                         // 黑方最终得分
    int whiteScoreFinal = 0;                         // 白方最终得分
    int winner = 0;                                  // 胜者 (BLACK/WHITE/0)
    std::string phase;                               // opening/midgame/endgame
    float rootEntropy = 0.0f;                        // 根节点 policy 熵
    float rootConfidence = 0.0f;                     // 根节点 policy 置信度
    float rootQ = 0.0f;                              // MCTS 根节点 Q 值 (当前玩家视角)
};

struct SelfPlayResult
{
    int blackScore = 0;
    int whiteScore = 0;
    int winner = 0;      // BLACK, WHITE, or 0(draw)
    int numSamples = 0;  // 本局生成的训练样本数
    int numMoves = 0;    // 总落子步数
};

class SelfPlayEngine
{
  public:
    // 运行 numGames 盘自对弈，将训练数据写入 outputDir
    void run(int numGames, int numSimulations, const std::string &outputDir);

    // 反向训练模式：startSearchStep 之前用贪心启发式，之后用 MCTS
    // 仅收集 MCTS 搜索的步骤作为训练数据
    void runBackward(int numGames, int numSimulations,
                     int startSearchStep, const std::string &outputDir);

    // 进行一盘自对弈，样本追加到 samples 中
    // temperature: 探索温度（前 temperatureMoves 步使用，之后贪心）
    // temperatureMoves: 前多少步使用温度采样
    SelfPlayResult playOneGame(int numSimulations,
                               float temperature, int temperatureMoves,
                               std::vector<SelfPlaySample> &samples);

    // 反向训练：startSearchStep 之前用贪心，之后用 MCTS
    SelfPlayResult playOneGameBackward(int numSimulations,
                                       int startSearchStep,
                                       float temperature, int temperatureMoves,
                                       std::vector<SelfPlaySample> &samples);

    // 设置 teacher 标识（记录数据来源）
    void setTeacher(const std::string &name) { teacherName = name; }

    // 黑方 sims 倍数: 黑方搜索更深以补偿先手劣势
    void setBlackSimsMultiplier(float mult) { blackSimsMult = mult; }

  private:
    // 将样本写入 JSONL 文件 (schema v2)
    void writeSamples(const std::string &filepath,
                      const std::vector<SelfPlaySample> &samples) const;

    // 判断局面阶段
    static std::string detectPhase(const Board &board);

    int gamesPlayed = 0;
    std::string teacherName = "heuristic";
    float blackSimsMult = 1.0f;
};
