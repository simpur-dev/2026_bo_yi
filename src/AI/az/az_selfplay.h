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
// 输出格式 (每行一条 JSON):
//   {
//     "board": [847 floats],       // 7×11×11 编码后的棋盘
//     "player": 1,                 // BLACK=1, WHITE=-1
//     "legal_mask": [60 floats],   // 合法动作 mask
//     "policy": [60 floats],       // PUCT 访问次数分布 π
//     "value": 1.0                 // 最终胜负（该玩家视角）
//   }

struct SelfPlaySample
{
    Tensor boardTensor;                              // 编码后的棋盘特征
    int player;                                      // 当前玩家
    std::array<float, AZ_ACTION_SIZE> legalMask;     // 合法动作 mask
    std::array<float, AZ_ACTION_SIZE> policy;        // PUCT 访问分布
    float value;                                     // 最终胜负（对局结束后回填）
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

    // 进行一盘自对弈，样本追加到 samples 中
    // temperature: 探索温度（前 temperatureMoves 步使用，之后贪心）
    // temperatureMoves: 前多少步使用温度采样
    SelfPlayResult playOneGame(int numSimulations,
                               float temperature, int temperatureMoves,
                               std::vector<SelfPlaySample> &samples);

  private:
    // 将样本写入 JSONL 文件
    static void writeSamples(const std::string &filepath,
                             const std::vector<SelfPlaySample> &samples);

    int gamesPlayed = 0;
};
