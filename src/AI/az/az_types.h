#pragma once
#include <array>
#include <vector>

// AlphaZero 搜索参数
constexpr int AZ_ACTION_SIZE = 60;       // 总动作数（30横边 + 30竖边）
constexpr int AZ_BOARD_SIZE = 11;        // 棋盘数组边长
constexpr int AZ_CHANNELS = 7;           // 神经网络输入通道数
constexpr int AZ_SIMULATIONS = 200;      // 每步 PUCT 模拟次数（神经网络模式下 200 已足够）
constexpr float AZ_C_PUCT = 1.5f;        // PUCT 探索常数
constexpr int AZ_TIME_LIMIT_MS = 1000;   // 搜索时间上限（毫秒）
constexpr float AZ_DIRICHLET_ALPHA = 0.3f;  // Dirichlet 噪声参数
constexpr float AZ_DIRICHLET_FRAC = 0.25f;  // 根节点噪声混合比例

// AI 开关：true 使用 AlphaZero，false 使用旧版 UCT
constexpr bool USE_ALPHAZERO_AI = true;

// MCTS 搜索配置
// self-play 模式需要噪声和温度以增加探索
// evaluation/gameplay 模式应关闭噪声，使用贪心选择
struct MCTSConfig {
    int simulations = AZ_SIMULATIONS;          // PUCT 模拟次数
    int timeLimitMs = AZ_TIME_LIMIT_MS;        // 搜索时间上限（毫秒）
    float cPuct = AZ_C_PUCT;                   // PUCT 探索常数
    float dirichletAlpha = AZ_DIRICHLET_ALPHA; // Dirichlet 噪声参数
    float dirichletFrac = AZ_DIRICHLET_FRAC;   // 根节点噪声混合比例
    bool addRootNoise = false;                 // 是否添加根节点 Dirichlet 噪声
    float temperature = 0.0f;                  // 动作选择温度
    int temperatureMoves = 0;                  // 前多少步使用温度采样

    // 预设：自对弈模式（有噪声、有温度）
    static MCTSConfig selfPlay(int sims = AZ_SIMULATIONS) {
        MCTSConfig cfg;
        cfg.simulations = sims;
        cfg.timeLimitMs = 0;           // 自对弈按模拟次数，不限时间
        cfg.addRootNoise = true;
        cfg.temperature = 1.0f;
        cfg.temperatureMoves = 20;
        return cfg;
    }

    // 预设：评估/实战模式（无噪声、贪心）
    static MCTSConfig evaluation(int sims = AZ_SIMULATIONS, int timeMs = AZ_TIME_LIMIT_MS) {
        MCTSConfig cfg;
        cfg.simulations = sims;
        cfg.timeLimitMs = timeMs;
        cfg.addRootNoise = false;
        cfg.temperature = 0.0f;
        cfg.temperatureMoves = 0;
        return cfg;
    }
};

// 神经网络输出
struct NetworkOutput {
    std::array<float, AZ_ACTION_SIZE> policy{};  // 策略先验概率
    float value = 0.0f;                           // 局面价值 [-1, 1]
};
