#pragma once
#include <array>
#include <vector>

// AlphaZero 搜索参数
constexpr int AZ_ACTION_SIZE = 60;       // 总动作数（30横边 + 30竖边）
constexpr int AZ_BOARD_SIZE = 11;        // 棋盘数组边长
constexpr int AZ_CHANNELS = 7;           // 神经网络输入通道数
constexpr int AZ_SIMULATIONS = 800;      // 每步 PUCT 模拟次数
constexpr float AZ_C_PUCT = 1.5f;        // PUCT 探索常数
constexpr int AZ_TIME_LIMIT_MS = 1000;   // 搜索时间上限（毫秒）
constexpr float AZ_DIRICHLET_ALPHA = 0.3f;  // Dirichlet 噪声参数
constexpr float AZ_DIRICHLET_FRAC = 0.25f;  // 根节点噪声混合比例

// AI 开关：true 使用 AlphaZero，false 使用旧版 UCT
constexpr bool USE_ALPHAZERO_AI = true;

// 神经网络输出
struct NetworkOutput {
    std::array<float, AZ_ACTION_SIZE> policy{};  // 策略先验概率
    float value = 0.0f;                           // 局面价值 [-1, 1]
};
