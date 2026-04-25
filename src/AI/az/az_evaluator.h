#pragma once
#include "../board.h"
#include "az_types.h"

// 策略价值评估器接口
// 初期使用启发式实现；后续替换为神经网络推理
class AZEvaluator
{
  public:
    virtual ~AZEvaluator() = default;

    // 评估当前局面，返回策略概率和价值（从 currentPlayer 视角）
    virtual NetworkOutput evaluate(const Board &board, int currentPlayer) = 0;
};

// 启发式评估器：不使用神经网络，基于手工规则生成 policy 和 value
class HeuristicEvaluator : public AZEvaluator
{
  public:
    NetworkOutput evaluate(const Board &board, int currentPlayer) override;
};

// 全局评估器实例（可在运行时替换为神经网络版本）
AZEvaluator &getEvaluator();
void setEvaluator(AZEvaluator *evaluator);
