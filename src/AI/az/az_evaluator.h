#pragma once
#include "../board.h"
#include "az_types.h"
#include "az_neural_net.h"
#include <string>

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

// 神经网络评估器：加载 MLP 权重进行推理
class NeuralNetEvaluator : public AZEvaluator
{
  public:
    // 从权重文件加载模型
    bool loadModel(const std::string &weightPath);
    bool isModelLoaded() const { return mlp.isLoaded(); }

    NetworkOutput evaluate(const Board &board, int currentPlayer) override;

  private:
    MLPInference mlp;
};

// 全局评估器实例（可在运行时替换为神经网络版本）
AZEvaluator &getEvaluator();
void setEvaluator(AZEvaluator *evaluator);

// 便捷函数：尝试加载神经网络，如加载失败则使用启发式评估器
bool tryLoadNeuralNet(const std::string &weightPath);
