#pragma once
#include "../board.h"
#include "az_types.h"
#include "az_neural_net.h"
#include <string>

// 策略价值评估器接口
// 支持三种实现：
//   1. HeuristicEvaluator - 手工规则
//   2. NeuralNetEvaluator - MLP 手写推理 (weights.bin)
//   3. ONNXEvaluator      - CNN/MLP ONNX 推理 (model.onnx) [需 USE_ONNX]
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

// 双模型支持: 按当前玩家颜色选择评估器
// 设置黑白各自专用的评估器, 之后 selectEvaluatorForColor() 自动切换
void setEvaluatorBlack(AZEvaluator *evaluator);
void setEvaluatorWhite(AZEvaluator *evaluator);
bool hasDualEvaluators();
void selectEvaluatorForColor(int player);  // BLACK -> black evaluator, WHITE -> white

// 便捷函数：尝试加载 MLP 神经网络，如加载失败则使用启发式评估器
bool tryLoadNeuralNet(const std::string &weightPath);

// 便捷函数：自动检测文件类型 (.onnx / .bin) 并加载对应评估器
// .onnx -> ONNXEvaluator (需编译时启用 USE_ONNX)
// .bin  -> NeuralNetEvaluator (MLP)
// 其他  -> 尝试作为 MLP 加载
bool tryLoadModel(const std::string &modelPath);
