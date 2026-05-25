#include "az_evaluator.h"
#include "az_onnx_evaluator.h"
#include "az_action.h"
#include "az_encoder.h"
#include "../define.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ========== 全局评估器管理 ==========

static HeuristicEvaluator defaultEvaluator;
static AZEvaluator *globalEvaluator = &defaultEvaluator;

AZEvaluator &getEvaluator()
{
    return *globalEvaluator;
}

void setEvaluator(AZEvaluator *evaluator)
{
    if (evaluator)
        globalEvaluator = evaluator;
}

// ========== 双模型支持 ==========

static AZEvaluator *blackEvaluator = nullptr;
static AZEvaluator *whiteEvaluator = nullptr;

void setEvaluatorBlack(AZEvaluator *evaluator)
{
    blackEvaluator = evaluator;
}

void setEvaluatorWhite(AZEvaluator *evaluator)
{
    whiteEvaluator = evaluator;
}

bool hasDualEvaluators()
{
    return blackEvaluator != nullptr && whiteEvaluator != nullptr;
}

void selectEvaluatorForColor(int player)
{
    if (player == BLACK && blackEvaluator)
        globalEvaluator = blackEvaluator;
    else if (player == WHITE && whiteEvaluator)
        globalEvaluator = whiteEvaluator;
}

// ========== 启发式评估器实现 ==========

// 对每条合法边评分的辅助函数
static float scoreAction(Board &board, int action, int player)
{
    LOC loc = actionToLoc(action);
    float score = 1.0f; // 基础分

    // 如果能立即吃到格子，给予高分
    if (board.ifEarnBox(loc))
    {
        score += 20.0f;
    }
    // 如果不会制造 C 型格（安全边），给予中等分
    else if (!board.ifMakeCBox(loc))
    {
        score += 10.0f;
    }
    // 会制造 C 型格的危险边，低分
    else
    {
        score += 0.5f;
    }

    return score;
}

// softmax 归一化
static void softmax(std::array<float, AZ_ACTION_SIZE> &logits,
                    const std::array<float, AZ_ACTION_SIZE> &mask)
{
    // 找合法动作中的最大值，防止 exp 溢出
    float maxVal = -1e9f;
    for (int i = 0; i < AZ_ACTION_SIZE; i++)
    {
        if (mask[i] > 0.5f && logits[i] > maxVal)
            maxVal = logits[i];
    }

    float sumExp = 0.0f;
    for (int i = 0; i < AZ_ACTION_SIZE; i++)
    {
        if (mask[i] > 0.5f)
        {
            logits[i] = std::exp(logits[i] - maxVal);
            sumExp += logits[i];
        }
        else
        {
            logits[i] = 0.0f;
        }
    }

    if (sumExp > 0.0f)
    {
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
            logits[i] /= sumExp;
    }
}

NetworkOutput HeuristicEvaluator::evaluate(const Board &board, int currentPlayer)
{
    NetworkOutput output;

    // ---- 策略（Policy）----
    auto mask = getLegalMask(board);
    Board mutableBoard = board; // 需要可变副本来调用非 const 方法

    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (mask[a] > 0.5f)
            output.policy[a] = scoreAction(mutableBoard, a, currentPlayer);
        else
            output.policy[a] = 0.0f;
    }

    // softmax 归一化为概率
    softmax(output.policy, mask);

    // ---- 价值（Value）----
    // 简单启发式：当前分数差 / 总格子数
    int myBoxes = mutableBoard.getPlayerBoxes(currentPlayer);
    int oppBoxes = mutableBoard.getPlayerBoxes(-currentPlayer);
    float scoreDiff = static_cast<float>(myBoxes - oppBoxes) / static_cast<float>(BOXNUM);

    // 统计安全边比例作为额外估计
    int totalLegal = 0;
    int safeCount = 0;
    for (int a = 0; a < AZ_ACTION_SIZE; a++)
    {
        if (mask[a] > 0.5f)
        {
            totalLegal++;
            LOC loc = actionToLoc(a);
            if (!mutableBoard.ifMakeCBox(loc))
                safeCount++;
        }
    }
    float safeRatio = (totalLegal > 0) ? static_cast<float>(safeCount) / static_cast<float>(totalLegal) : 0.0f;

    // 综合价值：分数差 + 安全边优势
    output.value = std::clamp(scoreDiff + 0.1f * safeRatio, -1.0f, 1.0f);

    return output;
}

// ========== 神经网络评估器实现 ==========

bool NeuralNetEvaluator::loadModel(const std::string &weightPath)
{
    return mlp.loadWeights(weightPath);
}

NetworkOutput NeuralNetEvaluator::evaluate(const Board &board, int currentPlayer)
{
    // 编码棋盘
    Tensor input = encodeBoard(board, currentPlayer);

    // 获取合法动作 mask
    auto legalMask = getLegalMask(board);

    // MLP 前向推理（内部已做 masked softmax）
    return mlp.forward(input, legalMask);
}

// ========== 便捷函数 ==========

static NeuralNetEvaluator *neuralNetInstance = nullptr;

bool tryLoadNeuralNet(const std::string &weightPath)
{
    if (!neuralNetInstance)
        neuralNetInstance = new NeuralNetEvaluator();

    if (neuralNetInstance->loadModel(weightPath))
    {
        setEvaluator(neuralNetInstance);
        std::cerr << "[AZ] Switched to NeuralNetEvaluator\n";
        return true;
    }
    else
    {
        std::cerr << "[AZ] Failed to load neural net, using HeuristicEvaluator\n";
        return false;
    }
}

// 自动检测模型文件类型并加载
bool tryLoadModel(const std::string &modelPath)
{
    // 检测文件扩展名
    std::string ext;
    auto dotPos = modelPath.rfind('.');
    if (dotPos != std::string::npos)
        ext = modelPath.substr(dotPos);

    if (ext == ".onnx")
    {
        // 使用 ONNX Runtime 加载 CNN 模型
        return tryLoadOnnxModel(modelPath);
    }
    else
    {
        // 默认按 MLP 二进制权重加载
        return tryLoadNeuralNet(modelPath);
    }
}
