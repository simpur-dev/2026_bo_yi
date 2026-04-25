#pragma once
#include "az_types.h"
#include "az_encoder.h"
#include <string>
#include <vector>

// C++ 手写 MLP 前向推理
//
// 对应 Python 端的 DotsAndBoxesMLP 网络结构:
//   Flatten(7×11×11=847)
//   -> FC(847, 256) + ReLU
//   -> FC(256, 128) + ReLU
//   -> Policy Head: FC(128, 60)
//   -> Value Head:  FC(128, 1) + Tanh
//
// 权重文件格式（由 export_weights.py 生成）:
//   每个参数: 4 字节 int32（元素数） + float32 数据
//   参数顺序:
//     fc1.weight   (256 × 847)
//     fc1.bias     (256)
//     fc2.weight   (128 × 256)
//     fc2.bias     (128)
//     policy_head.weight (60 × 128)
//     policy_head.bias   (60)
//     value_head.weight  (1 × 128)
//     value_head.bias    (1)

class MLPInference
{
  public:
    // MLP 结构常量
    static constexpr int INPUT_SIZE = AZ_CHANNELS * AZ_BOARD_SIZE * AZ_BOARD_SIZE; // 847
    static constexpr int HIDDEN1 = 256;
    static constexpr int HIDDEN2 = 128;

    MLPInference() = default;

    // 从二进制文件加载权重
    bool loadWeights(const std::string &weightPath);

    // 是否已加载权重
    bool isLoaded() const { return loaded; }

    // 前向推理：输入编码后的棋盘张量，输出 policy logits 和 value
    // legalMask 用于对 policy 做 mask + softmax
    NetworkOutput forward(const Tensor &input, const std::array<float, AZ_ACTION_SIZE> &legalMask) const;

  private:
    bool loaded = false;

    // 权重数据
    std::vector<float> fc1_weight;       // (HIDDEN1, INPUT_SIZE) = (256, 847)
    std::vector<float> fc1_bias;         // (HIDDEN1) = (256)
    std::vector<float> fc2_weight;       // (HIDDEN2, HIDDEN1)    = (128, 256)
    std::vector<float> fc2_bias;         // (HIDDEN2) = (128)
    std::vector<float> policy_weight;    // (ACTION_SIZE, HIDDEN2) = (60, 128)
    std::vector<float> policy_bias;      // (ACTION_SIZE) = (60)
    std::vector<float> value_weight;     // (1, HIDDEN2) = (1, 128)
    std::vector<float> value_bias;       // (1)

    // 线性层前向: output[j] = bias[j] + sum_i(weight[j * inF + i] * input[i])
    static void linearForward(const float *input, int inFeatures,
                              const float *weight, const float *bias,
                              float *output, int outFeatures);

    // ReLU 激活
    static void relu(float *data, int size);

    // 从文件流读取一个参数块
    static bool readParam(FILE *f, std::vector<float> &param, int expectedCount);
};
