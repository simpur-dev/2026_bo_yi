#include "az_neural_net.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>

// ========== 基础运算 ==========

void MLPInference::linearForward(const float *input, int inFeatures,
                                 const float *weight, const float *bias,
                                 float *output, int outFeatures)
{
    // output[j] = bias[j] + sum_i( weight[j * inFeatures + i] * input[i] )
    // weight 按行优先存储，每行 inFeatures 个元素，共 outFeatures 行
    for (int j = 0; j < outFeatures; j++)
    {
        float sum = bias[j];
        const float *wRow = weight + j * inFeatures;
        for (int i = 0; i < inFeatures; i++)
        {
            sum += wRow[i] * input[i];
        }
        output[j] = sum;
    }
}

void MLPInference::relu(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0.0f)
            data[i] = 0.0f;
    }
}

// ========== 权重加载 ==========

bool MLPInference::readParam(FILE *f, std::vector<float> &param, int expectedCount)
{
    int count = 0;
    if (fread(&count, sizeof(int), 1, f) != 1)
        return false;

    if (count != expectedCount)
    {
        std::cerr << "[MLPInference] Weight count mismatch: expected "
                  << expectedCount << ", got " << count << "\n";
        return false;
    }

    param.resize(count);
    if (fread(param.data(), sizeof(float), count, f) != static_cast<size_t>(count))
        return false;

    return true;
}

bool MLPInference::loadWeights(const std::string &weightPath)
{
    FILE *f = fopen(weightPath.c_str(), "rb");
    if (!f)
    {
        std::cerr << "[MLPInference] Cannot open weight file: " << weightPath << "\n";
        return false;
    }

    bool ok = true;

    // 按 export_weights.py 导出顺序读取
    // fc1.weight: (256, 847) -> 217,472 elements
    ok = ok && readParam(f, fc1_weight, HIDDEN1 * INPUT_SIZE);
    // fc1.bias: (256)
    ok = ok && readParam(f, fc1_bias, HIDDEN1);
    // fc2.weight: (128, 256) -> 32,768 elements
    ok = ok && readParam(f, fc2_weight, HIDDEN2 * HIDDEN1);
    // fc2.bias: (128)
    ok = ok && readParam(f, fc2_bias, HIDDEN2);
    // policy_head.weight: (60, 128) -> 7,680 elements
    ok = ok && readParam(f, policy_weight, AZ_ACTION_SIZE * HIDDEN2);
    // policy_head.bias: (60)
    ok = ok && readParam(f, policy_bias, AZ_ACTION_SIZE);
    // value_head.weight: (1, 128) -> 128 elements
    ok = ok && readParam(f, value_weight, 1 * HIDDEN2);
    // value_head.bias: (1)
    ok = ok && readParam(f, value_bias, 1);

    fclose(f);

    if (!ok)
    {
        std::cerr << "[MLPInference] Failed to load weights from: " << weightPath << "\n";
        loaded = false;
        return false;
    }

    loaded = true;
    std::cerr << "[MLPInference] Loaded weights from: " << weightPath << "\n";

    // 输出参数总量
    int totalParams = HIDDEN1 * INPUT_SIZE + HIDDEN1 +
                      HIDDEN2 * HIDDEN1 + HIDDEN2 +
                      AZ_ACTION_SIZE * HIDDEN2 + AZ_ACTION_SIZE +
                      1 * HIDDEN2 + 1;
    std::cerr << "[MLPInference] Total parameters: " << totalParams << "\n";

    return true;
}

// ========== 前向推理 ==========

NetworkOutput MLPInference::forward(const Tensor &input,
                                    const std::array<float, AZ_ACTION_SIZE> &legalMask) const
{
    NetworkOutput output;

    if (!loaded)
    {
        // 未加载权重时返回均匀 policy 和 0 value
        float legalCount = 0.0f;
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
            legalCount += legalMask[i];
        if (legalCount > 0.0f)
        {
            for (int i = 0; i < AZ_ACTION_SIZE; i++)
                output.policy[i] = legalMask[i] / legalCount;
        }
        output.value = 0.0f;
        return output;
    }

    // Layer 1: FC(847 -> 256) + ReLU
    std::vector<float> h1(HIDDEN1);
    linearForward(input.data(), INPUT_SIZE,
                  fc1_weight.data(), fc1_bias.data(),
                  h1.data(), HIDDEN1);
    relu(h1.data(), HIDDEN1);

    // Layer 2: FC(256 -> 128) + ReLU
    std::vector<float> h2(HIDDEN2);
    linearForward(h1.data(), HIDDEN1,
                  fc2_weight.data(), fc2_bias.data(),
                  h2.data(), HIDDEN2);
    relu(h2.data(), HIDDEN2);

    // Policy Head: FC(128 -> 60) + masked softmax
    std::vector<float> pLogits(AZ_ACTION_SIZE);
    linearForward(h2.data(), HIDDEN2,
                  policy_weight.data(), policy_bias.data(),
                  pLogits.data(), AZ_ACTION_SIZE);

    // 对非法动作施加大负数，然后 softmax
    float maxLogit = -1e30f;
    for (int i = 0; i < AZ_ACTION_SIZE; i++)
    {
        if (legalMask[i] < 0.5f)
            pLogits[i] = -1e9f;
        if (pLogits[i] > maxLogit)
            maxLogit = pLogits[i];
    }

    float expSum = 0.0f;
    for (int i = 0; i < AZ_ACTION_SIZE; i++)
    {
        pLogits[i] = std::exp(pLogits[i] - maxLogit);
        expSum += pLogits[i];
    }

    for (int i = 0; i < AZ_ACTION_SIZE; i++)
    {
        output.policy[i] = (expSum > 0.0f) ? pLogits[i] / expSum : 0.0f;
    }

    // Value Head: FC(128 -> 1) + Tanh
    float vRaw = value_bias[0];
    for (int i = 0; i < HIDDEN2; i++)
    {
        vRaw += value_weight[i] * h2[i];
    }
    output.value = std::tanh(vRaw);

    return output;
}
