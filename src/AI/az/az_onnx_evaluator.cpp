#ifdef USE_ONNX

#include "az_onnx_evaluator.h"
#include "az_action.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// ========== ONNXEvaluator 实现 ==========

ONNXEvaluator::ONNXEvaluator()
    : env(ORT_LOGGING_LEVEL_WARNING, "DotsAndBoxes"),
      memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
}

ONNXEvaluator::~ONNXEvaluator() = default;

bool ONNXEvaluator::loadModel(const std::string &onnxPath)
{
    try
    {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 优先使用 CUDA GPU 推理，失败则回退 CPU
        try
        {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            cudaOptions.arena_extend_strategy = 0;
            cudaOptions.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            std::cerr << "[ONNXEvaluator] CUDA provider enabled (GPU 0)\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ONNXEvaluator] CUDA unavailable, falling back to CPU: " << e.what() << "\n";
        }

#ifdef _WIN32
        // Windows 需要宽字符路径
        std::wstring widePath(onnxPath.begin(), onnxPath.end());
        session = std::make_unique<Ort::Session>(env, widePath.c_str(), sessionOptions);
#else
        session = std::make_unique<Ort::Session>(env, onnxPath.c_str(), sessionOptions);
#endif

        // 获取输入输出名
        Ort::AllocatorWithDefaultOptions allocator;

        // 输入名
        inputNamesOwned.clear();
        inputNames.clear();
        size_t numInputs = session->GetInputCount();
        for (size_t i = 0; i < numInputs; i++)
        {
            auto name = session->GetInputNameAllocated(i, allocator);
            inputNamesOwned.push_back(std::string(name.get()));
        }
        for (auto &s : inputNamesOwned)
            inputNames.push_back(s.c_str());

        // 输出名
        outputNamesOwned.clear();
        outputNames.clear();
        size_t numOutputs = session->GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++)
        {
            auto name = session->GetOutputNameAllocated(i, allocator);
            outputNamesOwned.push_back(std::string(name.get()));
        }
        for (auto &s : outputNamesOwned)
            outputNames.push_back(s.c_str());

        loaded = true;

        std::cerr << "[ONNXEvaluator] Loaded model: " << onnxPath << "\n";
        std::cerr << "[ONNXEvaluator] Inputs: " << numInputs;
        for (auto &n : inputNamesOwned) std::cerr << " " << n;
        std::cerr << "\n";
        std::cerr << "[ONNXEvaluator] Outputs: " << numOutputs;
        for (auto &n : outputNamesOwned) std::cerr << " " << n;
        std::cerr << "\n";

        return true;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[ONNXEvaluator] Failed to load model: " << e.what() << "\n";
        loaded = false;
        return false;
    }
}

NetworkOutput ONNXEvaluator::evaluate(const Board &board, int currentPlayer)
{
    NetworkOutput output;

    if (!loaded || !session)
    {
        // 回退: 均匀 policy
        auto mask = getLegalMask(board);
        float count = 0.0f;
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
            count += mask[i];
        if (count > 0.0f)
            for (int i = 0; i < AZ_ACTION_SIZE; i++)
                output.policy[i] = mask[i] / count;
        output.value = 0.0f;
        return output;
    }

    // 1. 编码棋盘 -> (7, 11, 11) flat array
    Tensor inputTensor = encodeBoard(board, currentPlayer);

    // 2. 准备输入: shape = [1, 7, 11, 11]
    std::array<int64_t, 4> inputShape = {1, AZ_CHANNELS, AZ_BOARD_SIZE, AZ_BOARD_SIZE};
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensor.data(),
        inputTensor.size(),
        inputShape.data(),
        inputShape.size());

    // 3. 推理
    try
    {
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputOrt,
            1,
            outputNames.data(),
            outputNames.size());

        // 4. 解析输出
        // output[0] = policy_logits: shape [1, 60]
        // output[1] = value: shape [1, 1]
        float *policyLogits = outputTensors[0].GetTensorMutableData<float>();
        float *valuePtr = outputTensors[1].GetTensorMutableData<float>();

        // 5. Masked softmax on policy
        auto legalMask = getLegalMask(board);

        float maxLogit = -1e30f;
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
        {
            if (legalMask[i] < 0.5f)
                policyLogits[i] = -1e9f;
            if (policyLogits[i] > maxLogit)
                maxLogit = policyLogits[i];
        }

        float expSum = 0.0f;
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
        {
            policyLogits[i] = std::exp(policyLogits[i] - maxLogit);
            expSum += policyLogits[i];
        }

        for (int i = 0; i < AZ_ACTION_SIZE; i++)
        {
            output.policy[i] = (expSum > 0.0f) ? policyLogits[i] / expSum : 0.0f;
        }

        // 6. Value (already tanh'd by the model)
        output.value = valuePtr[0];
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "[ONNXEvaluator] Inference error: " << e.what() << "\n";
        // 回退
        auto mask = getLegalMask(board);
        float count = 0.0f;
        for (int i = 0; i < AZ_ACTION_SIZE; i++)
            count += mask[i];
        if (count > 0.0f)
            for (int i = 0; i < AZ_ACTION_SIZE; i++)
                output.policy[i] = mask[i] / count;
        output.value = 0.0f;
    }

    return output;
}

// ========== 便捷函数 ==========

static ONNXEvaluator *onnxInstance = nullptr;

bool tryLoadOnnxModel(const std::string &onnxPath)
{
    if (!onnxInstance)
        onnxInstance = new ONNXEvaluator();

    if (onnxInstance->loadModel(onnxPath))
    {
        setEvaluator(onnxInstance);
        std::cerr << "[AZ] Switched to ONNXEvaluator\n";
        return true;
    }
    else
    {
        std::cerr << "[AZ] Failed to load ONNX model, using current evaluator\n";
        return false;
    }
}

ONNXEvaluator *createOnnxEvaluator(const std::string &onnxPath)
{
    ONNXEvaluator *evaluator = new ONNXEvaluator();
    if (evaluator->loadModel(onnxPath))
    {
        std::cerr << "[AZ] Created ONNXEvaluator for: " << onnxPath << "\n";
        return evaluator;
    }
    else
    {
        std::cerr << "[AZ] Failed to create ONNXEvaluator for: " << onnxPath << "\n";
        delete evaluator;
        return nullptr;
    }
}

#endif // USE_ONNX
