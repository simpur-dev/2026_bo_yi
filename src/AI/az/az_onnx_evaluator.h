#pragma once

// ONNX Runtime CNN 评估器
// 加载由 export_onnx.py 导出的 .onnx 模型进行推理
// 支持所有 CNN 架构 (resnet_s / resnet_m / resnet_l)
//
// 需要在编译时启用 ONNX Runtime:
//   cmake -DUSE_ONNX=ON ..
//
// ONNX Runtime 库应放置在 3rdparty/onnxruntime/ 目录下

#ifdef USE_ONNX

#include "az_types.h"
#include "az_encoder.h"
#include "az_evaluator.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

class ONNXEvaluator : public AZEvaluator
{
  public:
    ONNXEvaluator();
    ~ONNXEvaluator() override;

    // 加载 .onnx 模型文件
    bool loadModel(const std::string &onnxPath);

    // 是否已加载
    bool isModelLoaded() const { return loaded; }

    // 评估局面
    NetworkOutput evaluate(const Board &board, int currentPlayer) override;

  private:
    bool loaded = false;

    // ONNX Runtime 组件
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo;

    // 输入输出名
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    std::vector<std::string> inputNamesOwned;
    std::vector<std::string> outputNamesOwned;
};

// 便捷函数：尝试加载 ONNX 模型
bool tryLoadOnnxModel(const std::string &onnxPath);

#else

// ONNX 未启用时的空实现
inline bool tryLoadOnnxModel(const std::string &) { return false; }

#endif // USE_ONNX
