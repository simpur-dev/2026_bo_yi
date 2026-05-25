#include "AI/az/az_selfplay.h"
#include "AI/az/az_evaluator.h"
#include "AI/az/az_onnx_evaluator.h"
#include <iostream>
#include <cstdlib>
#include <string>

// BoxesZero 反向训练自对弈工具
//
// 用法:
//   backward_selfplay <对局数> <模拟次数> <start_step> <输出目录> [模型路径]
//     --black-model <路径>  黑方专用模型 (双模型模式)
//     --white-model <路径>  白方专用模型 (双模型模式)
//
// 参数:
//   start_step: 从第几步开始用 MCTS 搜索 (之前用贪心启发式)
//               50 = 只搜最后几步, 0 = 全程搜索
//
// 示例:
//   backward_selfplay 50 800 0 data/backward model.onnx              # 单模型
//   backward_selfplay 50 800 0 data/backward --black-model b.onnx --white-model w.onnx  # 双模型

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: backward_selfplay <games> <simulations> <start_step> <output_dir> [model_path]\n"
                  << "       --black-model <path>  --white-model <path>\n"
                  << "\n"
                  << "  start_step: move index to start MCTS (before this: greedy heuristic)\n"
                  << "              50 = search last ~10 moves (endgame training)\n"
                  << "               0 = search all moves (full self-play)\n"
                  << "  --black-model: model for BLACK player (dual model mode)\n"
                  << "  --white-model: model for WHITE player (dual model mode)\n";
        return 1;
    }

    int numGames = std::atoi(argv[1]);
    int numSims = std::atoi(argv[2]);
    int startStep = std::atoi(argv[3]);
    std::string outputDir = argv[4];

    // 解析可选参数
    std::string modelPath;
    std::string blackModelPath, whiteModelPath;
    float blackSimsMult = 1.0f;
    for (int i = 5; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--black-model" && i + 1 < argc)
            blackModelPath = argv[++i];
        else if (arg == "--white-model" && i + 1 < argc)
            whiteModelPath = argv[++i];
        else if (arg == "--black-sims-mult" && i + 1 < argc)
            blackSimsMult = std::atof(argv[++i]);
        else if (arg[0] != '-')
            modelPath = arg;
    }

    std::cerr << "=== Backward Self-Play (BoxesZero) ===\n"
              << "  Games:       " << numGames << "\n"
              << "  Simulations: " << numSims << "\n"
              << "  Start step:  " << startStep << " (greedy before, MCTS after)\n"
              << "  Output:      " << outputDir << "\n";

    // 双模型模式: 分别创建独立评估器实例
    if (!blackModelPath.empty() && !whiteModelPath.empty())
    {
        ONNXEvaluator *blackEval = createOnnxEvaluator(blackModelPath);
        if (!blackEval)
        {
            std::cerr << "  Black model load failed: " << blackModelPath << "\n";
            return 2;
        }
        setEvaluatorBlack(blackEval);

        ONNXEvaluator *whiteEval = createOnnxEvaluator(whiteModelPath);
        if (!whiteEval)
        {
            std::cerr << "  White model load failed: " << whiteModelPath << "\n";
            return 2;
        }
        setEvaluatorWhite(whiteEval);

        std::cerr << "  Evaluator:   Dual (BLACK=" << blackModelPath
                  << ", WHITE=" << whiteModelPath << ")\n";
    }
    // 单模型模式: 向后兼容
    else if (!modelPath.empty())
    {
        if (tryLoadModel(modelPath))
            std::cerr << "  Evaluator:   Model (" << modelPath << ")\n";
        else
        {
            std::cerr << "  Evaluator:   Model load failed: " << modelPath << "\n";
            return 2;
        }
    }
    else
    {
        std::cerr << "  Evaluator:   Heuristic\n";
    }

    std::cerr << "\n";

    SelfPlayEngine engine;
    engine.setBlackSimsMultiplier(blackSimsMult);

    // 设置 teacher 标识
    std::string teacher = (modelPath.empty() && blackModelPath.empty()) ? "heuristic" : "neuralnet";
    teacher += "_backward_st" + std::to_string(startStep) + "_" + std::to_string(numSims);
    engine.setTeacher(teacher);

    engine.runBackward(numGames, numSims, startStep, outputDir);

    return 0;
}
