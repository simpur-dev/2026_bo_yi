#include "AI/az/az_selfplay.h"
#include "AI/az/az_evaluator.h"
#include <iostream>
#include <cstdlib>
#include <string>

// BoxesZero 反向训练自对弈工具
//
// 用法:
//   backward_selfplay <对局数> <模拟次数> <start_step> <输出目录> [权重文件]
//
// 参数:
//   start_step: 从第几步开始用 MCTS 搜索 (之前用贪心启发式)
//               60 = 只搜最后几步, 0 = 全程搜索 (等价于普通自对弈)
//
// 示例:
//   backward_selfplay 100 400 50 data/backward          # 终局训练: 只搜后10步
//   backward_selfplay 100 400 40 data/backward model.onnx  # 中盘训练: 搜后20步
//   backward_selfplay 100 400 0  data/backward model.onnx  # 全程训练

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: backward_selfplay <games> <simulations> <start_step> <output_dir> [model_path]\n"
                  << "\n"
                  << "  start_step: move index to start MCTS (before this: greedy heuristic)\n"
                  << "              50 = search last ~10 moves (endgame training)\n"
                  << "              30 = search last ~30 moves (midgame training)\n"
                  << "               0 = search all moves (full self-play)\n";
        return 1;
    }

    int numGames = std::atoi(argv[1]);
    int numSims = std::atoi(argv[2]);
    int startStep = std::atoi(argv[3]);
    std::string outputDir = argv[4];
    std::string modelPath = (argc > 5) ? argv[5] : "";

    std::cerr << "=== Backward Self-Play (BoxesZero) ===\n"
              << "  Games:       " << numGames << "\n"
              << "  Simulations: " << numSims << "\n"
              << "  Start step:  " << startStep << " (greedy before, MCTS after)\n"
              << "  Output:      " << outputDir << "\n";

    // 加载模型
    if (!modelPath.empty())
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

    // 设置 teacher 标识
    std::string teacher = modelPath.empty() ? "heuristic" : "neuralnet";
    teacher += "_backward_st" + std::to_string(startStep) + "_" + std::to_string(numSims);
    engine.setTeacher(teacher);

    engine.runBackward(numGames, numSims, startStep, outputDir);

    return 0;
}
