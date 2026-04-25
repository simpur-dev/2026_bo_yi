#include "AI/az/az_selfplay.h"
#include "AI/az/az_evaluator.h"
#include <iostream>
#include <cstdlib>
#include <string>

// 自对弈训练数据生成工具
//
// 用法:
//   selfplay [对局数] [模拟次数] [输出目录] [权重文件]
//
// 示例:
//   selfplay                          # 默认: 100盘, 800次模拟
//   selfplay 500                      # 500盘
//   selfplay 500 400                  # 500盘, 400次模拟
//   selfplay 500 400 data/selfplay    # 指定输出目录
//   selfplay 500 800 data/selfplay model.bin  # 加载神经网络权重

int main(int argc, char *argv[])
{
    int numGames = 100;
    int numSims = 800;
    std::string outputDir = "data/selfplay";
    std::string weightPath = "";

    if (argc > 1)
        numGames = std::atoi(argv[1]);
    if (argc > 2)
        numSims = std::atoi(argv[2]);
    if (argc > 3)
        outputDir = argv[3];
    if (argc > 4)
        weightPath = argv[4];

    std::cerr << "=== Dots & Boxes Self-Play ===\n"
              << "  Games:       " << numGames << "\n"
              << "  Simulations: " << numSims << "\n"
              << "  Output:      " << outputDir << "\n";

    // 尝试加载神经网络权重
    if (!weightPath.empty())
    {
        if (tryLoadNeuralNet(weightPath))
            std::cerr << "  Evaluator:   NeuralNet (" << weightPath << ")\n";
        else
            std::cerr << "  Evaluator:   Heuristic (weight load failed)\n";
    }
    else
    {
        std::cerr << "  Evaluator:   Heuristic\n";
    }

    std::cerr << "\n";

    SelfPlayEngine engine;
    engine.run(numGames, numSims, outputDir);

    return 0;
}
