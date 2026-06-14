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
//     --black-model <路径>  黑方专用模型
//     --white-model <路径>  白方专用模型
//     --black-sims-mult N   黑方 sims 倍数 (默认 1.0)
//     --temperature F       探索温度 (默认 1.0, 0.0=贪心)
//     --temperature-moves N  温度采样步数 (默认 10, 0=全程贪心)
//
// 模型配置:
//   --black-model + --white-model  → 双模型各自独立
//   --black-model only             → 黑=模型, 白=启发式 (league)
//   --white-model only             → 黑=启发式, 白=模型
//   model_path only                → 单模型双方共用 (标准 selfplay)
//   无模型                         → 纯启发式双方
//
// 参数:
//   start_step: 从第几步开始用 MCTS 搜索 (之前用贪心启发式)
//               50 = 只搜最后几步, 0 = 全程搜索
//
// 示例:
//   backward_selfplay 50 800 0 data/backward model.onnx              # 单模型
//   backward_selfplay 50 800 0 data/backward --black-model b.onnx --white-model w.onnx  # 双模型
//   backward_selfplay 20 800 0 data/backward --black-model m.onnx    # league: 黑=模型, 白=启发式

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: backward_selfplay <games> <simulations> <start_step> <output_dir> [model_path]\n"
                  << "       --black-model <path>  --white-model <path>  --black-sims-mult N\n"
                  << "\n"
                  << "  start_step: move index to start MCTS (before this: greedy heuristic)\n"
                  << "              50 = search last ~10 moves (endgame training)\n"
                  << "               0 = search all moves (full self-play)\n"
                  << "  --black-model: model for BLACK (can be used alone for league mode)\n"
                  << "  --white-model: model for WHITE (can be used alone for league mode)\n";
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
    bool evalBlack = false;
    float temperature = 1.0f;
    int temperatureMoves = 10;
    for (int i = 5; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--black-model" && i + 1 < argc)
            blackModelPath = argv[++i];
        else if (arg == "--white-model" && i + 1 < argc)
            whiteModelPath = argv[++i];
        else if (arg == "--black-sims-mult" && i + 1 < argc)
            blackSimsMult = std::atof(argv[++i]);
        else if (arg == "--eval-black")
            evalBlack = true;
        else if (arg == "--temperature" && i + 1 < argc)
            temperature = std::atof(argv[++i]);
        else if (arg == "--temperature-moves" && i + 1 < argc)
            temperatureMoves = std::atoi(argv[++i]);
        else if (arg[0] != '-')
            modelPath = arg;
    }

    std::cerr << "=== Backward Self-Play (BoxesZero) ===\n"
              << "  Games:       " << numGames << "\n"
              << "  Simulations: " << numSims << "\n"
              << "  Start step:  " << startStep << " (greedy before, MCTS after)\n"
              << "  Output:      " << outputDir << "\n";

    // 双模型 / 单侧模型模式: 按颜色独立设置评估器
    //   --black-model + --white-model → 双模型 (各自独立网络)
    //   --black-model only           → 黑=模型, 白=启发式 (league: 模型执黑 vs 弱对手)
    //   --white-model only           → 黑=启发式, 白=模型 (league: 模型执白 vs 弱对手)
    //   model_path only              → 单模型双方共用 (标准 selfplay)
    //   无模型                       → 纯启发式双方
    bool hasBlackModel = false, hasWhiteModel = false;

    if (!blackModelPath.empty())
    {
        ONNXEvaluator *blackEval = createOnnxEvaluator(blackModelPath);
        if (!blackEval)
        {
            std::cerr << "  Black model load failed: " << blackModelPath << "\n";
            return 2;
        }
        setEvaluatorBlack(blackEval);
        hasBlackModel = true;
    }

    if (!whiteModelPath.empty())
    {
        ONNXEvaluator *whiteEval = createOnnxEvaluator(whiteModelPath);
        if (!whiteEval)
        {
            std::cerr << "  White model load failed: " << whiteModelPath << "\n";
            return 2;
        }
        setEvaluatorWhite(whiteEval);
        hasWhiteModel = true;
    }

    // 单侧模型模式: 另一侧显式使用启发式评估器
    // 确保 selectEvaluatorForColor() 对无模型一侧正确切换 (而非沿用上一轮的评估器)
    if (hasBlackModel && !hasWhiteModel)
        setEvaluatorWhite(new HeuristicEvaluator());
    if (!hasBlackModel && hasWhiteModel)
        setEvaluatorBlack(new HeuristicEvaluator());

    if (hasBlackModel && hasWhiteModel)
        std::cerr << "  Evaluator:   Dual (BLACK=" << blackModelPath
                  << ", WHITE=" << whiteModelPath << ")\n";
    else if (hasBlackModel)
        std::cerr << "  Evaluator:   Black=" << blackModelPath << ", White=Heuristic\n";
    else if (hasWhiteModel)
        std::cerr << "  Evaluator:   Black=Heuristic, White=" << whiteModelPath << "\n";
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
    engine.setEvalBlack(evalBlack);

    // 设置 teacher 标识
    std::string teacher = (modelPath.empty() && blackModelPath.empty() && whiteModelPath.empty()) ? "heuristic" : "neuralnet";
    teacher += "_backward_st" + std::to_string(startStep) + "_" + std::to_string(numSims);
    engine.setTeacher(teacher);

    engine.runBackward(numGames, numSims, startStep, outputDir,
                       temperature, temperatureMoves);

    return 0;
}
